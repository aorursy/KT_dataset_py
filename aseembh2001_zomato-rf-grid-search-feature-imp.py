# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Print multiple lines within same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Show all columns and rows
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import os
import itertools
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
zomato=pd.read_csv("../input/zomato.csv",encoding='latin1')
# Size of data and how it looks
zomato.shape
zomato.head()
# How many missing values and where
# How many missing values an which column
zomato.isnull().values.sum()
zomato.isnull().any()[zomato.isnull().any()==True]
# Cuisines is the only feature which has null values
# removing rows with missing values
zomato.dropna(inplace=True)
# Let's remove the features we do not want to use for our modelling
relvnt=zomato.drop(['Restaurant ID', 'Restaurant Name',
                        'Address','Locality Verbose', 'Longitude', 'Latitude', 'Rating color','Aggregate rating'],axis=1)
# Let's find out variables that are categorical:
categorical=[]
for i in relvnt.columns:
    if relvnt[i].dtype=='object':
        categorical.append(i)
# Which variables are categorical ?
print(categorical)
for i in categorical:
    if i!='Rating text':
        relvnt=pd.concat([relvnt,pd.get_dummies(relvnt[i],prefix=i)],axis=1)
        relvnt.drop(i,axis=1,inplace=True)
target=relvnt['Rating text']
target_map={'Excellent':1, 'Very Good':2, 'Good':3, 'Average':4, 'Not rated':5, 'Poor':6}
target=target.apply(lambda x: target_map[x])
predictors=relvnt.drop('Rating text',axis=1)
target=relvnt['Rating text']
sns.countplot(target)
# Splitting data into train and test set
X_train,X_test,Y_train,Y_test=train_test_split(predictors,target,test_size=.33)

from sklearn.model_selection import GridSearchCV

# Create Grid search funtion
def rfzomato(X,Y,nfolds):
    n_est=[100,300,500]
    min_samples_split=[10,50,100]
    param_grid={'n_estimators':n_est,'min_samples_split':min_samples_split}
    grid_search=GridSearchCV(RandomForestClassifier(random_state=42,n_jobs=-1),cv=nfolds,param_grid=param_grid)
    grid_search.fit(X,Y)
    return grid_search.best_params_
m=rfzomato(X_train,Y_train,5)
randFor=RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=m['n_estimators'],min_samples_split=m['min_samples_split'])
randFor.fit(X_train,Y_train)
Y_pred=randFor.predict(X_test)

acc_score=accuracy_score(Y_test,Y_pred)
cnf_mat=confusion_matrix(Y_test,Y_pred)
print ("accuracy score",acc_score)

# Function to plot confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    target_map={1:'Excellent',2:'Very Good', 3:'Good', 4:'Average', 5:'Not rated', 6:'Poor'}
    classes=pd.Series(classes)
    classes=classes.apply(lambda x: target_map[x])            
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot confusion matrix 
plt.figure()
plot_confusion_matrix(cnf_mat, classes=target.unique(),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_mat, classes=target.unique(), normalize=True,
                      title='Normalized confusion matrix')

plt.show()