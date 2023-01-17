# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# function read a text file & generate a DataFrame

# Input file is a text file with values separated using single space

# Function reads the text files and returns it as DataFrame



def read_files(file):

    

    df = pd.read_csv(file, sep = ' ', header = None)

    

    return df
# Genrating dataframes using the function defined earlier



X_train = read_files('/kaggle/input/human-activities-and-postural-transitions-data-set/X_train.txt')

y_train = read_files('/kaggle/input/human-activities-and-postural-transitions-data-set/y_train.txt')



X_test = read_files('/kaggle/input/human-activities-and-postural-transitions-data-set/X_test.txt')

y_test = read_files('/kaggle/input/human-activities-and-postural-transitions-data-set/y_test.txt')
X_train.head()
# generating list of features from text file



headers = []



with open('/kaggle/input/human-activities-and-postural-transitions-data-set/features.txt', 'r') as f:

    

    for each in f:

        

        headers.append(each)
# renaming the headers with roiginal column names



X_train.columns = headers

X_test.columns = headers
X_train.head()
# checking sample from the dataframe



X_train.describe(include = 'all').T[50:60]
# Checking for th dtype of all the columns - all are float 



X_train.dtypes.value_counts()
import seaborn as sns

%matplotlib inline



import matplotlib.pyplot as plt
# Label list and description from Textfile





label_desc = []

y_labels = []



with open('/kaggle/input/human-activities-and-postural-transitions-data-set/activity_labels.txt', 'r') as f:

    

    for each in f:

        

        label_desc.append(each.strip())



        

        

for each in label_desc:

    y_labels.append(each.split(' '))

        

        

print(label_desc)

print(y_labels)
# CHecking for the class representation -- there seem to be a class imbalance for 6 categories

# Ignore it for now and calculate the metrics

# Performance metircs can be compared after applying SMOTE, imblearn, etc.,,



y_train.columns = ['Activity']

y_train['Activity'].value_counts()
# Appending Activity name to the columns 



y_train['Activity_Name'] = y_train['Activity'].apply(lambda x : dict(y_labels)[str(x)])
y_train.head()
import matplotlib.gridspec as gridspec



import matplotlib.style as style



style.use('fivethirtyeight')



def plotting_charts(df, feature):

    

    plt.figure(constrained_layout = True, figsize = (12,8))



    ax = sns.stripplot(y_train['Activity_Name'], df.loc[:,feature], jitter = True)

    

    plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='large')
for i in range(10):

    plotting_charts(X_train, X_train.columns[i])

    
plt.figure(figsize = (12,8))

sns.swarmplot(y_train['Activity_Name'], X_train.iloc[:,3])

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

    fontsize='large')
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



# importing plain classifier with default params



sgd_clf = SGDClassifier(random_state = 42)

lg_reg = LogisticRegression(random_state = 42)
print(cross_val_score(sgd_clf, X_train, y_train['Activity'], cv = 5))
# checking for score with CV of 10



cross_val_score(lg_reg, X_train, y_train['Activity'], cv = 10)
# importing performance metrics



from sklearn.model_selection import cross_val_predict



from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
y_pred = cross_val_predict(lg_reg, X_train, y_train['Activity'], cv = 10)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score





print(classification_report(y_train['Activity'], y_pred))

print(confusion_matrix(y_train['Activity'], y_pred))
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier


#Creating a pipelibe of 2 classifier without PCA



pipe_wo_PCA = Pipeline([('classifier', LogisticRegression())])





# creating grid for search



param_grid = [

    {'classifier' : [LogisticRegression()],

    'classifier__penalty' : ['l1', 'l2'],

     'classifier__C' : np.logspace(-4, 4, 20),

     'classifier__solver' : ['liblinear']},

    {'classifier' : [RandomForestClassifier()],

    'classifier__n_estimators' : list(range(100,1000,100)),

    'classifier__max_features' : ['auto', 'log2', 'sqrt']}]





# Searching for best grid using gridsearch CV



clf = GridSearchCV(pipe_wo_PCA, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
clf.fit(X_train, y_train['Activity'])
best_clf = clf.best_estimator_
y_pred_best_lr = best_clf.predict(X_train)



print(accuracy_score(y_train['Activity'], y_pred_best_lr))

print(classification_report(y_train['Activity'], y_pred_best_lr))

print(confusion_matrix(y_train['Activity'], y_pred_best_lr))
y_test_pred_wo_pca = best_clf.predict(X_test)



print(accuracy_score(y_test, y_test_pred_wo_pca))

print(classification_report(y_test, y_test_pred_wo_pca))

print(confusion_matrix(y_test, y_test_pred_wo_pca))
import seaborn as sns



# Confusion matrix on the train data using sns.heatmap

plt.figure(figsize = (12,10))

sns.heatmap(confusion_matrix(y_train['Activity'], y_pred_best_lr), 

            cbar = False, 

            annot = True, 

            fmt = 'd', 

            xticklabels = dict(y_labels).values(), 

            yticklabels = dict(y_labels).values(),

           cmap="winter" )
# Pipeline with PCA





pipe_w_PCA = Pipeline([('pca', PCA()), ('classifier', LogisticRegression())])





# creating grid for search



param_grid = [{'pca__n_components': [4, 16, 32, 62, 128, 256],

              'classifier__penalty' : ['l1', 'l2'],

               'classifier__C' : np.logspace(-4, 4, 20),

               'classifier__solver' : ['liblinear']}]
clf_pca = GridSearchCV(pipe_w_PCA, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

clf_pca.fit(X_train, y_train['Activity'])

best_clf_pca = clf_pca.best_estimator_



best_clf_pca.fit(X_train, y_train['Activity'])



y_pred_best_lr_w_pca = best_clf_pca.predict(X_train)
print(accuracy_score(y_train['Activity'], y_pred_best_lr_w_pca))

print(classification_report(y_train['Activity'], y_pred_best_lr_w_pca))

print(confusion_matrix(y_train['Activity'], y_pred_best_lr_w_pca))
# Confusion Matrix for the model using PCA & Logistic Regression



plt.figure(figsize = (12,10))

sns.heatmap(confusion_matrix(y_train['Activity'], y_pred_best_lr_w_pca), 

            cbar = False, 

            annot = True, 

            fmt = 'd', 

            xticklabels = dict(y_labels).values(), 

            yticklabels = dict(y_labels).values(),

           cmap="winter" )
y_test_pred_w_pca = best_clf_pca.predict(X_test)



print(accuracy_score(y_test, y_test_pred_w_pca ))

print(classification_report(y_test, y_test_pred_w_pca))

print(confusion_matrix(y_test, y_test_pred_w_pca))



# 



plt.figure(figsize = (12,10))

sns.heatmap(confusion_matrix(y_test, y_test_pred_w_pca), 

            cbar = False, 

            annot = True, 

            fmt = 'd', xticklabels = dict(y_labels).values(), 

            yticklabels = dict(y_labels).values(),   

            cmap="Blues_r" )