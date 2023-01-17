# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv',header=0)
data.head()

data.info()
data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself 


data.drop("id",axis=1,inplace=True)

#check that there really are no missing values
data.isnull().sum()

# converting diagnosis to dummy variables
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.describe()

import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
%matplotlib inline
response = ['diagnosis']
 # list of variables who are NOT predictors
predictors=[x for x in list(data.columns) if x not in response] # building a list a predictors
data=data[response+predictors] # excluding variabes which we are not going to use

train = data.sample(frac=0.5, random_state=1)
test = data[data.index.isin(train.index)==False].copy()

#Normalise the data
mu=train[predictors].mean()
sigma=train[predictors].std()

train[predictors]=(train[predictors]-mu)/sigma
test[predictors]=(test[predictors]-mu)/sigma
#boxplot to see malignant vs begign, where 0 is malignant and 1 is begnign 
sns.countplot(train['diagnosis'],label="Count")
#Violin plot of means
# took the idea from : https://www.kaggle.com/kanncaa1/feat-select-corr-rfe-rfecv-pca-seaborn-randforest
train_mean = train[predictors]
train_mean = pd.concat([train[response],train[predictors].iloc[:,0:10]],axis=1)
train_mean = pd.melt(train_mean,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=train_mean,split=True, inner="quart")
plt.xticks(rotation=90)
#Violin plot of standard errors
train_se = train[predictors]
train_se = pd.concat([train[response],train[predictors].iloc[:,11:20]],axis=1)
train_se = pd.melt(train_se,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=train_se,split=True, inner="quart")
plt.xticks(rotation=90)
#Violin plot of worst case
train_worst = train[predictors]
train_worst = pd.concat([train[response],train[predictors].iloc[:,21:30]],axis=1)
train_worst = pd.melt(train_worst,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=train_worst,split=True, inner="quart")
plt.xticks(rotation=90)
#see any correlations
corr_matrix = train.corr()
corr_matrix
#visual representation, the more red, the higher the correlation
import statsmodels.api as sm
N, M = 15, 15
fig, ax = plt.subplots(figsize=(N, M))
sm.graphics.plot_corr(corr_matrix, xnames = train.columns, ax=ax)
plt.show
features = ['radius_mean', 'perimeter_mean', 'radius_worst', 'perimeter_worst', 'concave points_worst']
#These features appear to have a high correlation with diagnosis

for feature in features:
    plt.figure()
    sns.regplot(x="diagnosis", y=feature, data=train)

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Create the RFE object and rank each pixel    

rfe = RFECV(estimator=RandomForestClassifier() , step=1, cv=5,scoring='accuracy') 
rfe = rfe.fit(train[predictors], np.ravel(train[response]))

print('Number of predictors :', rfe.n_features_)
print('Best features :', train[predictors].columns[rfe.support_])

#Visually see how the optimal number was found
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation scores")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()

best_p = train[predictors].columns[rfe.support_] #best predictors
train[best_p]



import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import accuracy_score

#Confusion matrix function
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-
#auto-examples-model-selection-plot-confusion-matrix-py

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
columns=['Accuracy']
rows=['1R','Random Forrest', 'Random Forest FS', 'DT',
     'NB', '3NN','SVM']

results=pd.DataFrame(0.0, columns=columns, index=rows) 
count = 0


 # for the check the error and accuracy of the model
from sklearn import tree

r = tree.DecisionTreeClassifier(random_state=0, max_depth=1)
r.fit(train[predictors], train[response])
prediction=r.predict(test[predictors])# predict for the test data
results.iloc[count] = metrics.accuracy_score(prediction.round(),test[response])
count+=1

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train[predictors],train[response])
prediction=model.predict(test[predictors])# predict for the test data
results.iloc[count] =metrics.accuracy_score(prediction,test[response]) # to check the accuracy
count+=1


cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
#random forest with feature selection
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train[best_p],train[response])
prediction=model.predict(test[best_p])# predict for the test data
results.iloc[count] =metrics.accuracy_score(prediction,test[response]) # to check the accuracy
count+=1

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
dt = tree.DecisionTreeClassifier(random_state=0)
dt.fit(train[predictors], train[response])
prediction=dt.predict(test[predictors])# predict for the test data
results.iloc[count] = metrics.accuracy_score(prediction.round(),test[response])
count+=1

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(train[predictors], train[response])
prediction=gnb.predict(test[predictors])# predict for the test data
results.iloc[count] = metrics.accuracy_score(prediction.round(),test[response])
count+=1

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train[predictors], train[response])
prediction=neigh.predict(test[predictors])# predict for the test data
results.iloc[count] = metrics.accuracy_score(prediction.round(),test[response])
count+=1

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
from sklearn import svm 
model = svm.SVC()
model.fit(train[best_p], np.ravel(train[response]))
prediction=model.predict(test[best_p])
results.iloc[count] = metrics.accuracy_score(prediction,test[response])
count+=1

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
results.round(3)
from sklearn.dummy import DummyClassifier
model = DummyClassifier(strategy='most_frequent',random_state=0)
print(model.fit(train[predictors], np.ravel(train[response])))
prediction=model.predict(test[predictors])
metrics.accuracy_score(prediction,test[response])

cnf_matrix = confusion_matrix(test[response], prediction)
np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['malignant','begnign'], normalize=True,
                      title='Confusion matrix')

plt.show()
metrics.accuracy_score(prediction,test[response])
