# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns
sns.set(style='darkgrid')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
traindf=pd.read_csv("/kaggle/input/novartis-data/Train.csv")
testdf=pd.read_csv("/kaggle/input/novartis-data/Test.csv")
submissiondf=pd.read_csv("/kaggle/input/novartis-data/sample_submission.csv")
print(traindf.shape)
traindf.head()
traindf.info()
print(testdf.shape)
testdf.head()
# Multiple_offense is the target varaible, lets see how it is populated
## Plotting the bar char to identify the frequnecy of values
sns.countplot(traindf["MULTIPLE_OFFENSE"],color='black')
##prinitng number of values for each type
print(traindf["MULTIPLE_OFFENSE"].value_counts())

# Analysing NULL values in the features
for col in traindf.columns:
    if traindf[col].isnull().values.any():
        print(f"Train Dataset Feature - {col} contains {traindf[col].isna().sum()*100/traindf[col].sum()}% of Null Values")
    
for col in testdf.columns:
    if testdf[col].isnull().values.any():
        print(f"Test Dataset Feature - {col} contains {testdf[col].isna().sum()*100/testdf[col].sum()}% of Null Values")
tempdf=traindf.loc[traindf["X_12"].isnull()==True]["MULTIPLE_OFFENSE"]
sns.countplot(tempdf,color='black')
print(traindf.shape)
traindf=traindf.dropna(axis=0, subset=['X_12'])
print(traindf.shape)
traindf["DATE"].loc[0:5]
traindf['DATE'] = pd.to_datetime(traindf.DATE)
traindf['DATE'].head()
def Get_feature_from_DATE(df):
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day
    df['YEAR'] = df['DATE'].dt.year
    df['DAYOFWEEK'] = df['DATE'].dt.dayofweek
    df['WEEKEND'] = np.where(df['DATE'].dt.day_name().isin(['Sunday','Saturday']),1,0)
    df=df.drop(['DATE'],axis=1)
    return df
traindf=Get_feature_from_DATE(traindf)
traindf.head()
traindf.describe()
Numerical_features=['X_1','X_2','X_3','X_4','X_5','X_6']
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for variable, subplot in zip(Numerical_features, ax.flatten()):
    sns.boxplot(traindf[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(0)
Numerical_features=['X_7','X_8','X_9','X_10','X_11','X_12']
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for variable, subplot in zip(Numerical_features, ax.flatten()):
    sns.boxplot(traindf[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(0)
Numerical_features=['X_6','X_7','X_8','X_9','X_11']
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for variable, subplot in zip(Numerical_features, ax.flatten()):
    sns.distplot(traindf[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(0)
#removing outliers from X_8 variable
print(traindf.shape)
traindf = traindf[~((traindf['X_8']>6))]
print(traindf.shape)
#removing outliers from X_10 variable
print(traindf.shape)
traindf = traindf[~((traindf['X_10']>10))]
print(traindf.shape)
Numerical_features=['MONTH','DAY','YEAR','DAYOFWEEK','WEEKEND']
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for variable, subplot in zip(Numerical_features, ax.flatten()):
    sns.distplot(traindf[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(0)
traindf['X_1'].plot.hist()
print(traindf['X_1'].value_counts())
traindf["X_1_new"]=traindf["X_1"].apply(lambda x : x if x ==0  else 1)

traindf["X_1_new"].plot.hist()
traindf['X_10'].plot.hist()
print(traindf['X_10'].value_counts())
traindf["X_10_new"]=traindf["X_10"].apply(lambda x : x if x ==1  else 2)
traindf["X_10_new"].plot.hist()
traindf['X_12'].plot.hist()
print(traindf['X_12'].value_counts())
traindf["X_12_new"]=traindf["X_12"].apply(lambda x : x if x ==1  else 2)
traindf["X_12_new"].plot.hist()
# X_2 and X_3 are highly corelated
traindf.drop(['X_1','X_10','X_12','INCIDENT_ID'], axis=1, inplace=True)
traindf.head()
fig, ax = plt.subplots(figsize=(25,10))         # Sample figsize in inches
matrix = np.triu(traindf.corr())
sns.heatmap(traindf.corr(), annot=True, mask=matrix,linewidths=.5, ax=ax)
#Dropping those columns
traindf.drop(['X_3','X_1_new','DAYOFWEEK','X_7'], axis=1, inplace=True)
traindf.head()
fig, ax = plt.subplots(figsize=(25,10))         # Sample figsize in inches
matrix = np.triu(traindf.corr())
sns.heatmap(traindf.corr(), annot=True, mask=matrix,linewidths=.5, ax=ax)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

x = traindf.drop(["X_5","X_6","X_9","X_13","MONTH","DAY","MULTIPLE_OFFENSE"], axis=1)
y = traindf["MULTIPLE_OFFENSE"]
y.plot.hist()
y.value_counts()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
LR= LogisticRegression(penalty='none')
LR= LR.fit(x_train,y_train)
Y_pred = LR.predict(x_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, Y_pred), "\n")
print("accuracy", metrics.accuracy_score(y_test, Y_pred))
print("precision", metrics.precision_score(y_test,Y_pred))
print("recall", metrics.recall_score(y_test,Y_pred,average='binary'))
confusion=confusion_matrix(y_test,Y_pred)    
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("Sensitivity",TP / float(TP+FN))
# positive predictive value 
print ("Positive Predection Rate",TP / float(TP+FP))
# Negative predictive value
print ("Negative Predection rate",TN / float(TN+ FN))
# Calculate false postive rate - predicting churn when customer does not have churned
print("False positive Predection Rate",FP/ float(TN+FP))
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
seed = 7
# prepare models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisonTree', DecisionTreeClassifier()))
models.append(('NB', BernoulliNB()))
models.append(('SVM', SVC()))
models.append(('BaggingDecisonTree', BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))))
models.append(('Adaboost',AdaBoostClassifier()))
models.append(('Logistic', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('GradientBoosting', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)))

# evaluate each model in turn
results = []
names = []
performance=[]
scoring = 'recall_macro'
for name, model in models:
	kfold = model_selection.KFold(n_splits=15, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x_train,y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	performance.append(msg)
# boxplot algorithm comparison
fig = plt.figure(figsize=(20,8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results,widths = 0.5)
ax.set_xticklabels(names)
plt.show()

for perf in performance:
    print(perf)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, Y_pred), "\n")
print("accuracy", metrics.accuracy_score(y_test, Y_pred))
print("precision", metrics.precision_score(y_test,Y_pred))
print("recall", metrics.recall_score(y_test,Y_pred,average='binary'))
confusion=confusion_matrix(y_test,Y_pred)    
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("Sensitivity",TP / float(TP+FN))
# positive predictive value 
print ("Positive Predection Rate",TP / float(TP+FP))
# Negative predictive value
print ("Negative Predection rate",TN / float(TN+ FN))
# Calculate false postive rate - predicting churn when customer does not have churned
print("False positive Predection Rate",FP/ float(TN+FP))
from sklearn.ensemble import BaggingClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True)
bag_model=bag_model.fit(x_train,y_train)
Y_pred=bag_model.predict(x_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, Y_pred), "\n")
print("accuracy", metrics.accuracy_score(y_test, Y_pred))
print("precision", metrics.precision_score(y_test,Y_pred))
print("recall", metrics.recall_score(y_test,Y_pred,average='binary'))
confusion=confusion_matrix(y_test,Y_pred)    
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("Sensitivity",TP / float(TP+FN))
# positive predictive value 
print ("Positive Predection Rate",TP / float(TP+FP))
# Negative predictive value
print ("Negative Predection rate",TN / float(TN+ FN))
# Calculate false postive rate - predicting churn when customer does not have churned
print("False positive Predection Rate",FP/ float(TN+FP))
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2)
clf.fit(x_train, y_train)
Y_pred=clf.predict(x_test)
from sklearn import metrics
print(metrics.confusion_matrix(y_test, Y_pred), "\n")
print("accuracy", metrics.accuracy_score(y_test, Y_pred))
print("precision", metrics.precision_score(y_test,Y_pred))
print("recall", metrics.recall_score(y_test,Y_pred,average='binary'))
confusion=confusion_matrix(y_test,Y_pred)    
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("Sensitivity",TP / float(TP+FN))
# positive predictive value 
print ("Positive Predection Rate",TP / float(TP+FP))
# Negative predictive value
print ("Negative Predection rate",TN / float(TN+ FN))
# Calculate false postive rate - predicting churn when customer does not have churned
print("False positive Predection Rate",FP/ float(TN+FP))
