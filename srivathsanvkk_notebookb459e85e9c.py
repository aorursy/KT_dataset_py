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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
data=pd.read_csv("/kaggle/input/income/income.csv")
data.head()
data.shape
def value(a):
    print("counts for {}".format(a))
    print(data[a].value_counts())
value('workclass')
value('education')        #To check where are '?' is found in the dataset and to get a clear picture of the values in features
value('marital-status')                                             
value('occupation')
value('relationship')
value('race')
value('gender')
value('capital-gain')
value('capital-loss')
value('native-country')
#replacing '?' with 'nan' so that we can remove null values
data['native-country']=data['native-country'].replace('?',np.nan)
data['workclass']=data['workclass'].replace('?',np.nan)
data['occupation']=data['occupation'].replace('?',np.nan)
#checking null values in the dataset
data.isnull().sum()
#removing null values in the dataset
data.dropna(inplace=True)
#null values free dataset
data.isnull().sum()
#shape of the dataset after cleaning the dataset
data.shape
# removing capital loss and capital gain from the dataset as they both contain around 89% zeros in their columns
data=data.drop(['capital-gain','capital-loss'],axis=1)
# Since USA is predominant around 90% in native-country feature, i am categorising all other country as REST OF WORLD to avoid complexity 
for i in data['native-country']:
    if (i!='United-States'):
        data['native-country']=data['native-country'].replace(i,'rest of world')
data['native-country'].value_counts()
# For reducing the complexity i have renamed some of the terms to a single term in some feature set
for i in data['education']:
    if (i=='Preschool' or i=='1st-4th' or i=='5th-6th' or i=='7th-8th' or i=='9th'):
        data['education']=data['education'].replace(i,'school')
    elif (i=='10th' or i=='11th' or i=='12th'):
        data['education']=data['education'].replace(i,'HS-Drop')
    elif (i=='Assoc-acdm' or i=='Assoc-voc'):
        data['education']=data['education'].replace(i,'Assoc')
        
for i in data['workclass']:
    if (i=='Never-worked' or i=='Without-pay'):
        data['workclass']=data['workclass'].replace(i,'No-pay')
    elif (i=='Local-gov' or i=='State-gov'):
        data['workclass']=data['workclass'].replace(i,' Tier2Gov')
data['education'].value_counts()
data['workclass'].value_counts()
#Encoding Categorical Data
encoder=LabelEncoder()
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
#Dropping redundatnt features 
data.drop(['educational-num','fnlwgt'],axis=1,inplace=True)
data.head()
#Using get_dummies in some feature to avoid biasing since we cannot discriminate by race, gender and native.
gend=pd.get_dummies(data['gender'],drop_first=True)
native=pd.get_dummies(data['native-country'],drop_first=True)
race1=pd.get_dummies(data['race'],drop_first=True)
data['education']=encoder.fit_transform(data['education'])
data['workclass']=encoder.fit_transform(data['workclass'])
data=pd.concat([data,gend,native,race1],axis=1)
#adding and dropping some columns in the data
data.drop(['race','gender','native-country'],axis=1,inplace=True)
data.head()
data['income'].value_counts()# encoding target variable to numerical value 0 corresponds to <=50k and 1 corresponds to >50k
data['income']=encoder.fit_transform(data['income'])
data['income'].value_counts()
y=data['income']
X=data.drop(['income'],axis=1)
X_copy=X
#Normalising the data to get unbiased result
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
X
#Splitting the dataset
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=7)
#KNN Classifier for various values of n to know which n gives best model
acc=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain,ytrain)
    pred_i=knn.predict(xtest)
    acc.append(accuracy_score(ytest,pred_i))
print('In KNN classifier for n={} we got the best accuracy of {}'.format(acc.index(max(acc)),max(acc)))
kn1=KNeighborsClassifier(n_neighbors=25)
kn1.fit(xtrain,ytrain)
pred_knn=kn1.predict(xtest)
print(classification_report(ytest,pred_knn))
#Now using Logistic Regression
log=LogisticRegression(max_iter=1000)
log.fit(xtrain,ytrain)
log_pred=log.predict(xtest)
print('Using Logisitic Regression we got accuracy of {}'.format(accuracy_score(ytest,log_pred)))
print(classification_report(ytest,log_pred))
#Now using Naive Bayes Classifier
naive=GaussianNB()
naive.fit(xtrain,ytrain)
naive_pred=naive.predict(xtest)
print('Using Naive Bayes algorithm we got accuracy of {}'.format(accuracy_score(ytest,naive_pred)))
print(classification_report(ytest,naive_pred))
#Now using Decision Tree Classifier
tree=DecisionTreeClassifier(criterion='entropy')
tree.fit(xtrain,ytrain)
tree_pred=tree.predict(xtest)
print('Using Decision Tree algorithm we got accuracy of {}'.format(accuracy_score(ytest,tree_pred)))
print(classification_report(ytest,tree_pred))
#Using Random Forest Classifier
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=1,max_depth=10)
clf.fit(xtrain,ytrain)
clf_pred=clf.predict(xtest)
print('Using Random Forest Classifier we got accuracy of {}'.format(accuracy_score(ytest,clf_pred)))
print(classification_report(ytest,clf_pred))
#Using SVM AND GRID SEARCH CV
param_grid = {'C': [0.1,10,100,1000], 
      'gamma': [1, 0.1, 0.01], 
      'kernel': ['rbf']} 
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(xtrain,ytrain)
grid_svm_pred=grid.predict(xtest)
print('Using SVM with Grid Search CV we got an accuracy of {}'.format(accuracy_score(ytest,grid_svm_pred)))
print(classification_report(ytest,grid_svm_pred))
#getting best parameter of SVM with grid search CV
grid.best_params_
#Getting best estimator
grid.best_estimator_
#Finding feature importance using feature_importances command with Random forest classifier model.
feature_imp=pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp
#Getting a dataset with importance above 0.05 and fitting them
from sklearn.feature_selection import SelectFromModel
feat_sel=SelectFromModel(clf,threshold=0.05)
feat_sel.fit(xtrain,ytrain)
#Splittng the data as x_train and x_test to train only important feature
x_train=feat_sel.transform(xtrain)
x_test=feat_sel.transform(xtest)
#Using Random Forest only for important features
clf_imp=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=1,max_depth=7)
clf_imp.fit(x_train,ytrain)
imp_pred=clf_imp.predict(x_test)
print('Using Random forest with for important features we got an accuracy of {}'.format(accuracy_score(ytest,imp_pred)))
print(classification_report(ytest,imp_pred))