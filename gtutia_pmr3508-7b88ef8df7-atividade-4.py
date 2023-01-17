%matplotlib inline
import numpy as np # linear algebra
import pandas as pd
import os
print(os.listdir("../input"))
import matplotlib
import pandas as pd
adult = pd.read_csv('../input/train_data.csv',na_values='?')
adult.shape
adult.head()
adult_full = adult.dropna()
adult_full.shape
import seaborn as sns 
adult_full['age_range'] = pd.cut(adult_full['age'],bins=[0,20,40,60,80,100])
sns.countplot(x='age_range', data = adult_full)
sns.countplot(x='age_range',hue='income', data = adult_full)
adult_full['age'].describe()
adult_full['workclass'].value_counts()
sns.countplot(x='workclass',hue='income', data = adult_full)
adult_full[adult_full['income']=="<=50K"].workclass.value_counts()
adult_full[adult_full['income']==">50K"].workclass.value_counts()
adult_full['fnlwgt'].describe()
adult_full['education'].value_counts()
adult_full[adult_full["income"]=="<=50K"].education.value_counts()
adult_full[adult_full["income"]==">50K"].education.value_counts()
adult_full['education.num'].value_counts()
sns.countplot(x='education.num',hue='income', data = adult_full)
sns.countplot(x='marital.status',hue='income', data = adult_full)
adult_full['marital.status'].value_counts()
adult_full[adult_full["income"]==">50K"]['marital.status'].value_counts()
adult_full['occupation'].value_counts()
adult_full[adult_full["income"]==">50K"]['occupation'].value_counts()
adult_full['relationship'].value_counts()
sns.countplot(x='relationship',hue='income', data = adult_full)
sns.countplot(x='race',hue='income', data = adult_full)
adult_full['race'].value_counts()
adult_full[adult_full["income"]==">50K"]['race'].value_counts()
sns.countplot(x='sex', data = adult_full)
sns.countplot(x='sex',hue='income', data = adult_full)
adult_full['cg'] = 1
adult_full.loc[adult_full['capital.gain']==0,'cg']=0
adult_full.loc[adult_full['capital.gain']!=0,'cg']=1
adult_full["cg"].value_counts()
sns.countplot(x='cg',hue='income', data = adult_full)
adult_full['cl'] = 1
adult_full.loc[adult_full['capital.loss']==0,'cl']=0
adult_full.loc[adult_full['capital.loss']!=0,'cl']=1
adult_full["cl"].value_counts()
sns.countplot(x='cl',hue='income', data = adult_full)
adult_full[adult_full["income"]==">50K"]["hours.per.week"].describe()

adult_full[adult_full["income"]=="<=50K"]["hours.per.week"].describe()
adult_full['native.country'].value_counts()
adult_full[adult_full["income"]==">50K"]["native.country"].value_counts()
adult_full[adult_full["income"]=="<=50K"]["native.country"].value_counts()
from sklearn import preprocessing
num_adult = adult_full.apply(preprocessing.LabelEncoder().fit_transform)
num_adult.head()
adult_output = num_adult.income
adult_input = num_adult.drop(['Id','income','fnlwgt','education','occupation','native.country','age_range','cg','cl'],axis = 1)
adult_input.head()
from sklearn.model_selection import train_test_split
adult_input_train, adult_input_test, adult_output_train, adult_output_test = train_test_split(adult_input, adult_output, train_size=0.75,random_state=1)
from sklearn.ensemble import RandomForestRegressor

regressorRF = RandomForestRegressor(n_estimators=100, random_state=0)  
regressorRF.fit(adult_input_train, adult_output_train)  
RFpredict = regressorRF.predict(adult_input_test)
from sklearn.metrics import accuracy_score
accuracy_score(adult_output_test,RFpredict.round())


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0, solver='lbfgs',max_iter = 200).fit(adult_input_train, adult_output_train)
LRpredict = LR.predict(adult_input_test)
accuracy_score(adult_output_test,LRpredict.round())


from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(solver='lbfgs', random_state=0).fit(adult_input_train, adult_output_train)
NNpredict = NN.predict(adult_input_test)
accuracy_score(adult_output_test,NNpredict)
                 

from sklearn.ensemble import AdaBoostClassifier

ADA = AdaBoostClassifier(n_estimators=100).fit(adult_input_train, adult_output_train)
ADApredict = ADA.predict(adult_input_test)
accuracy_score(adult_output_test,ADApredict.round())
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=25).fit(adult_input_train, adult_output_train)
KNNpredict = KNN.predict(adult_input_test)
accuracy_score(adult_output_test,KNNpredict.round())
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('lr', LR), ('nn', NN),('ada',ADA),('knn',KNN)], voting='soft',weights=[1,2,3,1])
eclf1 = eclf1.fit(adult_input_train, adult_output_train)
eclf1.score(adult_input_test,adult_output_test)

from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('lr', LR), ('nn', NN),('ada',ADA),('knn',KNN)], voting='soft',weights=[1,1,3,2])
eclf1 = eclf1.fit(adult_input_train, adult_output_train)
eclf1.score(adult_input_test,adult_output_test)

from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('lr', LR), ('nn', NN),('ada',ADA),('knn',KNN)], voting='hard')
eclf1 = eclf1.fit(adult_input_train, adult_output_train)
eclf1.score(adult_input_test,adult_output_test)
