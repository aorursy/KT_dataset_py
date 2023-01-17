import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head(3)
#Checking and updating Null Values

df.isnull().sum()
#We can convert Age to Int

df.age=df.age.astype(int)

print(df.dtypes)

print('shape:',df.shape)

df.head(2)
sns.distplot(df.platelets)
#we can see that the death rate is high for people over 80

plt.scatter(df.age,df.DEATH_EVENT)
print('Death Rate for less than 50 y/o:',int(df[df['age']<=50].DEATH_EVENT.sum()*100/df[df['age']<=50].shape[0]),'%')

print('Death Rate for more than 80 y/o:',int(df[df['age']>=80].DEATH_EVENT.sum()*100/df[df['age']>=80].shape[0]),'%')

print('Death Rate for more than 50 and less than 80 y/o:',int(df[(df['age']>50)&(df['age']<80)].DEATH_EVENT.sum()*100/df[(df['age']>50)&(df['age']<80)].shape[0]),'%')
# We can see that there is a significant difference between the death rates for people below and above the 80 y/o mark

#so we split them into two categories Old and Not Old

bins = [0,80,100]

group_names = ['MiddleAged', 'Old']

df['Age_bin']=pd.cut(df['age'],bins,labels=group_names)

df.drop('age',axis=1,inplace=True)

age_variables = pd.get_dummies(df['Age_bin'])

age_variables.drop('Old',axis=1,inplace=True)

df=pd.concat([df,age_variables],axis=1)

df.tail(10)
df.drop('Age_bin',axis=1,inplace=True)
#model training

from sklearn.model_selection import train_test_split

X=df.drop('DEATH_EVENT',axis=1)

y=df['DEATH_EVENT']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.28,random_state=2)
#Metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
#Initially we do not perform any hyperparameter tuning, we use the default parameters itself

from sklearn.ensemble import RandomForestClassifier

rf_random=RandomForestClassifier()

rf_random.fit(X_train,y_train)

print('Accuracy score:',accuracy_score(y_test, rf_random.predict(X_test)))

print(classification_report(y_test,rf_random.predict(X_test)))

print(confusion_matrix(y_test,rf_random.predict(X_test)))
print(rf_random.feature_importances_)

feat_importances=pd.Series(rf_random.feature_importances_,index=X.columns)

feat_importances.nlargest(11).plot(kind='barh')

plt.show()

from sklearn.feature_selection import chi2

from sklearn.feature_selection import f_classif

print('Chi Square:')

print(chi2(X,y))

print('F:')

print(f_classif(X,y))
#We can see that Diabetes, Sex, Smoking have almost no correlation with Death, so we drop them.

X_train.drop(['diabetes','sex','smoking'],axis=1,inplace=True)

X_test.drop(['diabetes','sex','smoking'],axis=1,inplace=True)

print(X_train.shape)

print(X_test.shape)
rf_random=RandomForestClassifier()

rf_random.fit(X_train,y_train)

print('Accuracy score:',accuracy_score(y_test, rf_random.predict(X_test)))

print(classification_report(y_test,rf_random.predict(X_test)))

print(confusion_matrix(y_test,rf_random.predict(X_test)))
o=[]

for i in range (1,100):

    o.append(i)

from sklearn.model_selection import GridSearchCV

parameters=[{'n_estimators':o,'criterion':['gini','entropy']}]

grid_search=GridSearchCV(estimator=rf_random,param_grid=parameters,scoring='accuracy',cv=10)

grid_search=grid_search.fit(X_train,y_train)

grid_search.best_params_
rf_random=RandomForestClassifier(n_estimators=49,criterion='gini')

rf_random.fit(X_train,y_train)

print('Accuracy score:',accuracy_score(y_test, rf_random.predict(X_test)))

print(classification_report(y_test,rf_random.predict(X_test)))

print(confusion_matrix(y_test,rf_random.predict(X_test)))
class_weight=({0:1,1:1.8})

rf_random=RandomForestClassifier(n_estimators=59,criterion='entropy',class_weight=class_weight)

rf_random.fit(X_train,y_train)

print('Accuracy score:',accuracy_score(y_test, rf_random.predict(X_test)))

print(classification_report(y_test,rf_random.predict(X_test)))

print(confusion_matrix(y_test,rf_random.predict(X_test)))
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)

print('Accuracy score:',accuracy_score(y_test, logmodel.predict(X_test)))

print(classification_report(y_test,logmodel.predict(X_test)))

print(confusion_matrix(y_test,logmodel.predict(X_test)))
#we try feature scaling for improving the model performance

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train_scaled=sc.fit_transform(X_train)

X_test_scaled=sc.transform(X_test)

X_train_scaled
logmodel=LogisticRegression()

logmodel.fit(X_train_scaled,y_train)

print('Accuracy score:',accuracy_score(y_test, logmodel.predict(X_test_scaled)))

print(classification_report(y_test,logmodel.predict(X_test_scaled)))

print(confusion_matrix(y_test,logmodel.predict(X_test_scaled)))
#the performance doesn't improve, we now try the MinMax scaler

from sklearn import preprocessing

minmaxscaler=preprocessing.MinMaxScaler(feature_range=(0,1))

X_train_scaled=minmaxscaler.fit_transform(X_train)

X_test_scaled=minmaxscaler.transform(X_test)
logmodel=LogisticRegression()

logmodel.fit(X_train_scaled,y_train)

print('Accuracy score:',accuracy_score(y_test, logmodel.predict(X_test_scaled)))

print(classification_report(y_test,logmodel.predict(X_test_scaled)))

print(confusion_matrix(y_test,logmodel.predict(X_test_scaled)))
#by adding class weight to balance the dataset

class_weight=({0:1,1:1.8})

logmodel=LogisticRegression(class_weight=class_weight)

logmodel.fit(X_train_scaled,y_train)

print('Accuracy score:',accuracy_score(y_test, logmodel.predict(X_test_scaled)))

print(classification_report(y_test,logmodel.predict(X_test_scaled)))

print(confusion_matrix(y_test,logmodel.predict(X_test_scaled)))