import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
df = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df.head()
df.info() 
df.describe() # some statistical describes
df.isnull().sum().sort_values(ascending=False) #there are no null data
set(df['EDUCATION'])
df['MARRIAGE'].value_counts() 

# Normally there is no 0 tag in the marriage column.For this we will change to 0 as 3.
df['MARRIAGE'] = np.where(df['MARRIAGE']>0, df['MARRIAGE'], 3)
df['MARRIAGE'].value_counts() #we changed 
df['EDUCATION'].value_counts() #and there are no 0,5,6 .we will handle it
df['EDUCATION'] = np.where(df['EDUCATION']<5, df['EDUCATION'], 4)
df['EDUCATION'] = np.where(df['EDUCATION']!=0, df['EDUCATION'], 4)
df['EDUCATION'].value_counts() 
set(df['PAY_0']) 
set(df['PAY_6'])
plt.figure(figsize=(16,8))

sns.countplot(x='default.payment.next.month', data = df) 
plt.figure(figsize=(16,8))

sns.distplot(df['LIMIT_BAL'], kde=True, bins = 180)
df['LIMIT_BAL'].value_counts().head(5)
plt.figure(figsize=(16,8))

sns.boxplot(x= 'default.payment.next.month', y='LIMIT_BAL', hue='SEX',data=df) #plot of credict paymnet as sex
plt.figure(figsize=(16,8))

sns.boxplot(x= 'MARRIAGE', y='AGE',hue='SEX',showfliers=False ,data=df)
plt.figure(figsize=(16,8))

sns.boxplot(x= 'EDUCATION', y='AGE',hue='MARRIAGE',showfliers=False ,data=df)
plt.figure(figsize=(16,8))

sns.boxplot(x= 'EDUCATION', y='LIMIT_BAL',hue='MARRIAGE',showfliers=True ,data=df) 

df.head()
df['PAY_0']= np.where(df['PAY_0']<0, 0, df['PAY_0'])

df['PAY_2']= np.where(df['PAY_2']<0, 0, df['PAY_2'])

df['PAY_3']= np.where(df['PAY_3']<0, 0, df['PAY_3'])

df['PAY_4']= np.where(df['PAY_4']<0, 0, df['PAY_4'])

df['PAY_5']= np.where(df['PAY_5']<0, 0, df['PAY_5'])

df['PAY_6']= np.where(df['PAY_6']<0, 0, df['PAY_6'])
# we will get the differences between Bill amount and pay amount 





df['diff1']=df['BILL_AMT1']- df['PAY_AMT1']

df['diff2']=df['BILL_AMT2']- df['PAY_AMT2']

df['diff3']=df['BILL_AMT3']- df['PAY_AMT3']

df['diff4']=df['BILL_AMT4']- df['PAY_AMT4']

df['diff5']=df['BILL_AMT5']- df['PAY_AMT5']

df['diff6']=df['BILL_AMT6']- df['PAY_AMT6']
set(df['PAY_0'])
df['PAY_0'].value_counts().plot.barh() 



#those are nominal scales. that means there is no differences between each values

#we will convert to ordinal scales of those nominal scales
categ= [8,7,6,5,4,3,2,1,0]
from pandas.api.types import CategoricalDtype

df.PAY_0 = df.PAY_0.astype(CategoricalDtype(categories = categ, ordered=True))  

df.PAY_2 = df.PAY_2.astype(CategoricalDtype(categories = categ, ordered=True))  

df.PAY_3 = df.PAY_3.astype(CategoricalDtype(categories = categ, ordered=True))  

df.PAY_4 = df.PAY_4.astype(CategoricalDtype(categories = categ, ordered=True))  

df.PAY_5 = df.PAY_5.astype(CategoricalDtype(categories = categ, ordered=True))  

df.PAY_6 = df.PAY_6.astype(CategoricalDtype(categories = categ, ordered=True))  
df.PAY_0.head() 
df.info() #as we can see we converted the types 
df.columns
df.EDUCATION = df.EDUCATION.astype(CategoricalDtype())

df.SEX = df.SEX.astype(CategoricalDtype())

df.MARRIAGE = df.MARRIAGE.astype(CategoricalDtype())
df.columns
df.info()
y = df['default.payment.next.month']

X= df.drop(columns=['default.payment.next.month','ID','BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
# we prepared the data for the model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=200)

rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,rf_pred))

print(classification_report(y_test,rf_pred))  
from sklearn.metrics import accuracy_score

print(accuracy_score(rf_pred,y_test))
from sklearn.tree import DecisionTreeClassifier

dt_model=DecisionTreeClassifier()

dt_model.fit(X_train,y_train)

dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))

print(classification_report(y_test,dt_pred))
print(accuracy_score(dt_pred,y_test))
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions)) #lojistik regresyon bütün değerlere 0 etiketini verdi tahmin yanlış 
print(accuracy_score(predictions,y_test))
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df = pd.get_dummies(df,columns=categorical_features)
y = df['default.payment.next.month']

X= df.drop(columns=['default.payment.next.month','ID'])
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
import lightgbm as lgb

d_train = lgb.Dataset(x_train, label=y_train)

params = {}

params['learning_rate'] = 0.2

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 10

params['min_data'] = 5000

params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)
#Prediction

y_pred=clf.predict(x_test)
len(y_pred)
#convert into binary values

for i in range(len(y_pred)):

    if y_pred[i]>=.5:       # setting threshold to .5

       y_pred[i]=1

    else:  

       y_pred[i]=0
y_pred
#Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

#Accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred,y_test)
cm
accuracy