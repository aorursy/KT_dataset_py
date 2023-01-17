import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/titanic/train.csv')

df_t = pd.read_csv('../input/titanic/test.csv')
df.head(15)
df_t.head()
df.info()
df_t.info()
df['Cabin'].nunique()
df['Survival'] = df['Survived']

df.drop(['Survived','SibSp','Parch','Cabin','Name','Ticket'],axis = 1,inplace = True)

df_t.drop(['SibSp','Parch','Cabin','Name','Ticket'],axis = 1,inplace = True)

df.rename(columns ={'Pclass':'class','Sex':'gender'},inplace = True)

df_t.rename(columns ={'Pclass':'class','Sex':'gender'},inplace = True)

df_t.head()
df.head()
df.info()
df_t.info()
df.isnull().sum()
df_t.isnull().sum()
df.describe(include = 'all')
df.Age.fillna(df.Age.median(),inplace = True)

df_t.Age.fillna(df.Age.median(),inplace = True)
df_t.Fare.fillna(df.Fare.mean(),inplace = True)
mode =df['Embarked'].mode()

mode
df_t.head()
df.Embarked.fillna(mode[0],inplace = True)

group1 = df.groupby('gender').Survival.sum().reset_index()

group1
group1.set_index('gender',inplace = True)
group1.plot(kind = 'bar',figsize = (10,6),color = ['coral','blue'])

plt.title('survival based on gender')

plt.xlabel('gender',size = 14)

plt.ylabel('number of indiviual survived',size = 14)

plt.xticks(size = 14)

plt.show()
group2 = df.groupby(['gender','class']).Survival.sum().reset_index()
group2
plt.figure(figsize =(20,6))

sns.set(font_scale = 1.5)

ax = sns.catplot(x = 'gender',y = 'Survival',hue = 'class',kind = 'bar',data = group2)

plt.figure(figsize =(12,6))

sns.catplot(x = 'class',y = 'Survival',hue = 'gender',kind = 'bar',data = group2)

sns.heatmap(df.corr(),annot = True)
sns.catplot(x = 'class',kind = 'count',palette="ch:.25",data = df)
plot1 = df.groupby(['gender','Survival']).PassengerId.count().reset_index()
plot1.replace([0,1],['died','survived'],inplace = True)
plot1.rename(columns= {'PassengerId':'count'},inplace = True)
plot1

sns.catplot(x='gender',y='count',hue = 'Survival',kind = 'bar',data = plot1 )
df['class'].value_counts()
plot2 = df.groupby(['class','Survival']).PassengerId.count().reset_index()
plot2['Survival'].replace([0,1],['died','survived'],inplace = True)
plot2.rename(columns= {'PassengerId':'count'},inplace = True)

plot2
sns.catplot(x = 'class',y = 'count',kind = 'bar',hue = 'Survival',data =plot2)
sns.countplot(x='class',hue = 'gender',data = df)
sns.catplot(x= 'class',y = 'Survival',hue = 'gender',kind = 'violin',data = df)
ax1 = plt.figure(figsize = (10,10))

ax1 = sns.catplot(x= "class",y = 'Survival',hue = 'gender',kind = 'swarm',data = df)

plt.show()
gender_tr = pd.get_dummies(df['gender'])

gender_te = pd.get_dummies(df_t['gender'])
embarked_tr =pd.get_dummies(df['Embarked'])

embarked_te = pd.get_dummies(df_t['Embarked'])
new_tr = df.join(gender_tr)

new_te = df_t.join(gender_te)
new_tr = new_tr.join(embarked_tr)

new_te = new_te.join(embarked_te)
new_tr.drop(['Embarked','gender'],axis = 1,inplace = True)

new_te.drop(['Embarked','gender'],axis = 1,inplace = True)
new_tr.head()

new_te.shape
new_tr.corr()


ax = plt.figure(figsize = (10,10))

ax = sns.heatmap(round(new_tr.corr(),2),cmap="YlGnBu",annot = True)
new_tr.Age.median()
age_df = new_tr[new_tr['Age']<= 28]
age_df.head()
sns.catplot(x= 'Survival',y = 'Age',kind= 'swarm',data = age_df)
y = new_tr['Survival']

new_tr.drop(['Survival','PassengerId'],axis = 1,inplace = True)

new_te.drop('PassengerId',axis = 1,inplace = True)

x = new_tr
test_set = new_te

new_te.shape
x_array = np.asarray(x)

y_array = np.asarray(y)

test_set_array = np.asarray(test_set)

print(x_array.shape)

print(y_array.shape)

print(test_set.shape)
from sklearn import preprocessing

x_normalized = preprocessing.StandardScaler().fit(x_array).transform(x_array)

print(x_normalized.shape)

test_set_normalized = preprocessing.StandardScaler().fit(test_set_array).transform(test_set_array)

print(test_set_normalized.shape)
print(x_normalized[0:5])

print(test_set_normalized[0:5])
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C = 0.01, solver = 'liblinear')
Rcross = cross_val_score(LR,x_normalized,y_array, cv = 6)
Rcross.shape
Rcross
print('mean r2 value is: ',Rcross.mean())

print('deviation of score value:', Rcross.std())
x_train,x_test,y_train,y_test = train_test_split(x_normalized,y_array,test_size = 0.2,random_state = 2)

print('x_train shape: ',x_train.shape)

print('x_test shape:',x_test.shape)
LR2 = LogisticRegression(C = 0.01, solver = 'liblinear').fit(x_train,y_train)

LR2
Yhat = LR2.predict(x_test)

Yhat
yhat_prob = LR2.predict_proba(x_test)

yhat_prob 


from sklearn.metrics import classification_report

print(classification_report(y_test,Yhat))
prediction = LR2.predict(test_set_normalized)

prediction
perd_df = pd.DataFrame({'PassengerId':df_t.PassengerId,'Survived':prediction})

perd_df.head()
perd_df.shape
perd_df.to_csv('titanic_survival_classification.csv',index = False)