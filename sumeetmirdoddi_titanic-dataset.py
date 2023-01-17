# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics
#Read the datasets

train = pd.read_csv(r'/kaggle/input/train.csv')

test = pd.read_csv(r'/kaggle/input/test.csv')
train.head()
train.info()
#Since for cabin there are only 204 rows that have values we can say that this coloum is better dropped

train.drop('Cabin',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)
#For Embarked there are 2 rows that are having null values

train[train.Embarked.isnull()]
#Since we can see that Both are form Pclass 1 we can just impute the most Embarked values from Pclass

train[['Pclass','Embarked']][train.Pclass==1].groupby('Embarked').count()
#So Changing NaN to 'S'

train.Embarked = train.Embarked.fillna('S')
#Now Age has 19% of missing values, so the logic is impute it with median by grouping Parch,Sex

train.Age = train[['Parch','Sex','Age']].groupby(['Parch','Sex']).transform(lambda group: group.fillna(group.median()))

test.Age = test[['Parch','Sex','Age']].groupby(['Parch','Sex']).transform(lambda group: group.fillna(group.median()))
#Since we cant impute the age dropping the rows having missing values

train.dropna(subset=['Age'],inplace=True)
#Converting binary to numeric

train.Sex = train.Sex.map({'male':1,'female':0})

test.Sex = test.Sex.map({'male':1,'female':0})
#One-hot encoding for Embarked

Embark = pd.get_dummies(train.Embarked)

Embark.drop('S',axis=1,inplace=True)

train = pd.concat([train,Embark],axis=1)

Embark1 = pd.get_dummies(test.Embarked)

Embark1.drop('S',axis=1,inplace=True)

test = pd.concat([test,Embark1],axis=1)
train.describe(percentiles=[.25,.5,.75,.9,.95,.98])
#Plotting the scatter plot

plt.scatter(train.Fare,train.Fare)
#Hence deleting the coloum

train = train[train.Fare!=512.329200]
#Dropping the unrequired colums

trainbkp = train[['PassengerId']]

train.drop(['Name','Ticket','Embarked','PassengerId'],axis=1,inplace=True)

testbkp = test[['PassengerId']]

test.drop(['Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
y = train.pop('Survived')

x = train

x_test = test
plt.figure(figsize=(10,5))

sns.heatmap(x.corr(),annot=True)
x.drop('Parch',axis=1,inplace=True)

x_test.drop('Parch',axis=1,inplace=True)
#checking the corelation matix once again

sns.heatmap(x.corr(),annot=True)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x[['Pclass','Age','SibSp','Fare']] = scaler.fit_transform(x[['Pclass','Age','SibSp','Fare']])

x_test[['Pclass','Age','SibSp','Fare']] = scaler.transform(x_test[['Pclass','Age','SibSp','Fare']])
x.describe()
lm1 = sm.GLM(y,(sm.add_constant(x)),family= sm.families.Binomial())

lm1.fit().summary()
lgrg = LogisticRegression()

rfe =RFE(lgrg,5)
rfe = rfe.fit(x,y)
rfe.support_
list(zip(x.columns,rfe.support_))
#Hence considering the colums

cols = x.columns[rfe.support_]

x_sm = sm.add_constant(x[cols])
lm2 = sm.GLM(y,x_sm,family=sm.families.Binomial())

md1 = lm2.fit()
y_pred = md1.predict(x_sm)

y_pred = y_pred.values.reshape(-1)
#Crearting the dataframe with survied,predicated,passenger

answer = pd.DataFrame({'PassengerID':trainbkp.PassengerId,'Survived':y,'Predicted':y_pred})

answer['Predicted5'] = answer['Predicted'].map(lambda x:1 if x>0.5 else 0)
#checking the VIF

vif = pd.DataFrame()

vif['Features'] = x_sm.columns

vif['VIF'] = [variance_inflation_factor(x_sm.values,i)for i in range(x_sm.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by='VIF',ascending=False)

vif
metrics.accuracy_score(answer.Survived,answer.Predicted5)
numbers = [float (x)/10 for x in range(10)]

for i in numbers:

        answer[i] = answer.Predicted.map(lambda x: 1 if x>i else 0)


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(answer['Survived'], answer[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
asnwer = answer[['PassengerID','Survived',0.4]]
metrics.accuracy_score(answer['Survived'],answer[0.4])
x_test = x_test[cols]
x_test = sm.add_constant(x_test)

y_test = md1.predict(x_test)

y_test = y_test.values.reshape(-1)
sol = pd.DataFrame()

sol['PassengerID'] = testbkp['PassengerId']

sol['y_test'] = y_test

sol['Survived'] = sol['y_test'].map(lambda x: 1 if x>0.4 else 0)

sol.drop('y_test',axis=1,inplace=True)
sol.columns = ['PassengerId','Survived']
sol.to_csv(r'gender_submission.csv')