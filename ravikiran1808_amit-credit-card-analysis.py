import os

os.chdir("../input")

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")

data.head()
data.shape
f_variables = data.iloc[:,0:30]

f_variables.dtypes
data['Class'].dtypes
#f_variables.isnull().any

f_variables.describe()

#by observing count values, all the variables have same count which means no missing values
#f_variables.isnull().values.any() gives if there any missing values(true/false)

#f_variables.isnull().sum() gives number of missing values by each column
### There are no missing values in the data 
from numpy import percentile

Q1 = f_variables.quantile(0.25)

Q3 = f_variables.quantile(0.75)

IQR = Q3-Q1

#print("quartiles are ",Q3,Q1)

print(IQR)
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

plt.figure(figsize = (15,100))

gs = gridspec.GridSpec(10,3)

for i,cn in enumerate(f_variables.columns):

    ax = plt.subplot(gs[i])

    sns.boxplot(f_variables[cn])

    ax.set_xlabel('')

    ax.set_title('feature: ' + str(cn))

plt.show() 
lower_band = IQR - 1.5*Q1

upper_band = IQR +1.5*Q3

f_variables.clip(lower = lower_band, upper = upper_band, axis = 1)

f_variables.shape
#9.1

f_variables.hist(figsize = (20,20))

plt.show()
data['Class'].value_counts()
sns.countplot(x = 'Class', data= data)
##10.1)

corr = data.corr()

corr
plt.figure(figsize = (30,10))

sns.heatmap(corr,xticklabels = corr.columns, yticklabels = corr.columns)
plt.figure(figsize = (15,100))

gs = gridspec.GridSpec(10,3)

for i, c in enumerate(f_variables.columns):

    plt.subplot(gs[i])

    plt.scatter(f_variables['V1'], f_variables[c])

    plt.xlabel("V1")

    plt.ylabel("feature Variable:"+str(c))
import numpy as np

colors = np.where(data['Class']==0,'red','green')
plt.figure(figsize = (15,100))

gs = gridspec.GridSpec(10,3)

for i, c in enumerate(f_variables.columns):

    plt.subplot(gs[i])

    plt.scatter(f_variables['V1'], f_variables[c], c = colors)

    plt.xlabel("V1")

    plt.ylabel("feature Variable:" +str(c))

plt.show()
f_variables['Class'] = data['Class']

f_variables.head()
split1 = int(0.8*len(f_variables))

split2 = int(0.9*len(f_variables))

train = f_variables[:split1]

validation = data[split1:split2]

test = data[split2:]
x_train = train.drop('Class', axis = 1)

y_train = train['Class']

x_validation = validation.drop('Class', axis =1)

y_validation = validation['Class']

x_test = test.drop('Class', axis = 1)

y_test = test['Class']
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 40, random_state = 10)

rf.fit(x_train, y_train)
pred = rf.predict(x_validation)

pred
from sklearn.metrics import accuracy_score

score = accuracy_score(y_validation, pred.round())

score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

c_pred = clf.predict(x_validation)

score = accuracy_score(y_validation, c_pred)

score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)
log_pred = logreg.predict(x_validation)

score = accuracy_score(y_validation, log_pred)

score