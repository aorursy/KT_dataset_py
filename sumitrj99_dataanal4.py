import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.hist()

plt.show()
sns.kdeplot(df[df.columns[0]])
sns.kdeplot(df[df.columns[1]])
sns.kdeplot(df[df.columns[2]])
sns.kdeplot(df[df.columns[3]])
sns.kdeplot(df[df.columns[4]])
sns.kdeplot(df[df.columns[5]])
sns.kdeplot(df[df.columns[6]])
sns.kdeplot(df[df.columns[7]])
sns.kdeplot(df[df.columns[8]])
fig = plt.figure(figsize =(10, 7)) 

  

# Creating axes instance 

ax = fig.add_axes([0, 0, 1, 1]) 

  

# Creating plot 

bp = ax.boxplot([df[i].values for i in df.columns]) 

  

# show plot 

plt.show() 
df.corr()
import seaborn as sns

sns.pairplot(df)
df.describe()
nvs = sklearn.preprocessing.normalize(df.values)

dfn = pd.DataFrame(nvs)

dfn.describe()
diabetes_dataset = df

from sklearn.model_selection import train_test_split 



train,test = train_test_split(diabetes_dataset, test_size=0.25, random_state=0, stratify=diabetes_dataset['Outcome']) 



train_X = train[train.columns[:8]]

test_X = test[test.columns[:8]]

train_Y = train['Outcome']

test_Y = test['Outcome']
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

from sklearn import metrics



print(metrics.accuracy_score(test_Y, prediction))