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
data=pd.read_csv('../input/model-trap/train.csv')

datatest= pd.read_csv('../input/model-trap/train.csv')
data.head(10)
data.isnull()
data['income'].max()
data['income'].mean()
data['income'].median()
fig, ax = plt.subplots(figsize=(8,4))

ax.scatter(data['income'], data['Unnamed: 0'])

ax.set_xlabel('income')

plt.show()
# import numpy as np; np.random.seed(0)

import seaborn as sns

cor= data.corr()

ax = sns.heatmap(cor, cmap="YlGnBu")
data['default'].value_counts()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

x= ['True','False']

y_pos = np.arange(len(val))

plt.bar(y_pos, val, align='center', alpha=0.5)

plt.xticks(y_pos,x)

plt.title('Defaulter')

plt.show()
data.describe()
data.columns
data['label'] = ['1' if star == True else '0' for star in data['default']];

data
cols = [0,3,11,12,13]

data.drop(data.columns[cols],axis=1,inplace=True)

data.head()
data.columns
datanew=pd.get_dummies(data)

datanew
X=datanew.iloc[:,0:9]

Y=datanew['label_1']
datatest= pd.read_csv('../input/model-trap/test.csv')

datatest
datatest.columns
cols = [0,1,4,11,14]

datatest.drop(datatest.columns[cols],axis=1,inplace=True)

datatest.head()
datatest.columns
datatest['label'] = ['1' if star == True else '0' for star in datatest['default']];

datatest
cols = [9]

datatest.drop(datatest.columns[cols],axis=1,inplace=True)

datatest.head()
datatestnew=pd.get_dummies(datatest)

datatestnew
Xtest = datatestnew.iloc[:,:9]

Ytest = datatestnew['label_1']
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X,Y)

model.score(X,Y)

predicted= model.predict(Xtest)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Ytest, predicted))

print('\n')

print(classification_report(Ytest, predicted))

model.score(X,Y)
from sklearn.ensemble import GradientBoostingClassifier

model1= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

model1.fit(X,Y)

predicted= model1.predict(Xtest)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Ytest, predicted))

print('\n')

print(classification_report(Ytest, predicted))

model1.score(X,Y)