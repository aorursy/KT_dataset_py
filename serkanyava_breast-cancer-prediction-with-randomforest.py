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

data=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data
data.describe()
del data['Unnamed: 32']
t=data["diagnosis"].value_counts()

t
import matplotlib.pyplot as plt

import seaborn as sns

sns.barplot(x="diagnosis", y='id', data=data);

out=data.copy()

del out['diagnosis']
for column in out.columns[1:]:

    sns.scatterplot(data=out, x=column,y='id')

    plt.show()
data.isna().sum()
out
for column in out.columns[:]:

    col=out[column]

    std=col.std()

    avg=col.mean()

    three_sigma_plus = avg + (3 * std)

    three_sigma_minus =  avg - (3 * std)

        

    outliers = col[((out[column] > three_sigma_plus) | (out[column] < three_sigma_minus))].index

    print(column, outliers)


for column in out.columns[1:]:

    sns.scatterplot(data=out, x=column, y="id"

                   )

    plt.show()
data.drop(index=outliers, inplace=True)
for column in data.columns[2:]:

    sns.scatterplot(data=data, x="id", y=column

                   )

    plt.show()
x=data.iloc[:,2:]

y=data.iloc[:,1:2]
x
y


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.25, random_state=0)
x_train.shape
y_train.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

parameters = {

    'n_estimators'      : [20,30,40],

    'max_depth'         : [5, 6, 7],

    'random_state'      : [0],

    'max_features': ['auto'],

    'criterion' :['gini']

}

clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, n_jobs=-1)

clf.fit(x_train, y_train)



print(clf.score(x_train, y_train))

print(clf.best_params_)


rf=RandomForestClassifier(n_estimators=20, random_state=0, max_depth=6 ,max_features ='auto', criterion = 'gini')

a=rf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

pred=a.predict(x_test)

print(accuracy_score(pred,y_test))

pred
predf = pd.DataFrame()

predf['predictions'] = pred.tolist()

predf.head()
y_test.head()
predf.to_csv('breastcancerprediction.csv', index=False)