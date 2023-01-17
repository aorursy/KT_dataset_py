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
dataset = pd.read_csv('/kaggle/input/titanicdataset-traincsv/train.csv')

dataset.head()
dataset.info()
dataset.isnull().sum()
def impute_sex(col):

    Age = col[0]

    Sex = col[1]

    

    if Age <= 12: return 'Child'

    else: return Sex

    



dataset['Sex'] = dataset[['Age','Sex']].apply(impute_sex, axis=1)
dataset['Sex'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot(x= 'Survived', hue='Sex', data=dataset)

plt.show()
sns.countplot(x= 'Pclass', hue='Survived', data=dataset)

plt.show()
dataset['Age'].plot.hist(bins=20)
dataset.boxplot(by='Pclass', column=['Age'])
def impute_age(col):

    Age = col[0]

    Pclass = col[1]

    

    if pd.isnull(Age):

        if Pclass ==1 : return 38

        elif Pclass ==2: return 29

        else: return 24

    else: return Age

    



dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age, axis=1)
dataset['Embarked'].value_counts()
dataset['Embarked'] = dataset['Embarked'].fillna('S')
one_hot = pd.get_dummies(dataset , columns=['Sex', 'Embarked'], drop_first=True)

one_hot.head()
x = one_hot.drop(['PassengerId','Survived', 'Name','Ticket','Cabin'], axis=1)

y = one_hot['Survived']
from sklearn.model_selection import train_test_split

xtr, xts, ytr, yts = train_test_split(x, y, test_size=0.3, random_state=101)
print(xtr.shape)

print(xts.shape)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression(max_iter=5000)

model.fit(xtr, ytr)

ypr = model.predict(xts)

model_score = model.score(xts, yts)
model_score
print(model.predict_proba([xts.iloc[0,:]]))

print(model.predict([xts.iloc[0,:]]))
param_grid = [

    {

        'penalty': ['l1', 'l2', 'elasticnet', 'none'],

        'C' : [0.01, 0.1, 1, 10 ,100],

        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

        'max_iter' : [100, 1000, 5000]

    }

]
from sklearn.model_selection import GridSearchCV



clf = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

clf.fit(xtr, ytr)
clf.best_estimator_
clf.best_score_