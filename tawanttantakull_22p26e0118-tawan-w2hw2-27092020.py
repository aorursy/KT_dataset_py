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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

#test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
df.head(5)
df.info()
df.nunique()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.heatmap(df.corr(), linewidth=0.5, linecolor='white', annot=True)
cols = ['Pclass', 'SibSp', 'Parch', 'Embarked']

for col in cols:

    plot = sns.catplot(x=col, y='Age',

           hue='Sex', col='Survived',

           data=df, kind="strip",

           height=6, aspect=.8, 

           alpha=.7, size=5, 

           palette='seismic');
age_cat = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']

df['Age_Group'] = pd.cut(df.Age,

                         bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.Inf],

                         labels = age_cat,

                         right = True,

                         include_lowest = True)
fig = plt.figure(figsize=(10,7))

sns.countplot(data=df, x='Age_Group', hue='Survived', palette='seismic', alpha=.8)

plt.title('Passenger Age Group - Survived vs Non-Survived', fontsize=20, pad=40)

plt.ylabel('Number of Passenger Survived', fontsize=15, labelpad=20)

plt.xlabel('Passenger Age Group', fontsize=15,labelpad=20);
df_clean = df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
df.head()
df.isnull().sum()
#fill 'Age' with mean

age_mean = df_clean['Age'].mean()

df_clean['Age'].fillna(age_mean, inplace=True)
#Create 'Age_Group' feature

age_cat = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']

df_clean['Age_Group'] = pd.cut(df_clean.Age,

                         bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, np.Inf],

                         labels = age_cat,

                         right = True,

                         include_lowest = True)

df_clean.drop('Age', axis=1, inplace=True)
df_clean.dropna(inplace=True)
df_clean.info()
df_dum = pd.get_dummies(df_clean, drop_first=True)
df_dum.info()
from sklearn.preprocessing import MinMaxScaler



X = df_dum.drop('Survived', axis=1)

y = df_dum['Survived']



scaler = MinMaxScaler()

X = scaler.fit_transform(X)

y = np.array(y)
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

from sklearn.metrics import confusion_matrix



dt = DecisionTreeClassifier()

nb = GaussianNB()

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)



models_lst = [dt, nb, mlp]
from sklearn.model_selection import KFold



for model in models_lst:

    rec_lst = []

    pre_lst = []

    f1_lst = []

    name = model.__class__.__name__

    print(name)

    print('************************************')

    kf = KFold(n_splits=5, random_state=69, shuffle=True)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        

        rec = recall_score(y_test, pred)

        pre = precision_score(y_test, pred)

        f1 = precision_score(y_test, pred)

        

        rec_lst.append(rec)

        pre_lst.append(pre)

        f1_lst.append(f1)

    print('Racall:', rec_lst)

    print('Precision:', pre_lst)

    print('AVG F-Measure:', np.array(f1_lst).mean())

    print('************************************')

    print('\n')