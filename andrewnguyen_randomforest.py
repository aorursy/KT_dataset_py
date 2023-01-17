# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/train.csv', header=0)



df_test = pd.read_csv('../input/test.csv', header=0)



df.head(3)



df.info()



df.describe()



df[0:10]



df.Age.dropna().hist(bins=8, range=(0,80))

plt.show()
df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)



df_test['Gender'] = df_test['Sex'].map({'female':0, 'male':1}).astype(int)



median_ages = np.zeros((2,3))

median_ages_test = np.zeros((2,3))



for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1) ]['Age'].dropna().median()

        median_ages_test[i,j] = df_test[(df_test['Gender'] == i) & (df_test['Pclass'] == j+1) ]['Age'].dropna().median()



df['AgeFill'] = df['Age']

df_test['AgeFill'] = df_test['Age']



for i in range(0,2):

    for j in range(0, 3):

        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1), 'AgeFill'] = median_ages_test[i,j]



df['AgeIsNull'] = df['Age'].isnull().astype(int)

df_test['AgeIsNull'] = df_test['Age'].isnull().astype(int)



df['FamilySize'] = df['SibSp'] + df['Parch']

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']



df = df.drop(['PassengerId', 'Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'AgeIsNull'], 1)

df_test = df_test.drop(['Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'AgeIsNull'], 1)



# transform embarked

df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

df_test['Embarked'] = df_test['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})



df = df.dropna()

df.info()

df_test = df_test.dropna()

#df.head(10)
from sklearn.ensemble import RandomForestClassifier 

import csv



train_data = df.values

train_data 



test_data = df_test.values



forest = RandomForestClassifier(n_estimators = 100)



forest.fit(train_data[0::, 1::], train_data[0::, 0])



#forest.predict(test_data)



#write submission file

with open("randomforestmodel.csv", "w") as predictions_file:

    p = csv.writer(predictions_file)

    p.writerow(["PassengerId", "Survived"])



    for row in test_data:

        p.writerow( [ int(row[0]), forest.predict(row[1::])[0]] )