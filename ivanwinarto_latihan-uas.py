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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv')



data_train.head()
plt.style.use('fivethirtyeight')

sns.barplot(x='Sex' , y='Survived' , data=data_train)
sns.pointplot(x='Pclass' , y='Survived' , hue='Sex' , data=data_train)
data_train['Age'].hist(bins=70)
data_train['Age'].quantile([0, 0.25, 0.75, 0.9])
def age_cat(age):

    if age < 20.125:

        return 'Young'

    elif 20.125 <= age <= 38.000:

        return 'Adult'

    else:

        return 'Senior'



data_train['Age'] = data_train['Age'].fillna(data_train['Age'].mean())

data_train['AgeCat'] = data_train['Age'].apply(age_cat)

data_train.head(20)
sns.barplot(x='AgeCat' , y='Survived' , data=data_train)
sns.factorplot(x='AgeCat' , kind='count' , hue='Survived' , data=data_train)
sns.distplot(data_train.Fare)
data_train.Fare.quantile([.33, .67])
def fare_cat(fare):

    if fare < 8.61295:

        return 'Low'

    elif 8.61295 <= fare <= 26.25000:

        return 'Middle'

    else:

        return 'High'

    

data_train['Fare'] = data_train['Fare'].fillna(data_train['Fare'].mean())

data_train['FareCat'] = data_train['Fare'].apply(fare_cat)

data_train.head(20)
sns.barplot(x='FareCat' , y='Survived' , data=data_train)
sns.barplot(x='Embarked' , y='Survived' , data=data_train)
data_train['Is_Alone'] = data_train['SibSp'] + data_train['Parch'] == 0

sns.barplot(x='Is_Alone' , y='Survived' , data=data_train)
dfTrain = pd.read_csv('/kaggle/input/titanic/train.csv')

dfTest = pd.read_csv('/kaggle/input/titanic/test.csv')



dfTest.head()
def preprocessing_data(df):

    df['Age'] = df['Age'].fillna(df['Age'].mean())

    df['Age_Category'] = (df['Age'].apply(age_cat)).map({'Young':0, 'Adult':1, 'Senior':2})

    df['Fare_Category'] = (df.Fare.apply(fare_cat)).map({'Low':0, 'Middle':1, 'High':2})

    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

    df['Sex'] = df['Sex'].map({'male':0, 'female':1})

    df_final = df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

    return df_final



trainFinal = preprocessing_data(dfTrain)

testFinal = preprocessing_data(dfTest)



trainFinal
x_train = trainFinal.drop('Survived' , axis=1).fillna(0.0)

y_train = trainFinal['Survived']

x_test = testFinal.copy()

x_train.shape, y_train.shape, x_test.shape
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred_log = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
svc = SVC()

svc.fit(x_train, y_train)

y_pred_svc = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred_gaussian = gaussian.predict(x_test)

acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred_perceptron = perceptron.predict(x_test)

acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred_linear_svc = linear_svc.predict(x_test)

acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred_sgd = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)

acc_sgd
tree = DecisionTreeClassifier()

tree.fit(x_train, y_train)

y_pred_tree = tree.predict(x_test)

acc_tree = round(tree.score(x_train, y_train) * 100, 2)

acc_tree
forest = RandomForestClassifier(n_estimators=100)

forest.fit(x_train, y_train)

y_pred_forest = forest.predict(x_test)

acc_forest = round(forest.score(x_train, y_train) * 100, 2)

acc_forest
models = pd.DataFrame({

    'Model':['Logistic Regression' , 'SVM' , 'KNN' , 'Gaussian' , 'Perceptron' , 'Linear SVC' , 'SGD' , 'Tree' , 'Forest'],

    'Score':[acc_log , acc_svc , acc_knn , acc_gaussian , acc_perceptron , acc_linear_svc , acc_sgd , acc_tree , acc_forest]

})

models.sort_values(by='Score' , ascending=False)
result = pd.DataFrame({

    'PassangerId': dfTest['PassengerId'],

    'Survived': y_pred_tree

})

result.head(20)
dfTest['Survived #Tree'] = y_pred_tree

dfTest
dfTest.to_csv('result_tree.csv' , index=False)