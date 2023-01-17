# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn import linear_model
traindata = pd.read_csv('../input/train.csv')

testdata = pd.read_csv('../input/test.csv')
testdata.head()
traindata.describe()
traindata.info()
traindata.isnull().sum()
traindata.hist(bins=10,figsize=(9,7),grid=False);

# very different scale for age and fare, need feature scaling
import seaborn as sns

import matplotlib.pyplot as plt

g = sns.FacetGrid(traindata, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple");
traindata.corr()['Survived']
# Imputation

traindata[traindata['Embarked'].isnull()]

sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=traindata);
traindata['Embarked'] = traindata['Embarked'].fillna('C')
traindata.isnull().sum()
testdata.isnull().sum()
#Impute 'Age'

agemedian = traindata['Age'].median()

# print(agemedian)

traindata['Age'] = traindata['Age'].fillna(agemedian)

testdata['Age'] = testdata['Age'].fillna(agemedian)
#Impute 'Fare'

faremedian = traindata['Fare'].median()

# print(agemedian)

testdata['Fare'] = testdata['Fare'].fillna(agemedian)

testdata.isnull().sum()
#Impute Cabin

# from sklearn.preprocessing import Imputer

# s = traindata['Cabin']

# imp = Imputer(missing_values='NaN', strategy = 'most_frequent' ,axis=0)

# imp.fit(s)

# traindata['Cabin'] = imp.transform(s)

# testdata['Cabin'] = imp.transform(s)

traindata.isnull().sum()

#testdata.isnull().sum()

# Convert categorical variables to integer variables

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(traindata['Sex'])

print(le.classes_)

train_sex = le.transform(traindata['Sex'])

test_sex = le.transform(testdata['Sex'])

print(train_sex[:5])

print(test_sex[:5])

#traindata.head()

#testdata.head()

traindata['Sex'] = train_sex

testdata['Sex'] = test_sex

traindata.head()

#testdata.head()
le = preprocessing.LabelEncoder()

le.fit(traindata['Embarked'])

print(le.classes_)

train_Embarked = le.transform(traindata['Embarked'])

test_Embarked = le.transform(testdata['Embarked'])

print(train_Embarked[:5])

print(test_Embarked[:5])

#traindata.head()

#testdata.head()

traindata['Embarked'] = train_Embarked

testdata['Embarked'] = test_Embarked
X = traindata[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

Y = traindata['Survived']
from sklearn.model_selection import train_test_split



num_test = 0.2

X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=num_test, random_state=23)



logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_train, Y_train)

logreg.score(X_train, Y_train)

#logreg.score(X_cv, Y_cv)

logreg.score(X_cv, Y_cv)

Xtest = testdata[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

Y = logreg.predict(Xtest)

ids = testdata['PassengerId']



output_df = pd.DataFrame({'PassengerId' : ids, 'Survived': Y})

output_df.head()