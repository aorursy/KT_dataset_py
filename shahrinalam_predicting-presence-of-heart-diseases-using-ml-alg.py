#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart-disease/heart.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, fmt='.1f')

plt.show()
#age analysis

df.age.value_counts()[:10]
sns.barplot(x= df.age.value_counts()[:10].index, y= df.age.value_counts()[:10].values  )

plt.xlabel('Age')

plt.ylabel("Age counter")

plt.title("Age Analysis")

plt.show
df.target.value_counts()
countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of patients dont have heart disease: {:.2f}%".format((countNoDisease/(len(df.target)))*100))

print("Percentage of patients have heart disease: {:.2f}%".format((countHaveDisease/(len(df.target)))*100))
countFemale= len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("% of Female Patients: {:.2f}%".format((countFemale/(len(df.sex))*100)))

print("% of male Patients: {:.2f}%".format((countMale/(len(df.sex))*100)))
young_ages = df[(df.age>=29)&(df.age<40)]

middle_ages = df[(df.age>=40)&(df.age<55)]

elderly_ages = df[(df.age>=55)]

print("young ages", len(young_ages))

print("middle ages", len(middle_ages))

print("elderly ages", len(elderly_ages))
colors = ['blue','green', 'red']

explode= [1,1,1]

plt.figure(figsize= (8,8))

plt.pie([len(young_ages), len(middle_ages), len(elderly_ages)], labels=['young ages', 'middle ages', 'elderly ages'])

plt.show()
#chest pain analysis

df.cp.value_counts()
df.target.unique()
sns.countplot(df.target)

plt.xlabel('Target')

plt.ylabel('Count')

plt.title('Target 1 & 0')

plt.show()
df.corr()
# Model Building
from sklearn.linear_model import LogisticRegression

x_data = df.drop(['target'], axis = 1)

y = df.target.values
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.2, random_state= 0)
lr = LogisticRegression()

lr.fit(x_train, y_train)

print('Test Accuracy {:.2f}%'.format(lr.score(x_test, y_test)*100))
# Logistic Regression Test Accuracy 85.25%

#KNN model

from sklearn.neighbors import KNeighborsClassifier

knn =  KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

print("KNN accuracy: {:.2f}%".format(knn.score(x_test, y_test)*100))
# support vector

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train, y_train)

print("SVC accuracy: {:.2f}%".format(svm.score(x_test, y_test)*100))
# Naive Bayes



from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

print("NB accuracy: {:.2f}%".format(nb.score(x_test, y_test)*100))
# Random forset



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state= 1)

rf.fit(x_train, y_train)

print("Random Forest accuracy: {:.2f}%".format(rf.score(x_test, y_test)*100))