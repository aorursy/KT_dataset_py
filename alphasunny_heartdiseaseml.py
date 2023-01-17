# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')
df.head()
df.describe()
df.isna().any()
df.info()
# Check the count of with disease and without disease

fig, ax = plt.subplots(figsize=(6.,5.))

sns.countplot(x= 'target', data=df, palette='Accent')
# use the pair plot try to find the inner relationship

sns.pairplot(data = df)
# The disease rate difference between genders

plt.figure(figsize=(12, 8))

sns.countplot(x = 'target', hue='sex', data = df,palette='bwr')

plt.xlabel("Sex (0 = female, 1= male)")

plt.title('Heart Frequency for Sex')

plt.legend(['No Disease', 'Disease'])

plt.ylabel('Frequency')
# what about the age

plt.figure(figsize=(25,8), dpi=100)

sns.countplot(x = 'age', hue='target', data=df)

plt.title('Heart Disease Frequency for Ages')

plt.xticks(rotation=0)

plt.xlabel('Age')

plt.ylabel('Frequency')
# chest pain type

print("There are {} types of chest pain".format(len(df["cp"].unique())))
plt.figure(figsize=(12, 8))

sns.countplot(x ="cp", hue= "target", data=df)

plt.title("Different chest type and thier disease count")

plt.legend(['No disease', 'Disease'])

plt.xlabel("Chest pain type")
df.groupby('target')['trestbps'].mean()
print("With disease, the average blood pressure is {}".format(df.groupby('target')['trestbps'].mean()[1]))

print("Normal, the average blood pressure is {}".format(df.groupby('target')['trestbps'].mean()[0]))
# The blood pressuer distribution

plt.figure(figsize=(8, 8))

sns.violinplot(x = 'target', y ='trestbps' ,data = df)

plt.title("Blood pressure difference")

plt.ylabel("Resting blood pressure")

plt.xlabel("Target (0 = No disease, 1= Disease)")
print("With disease, the average blood pressure is {}".format(df.groupby('target')['thalach'].mean()[1]))

print("Normal, the average blood pressure is {}".format(df.groupby('target')['thalach'].mean()[0]))
# The blood pressuer distribution

plt.figure(figsize=(6,7))

sns.violinplot(x = 'target', y ='thalach' ,data = df)

plt.title("Maximun heart rate difference")

plt.ylabel("Maximum heart rate")

plt.xlabel("Target (0 = No disease, 1= Disease)")
sns.scatterplot(x = 'age', y = 'thalach',hue='target',data = df, palette='bwr')

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
# handle the dummy data, there are three dummy datas: cp, slope, thal

a = pd.get_dummies(df['cp'], prefix='cp')

b = pd.get_dummies(df['slope'], prefix='slope')

c = pd.get_dummies(df['thal'], prefix='thal')



# new frame

frames = [df, a, b, c]

df_dummyed = pd.concat(frames, axis=1)

df_dummyed.drop(['cp', 'slope', 'thal'], axis=1, inplace= True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(df_dummyed.drop(['target','cp_0', 'cp_1', 'cp_2', 'cp_3', 'thal_0',

       'thal_1', 'thal_2', 'thal_3', 'slope_0', 'slope_1', 'slope_2'], axis=1))

scaled_features = scaler.transform(df_dummyed.drop(['target','cp_0', 'cp_1', 'cp_2', 'cp_3', 'thal_0',

       'thal_1', 'thal_2', 'thal_3', 'slope_0', 'slope_1', 'slope_2'], axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df_dummyed.columns[:-12])

df_feat = df_feat.join(df_dummyed[['cp_0', 'cp_1', 'cp_2', 'cp_3', 'thal_0',

       'thal_1', 'thal_2', 'thal_3', 'slope_0', 'slope_1', 'slope_2']])

df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, df['target'], test_size= 0.20, random_state=0)
from sklearn.metrics import confusion_matrix, classification_report
precisions = [] 

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

print("Test Accuracy {:.2f}%".format(lr.score(X_test,y_test)*100))

precisions.append(lr.score(X_test,y_test)*100)
pred_y = lr.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)



print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, y_test)*100))
# try ro find best k value

scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(X_train, y_train)

    scoreList.append(knn2.score(X_test, y_test))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()





print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)  # n_neighbors means k

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)



print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, y_test)*100))

precisions.append(knn.score(X_test,y_test)*100)
pred_y = knn.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train, y_train)

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(X_test,y_test)*100))

precisions.append(svm.score(X_test,y_test)*100)
pred_y = svm.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(X_test,y_test)*100))

precisions.append(nb.score(X_test,y_test)*100)
pred_y = nb.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

print("Decision Tree Test Accuracy {:.2f}%".format(dtc.score(X_test, y_test)*100))

precisions.append(dtc.score(X_test, y_test)*100)
pred_y = dtc.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(X_train, y_train)

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(X_test,y_test)*100))

precisions.append(rf.score(X_test,y_test)*100)
pred_y = rf.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(n_estimators=100)

abc.fit(X_train, y_train)

print("AdaBoost Accuracy Score : {:.2f}%".format(abc.score(X_test,y_test)*100))

precisions.append(abc.score(X_test,y_test)*100)
pred_y = rf.predict(X_test)

print("Classification report:\n")

print(classification_report(y_test, pred_y))



print("Confusion matrix:\n")

print(confusion_matrix(y_test, pred_y))
methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest", "Adaboost"]

sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=methods, y=precisions, palette="gnuplot")

plt.show()