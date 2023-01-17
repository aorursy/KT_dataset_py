# importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import os

df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.info()
df.describe()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt=" .1f")
plt.show()
# age analysis
df.age.value_counts()[:10]
sns.barplot(x=df.age.value_counts()[:10].index, y=df.age.value_counts()[:10].values)
df.target.value_counts()
countFemale = len(df[df.sex== 0])
countMale = len(df[df.sex== 1])
print("percentage of  Female patients with heart disease: {:.2f}%".format((countFemale/len(df.target))*100))
print("percentage of Male patients with heart disease: {:.2f}%".format((countMale/len(df.target))*100))

countHaveDisease = len(df[df.target ==1])
countNoDisease = len(df[df.target ==0])
print("percentage of patient who have disease: {:.2f}%".format((countHaveDisease/len(df.target))*100))
print("percentage of patient who do not have disease: {:.2f}%".format((countNoDisease/len(df.target))*100))
young_ages = df[(df.age>=29) & (df.age<40)]
middle_ages = df[(df.age>=40) & (df.age<55)]
elderly_ages = df[(df.age>=55)]
print("young_ages:", len(young_ages))
print("middle_ages:", len(middle_ages))
print("elderly_ages:", len(elderly_ages))

colors = ["blue","green","red"]
explode = [0.1, 0.1, 0.1]
plt.figure(figsize=(8,8))
plt.pie([len(young_ages), len(middle_ages), len(elderly_ages)] , labels= ["young_ages","middele_ages","elderly_ages"])
plt.show()
sns.countplot(df.target)
plt.xlabel("Target")
plt.ylabel("count")
plt.show()
#model building
#logistic Regression
from sklearn.linear_model import LogisticRegression
x_data = df.drop(["target"],axis=1)
y= df.target.values
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size= 0.2, random_state=0 )
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Accuracy of logistic regression is: {:.2f}%".format(lr.score(x_test,y_test)*100))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
print("Accuracy of KNN is: {:.2f}%".format(knn.score(x_test,y_test)*100))

#Support Vector
from sklearn.svm import SVC
svc = SVC(random_state=1)
svc.fit(x_train,y_train)
print("Accuracy of svc is: {:.2f}%".format(svc.score(x_test,y_test)*100))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Accuracy of nb is: {:.2f}%".format(nb.score(x_test,y_test)*100))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, random_state=1)
rf.fit(x_train,y_train)
print("Accuracy of Random Forest is: {:.2f}%".format(rf.score(x_test,y_test)*100))
