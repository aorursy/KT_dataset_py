# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn import metrics
dataset = pd.read_csv("../input/heart.csv")
dataset.shape
dataset.dtypes
dataset.target.value_counts()
sns.countplot(x='target', data=dataset, palette="RdBu")
plt.show()
maleCount = len(dataset[dataset.sex == 1])
femaleCount = len(dataset[dataset.sex == 0])
print("Male Patients = {:.2f} %".format(maleCount/len(dataset) * 100))
print("Female Patients = {:.2f} %".format(femaleCount/len(dataset) * 100))
pd.crosstab(dataset.sex, dataset.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency w.r.t sex')
plt.ylabel('Frequency')
plt.xlabel('SEX(0 = male, 1 = female)')
plt.legend(["Disease","No Disease"])
plt.show()
df=dataset
#scatter plot
plt.scatter(x=df.age[df.target==1], y=df.thalach[df.target==1], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[df.target==0], c="green")
plt.xlabel("Age")
plt.ylabel("Maaximum Heart Rate")
plt.show()
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#FAF7A6','#FD5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#DDCF00','#F81845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()
pd.crosstab(df.cp, df.target).plot(kind="bar",figsize=(15,6),color=['#DDCF00','#F81845' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('CPT - Chest Pain Type ')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()
X = df.drop(['target'], axis=1)
y = df['target']
y.head()
logreg=LogisticRegression()
predicted = cross_val_predict(logreg, X, y, cv=10)
metrics.accuracy_score(y, predicted)
print(metrics.classification_report(y, predicted))
X = df.drop(['target'], axis=1)
y = df['target']
# Normalize
x = (X - np.min(X)) / (np.max(X) - np.min(X)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test.T,y_test.T)*100))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("{} NN Score: {:.2f}%".format(3, knn.score(x_test.T, y_test.T)*100))
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()


print("Maximum KNN Score is {:.2f}%".format((max(scoreList))*100))
