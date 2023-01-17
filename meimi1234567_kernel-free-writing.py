import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heartcsv/heart.csv')

df.head()
df.target.value_counts()
sns.countplot(x="target", data=df)

plt.show()

countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))

#:.2f取到小數點第二位
sns.countplot(x='sex', data=df, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()

countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
#了解target的平均

df.groupby('target').mean()
#年齡和目標可視化

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(30,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
#性別和目標可視化

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

#plt.legend是標籤名稱定義

plt.ylabel('Frequency')

plt.show()
#有沒有疾病和最大心率及年齡關係（有疾病心跳偏高，年齡落在40-60間）

plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

#scatter散布圖

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')

plt.legend(["Disease", "Not Disease"])

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()
#血糖有沒有超過120，有病的通常血糖都超過120

pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
#有無病和胸痛等級關系

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
#pd.get_dummies類別變量轉換為標籤變量

a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]

df = pd.concat(frames, axis = 1)#合併frames

df = df.drop(columns = ['cp', 'thal', 'slope'])#去掉原始東西

df.head()
y = df.target.values

#區分x 和 y data

x_data = df.drop(['target'], axis = 1)
# Normalize(歸一化)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
#區分測試及訓練集

#有20%的資料會用在測試集中

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
accuracies = {}



lr = LogisticRegression()

lr.fit(x_train,y_train)#fit訓練訓練集

acc = lr.score(x_test,y_test)*100  #輸出準確率



accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))  
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)



print("{} NN Score: {:.2f}%".format(2, knn.score(x_test, y_test)*100))
# try ro find best k value(找最佳的k值)

scoreList = []

for i in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn2.fit(x_train, y_train)

    scoreList.append(knn2.score(x_test, y_test))

    

plt.plot(range(1,20), scoreList)#print 畫圖不能放到for裡面

plt.xticks(np.arange(1,20,1))   #距離在1~19之間（算頭不算尾）

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()



acc = max(scoreList)*100

accuracies['KNN'] = acc

print("Maximum KNN Score is {:.2f}%".format(acc))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train, y_train)

#acc算準確率

acc = svm.score(x_test,y_test)*100

accuracies['SVM'] = acc

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)



acc = nb.score(x_test,y_test)*100

accuracies['Naive Bayes'] = acc

print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)



acc = dtc.score(x_test, y_test)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train, y_train)



acc = rf.score(x_test,y_test)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()