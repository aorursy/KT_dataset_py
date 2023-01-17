# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sys

import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn import utils

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/Admission_Predict.csv")

print("there are", len(data.columns), "columns: ")

print (data.columns)

data.columns = [c.replace(' ', '_') for c in data.columns]

data.info()
data.GRE_Score.plot(kind="line",color="green",label="GRE Score",grid=True,linestyle=":")

data.TOEFL_Score.plot(kind="line",color="purple",label="TOEFL Score",grid=True)

data.University_Rating.plot(kind="line",color="pink",label="UNI rank",grid=True)

data.SOP.plot(kind="line",color="orange",label="SOP",grid=True)

data.LOR_.plot(kind="line",color="blue",label="SOP",grid=True)

plt.legend(loc="upper right") #legend: puts feature label into plot

plt.xlabel("indexes")

plt.ylabel("Features")

plt.title("GRE SCore")

plt.show()



#Using Scatter plot for GRE score and Chance of admit.


data.plot(kind="scatter" , x="GRE_Score", y="Chance_of_Admit_", alpha=0.5, color ="red")

plt.xlabel("GRE_Score")

plt.ylabel("Chance_of_Admit_")

plt.title("GRE and Chance of Admit with scatter")

plt.show()

data.plot(kind="scatter" , x="LOR_", y="Chance_of_Admit_", alpha=0.5, color ="red")

plt.xlabel("LOR")

plt.ylabel("Chance_of_Admit_")

plt.title("GRE and Chance of Admit with scatter")

plt.show()
data.plot(kind="scatter" , x="SOP", y="Chance_of_Admit_", alpha=0.5, color ="red")

plt.xlabel("SOP")

plt.ylabel("Chance_of_Admit_")

plt.title("GRE and Chance of Admit with scatter")

plt.show()
data.describe()

#We can see visualization of statistical calculations.

data.boxplot(column="GRE_Score", by="Research")

# ages value by sex

plt.show()

data.describe()

#We can see visualization of statistical calculations.

data.boxplot(column="Chance_of_Admit_", by="Research")

# ages value by sex

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,fmt='.1f')

plt.show()
from scipy import stats 

#I include it for see pearsonr value.

g= sns.jointplot(data['GRE_Score'],data['Chance_of_Admit_'],kind="kde",height=5)

g = g.annotate(stats.pearsonr)

plt.savefig('graph.png')

plt.show()
y = data.Chance_of_Admit_.values

print(len(y))

for i in range(len(y)):

    if y[i] <= 0.8:

            y[i] = 0

    else:

            y[i] = 1



x = data.drop(['Chance_of_Admit_','Serial_No.'], axis = 1)

#y =np.ones((80))

#y=np.random.randint(2, size=80)

y=y.T

#x=np.ones((80,6))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
lr = LogisticRegression()

lr.fit(x_train,y_train)

print("Test Accuracy Logistics Reg {:.2f}%".format(lr.score(x_test,y_test)*100))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train, y_train)

print("Test Accuracy of SVM Algorithm: {:.2f}%".format(svm.score(x_test,y_test)*100))


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test, y_test)*100))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test,y_test)*100))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train, y_train)

print("Decision Tree Test Accuracy {:.2f}%".format(dtc.score(x_test, y_test)*100))



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf.fit(x_train, y_train)

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(rf.score(x_test,y_test)*100))
#This is my first attemt to visualize data in this field.

# Please suggest ways in which I can Improve.

# Upvote if you find this useful.