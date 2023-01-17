# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df2 = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

print(plt.style.available)

plt.style.use("ggplot")
df.info()

# its showing data's info as you can see data's  not null value
df.head()
df.describe()



# df["class"].value_counts()

sns.countplot(x = "class", data = df)

plt.show()

df["class"].value_counts()

df.corr()
# split data

from sklearn.model_selection import train_test_split



x,y = df.loc[:,df.columns != "class"],df.loc[:,"class"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)



# find the best "k" value

from sklearn.neighbors import KNeighborsClassifier

train_accuracy = []

test_accuracy = []

for k in range(1,30):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train,y_train))

    test_accuracy.append(knn.score(x_test,y_test))



print("For K: {},  best accuracy : {}:".format(1+test_accuracy.index(max(test_accuracy)),max(test_accuracy)))

print("")

knn_score = max(test_accuracy)



# confusion matrix and classification report

from sklearn.metrics import classification_report, confusion_matrix

predict = knn.predict(x_test)

cm = confusion_matrix(y_test,predict)

print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))

print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))

print("")

print("Classification Report:\n\n",classification_report(y_test,predict))



sns.pairplot(data= df,hue="class", kind = "reg")

plt.show()
from sklearn.tree import DecisionTreeClassifier



d_tree = DecisionTreeClassifier()

d_tree.fit(x_train,y_train)

d_score = d_tree.score(x_test,y_test)

print("Decision Tree Score:",d_score)
from sklearn.ensemble import RandomForestClassifier

r_forest = RandomForestClassifier(n_estimators=100,random_state=1)

r_forest.fit(x_train,y_train)

rf_score = r_forest.score(x_test,y_test)

print("Random Forest Score:",rf_score)
from sklearn.linear_model import LogisticRegression

l_reg = LogisticRegression(random_state=42,max_iter=100)

l_reg.fit(x_train,y_train)

lr_score = l_reg.score(x_test,y_test)

print("Logistic Regression score:",lr_score)
df["class"] = [1 if i=="Abnormal" else 0 for i in df["class"]]

x,y = df.loc[:,df.columns != "class"],df.loc[:,"class"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
from sklearn.linear_model import Ridge, Lasso





ridg = Ridge(alpha = 0.1,normalize = True)

ridg.fit(x_train,y_train)

#ridg_predict = ridg.predict(x_test)

rg_score = ridg.score(x_test,y_test)

print("Ridge Score:",rg_score )
rg_score

lr_score

rf_score

d_score

knn_score



score_data_ = {"Ridge":[rg_score],"Logistic Regression":[lr_score],

               "Random Forrest Classifer":[rf_score],"Decision Tree":[d_score],

              "KNN":[knn_score]}



score_df = pd.DataFrame(score_data_)
plt.figure(figsize = [10,6])

sns.barplot(data = score_df)

plt.xlabel("Algorithms")

plt.ylabel("Score")

plt.title("ML Algorithms Scores")

plt.show()