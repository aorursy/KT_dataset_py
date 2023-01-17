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
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
data = pd.read_csv("../input/drug-classification/drug200.csv")
data.head()
data.info()
data.describe()
sns.countplot(data["Drug"])
print(data.Drug.value_counts())
plt.plot(data.Na_to_K[data.Drug == "DrugY"]) # Diğerlerinden kolayca ayrılıyor.
plt.plot(data.Na_to_K[data.Drug == "drugx"])
plt.plot(data.Na_to_K[data.Drug == "drugA"])
plt.plot(data.Na_to_K[data.Drug == "drugB"])
plt.plot(data.Na_to_K[data.Drug == "drugC"])

plt.show()
sns.countplot(data["Cholesterol"])
print(data.Cholesterol.value_counts())
sns.countplot(data["Age"])
plt.show()
sns.countplot(data["BP"])
print(data.BP.value_counts())
data.isnull().sum()
data['Sex'].replace({'M', 'F'},{1, 0}, inplace=True)
data['BP'].replace({'HIGH', 'LOW', 'NORMAL'},{1, 2, 3}, inplace=True)
data['Cholesterol'].replace({'HIGH', 'NORMAL'},{1, 0}, inplace=True)
data.boxplot(column="Na_to_K")
plt.show()
describe = data.describe()
Na_to_K_desc = describe["Na_to_K"]
Na_to_K_desc
data.drop(data[data["Na_to_K"] > 31].index, inplace=True)
data.boxplot(column="Na_to_K")
plt.show()
x = data.drop(['Drug'], axis=1)
y = data['Drug']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.metrics import accuracy_score, plot_confusion_matrix
print("X_train shape:",x_train.shape)
print("X_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
data_stan = scaler.fit_transform(x)

data_stan = pd.DataFrame(data_stan, columns=x.columns)
data_stan.head()
# knn model -> k = 3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

# knn model -> k = 1
knn = KNeighborsClassifier(n_neighbors = 1) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(1,knn.score(x_test,y_test)))
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(x_train, y_train) 

print("score: ", gnb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("score: ", dt.score(x_test,y_test))
from sklearn.svm import SVC
 
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)

print("score: ", svm.score(x_test,y_test))
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

voting = VotingClassifier(estimators=[("dt",dt),("knn",knn),("svm",svm),("gnb",gnb)])

for i in (dt, knn, svm, voting):
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    print(i, "= ",accuracy_score(y_test, y_pred))