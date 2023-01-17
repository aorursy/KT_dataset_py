#import required packages

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)
df.head()
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', 1), df['diagnosis'], test_size = .2, random_state=10)
#Beginning of dictionary to keep track of model scores

accuracies = {}
lr = LogisticRegression()

lr.fit(X_train,y_train)

acc = lr.score(X_test,y_test)*100



accuracies['Logistic Regression'] = acc

print("Test Accuracy {:.2f}%".format(acc))
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)



print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, y_test)*100))
# try to find best k value

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



acc = max(scoreList)*100

accuracies['KNN'] = acc

print("Maximum KNN Score is {:.2f}%".format(acc))
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)



acc = dtc.score(X_test, y_test)*100

accuracies['Decision Tree'] = acc

print("Decision Tree Test Accuracy {:.2f}%".format(acc))
model = RandomForestClassifier(max_depth=5)

model.fit(X_train, y_train)



acc = model.score(X_test,y_test)*100

accuracies['Random Forest'] = acc

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))

plt.show()