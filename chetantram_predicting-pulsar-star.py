import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.metrics import accuracy_score,cohen_kappa_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
data=pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")
data.head()
data.describe().T
data.isna().sum()
data.info()
corr=data.corr()
plt.figure(figsize=(10,7))

sns.heatmap(corr,annot=True)
k=sns.countplot(data["target_class"])



for b in k.patches:

    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))

#Model Building

x=data.drop("target_class",axis=1)

y=data.target_class
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3, random_state=100)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)
model_tree=dtree.predict(xtest)
print("Decision Tree Acc Score",accuracy_score(ytest,model_tree))

print("Decision Tree Kappa score",cohen_kappa_score(ytest,model_tree))
#Random Forest

rf = RandomForestClassifier()
rf.fit(xtrain,ytrain)
model_forest=rf.predict(xtest)
print("Random Forest Acc Score",accuracy_score(ytest,model_forest))

print("Random Forest Kappa Score",cohen_kappa_score(ytest,model_forest))
#Naive Bayes

nb = GaussianNB()

nb.fit(xtrain, ytrain)

model_nb=nb.predict(xtest)
print("Naive Bayes Acc Score",accuracy_score(ytest,model_nb))

print("Naive Bayes Kappa Score",cohen_kappa_score(ytest,model_nb))
#K-Nearest Neigbors

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(xtrain, ytrain)

model_knn = knn.predict(xtest)
print("KNN Acc Score",accuracy_score(ytest,model_knn))

print("KNN Kappa Score",cohen_kappa_score(ytest,model_knn))