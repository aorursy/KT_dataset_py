import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train_data.csv")
train.head()
train.describe()
train = train.dropna()
from sklearn import preprocessing
Xtrain = train.drop(columns=["income"])
Xtrain = Xtrain.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain.head()
Ytrain = train["income"]
Ytrain.head()
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
sel.fit_transform(Xtrain)
from sklearn import feature_selection
a = list(Xtrain)
feature_scores = feature_selection.mutual_info_classif(Xtrain,Ytrain)
plt.bar(a,feature_scores)
plt.xticks(rotation=90)
plt.show()
best_score = []
i = 0
for x in range(0,len(a)):
    if feature_scores[x]>0.02 or a[x] == "Id" :
        best_score.insert(i,a[x])
        i+=1      
Xtrain = Xtrain[best_score]
Xtrain.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(Xtrain, Ytrain)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(Xtrain, Ytrain)
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(Xtrain, Ytrain)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
scoresrandom = cross_val_score(rf, Xtrain, Ytrain, cv=10)
scoresrandom
scoresLDA = cross_val_score(lda, Xtrain, Ytrain, cv=10)
scoresLDA
scoresDecisionTree = cross_val_score(decision_tree, Xtrain, Ytrain, cv=10)
scoresDecisionTree
Xtest = pd.read_csv("../input/test_data.csv")
Xtest = Xtest.apply(preprocessing.LabelEncoder().fit_transform)
Xtest = Xtest[best_score]
Xtest.head()
YtestPred = lda.predict(Xtest)
YtestPred
pred = pd.DataFrame(Xtest.Id)
pred["income"] = YtestPred
pred.to_csv("prediction.csv", index=False)