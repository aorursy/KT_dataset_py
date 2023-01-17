import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
sklearn.utils.check_random_state(42)
train = pd.read_csv('../input/train_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')

test = pd.read_csv('../input/test_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')
train.info()
train.head()
train.describe()
train.isnull().any(axis=0)
pd.get_dummies(train['workclass'])
for col in train.columns:
    if train[col].dtype.name == 'object':
        train[col].value_counts().plot(title=col, kind='bar', label='')
        plt.show()
    else:
        train[col].hist(bins=15).set_title(col)
        plt.show()
print('Zero value frequencies:')
print('capital.gain = {0:.2f}%'.format(sum(train['capital.gain'] == 0) / train.shape[0] * 100))
print('capital.loss = {0:.2f}%'.format(sum(train['capital.loss'] == 0) / train.shape[0] * 100))
train = train.dropna()
Xtrain = train.drop(columns=["income"])
Xtrain = Xtrain.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain.head()
Ytrain = train["income"]
Ytrain.head()
sel = VarianceThreshold(threshold=0.01)
sel.fit_transform(Xtrain)
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
knn = KNeighborsClassifier(n_neighbors=25)
rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(Xtrain, Ytrain)
lda = LinearDiscriminantAnalysis()
lda.fit(Xtrain, Ytrain)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(Xtrain, Ytrain)
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