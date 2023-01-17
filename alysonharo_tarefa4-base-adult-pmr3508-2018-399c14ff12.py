import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import linear_model
train = pd.read_csv("../input/train_data.csv")
train.head()
test = pd.read_csv("../input/test_data.csv")
dropTrain = train.dropna()
dropTest = test.dropna()
numtrain = dropTrain.apply(preprocessing.LabelEncoder().fit_transform)
numtest = dropTrain.apply(preprocessing.LabelEncoder().fit_transform)
train.describe()
numtrain.corr()
correlacao = numtrain.corr()
correlacao_value = correlacao["income"].abs()
correlacao_value.sort_values()
Xtrain = numtrain[["age","education.num","capital.gain","relationship","hours.per.week","sex", "marital.status","capital.loss"]]
Xtrain
Ytrain = numtrain.income
Ytrain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
maxscore=0
n=0
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    meanscore=np.mean(scores)
    if meanscore>maxscore:
        maxscore=meanscore
        n=i
maxscore
n
knn = KNeighborsClassifier(n_neighbors=n)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
mean=np.mean(scores)
mean
imptrain = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
imptest = test.apply(lambda x:x.fillna(x.value_counts().index[0]))
numImptrain = imptrain.apply(preprocessing.LabelEncoder().fit_transform)
numImptest = imptest.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = numImptrain[["age","education.num","capital.gain","relationship","hours.per.week","sex", "marital.status","capital.loss"]]
Ytrain = numImptrain.income
Xtest = numImptest[["age","education.num","capital.gain","relationship","hours.per.week","sex", "marital.status","capital.loss"]]
decision = tree.DecisionTreeClassifier(max_depth=6)
scores = cross_val_score(decision, Xtrain, Ytrain, cv=10)
scores.mean()
classifier = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(40,), random_state=1)
scores = cross_val_score(classifier, Xtrain, Ytrain, cv=10)
scores.mean()
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 50)
score = cross_val_score(lr, Xtrain, Ytrain, cv=5)
score.mean()