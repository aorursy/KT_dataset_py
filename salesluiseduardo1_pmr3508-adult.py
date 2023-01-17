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
ntrain = train.dropna()
ntest= test.dropna()#retiraremos os dados faltantes
numtrain = ntrain.apply(preprocessing.LabelEncoder().fit_transform)
numtest = ntest.apply(preprocessing.LabelEncoder().fit_transform)#transformando para valores numéricos
train.describe()
numtrain.corr()
plt.matshow(train.corr())
plt.colorbar()
correlacao = numtrain.corr()# conseguiremos ver nesse passo as features com maior correlação com a label
correlacao_value = correlacao["income"].abs()
correlacao_value.sort_values()
Xtrain = numtrain[["age","education.num","capital.gain","relationship","hours.per.week","sex"]]
Ytrain = numtrain.income #usando apenas as features com correlação com a label de valores acima de 0.2
knn = KNeighborsClassifier(n_neighbors=40)#utilizando k=40, veremos o resultado da knn
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
scores.mean()
imptrain = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
imptest = test.apply(lambda x:x.fillna(x.value_counts().index[0]))
numImptrain = imptrain.apply(preprocessing.LabelEncoder().fit_transform)
numImptest = imptest.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = numImptrain[["capital.gain", "education.num", "relationship", "age", "hours.per.week", "sex"]]
Ytrain = numImptrain.income
Xtest = numImptest[["capital.gain", "education.num", "relationship", "age", "hours.per.week", "sex"]]
decision = tree.DecisionTreeClassifier(max_depth=5)# decision tree
scores = cross_val_score(decision, Xtrain, Ytrain, cv=10)
scores.mean()
classifier = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(40,), random_state=1) #Rede neural
scores = cross_val_score(classifier, Xtrain, Ytrain, cv=10)
scores.mean()
lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 50) #logistic regression, limitamos o número de iterações.
score = cross_val_score(lr, Xtrain, Ytrain, cv=5)
score.mean()
"Tivemos o melhor resultado com a árvore de decisão"