import pandas as pd
import sklearn
# Target - the target is an ordinal variable indicating groups of income levels. 
# 1 = extreme poverty 
# 2 = moderate poverty 
# 3 = vulnerable households 
# 4 = non vulnerable households

train = pd.read_csv("../input/train.csv")
train
from sklearn import preprocessing
list(train)
#revela quais parâmetros possuem muitos NaN
a = train.isnull().sum().tolist()
for x in range(0,len(a)):
    if a[x] > 10:
        print(x)
a
train["v2a1"].value_counts(dropna = False)
train["v18q1"].value_counts(dropna = False)
train["rez_esc"].value_counts(dropna = False)
na_train = train.drop(columns=['v2a1','v18q1','rez_esc','edjefe','edjefa','dependency','idhogar'],axis=1)
na_train
na_train = na_train.dropna()
na_train
test = pd.read_csv("../input/test.csv")
na_test = test.drop(columns=['v2a1','v18q1','rez_esc','edjefe','edjefa','dependency','idhogar'],axis=1)
na_test = na_test.fillna(na_test.mean())
b = na_test.isnull().sum().tolist()
b
Xtrain = na_train.loc[:, na_train.columns != "Target"]
Xtrain = Xtrain.loc[:, Xtrain.columns != "Id"]
Xtest = na_test
Ytrain = na_train["Target"]
from sklearn.neighbors import KNeighborsClassifier
Xtest = na_test.loc[:,na_test.columns != "Id"]
knn = KNeighborsClassifier(n_neighbors=10)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict(Xtest)
YtestPred
pred = pd.DataFrame(na_test.Id)
pred["Target"] = YtestPred
pred
pred.to_csv("prediction.csv", index=False)
