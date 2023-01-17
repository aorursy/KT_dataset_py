import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train["idhogar"]
train["parentesco1"].value_counts()
train_only_heads = train.drop(train[train["parentesco1"] == 0].index)
train_only_heads["parentesco1"]
train_hna = train_only_heads.dropna(thresh=len(train_only_heads["parentesco1"])/2, axis="columns")
train_hna = train_hna.dropna()
train_hna[(train_hna == 'yes') | (train_hna == 'no')].dropna(axis = 'columns', how='all')
train_hna
Xtrain_h = train_hna.drop(['Target','Id','idhogar','dependency','edjefe','edjefa'] ,axis = 'columns')
Ytrain_h = train_hna.Target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn, Xtrain_h, Ytrain_h, cv=10)
scores.mean()
score_array = []
for i in range(50):
    knn = KNeighborsClassifier(n_neighbors=i+1)
    scores = cross_val_score(knn, Xtrain_h, Ytrain_h, cv=5)
    score_array.append(scores.mean())
plt.plot(score_array, 'ro')
knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(Xtrain_h, Ytrain_h)
Xtest = test.drop(['Id','idhogar','dependency','edjefe','edjefa','rez_esc', 'v18q1', 'v2a1'] ,axis = 'columns')
Xtest
Xtest = Xtest.fillna(0)
pred = knn.predict(Xtest)
prediction = pd.DataFrame(test.Id)
prediction['Target'] = pred
prediction
prediction.to_csv("submition.csv",index = False)