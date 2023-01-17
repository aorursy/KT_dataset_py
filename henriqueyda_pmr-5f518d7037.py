import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import seaborn as sns # data visualization library  
train = pd.read_csv("../input/train-data/train_data.csv")
train.head()
Ytrain = train["ham"]
a = sns.countplot(Ytrain,label="Count") 
S, NS = Ytrain.value_counts()
print('Number of Spams: ',S)
print('Number of Not Spams : ',NS)
from sklearn import feature_selection
Xtrain = train.drop(columns=["Id","ham"])
Xtrain.describe()
a = list(Xtrain)
feature_scores = feature_selection.mutual_info_classif(Xtrain,Ytrain)
plt.bar(a,feature_scores)
plt.xticks(rotation=90)
plt.show()
best_score = []
i = 0
for x in range(0,len(a)):
    if feature_scores[x]>0.06 :
        best_score.insert(i,a[x])
        i+=1      
best_score
Xtrain = train[best_score]
Xtrain.head()
corrTest = Xtrain.join(Ytrain)
corrTest.head()
plt.matshow(corrTest.corr())
correlation = corrTest.corr(method = 'pearson').ham
correlation
j = 0
best_pearson = []
for x in range(0, len(correlation)):
    if abs(correlation[x]) < 0.06 :
        best_pearson.insert(j,x)
        j+=1
best_pearson
Xtrain = Xtrain.drop(Xtrain.columns[best_pearson],axis=1)
Xtrain.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xtrain, Ytrain)
Nb = GaussianNB()

Nb.fit(Xtrain, Ytrain)
from sklearn.model_selection import cross_val_score
scoresKnn = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scoresKnn
scoresNb = cross_val_score(Nb, Xtrain, Ytrain, cv=10)
scoresNb
test = pd.read_csv("../input/test-features/test_features.csv")
Xtest = test[best_score]
Xtest = Xtest.drop(Xtest.columns[best_pearson],axis=1)
Xtest.head()
test_pred = Nb.predict(Xtest)
test_pred
pred = pd.DataFrame(test.Id)
pred["ham"] = test_pred
pred.head()

pred.to_csv("prediction.csv", index=False)