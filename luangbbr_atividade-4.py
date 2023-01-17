import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#leitura dataset de treino
train_raw = pd.read_csv("../input/train_data.csv", sep = ",")
train = train_raw.copy(deep=True)
target = train_raw["income"]
display(train_raw.shape,
train_raw.head())
#leitura do dataset de teste
test_raw = pd.read_csv("../input/test_data.csv", sep = ",",header=0)
test = test_raw.copy()
display(test_raw.shape,test_raw.head())
train['income']=train['income'].map({'<=50K': 0, '>50K': 1})
num_target=target.map({'<=50K': 0, '>50K': 1})

train["sex"]=train['sex'].map({"Female": 0, 'Male': 1})
train["native.country"]=train["native.country"].map(lambda x: 1 if x=="United-States" else 0)
test["sex"]=test['sex'].map({"Female": 0, 'Male': 1})
test["native.country"]=test["native.country"].map(lambda x: 1 if x=="United-States" else 0)
train.corr()["income"].abs().sort_values(ascending=False)

no_target=train.reindex(["education.num","age","hours.per.week","capital.gain","capital.loss","sex"],axis=1)
pca = PCA(n_components=2)
pca.fit(no_target)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
pca.explained_variance_ratio_
train_pca=pd.DataFrame(pca.transform(no_target))
testFeatures=test.reindex(["education.num","age","hours.per.week","capital.gain","capital.loss","sex"],axis=1)
test_pca=pd.DataFrame(pca.transform(testFeatures))
plt.scatter(train_pca[1],train_pca[0],s=1,cmap="magma",c=-1*num_target)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=30)
from sklearn.model_selection import cross_val_score
score=cross_val_score(knn, train_pca, num_target, cv=10)
np.array(score).mean()
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(n_estimators=100, max_depth=11,random_state=0)
score_clf=cross_val_score(clf, train_pca, num_target, cv=10)
np.array(score_clf).mean()
from catboost import CatBoostClassifier
CBC = CatBoostClassifier(verbose=False)
score_cbc=cross_val_score(CBC, train_pca, num_target, cv=2)
np.array(score_cbc).mean()
clf.fit(train_pca,num_target)
R = clf.predict(test_pca)
R = pd.Series(R).map({0:'<=50K', 1:'>50K'})
R.head()
#predict
arq=open("result.csv","w")
arq.write("Id,income\n")
for i,j in zip(list(R),list(test_raw["Id"])):
    arq.write(str(j)+","+str(i)+"\n")
arq.close()
