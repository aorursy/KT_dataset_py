import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", na_values="?")

train.head()
[lin_i, col_i] = train.shape

train.dropna(inplace=True)
[lin_f, col_f] = train.shape

print('O número incial de linhas era de', lin_i,'e o número final é de',lin_f,'\nUma redução de',(lin_i-lin_f)/lin_i *100,'%')
train.describe()
over_50k  = train[train['income'] == '>50K']

under_50k  = train[train['income'] == '<=50K']
over_50k.describe()
under_50k.describe()
train['age'].hist()
train['native.country'].value_counts().plot(kind='bar')
train['workclass'].value_counts().plot(kind='bar')
pd.crosstab(train["age"],train["income"],normalize="index").plot(figsize=(8,5))
pd.crosstab(train["sex"],train["income"],normalize="index").plot.barh(stacked="True")
pd.crosstab(train["race"],train["income"],normalize="index").plot.barh()
pd.crosstab(train["education.num"],train["income"],normalize="index").plot.barh(stacked=True, figsize=(9,5))
train = train.drop(['native.country', 'Id'], axis=1)
from sklearn.preprocessing import LabelEncoder

categorical = ['workclass', 'education',

       'marital.status', 'occupation', 'relationship', 'race', 'sex']

names = ['occupation', 'relationship', 'race']



for i in categorical: 

    le = LabelEncoder()

    train[i].astype("str")

    train[i] = le.fit_transform(train[i])



train.head()
X = train.drop(["income", "fnlwgt", "workclass", "race", 'age'], axis =1)

Y = train["income"]

X.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



scores = []

score_max = 0

k_max = 0





for k in range(1, 30):    

    KNN = KNeighborsClassifier(n_neighbors=k, p = 1)

    score = cross_val_score(KNN, X, Y).mean()

    

    if score > score_max:

        k_max = k

        score_max = score

print('O valor de k que maximixa a curácia é o', k_max)
knn = KNeighborsClassifier(n_neighbors=10, p = 1)

knn.fit(X, Y)
teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values='?')

test = teste.drop(["Id", "fnlwgt", "workclass", "race", "native.country", 'age'], axis=1)
from sklearn.impute import SimpleImputer, KNNImputer

for i in ["occupation"]:

    imp_mf = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



    imp_mf.fit(np.array(test[i]).reshape(-1, 1))

    test[i] = imp_mf.transform(np.array(test[i]).reshape(-1, 1))
from sklearn.preprocessing import LabelEncoder

categorical_teste = [ 'education',

       'marital.status', 'occupation', 'relationship', 'sex']





for i in categorical_teste: 

    le = LabelEncoder()

    test[i].astype("str")

    test[i] = le.fit_transform(test[i])
test.head()
X_test = test.values
y_pred = knn.predict(X_test)
pred = pd.DataFrame(columns = ["Id","income"])



pred.Id = teste["Id"]

pred.income=y_pred

pred.income.replace({0:"<=50K", 1:">50K" }, inplace=True)

pred.to_csv("submission.csv", index=False)