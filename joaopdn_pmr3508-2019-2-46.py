import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score as vdcruz

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
train = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        engine='python',

        na_values="?")

test = pd.read_csv("../input/adult-pmr3508/test_data.csv",na_values="?")
train.head()

index = test.Id
train["sex"].value_counts().plot(kind="bar")

train["education.num"].value_counts().plot(kind="bar")

train["relationship"].value_counts().plot(kind="bar")
train["education.num"].value_counts().plot(kind="box")
for df in [train,test]:

    df.set_index('Id',inplace=True)

train['income'].astype('category')
test.head()
train.head()
total = train.isnull().sum().sort_values(ascending = False)

percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending = False)

train_faltante = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])

train_faltante.head()
Xtrain = train.drop(columns='income')

Ytrain = train.income

Ytrain.head()
Xtrain.head()
print(len(Xtrain))

print(len(Xtrain.dropna()))
for A in Xtrain.columns:

    Xtrain[A].fillna(Xtrain[A].mode()[0], inplace=True)

for A in test.columns:

    test[A].fillna(test[A].mode()[0], inplace=True)
Xtrain.shape

Xtrain_nb = Xtrain
One_Hot_Xtrain = pd.get_dummies(Xtrain)

One_Hot_test = pd.get_dummies(test)

Xtrain, test = One_Hot_Xtrain.align(One_Hot_test,join='left',axis=1)

Xtrain.head()
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

Xtrain=sc_X.fit_transform(Xtrain)

test=sc_X.transform(test)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

Xtrain = my_imputer.fit_transform(Xtrain)

test = my_imputer.transform(test)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, random_state=0,test_size=0.2)
print(len(Xtrain))

print(len(Ytrain))
knn = KNeighborsClassifier(n_neighbors=49)

knn.fit(Xtrain,Ytrain)
accuracy_score(Ytest,knn.predict(Xtest))
resultado_knn = vdcruz(knn, Xtrain, Ytrain, cv=5)

resultado_knn.mean()
matriz_de_confusao = confusion_matrix(knn.predict(Xtest),Ytest)

print(matriz_de_confusao)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(Xtrain,Ytrain)
accuracy_score(Ytest,rf.predict(Xtest))
resultado_rf = vdcruz(rf, Xtrain, Ytrain, cv=5)

resultado_rf.mean()
matriz_de_confusao = confusion_matrix(rf.predict(Xtest),Ytest)

print(matriz_de_confusao)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(Xtrain, Ytrain)

nb.score
accuracy_score(Ytest,nb.predict(Xtest))
matriz_de_confusao = confusion_matrix(nb.predict(Xtest),Ytest)

print(matriz_de_confusao)
resultado_nb = vdcruz(nb, Xtrain, Ytrain, cv=10)

resultado_nb.mean()
from sklearn.svm import SVC

svc=SVC(gamma='auto')

svc.fit(Xtrain,Ytrain)
accuracy_score(Ytest,svc.predict(Xtest))
matriz_de_confusao = confusion_matrix(svc.predict(Xtest),Ytest)

print(matriz_de_confusao)
resultado_svc = vdcruz(svc, Xtrain, Ytrain, cv=5)

resultado_svc.mean()
Ytest_KNN = knn.predict(test)

Ytest_rf = rf.predict(test)

Ytest_nb = nb.predict(test)

Ytest_svc = svc.predict(test)

Ctrain = pd.read_csv("../input/california/Ctrain.csv",

        engine='python',

        na_values="?")

Ctest = pd.read_csv("../input/california/Ctest.csv",na_values="?")
Ctrain.head()
Ctest.head()
for df in [Ctrain,Ctest]:

    df.set_index('Id',inplace=True)
XCtrain = Ctrain.drop(columns='median_house_value')

YCtrain = Ctrain.median_house_value

YCtrain.head()
print(len(XCtrain))

print(len(XCtrain.dropna()))
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(XCtrain,YCtrain)
reg.score(XCtrain,YCtrain)
reg.coef_
reg.intercept_
resultado_reg = vdcruz(reg, XCtrain, YCtrain, cv=10)

resultado_reg.mean()
YCtest = reg.predict(Ctest)

YCtest
from sklearn import linear_model

lasso = linear_model.Lasso(alpha=0.1)

lasso.fit(XCtrain,YCtrain)



resultado_lasso = vdcruz(lasso, XCtrain, YCtrain, cv=10)

resultado_lasso.mean()
from sklearn.tree import DecisionTreeRegressor
scores_mean = []

scores_std = []



k_lim_inf = 1

k_lim_sup = 30



folds = 10



k_max = None

max_acc = 0



i = 0

print('Finding best k...')

for k in range(k_lim_inf, k_lim_sup):

    

    regr = DecisionTreeRegressor(max_depth=k)

    

    score = vdcruz(regr, XCtrain, YCtrain, cv = folds)

    

    scores_mean.append(score.mean())

    scores_std.append(score.std())

    

    if scores_mean[i] > max_acc:

        k_max = k

        max_acc = scores_mean[i]

    i += 1

    print('   k = {0} | Best CV acc = {1:2.2f}% (best k = {2})'.format(k, max_acc*100, k_max))

print('\nBest k: {}'.format(k_max))