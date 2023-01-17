import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
test = pd.read_csv("../input/test_data.csv",
                   sep=r'\s*,\s*',
                   engine='python',
                   na_values="?")
train = pd.read_csv("../input/train_data.csv",
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
train
features = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation',
            'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']
train['income'] = train['income'].replace(['<=50K','>50K'], [0,1])
train['workclass'].value_counts(normalize=True, dropna=False)
train['education'].value_counts(normalize=True, dropna=False)
train['marital.status'].value_counts(normalize=True, dropna=False)
train['occupation'].value_counts(normalize=True, dropna=False)
train['relationship'].value_counts(normalize=True, dropna=False)
train['race'].value_counts(normalize=True, dropna=False)
train['sex'].value_counts(normalize=True, dropna=False)
train['native.country'].value_counts(normalize=True, dropna=False)
train['workclass'] = train['workclass'].replace(100*train['workclass'].value_counts(normalize=True, dropna=False))
train['education'] = train['education'].replace(100*train['education'].value_counts(normalize=True, dropna=False))
train['marital.status'] = train['marital.status'].replace(100*train['marital.status'].value_counts(normalize=True, dropna=False))
train['occupation'] = train['occupation'].replace(100*train['occupation'].value_counts(normalize=True, dropna=False))
train['relationship'] = train['relationship'].replace(100*train['relationship'].value_counts(normalize=True, dropna=False))
train['race'] = train['race'].replace(100*train['race'].value_counts(normalize=True, dropna=False))
train['sex'] = train['sex'].replace(100*train['sex'].value_counts(normalize=True, dropna=False))
train['native.country'] = train['native.country'].replace(100*train['native.country'].value_counts(normalize=True, dropna=False))
train
dataset = train.iloc[:,1:]
pearson = dataset.corr(method='pearson')
pearson
pearson_values = abs(pearson.income)
pearson_values.pop('income')
pearson_values.plot(kind='bar')
test['workclass'] = test['workclass'].replace(100*test['workclass'].value_counts(normalize=True, dropna=False))
test['education'] = test['education'].replace(100*test['education'].value_counts(normalize=True, dropna=False))
test['marital.status'] = test['marital.status'].replace(100*test['marital.status'].value_counts(normalize=True, dropna=False))
test['occupation'] = test['occupation'].replace(100*test['occupation'].value_counts(normalize=True, dropna=False))
test['relationship'] = test['relationship'].replace(100*test['relationship'].value_counts(normalize=True, dropna=False))
test['race'] = test['race'].replace(100*test['race'].value_counts(normalize=True, dropna=False))
test['sex'] = test['sex'].replace(100*test['sex'].value_counts(normalize=True, dropna=False))
test['native.country'] = test['native.country'].replace(100*test['native.country'].value_counts(normalize=True, dropna=False))
test
X = train[features]
Y = train.income.replace([0,1], ['<=50K','>50K'])
Xtest = test[features]
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X,Y)
scores = cross_val_score(knn, X, Y, cv=10)
scores.mean()
RFC = RandomForestClassifier(n_estimators=100, max_features='sqrt')
RFC.fit(X,Y)
scores = cross_val_score(RFC, X, Y, cv=10)
scores.mean()
SVM = LinearSVC()
SVM.fit(X,Y)
scores = cross_val_score(SVM, X, Y, cv=10)
scores.mean()
X_PCA = PCA(n_components=1).fit_transform(X)
X_PCA
SVM_PCA = LinearSVC()
SVM_PCA.fit(X_PCA,Y)
scores = cross_val_score(SVM_PCA, X_PCA, Y, cv=10)
scores.mean()
Ytest = RFC.predict(Xtest)
result = np.vstack((test['Id'], Ytest)).T
x = ["Id","income"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("resultadosRFC.csv", index = False)
Resultado
