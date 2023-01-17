import numpy as np

import pandas as pd
p = pd.read_csv('train.csv')
from sklearn.model_selection import train_test_split

x = p.drop(['label'], axis=1)

y = p['label']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
from sklearn import metrics

print(metrics.classification_report(ytest, predicted))
from sklearn.metrics import accuracy_score, cohen_kappa_score

accuracy_score(ytest,predicted)
cohen_kappa_score(ytest,predicted)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
newp = sc.fit_transform(p.drop('label', axis=1))
newp = pd.DataFrame(newp, columns = p.columns[1:786])
from sklearn.decomposition import PCA
pc = PCA(n_components=80)

pca_val = pc.fit_transform(newp)
a = np.cumsum(pc.explained_variance_ratio_)

import matplotlib.pyplot as plt

plt.plot(a,'-o', linestyle = 'dashed', markerfacecolor = 'red', markersize=10)

plt.xlabel('no of components')

plt.ylabel('explained variance')

plt.title('finding the no of components')

plt.show()
pca_val = pc.fit_transform(newp)

pca_p=pd.DataFrame(pca_val)
xtrain, xtest, ytrain, ytest = train_test_split(pca_p, y, test_size=0.30, random_state=0)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
from sklearn.metrics import accuracy_score, cohen_kappa_score

accuracy_score(ytest,predicted)
cohen_kappa_score(ytest,predicted)