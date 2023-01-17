%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import os

os.listdir('../input/banknote-authentication')
columns = ["var","skewness","curtosis","entropy","class"]

df = pd.read_csv("../input/banknote-authentication/data_banknote_authentication.txt",index_col=False, names = columns)
df.head(3)
f, ax = plt.subplots(1, 4, figsize=(10,3))

vis1 = sns.distplot(df["var"],bins=10, ax= ax[0])

vis2 = sns.distplot(df["skewness"],bins=10, ax=ax[1])

vis3 = sns.distplot(df["curtosis"],bins=10, ax= ax[2])

vis4 = sns.distplot(df["entropy"],bins=10, ax=ax[3])

f.savefig('subplot.png')

sns.pairplot(df)
X = df.iloc[:,0:4].values

y = df.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)
sc = StandardScaler()

X_train_sd = sc.fit_transform(X_train)

X_test_sd = sc.transform(X_test)
cov_matrix = np.cov(X_train_sd.T)

print('Covariance Matrix \n%s', cov_matrix)



e_vals, e_vecs = np.linalg.eig(cov_matrix)

print('Eigenvectors \n%s' %e_vecs)

print('\nEigenvalues \n%s' %e_vals)
tot = sum(e_vals)

var_exp = [( i /tot ) * 100 for i in sorted(e_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)



print(var_exp)

print("Cumulative Variance Explained", cum_var_exp)
# Ploting 

plt.figure(figsize=(10 , 5))

plt.bar(range(1, e_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')

plt.step(range(1, e_vals.size + 1), cum_var_exp, where='mid', label = 'Cumulative explained variance')

plt.ylabel('Explained Variance Ratio')

plt.xlabel('Principal Components')

plt.legend(loc = 'best')

plt.tight_layout()

plt.show()
eigen_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in range(len(e_vals))]

eigen_pairs.sort(reverse=True)

eigen_pairs[:5]
w = np.hstack((eigen_pairs[0][1].reshape(4,1), eigen_pairs[1][1].reshape(4,1)))

print(w.shape)

print('Matrix W:\n', w)

X_sd_pca = X_train_sd.dot(w)
X_train_sd.shape, w.shape, X_sd_pca.shape
X_test_sd_pca = X_test_sd.dot(w)

X_test_sd.shape, w.shape, X_test_sd_pca.shape
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
clf = SVC()

clf.fit(X_train_sd, y_train)

print ('score', clf.score(X_test_sd, y_test))

from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver='lbfgs' , max_iter=5000 , multi_class='multinomial')

model = LogisticRegression()

model.fit(X_train_sd, y_train)
model.score(X_test_sd , y_test)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(X_train_sd, y_train)
model.score(X_test_sd , y_test) 
clf = SVC()

clf.fit(X_sd_pca, y_train)

print ('score', clf.score(X_test_sd_pca, y_test))
from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver='lbfgs' , max_iter=5000 , multi_class='multinomial')

model = LogisticRegression()

model.fit(X_sd_pca, y_train)
model.score(X_test_sd_pca , y_test)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(X_sd_pca, y_train)
model.score(X_test_sd_pca , y_test) 
w3 = np.hstack(

    (

        eigen_pairs[0][1].reshape(4,1), 

        eigen_pairs[1][1].reshape(4,1), 

        eigen_pairs[2][1].reshape(4,1)

    )

)

print('Matrix W:\n', w3)

X_train_sd_pca = X_train_sd.dot(w3)
X_train_sd.shape, w3.shape, X_train_sd_pca.shape
X_test_sd_pca = X_test_sd.dot(w3)

X_test_sd.shape, w3.shape, X_test_sd_pca.shape
clf = SVC()

clf.fit(X_train_sd_pca, y_train)

print ('score', clf.score(X_test_sd_pca, y_test))

from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver='lbfgs' , max_iter=5000 , multi_class='multinomial')

model = LogisticRegression()

model.fit(X_train_sd_pca, y_train)
model.score(X_test_sd_pca , y_test)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(X_train_sd_pca, y_train)
model.score(X_test_sd_pca , y_test) 
w4 = np.hstack(

    (

        eigen_pairs[0][1].reshape(4,1), 

        eigen_pairs[1][1].reshape(4,1), 

        eigen_pairs[2][1].reshape(4,1), 

        eigen_pairs[3][1].reshape(4,1)

    )

)

print('Matrix W:\n', w4)

X_train_sd_pca = X_train_sd.dot(w4)
X_train_sd.shape, w4.shape, X_train_sd_pca.shape
X_test_sd_pca = X_test_sd.dot(w4)

X_test_sd.shape, w4.shape, X_test_sd_pca.shape
clf = SVC()

clf.fit(X_train_sd_pca, y_train)

print ('score', clf.score(X_test_sd_pca, y_test))

from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(solver='lbfgs' , max_iter=5000 , multi_class='multinomial')

model = LogisticRegression()

model.fit(X_train_sd_pca, y_train)
model.score(X_test_sd_pca , y_test)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

model.fit(X_train_sd_pca, y_train)
model.score(X_test_sd_pca , y_test) 