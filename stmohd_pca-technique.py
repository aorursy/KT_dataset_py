#

import numpy as np # linear algebra

import pandas as pd # data processing



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import boxcox,skew,norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

# data

columns = ["Class", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"]

wine_data =pd.read_csv('../input/wine.data',names=columns)
wine_data.head()
wine_data.shape#no of rowsa and columns
wine_data.info()
wine_data.isna().any().sum()
wine_data.describe(include='all').T#Decriptive Statistics
#checking the overall distribution
sns.pairplot(wine_data,diag_kind='kde')
#splitting data into 70 :30
from sklearn.model_selection import train_test_split
x= wine_data.drop('Class',axis=1)

y =wine_data.Class



x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=3)
sc = StandardScaler()#scaling data
x_train_std = sc.fit_transform(x_train)

x_test_sd = sc.transform(x_test)
#constructing covariance matrics and finding eigen values and eigen vectors
cov_mat = np.cov(x_train_std.T);

print(cov_mat)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('Eigen Vectors \n', eigen_vecs)

print('\n Eigen Values \n', eigen_vals)
tot = sum(eigen_vals);

var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)];

cum_var_exp = np.cumsum(var_exp);

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')

plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal component index')

plt.legend(loc='best')

plt.show()
#60% of variance is explained by first 2  PCA
# Selecting k eigenvectors that correspond to the k largest eigenvalues,  

# where k is the dimensionality of the new feature subspace ( k≤d ).
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[ :, i]) for i in range(len(eigen_vals))]

eigen_pairs
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
eigen_pairs
#  projection matrix W from first  2 eigen vector

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))

print('Matrix W:\n', w)
x_train_pca = x_train_std.dot(w)
x_test_pca =x_test_sd.dot(w)
x_train_pca.shape
#  visualizing the transformed Wine training set
colors = ['r', 'b', 'g']

markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):plt.scatter(x_train_pca[y_train==l, 0],x_train_pca[y_train==l, 1],c=c, label=l, marker=m)

plt.xlabel('PC 1')

plt.ylabel('PC 2')

plt.legend(loc='lower left')

plt.show()
#LR model 
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA
lr = LogisticRegression()

pca =PCA(n_components=2)
# x_train_pca = pca.fit_transform(x_train_pca)
lr.fit(x_train_pca,y_train)#fitting with train data
lr.score(x_train_pca,y_train)#train accuracy score
y_predit =lr.predict(x_test_pca)#predictd test values
import sklearn.metrics as metrics
accuracy = metrics.accuracy_score(y_predit,y_test)

print(accuracy)#test accuracy
#confusion metrics

print(metrics.confusion_matrix(y_predit,y_test))
# classification report

print(metrics.classification_report(y_predit,y_test))