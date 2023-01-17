import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
df_wine = pd.read_csv('../input/Wine.csv');
df_wine.head()
df_wine.columns = [  'name'
                 ,'alcohol'
             	,'malicAcid'
             	,'ash'
            	,'ashalcalinity'
             	,'magnesium'
            	,'totalPhenols'
             	,'flavanoids'
             	,'nonFlavanoidPhenols'
             	,'proanthocyanins'
            	,'colorIntensity'
             	,'hue'
             	,'od280_od315'
             	,'proline'
                ]
df_wine.head()
#make train-test sets
from sklearn.model_selection import train_test_split;
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values;
#print(np.unique(y))
#split with stratify on y for equal proportion of classes in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, stratify = y,random_state = 0);

#standardize the features with same model on train and test sets
from sklearn.preprocessing import StandardScaler;
sc = StandardScaler();
X_train_std = sc.fit_transform(X_train);
X_test_sd = sc.transform(X_test);

cov_mat = np.cov(X_train_std.T);
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat);
print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals);
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)];
cum_var_exp = np.cumsum(var_exp);
plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
#Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#chossing k = 2 for better representation via 2-dimensional scatter plot.
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):plt.scatter(X_train_pca[y_train==l, 0],X_train_pca[y_train==l, 1],c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
