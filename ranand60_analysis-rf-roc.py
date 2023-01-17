# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
input_path = '../input/binaryclassification/data.csv'
input_data = pd.read_csv(input_path)
data = input_data.head()
print(data)
input_data.shape
input_data.info()
input_data.describe()
input_data["label"].value_counts()
input_data.isnull().sum()
input_data["f21"].value_counts()
x=input_data.drop(labels=['label'],axis=1)
y = input_data['label']
x.head(2)
y.head(5)
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(x)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(28), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(28,1), 
                      eig_pairs[1][1].reshape(28,1)
                    ))
print('Matrix W:\n', matrix_w)
from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,28,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=28)
x_pca  = sklearn_pca.fit_transform(X_std)

print(x_pca )
x_pca.shape
y.shape
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with 20% test rate
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Training model
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=101)
RFC.fit(X_train,y_train)

y_pred = RFC.predict(X_test)
# Import 4 metrics from sklearn for testing
from sklearn.metrics import accuracy_score
print ("Accuracy on testing data of RandomForestClassifier: {:.4f}".format(accuracy_score(y_test, y_pred)))
from sklearn import metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
confusion_matrix
roc=metrics.classification_report(y_test,y_pred)
roc
roc_score = metrics.roc_auc_score(y_test,y_pred)
roc_score
from sklearn.metrics import roc_curve, auc
FPR, TPR, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(FPR, TPR)
roc_auc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('ROC Curve')
plt.plot(FPR, TPR, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')