# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Forked from https://www.kaggle.com/nirajvermafcb/principal-component-analysis-explained



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
columns_names=df.columns.tolist()

print("Columns names:")

print(columns_names)
df.shape
df.head()
df.corr()
correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
df['sales'].unique()
sales=df.groupby('sales').sum()

sales
df['sales'].unique()
groupby_sales=df.groupby('sales').mean()

groupby_sales
IT=groupby_sales['satisfaction_level'].IT

RandD=groupby_sales['satisfaction_level'].RandD

accounting=groupby_sales['satisfaction_level'].accounting

hr=groupby_sales['satisfaction_level'].hr

management=groupby_sales['satisfaction_level'].management

marketing=groupby_sales['satisfaction_level'].marketing

product_mng=groupby_sales['satisfaction_level'].product_mng

sales=groupby_sales['satisfaction_level'].sales

support=groupby_sales['satisfaction_level'].support

technical=groupby_sales['satisfaction_level'].technical

technical


department_name=('sales', 'accounting', 'hr', 'technical', 'support', 'management',

       'IT', 'product_mng', 'marketing', 'RandD')

department=(sales, accounting, hr, technical, support, management,

       IT, product_mng, marketing, RandD)

y_pos = np.arange(len(department))

x=np.arange(0,1,0.1)



plt.barh(y_pos, department, align='center', alpha=0.8)

plt.yticks(y_pos,department_name )

plt.xlabel('Satisfaction level')

plt.title('Mean Satisfaction Level of each department')
df.head()
df_drop=df.drop(labels=['sales','salary'],axis=1)

df_drop.head()
cols = df_drop.columns.tolist()

cols
cols.insert(0, cols.pop(cols.index('left')))
cols
df_drop = df_drop.reindex(columns= cols)
X = df_drop.iloc[:,1:8].values

y = df_drop.iloc[:,0].values

X
y
np.shape(X)
np.shape(y)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
plt.figure(figsize=(8,8))

sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different features')
eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)

# 6) Selecting Principal Components¶
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(7), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 

                      eig_pairs[1][1].reshape(7,1)

                    ))

print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)

Y
from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,7,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=6)

Y_sklearn = sklearn_pca.fit_transform(X_std)
print(Y_sklearn)
Y_sklearn.shape