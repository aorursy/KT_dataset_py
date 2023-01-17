# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

### data is from https://www.kaggle.com/crawford/gene-expression/kernels
#### similar work: 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
labels_df = pd.read_csv('../input/actual.csv', index_col = 'patient')
test_df = pd.read_csv('../input/data_set_ALL_AML_independent.csv')
data_df = pd.read_csv('../input/data_set_ALL_AML_train.csv')
print('train_data: ',  data_df.shape, '\n test_data: ',  test_df.shape, '\n labels: ', labels_df.shape)
# labels_df.shape
data_df.head()
####### clean up the data

test_cols_to_drop = [c for c in test_df.columns if 'call' in c]
test_df = test_df.drop(test_cols_to_drop, axis=1)
test_df = test_df.drop(['Gene Description', 'Gene Accession Number'], axis=1 )

data_cols_to_drop = [c for c in data_df.columns if 'call' in c]
data_df = data_df.drop(data_cols_to_drop, axis=1)
data_df = data_df.drop(['Gene Description', 'Gene Accession Number'], axis=1 )
print('train_data ', data_df.shape, '\n test_data: ',  test_df.shape,  '\n labels: ', labels_df.shape)
data_df.head()

labels_df = labels_df.replace({'ALL':0, 'AML':1})
train_labels = labels_df[labels_df.index <= 38]
test_labels = labels_df[labels_df.index > 38]
print(train_labels.shape, test_labels.shape)
# labels_df.index
test_df = test_df.T
train_df = data_df.T

##### check if there is any null values (inf or nan)
print('Columns containing null values in train and test data are ', data_df.isnull().values.sum(),  test_df.isnull().values.sum())
# df.isnull().values.sum()
full_df = train_df.append(test_df, ignore_index=True)
print(full_df.shape)
full_df.head()
### Standardization
from sklearn import preprocessing
X_std = preprocessing.StandardScaler().fit_transform(full_df)
### Check how the standardized data look like
gene_index = 1
print('mean is : ',  np.mean(X_std[:, gene_index] ) )
print('std is :', np.std(X_std[:, gene_index]))

fig= plt.figure(figsize=(10,10))
plt.hist(X_std[:, gene_index], bins=10)
plt.xlim((-4, 4))
plt.xlabel('rescaled expression', size=30)
plt.ylabel('frequency', size=30)
### PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_std)
print(X_pca.shape)
cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100

fig = plt.figure(figsize=(10,10))
plt.bar(range(50), cum_sum)
plt.xlabel('PCA', size=30)
plt.ylabel('Cumulative Explained Varaince', size=30)
plt.title("Around 90% of variance is explained by the First 50 columns ", size=30)
labels = labels_df['cancer'].values

colors = np.where(labels==0, 'red', 'blue')


from mpl_toolkits.mplot3d import Axes3D
plt.clf()
fig = plt.figure(1, figsize=(15,15 ))
ax = Axes3D(fig, elev=-150, azim=110,)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, cmap=plt.cm.Paired,linewidths=10)
ax.set_title("First three PCA directions")
ax.set_xlabel("PCA1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("PCA2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("PCA3")
ax.w_zaxis.set_ticklabels([])
plt.show()
X = X_pca
y = labels
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred))


# print(X_train.shape)
# print(len(rfc.feature_importances_))
labs = ['PCA'+str(i+1) for i in range(X_train.shape[1])]
importance_df = pd.DataFrame({
    'feature':labs,
    'importance': rfc.feature_importances_
})

importance_df_sorted = importance_df.sort_values('importance', ascending=False)
importance_df_sorted.head()

fig = plt.figure(figsize=(25,10))
sns.barplot(data=importance_df_sorted, x='feature', y='importance')
plt.xlabel('PCAs', size=30)
plt.ylabel('Feature Importance', size=30)
plt.title('RF Feature Importance', size=30)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
### Normalize cm, np.newaxis makes to devide each row by the sum
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


print(np.newaxis)
cmap=plt.cm.Blues

plt.imshow(cm, interpolation='nearest', cmap=cmap)
print(cm)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)
y_gbc_pred = gbc.predict(X_test)
print(accuracy_score(y_test, y_gbc_pred))
