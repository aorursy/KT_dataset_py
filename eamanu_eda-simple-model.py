#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

print(os.listdir("../input"))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
print(train.columns)
print(train.shape)
print(train.info())
# Null values?
nulls_data = train.isnull().sum().sum()
print("There are {} null data on the dataset".format(nulls_data))
test.head(5)
print(test.columns)
print(test.shape)
print('test info:')
print(test.info())
# Null values?
nulls_data = test.isnull().sum().sum()

print("There are {} null data on the dataset".format(nulls_data))
print(train.target.describe())
train.target.plot.hist()
target_log = np.log(train.target)
target_subx = 1/train.target
target_square = np.square(train.target)
print(target_log.skew())

print(target_square.skew())
target_log.plot.hist()
train.target = target_log
columns = train.columns
print(len(train[train[columns[2]] == 0])/len(train[columns[2]]))
print(len(train[columns[2]]))
list_zeros = [len(train[train[d] == 0])/4459. for d in columns]
# list_zeros = []
#for d in columns:
#    zeros = len(train[train[d] == 0])
#    total = 4459.
#    list_zeros.append(zeros/total)
sns.distplot(list_zeros, bins=100)
# df = df.loc[:, df.var() == 0.0]
# obj_df = train.select_dtypes(include=['object'])
obj_df = train.iloc[:, :2]
# num_df = train.select_dtypes(exclude=['object'])
num_df = train.iloc[:,2:]
var = num_df.var()
l_keys_notzeros = []
l_values_notzeros = []
for k, v in var.items():
    if v != 0.0:
        l_keys_notzeros.append(k)
        l_values_notzeros.append(v)
# foo = num_df.loc[:, num_df.var() != 0.0]
foo = num_df[l_keys_notzeros]
new_train_without_zeros = pd.concat([obj_df, foo], axis=1) # new data without zero variance
print(new_train_without_zeros.shape)
obj_df = test.iloc[:, :1]
num_df = test.iloc[:,1:]
foo = num_df[l_keys_notzeros]
new_test_without_zeros = pd.concat([obj_df, foo], axis=1) # new data without zero variance
print(new_test_without_zeros.shape)
del obj_df
del num_df
del foo
# Remove duplicated columns
col_to_remove = list()
col_scanned = list()
dup_list = dict()

cols = new_train_without_zeros.columns

for i in range(len(cols) - 1):
    v = new_train_without_zeros[cols[i]].values
    dup_cols = list()
    for j in range(i+1, len(cols)):
        if np.array_equal(v, new_train_without_zeros[cols[j]].values):
            col_to_remove.append(cols[j])
            if cols[j] not in col_scanned:
                dup_cols.append(cols[j]) 
                col_scanned.append(cols[j])
                dup_list[cols[i]] = dup_cols
print(col_to_remove)    
cols = [c for c in cols if c not in col_to_remove]
cols_test = [c for c in cols if c != 'target']
new_train = new_train_without_zeros[cols]
new_test = new_test_without_zeros[cols_test]

print(new_train.shape)
print(new_test.shape)
del new_train_without_zeros
del new_test_without_zeros
del train
del test
del col_to_remove
del col_scanned
del dup_list
del cols
id_target_train = new_train.iloc[:,:2]
new_train = new_train.iloc[:,2:].values

id_test = new_test.iloc[:,:1]
new_test = new_test.iloc[:, 1:].values
print('Shape of train: ',new_train.shape)
print('Shape of test: ', new_test.shape)
#print('Shape of target: ',log_target.shape)
#print('Shape of test: ',test.shape)
def transform (dataframe):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled_data)
new_train = transform(new_train)
new_test = transform(new_test)
# num_data = ttrain.select_dtypes(exclude='object')
num_data = new_train
pca = PCA(copy=True, n_components=2000, whiten=False)
new = pca.fit(num_data).transform(num_data)
print(pca.explained_variance_ratio_) 
len_pca = len(pca.explained_variance_ratio_)
print("The first {} PCA explain {}".format(len_pca, pca.explained_variance_ratio_.sum()*100))
# var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var = pca.explained_variance_ratio_.cumsum()
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.style.context('seaborn-whitegrid')

plt.plot(var)
plt.show()

pca_train = pd.DataFrame(data=new, columns=['pca{}'.format(i) for i in range(2000)])
pca_train = pd.concat([id_target_train[['ID','target']], pca_train], axis = 1)
print(pca_train.head(1))
num_data = new_test
new = pca.transform(num_data)
pca_test = pd.DataFrame(data=new, columns=['pca{}'.format(i) for i in range(2000)])
pca_test = pd.concat([id_test[['ID']], pca_test], axis=1)
print(pca_test.head(1))
x_train = pca_train.iloc[:, 2:]
y_train = pca_train.iloc[:, 1:2]

x_test = pca_test.iloc[:, 1:]
linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
print(linear_regression.coef_)
target_test  = linear_regression.predict(x_test)
target_test = pd.DataFrame(data=target_test, columns=['target'])
print(target_test.head(1))
to_submit = pd.concat([pca_test['ID'], target_test['target']], axis=1)
print(to_submit.head(1))
to_submit.to_csv('ols.csv', columns=['ID','target'], index=False)