import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from itertools import chain

from sklearn import preprocessing, ensemble





pd.options.mode.chained_assignment = None  # default='warn'

pd.options.display.max_columns = 999



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)


train_df.head()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.hist(train_df.y.values, bins=50)

plt.xlabel('y value', fontsize=12)

plt.show()

plt.figure(figsize=(12,8))

plt.hist(train_df.y.values, bins=50,log=True )

plt.xlabel('y value', fontsize=12)

plt.show()
dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df.ix[:10,:]
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df
unique_values_dict = {}

for col in train_df.columns:

    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        unique_value = str(np.sort(train_df[col].unique()).tolist())

        tlist = unique_values_dict.get(unique_value, [])

        tlist.append(col)

        unique_values_dict[unique_value] = tlist[:]

for unique_val, columns in unique_values_dict.items():

    print("Columns containing the unique values : ",unique_val)

    print(columns)

    print("--------------------------------------------------")

        
correlation_threshold = 0.99 # can be switched. Default value 0.99



train_integer = train_df.drop(["ID",	"y",	"X0",	"X1",	"X2",	"X3",	"X4",	"X5",

                               "X6",	"X8"],axis=1)



cor = train_integer.corr()

cor.loc[:,:] =  np.tril(cor, k=-1)

cor = cor.stack()

correlated = cor[cor > correlation_threshold].reset_index().loc[:,['level_0','level_1']]

correlated = correlated.query('level_0 not in level_1')

correlated_array =  correlated.groupby('level_0').agg(lambda x: set(chain(x.level_0,x.level_1))).values

correlated_array
correlated_features = []

for sets in correlated_array:

    element_list = list(sets[0])

    for idx, el in enumerate(element_list):

        if idx is not 0:

            correlated_features.append(el)

correlated_features.sort(key = lambda x: int(x[1:]) )

print (correlated_features)
non_cor_train_df = train_df.drop(correlated_features, axis=1 )

non_cor_train_df.shape
non_cor_train_df.head()
non_cor_test_df = test_df.drop(correlated_features, axis=1 )

non_cor_test_df.shape
categorical = ["X0",  "X1",  "X2", "X3", "X4",  "X5", "X6", "X8"]

for f in categorical:

        if non_cor_train_df[f].dtype=='object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(non_cor_train_df[f].values) + list(non_cor_test_df[f].values))

            non_cor_train_df[f] = lbl.transform(list(non_cor_train_df[f].values))

            non_cor_test_df[f] = lbl.transform(list(non_cor_test_df[f].values))

non_cor_train_df.head()
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2)

pca2_results = pca2.fit_transform(non_cor_train_df.drop(["y"], axis=1))
cmap = sns.cubehelix_palette(n_colors=10,as_cmap=True)

f, ax = plt.subplots(figsize=(20,15))

points = ax.scatter(pca2_results[:,0], pca2_results[:,1], c=non_cor_train_df.y, s=50, cmap=cmap)

f.colorbar(points)

plt.show()
from sklearn.decomposition import PCA

pca2 = PCA(n_components=5)

pca2_results = pca2.fit_transform(non_cor_train_df.drop(["y", "ID"], axis=1))

non_cor_train_df['pca0']=pca2_results[:,0]

non_cor_train_df['pca1']=pca2_results[:,1]

non_cor_train_df['pca2']=pca2_results[:,2]

non_cor_train_df['pca3']=pca2_results[:,3]

non_cor_train_df['pca4']=pca2_results[:,4]

pca2_results = pca2.transform(non_cor_test_df.drop(["ID"], axis=1))

non_cor_test_df['pca0']=pca2_results[:,0]

non_cor_test_df['pca1']=pca2_results[:,1]

non_cor_test_df['pca2']=pca2_results[:,2]

non_cor_test_df['pca3']=pca2_results[:,3]

non_cor_test_df['pca4']=pca2_results[:,4]
usable_columns = list(set(non_cor_train_df.columns) - set(['ID', 'y']))

usable_columns.sort(key = lambda x: int(x[1:]) if x[0]=="X" else int(x[len("pca"):]))



y_train = non_cor_train_df['y'].values

id_test = non_cor_test_df['ID'].values



x_train = non_cor_train_df[usable_columns]

x_test = non_cor_test_df[usable_columns]
x_train.head()
import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(x_test)



params = {}

params['objective'] = 'reg:linear'

params['eta'] = 0.02

params['max_depth'] = 4

params["subsample"] = 0.95





def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, 

                verbose_eval=10)
p_test = clf.predict(d_test)



sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = p_test

sub.to_csv('simple_xgb_pca_1.csv', index=False)
sub.head()
plt.figure(figsize=(8,6))

plt.scatter(range(sub.shape[0]), np.sort(sub.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
plt.figure(figsize=(12,8))

plt.hist(sub.y.values, bins=50)

plt.xlabel('y value', fontsize=12)

plt.show()