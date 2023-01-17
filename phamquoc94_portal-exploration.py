import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA, KernelPCA

import xgboost as xgb

from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE



color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
train_df = pd.read_csv("../input/train.csv")
print("train dataset shape: ", train_df.shape)

train_df.head()
train_df['ps_car_15'].min()
train_df.isnull().values.any()
missing_df = np.sum(train_df==-1, axis=0)

missing_df.sort_values(ascending=False, inplace=True)



plt.figure(figsize=(10, 20))

sns.barplot(x=missing_df.values, y=missing_df.index)

plt.title("Number of missing values in each column")

plt.xlabel("Count of missing values")

plt.show()
sns.countplot(x="target", data=train_df)

plt.show()
bin_vars = []

for col in train_df.columns:

    if col.endswith("bin"):

        bin_var = train_df.groupby(col).size()  

        bin_vars.append(bin_var)

        

bin_df = pd.concat(bin_vars, axis=0, keys=[s.index.name for s in bin_vars]).unstack()



_ = bin_df.plot(kind='bar', stacked=True, grid=False, figsize=(10, 8))

bin_vars = []

for col in train_df.columns:

    if col.endswith("cat"):

        bin_var = train_df.groupby(col).size()  

        bin_vars.append(bin_var)

        

bin_df = pd.concat(bin_vars, axis=0, keys=[s.index.name for s in bin_vars]).unstack()



_ = bin_df.plot(kind='bar', stacked=True, grid=False, figsize=(10, 8), legend=False)
corr = train_df.corr()



plt.figure(figsize=(20,15))

sns.heatmap(corr)

plt.show()
ignore_columns = ['target', 'id', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'] + ['ps_calc_{:02d}'.format(i) for i in range(1, 15)] + ['ps_calc_{:02d}_bin'.format(i) for i in range(15, 21)]

train_columns = [col for col in train_df.columns if col not in ignore_columns]


X = train_df[train_columns].values

target = train_df.target

print("Training data shape: ", X.shape)



pca = PCA(n_components=2)

reduced_dim = pca.fit_transform(X)

reduced_dim = reduced_dim[np.random.randint(0, len(reduced_dim), size=10000)]

                                            

reduced_df = pd.DataFrame(data=reduced_dim, columns=['x', 'y'])

reduced_df['target'] = target



plt.figure(figsize=(20, 8))

sns.jointplot(x='x', y='y', data=reduced_df, size=7, color="g")

plt.show()
plt.figure(figsize=(20,15))

sns.lmplot(x='x', y='y', hue='target', data=reduced_df, size=7, fit_reg=False)

plt.show()
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score




X = train_df[train_columns]

y = train_df.target

x_train = X[:-100000]

y_train = y[:-100000]

x_val = X[-100000:]

y_val = y[-100000:]



dtrain = xgb.DMatrix(x_train, y_train)

dval = xgb.DMatrix(x_val, y_val)

watchlist = [(dtrain, 'train'), (dval, 'valid')]



xgb_params = {

        'eta': 0.037,

        'max_depth': 5,

        'subsample': 0.80,

        'objective': 'binary:logistic',

        'eval_metric': 'mae',

        'lambda': 0.8,   

        'alpha': 0.4, 

        'base_score': 0.0364,

        'silent': 1

    }



num_boost_rounds = 250

model = xgb.train(dict(xgb_params, silent=1), dtrain, evals=watchlist, feval=gini_xgb, num_boost_round=num_boost_rounds, verbose_eval=20)

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
predict = model.predict(dval)

idx = np.abs(y_val - predict).nlargest(1400).index.values

y_val[idx].value_counts()

#predict1 = predict > 0.1

#confusion_matrix(y_val, predict1)
#cat_columns = [col for col in train_df.columns if col.endswith('cat') and (col!='ps_car_11_cat')]

#train_df = pd.get_dummies(train_df, columns=cat_columns, prefix=cat_columns)



ignore_columns = ['target', 'id', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'] + ['ps_calc_{:02d}'.format(i) for i in range(1, 15)] + ['ps_calc_{:02d}_bin'.format(i) for i in range(15, 21)]

train_columns = [col for col in train_df.columns if col not in ignore_columns]

#X = train_df[train_columns]

#y = train_df.target



#x_train = X[:-100000]

#y_train = y[:-100000]

positive = train_df[train_df.target==1].head(20000)

negative = train_df[train_df.target==0].head(50000)

train_df = pd.concat([positive, negative], axis=0)

# Performing one hot encoding





train_df = train_df.sample(frac=1.0)

X = train_df[train_columns]

y = train_df.target



print(positive.shape)



x_train = X[:-5000]

y_train = y[:-5000]

x_val = X[-5000:]

y_val = y[-5000:]



dtrain = xgb.DMatrix(x_train, y_train)

dval = xgb.DMatrix(x_val, y_val)

watchlist = [(dtrain, 'train'), (dval, 'valid')]



xgb_params = {

        'eta': 0.037,

        'max_depth': 5,

        'subsample': 0.80,

        'objective': 'reg:logistic',

        'eval_metric': 'auc',

        'lambda': 0.8,   

        'alpha': 0.4, 

        'base_score': 0.01,

        'silent': 1

    }



num_boost_rounds = 250

model = xgb.train(dict(xgb_params, silent=1), dtrain, evals=watchlist, feval=gini_xgb, num_boost_round=num_boost_rounds, verbose_eval=20)

predict = model.predict(dval)

idx = np.abs(y_val - predict).nlargest(1400).index.values

y_val[idx].value_counts()

predict1 = predict > 0.

confusion_matrix(y_val, predict1)
ignore_columns = ['target', 'id', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'] + ['ps_calc_{:02d}'.format(i) for i in range(1, 15)] + ['ps_calc_{:02d}_bin'.format(i) for i in range(15, 21)] #+ [col for col in train_df.columns if col.endswith('cat')]

cat_columns = [col for col in train_df.columns if col.endswith('cat')]

train_df = pd.get_dummies(train_df, columns=cat_columns, prefix=cat_columns)



train_columns = [col for col in train_df.columns if col not in ignore_columns]



#for col in train_df.columns:

#    if col.endswith('cat'):

#        count = train_df[col].value_counts()

#        train_df[col] = train_df.replace({col:count})





#log_columns = ['ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03']

#log_columns = [col for col in train_df.columns if 'reg' in col]    

#for col in log_columns:

#    train_df[col] = np.square(train_df[col] +0.00001)

    

X = train_df[train_columns].values[:10000]

target = train_df.target[:10000]

print("Training data shape: ", X.shape)



#pca = KernelPCA(n_components=2, kernel='linear')

#reduced_dim = pca.fit_transform(X)

reduced_dim = TSNE(n_components=2).fit_transform(X)

reduced_dim = reduced_dim[np.random.randint(0, len(reduced_dim), size=10000)]

                                            

reduced_df = pd.DataFrame(data=reduced_dim, columns=['x', 'y'])

reduced_df['target'] = target



plt.figure(figsize=(20, 8))

sns.jointplot(x='x', y='y', data=reduced_df, size=7, color="g")

plt.show()



plt.figure(figsize=(20,15))

sns.lmplot(x='x', y='y', hue='target', data=reduced_df, size=7, fit_reg=False)

plt.show()