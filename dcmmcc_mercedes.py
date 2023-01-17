import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import xgboost as xgb

# R2 决定系数（拟合优度）,模型越好：r2→1,模型越差：r2→0

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



color = sns.color_palette()



%matplotlib inline



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

# 散点图

plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()
ulimit = 180

# loc 选取特定行和列的切片，可以使用 boolean array，这里只切片了行

train_df['y'].loc[train_df['y']>ulimit] = ulimit



plt.figure(figsize=(12,8))

sns.distplot(train_df.y.values, bins=50, kde=False)

plt.xlabel('y value', fontsize=12)

plt.show()
# count 数据类型

dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
# 这些 object 都是 str

dtype_df.iloc[:10,:]
# 检查 null 数据

missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')

missing_df
# 分析 integer 类型的列的数值特点，发现发部分都是 0-1 数据，还有些是毫无意义的 0

unique_values_dict = {}

for col in train_df.columns:

    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        # unique 将 list 变为 set

        unique_value = str(np.sort(train_df[col].unique()).tolist())

        tlist = unique_values_dict.get(unique_value, [])

        tlist.append(col)

        unique_values_dict[unique_value] = tlist[:]

for unique_val, columns in unique_values_dict.items():

    print("Columns containing the unique values : ",unique_val)

    print(columns)

    print("--------------------------------------------------")

        
# 使用 sns 统计类别列的特征

var_name = "X0"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

# Draw a scatterplot where one variable is categorical.

sns.stripplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X4"

col_order = np.sort(train_df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train_df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
zero_count_list = []

one_count_list = []

cols_list = unique_values_dict['[0, 1]']

for col in cols_list:

    zero_count_list.append((train_df[col]==0).sum())

    one_count_list.append((train_df[col]==1).sum())



N = len(cols_list)

ind = np.arange(N)

width = 0.35



plt.figure(figsize=(6,100))

# Make a horizontal bar plot.

p1 = plt.barh(ind, zero_count_list, width, color='red')

p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color="blue")

plt.yticks(ind, cols_list)

plt.legend((p1[0], p2[0]), ('Zero count', 'One Count'))

plt.show()
var_name = "ID"

plt.figure(figsize=(12,6))

# Plot data and a linear regression model fit.

# scatter_kws are the additional keywords to scatter

# alpha is Proportional opacity of the points

sns.regplot(x=var_name, y='y', data=train_df, scatter_kws={'alpha':0.5})

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
plt.figure(figsize=(6,10))

train_df['eval_set'] = "train"

test_df['eval_set'] = "test"

full_df = pd.concat([train_df[["ID","eval_set"]], test_df[["ID","eval_set"]]], axis=0)



plt.figure(figsize=(12,6))

sns.violinplot(x="eval_set", y='ID', data=full_df)

plt.xlabel("eval_set", fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of ID variable with evaluation set", fontsize=15)

plt.show()
for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        # encode the labels to normalized numerical representations with

        # value between 0 and n_classes-1

        # LabelEncoder can be used to normalize labels.

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values)) 

        train_df[f] = lbl.transform(list(train_df[f].values))



train_y = train_df['y']

# drop 表示去掉

train_x = train_df.drop(["ID", "y", "eval_set"] + 

                        ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293',

                         'X297', 'X330', 'X347'], axis=1)



# 将 train_X 划分为 train test

# 参数stratify： 依据标签y，按原数据y中各类比例，分配给train和test，

# 使得train和test中各类数据的比例与原数据集一样。

train_X, test_X, train_Y, test_Y = train_test_split(train_x, train_y, test_size=0.3, 

                                                    # stratify=train_y,

                                                    shuffle=True, random_state=1)



# Thanks to anokas for this #

def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)



xgb_params = {

    # 'gbtree'： 表示采用xgboost (默认值)

    # 'gblinear'： 表示采用线性模型。

    # 'gblinear' 使用带l1,l2 正则化的线性回归模型作为基学习器。因为boost 算法是一个线性叠加的过程，

    # 而线性回归模型也是一个线性叠加的过程。因此叠加的最终结果就是一个整体的线性模型，xgboost 

    # 最后会获得这个线性模型的系数。

    # 'dart'： 表示采用dart booster

    'booster': 'gbtree',

    # 也称作学习率。默认为 0.3 。范围为 [0,1]

    'eta': 0.05,

    #  也称作最小划分损失min_split_loss。 它刻画的是：对于一个叶子节点，当对它采取划分之后，

    # 损失函数的降低值的阈值。

    # 如果大于该阈值，则该叶子节点值得继续划分

    # 如果小于该阈值，则该叶子节点不值得继续划分

    # 该值越大，则算法越保守（尽可能的少划分）。默认值为 0

    'gamma': 0.,

    # 每棵子树的最大深度。其取值范围为， 0 表示没有限制，默认值为6。

    # 该值越大，则子树越复杂；值越小，则子树越简单。

    'max_depth': 6,

    # 对训练样本的采样比例。取值范围为 (0,1]，默认值为 1 。

    # 如果为 0.5， 表示随机使用一半的训练样本来训练子树。它有助于缓解过拟合。

    'subsample': 0.7,

    # 构建子树时，对特征的采样比例。取值范围为 (0,1]， 默认值为 1。

    # 如果为 0.5， 表示随机使用一半的特征来训练子树。它有助于缓解过拟合。

    # 要依据特征个数来判断

    'colsample_bytree': 0.7,

    # 目标函数的选择要根据问题确定，如果是回归问题 ，一般是 reg:linear ,

    # reg:logistic , count:poisson 如果是分类问题，一般是binary:logistic ,rank:pairwise

    # 多分类：'objective': 'multi:softmax', 配合 'num_class': 3,

    'objective': 'reg:linear',

    # silent： 如果为 0（默认值），则表示打印运行时的信息；如果为 1，

    # 则表示silent mode（ 不打印这些信息）

    'silent': 0,

    # nthread： 指定了运行时的并行线程的数量。如果未设定该参数，则默认值为可用的最大线程数。

    # lambda： L2 正则化系数（基于weights的正则化），默认为 1。 该值越大则模型越简单

    # alpha： L1 正则化系数（基于weights的正则化），默认为 0。 该值越大则模型越简单

    # tree_method： 指定了构建树的算法，可以为下列的值：（默认为'auto' )

    # 'auto'： 使用启发式算法来选择一个更快的tree_method：

    # 对于小的和中等的训练集，使用exact greedy 算法分裂节点

    # 对于非常大的训练集，使用近似算法分裂节点

    # 旧版本在单机上总是使用exact greedy 分裂节点

    # 'exact'： 使用exact greedy 算法分裂节点

    # 'approx'： 使用近似算法分裂节点

    # 'hist'： 使用histogram 优化的近似算法分裂节点（比如使用了bin cacheing 优化）

    # 'gpu_exact'： 基于GPU 的exact greedy 算法分裂节点

    # 'gpu_hist'： 基于GPU 的histogram 算法分裂节点

    'tree_method': 'auto',

    # early_stopping_rounds：一个整数，表示早停参数。

    # 如果在early_stopping_rounds 个迭代步内，验证集的验证误差没有下降，则训练停止。

    # 该参数要求evals 参数至少包含一个验证集。如果evals 参数包含了多个验证集，则使用最后的一个。

    # 返回的模型是最后一次迭代的模型（而不是最佳的模型）。

    # 如果早停发生，则模型拥有三个额外的字段：

    # .best_score： 最佳的分数

    # .best_iteration： 最佳的迭代步数

    # .best_ntree_limit： 最佳的子模型数量

    'eval_metric': ['rmse'],

}

# 调参思路

# 设置num_round 足够大（比如100000），以至于你能发现每一个round 的验证集预测结果，

# 如果在某一个round后 validation set 的预测误差上升了，你就可以停止掉正在运行的程序了。

# 然后开始逐个调参了。

# 首先调整max_depth ,通常max_depth 这个参数与其他参数关系不大，初始值设置为10，找到一个最好的误差值，

# 然后就可以调整参数与这个误差值进行对比。比如调整到8，如果此时最好的误差变高了，那么下次就调整到12；

# 如果调整到12,误差值比10 的低，那么下次可以尝试调整到15.

# 在找到了最优的max_depth之后，可以开始调整subsample,初始值设置为1，然后调整到0.8 

# 如果误差值变高，下次就调整到0.9，如果还是变高，就保持为1.0

# 接着开始调整min_child_weight , 方法与上面同理

# 再接着调整colsample_bytree

# 经过上面的调整，已经得到了一组参数，这时调整eta 到0.05，然后让程序运行来得到一个最佳的num_round,

# (在 误差值开始上升趋势的时候为最佳 )





# 这里 feature_names 表示 Set names for features.

# DMatrix is a internal data structure that used by XGBoost which is optimized for both memory

# efficiency and training speed.

dtrain = xgb.DMatrix(train_X, train_Y, feature_names=train_X.columns.values)

dval = xgb.DMatrix(test_X, test_Y, feature_names=test_X.columns.values)

watchlist = [(dtrain,'train'),(dval,'val')]

# feval is a Custom evaluation function

model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=200,

                  evals=watchlist,

                  # 一个布尔值或者整数。

                  # 如果为True，则evalutation metric  将在每个boosting stage 打印出来

                  # 如果为一个整数，则evalutation metric  将在每隔verbose_eval个boosting stage 打印出来。

                  # 另外最后一个boosting stage，以及早停的boosting stage 的 evalutation metric  也会被打印

                  verbose_eval=20,

                  feval=xgb_r2_score, maximize=True)

# test

# y_pred = model.predict(dtest)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
from sklearn import ensemble

model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10,

                                       min_samples_leaf=4, max_features=0.2,

                                       n_jobs=-1, random_state=0)

model.fit(train_X, train_Y)

feat_names = train_x.columns.values



pred_test = model.predict(test_X)

plt.figure(figsize=(8,6))

width = 50

start = np.random.random_integers(0, pred_test.size - width)

end = start + width

# 散点图

plt.scatter(range(width), pred_test[start:end], alpha=0.5, s=20, color='blue')

plt.scatter(range(width), test_Y[start:end], alpha=0.5, s=20, color='red')

plt.bar(x=range(width), height=pred_test[start:end]-test_Y[start:end], width=0.4, alpha=0.4,

        bottom=test_Y[start:end], color='green')

plt.xlabel('index', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.show()

print('Test results of RandomForest: R2 socre:', r2_score(test_Y, pred_test))



## plot the importances ##

importances = model.feature_importances_

# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA

from sklearn.decomposition import FastICA

from sklearn.decomposition import TruncatedSVD

# lightgbm 是 MS 开发的类似 xgboost 的框架

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error as MSE

from keras.layers import Input, Dense

from keras.models import Model



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# convert catagory variables to numberical variables

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))

train_y=train['y'] 

train.drop(['y', 'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289',

                    'X290', 'X293', 'X297', 'X330', 'X347'],inplace=True,axis=1)

test.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289',

                    'X290', 'X293', 'X297', 'X330', 'X347'],inplace=True,axis=1)

# convert object variables into one-hot variables

combine=pd.concat([train,test])

columns=['X0', 'X1','X2','X3','X4','X5','X6','X8']

for column in columns:

    temp=pd.get_dummies(pd.Series(combine[column]))

    print('cnovert categorical variable to one-hot dummy/indicator variables for {} with shape:'.format(

        column), temp.shape)

    combine=pd.concat([combine,temp],axis=1)

    combine= combine.drop([column], axis=1)



train=combine[:train.shape[0]]

test=combine[train.shape[0]:]



# assert train.columns.shape[0] == len(set(train.columns.values)), 'error need uniqueify'



# 把上述 one-hot 出来的重复的 column 名称 uniquify

def df_column_uniquify(df):

    df_columns = df.columns

    new_columns = []

    for item in df_columns:

        counter = 0

        newitem = item

        while newitem in new_columns:

            counter += 1

            newitem = "{}_{}".format(item, counter)

        new_columns.append(newitem)

    df.columns = new_columns

    return df



train = df_column_uniquify(train)  

test = df_column_uniquify(test)   

# train['y'] = y

assert train.columns.shape[0] == len(set(train.columns.values)), 'error after uniqueify'



# drop 表示去掉

# train_x = train_df.drop(["ID"], axis=1)

train_X, test_X, train_y, test_Y = train_test_split(train[list(set(test.columns) - {'y', 'ID'})], train_y, test_size=0.3, 

                                                    # stratify=train_y,

                                                    shuffle=True, random_state=1)

assert 'y' not in list(train_X.columns)

assert 'y' not in list(test_X.columns)

# train_ = train_X

# train_['y'] = train_y

# test_ = test_X

# 打乱顺序

# train_ = train_.sample(frac=1,random_state=420)



# Reduce dimensionality

n_comp = 20

col = list(test.columns)

col.remove('ID')

reduced_dim_cols = []

# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train_X[col])

tsvd_results_test = tsvd.transform(test_X[col])

# PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(train_X[col])

pca2_results_test = pca.transform(test_X[col])

# ICA

# tol is A positive scalar giving the tolerance at which the un-mixing matrix is considered to have converged.

ica = FastICA(n_components=n_comp, tol=0.03, random_state=420)

ica2_results_train = ica.fit_transform(train_X[col])

ica2_results_test = ica.transform(test_X[col])

# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train_X[col])

grp_results_test = grp.transform(test_X[col])

# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train_X[col])

srp_results_test = srp.transform(test_X[col])

# Autoencoder

ncol = len(col)

# print(set(train_X.columns) - (set(col) & set(train_X.columns)))

input_dim = Input(shape = (ncol, ))

# Encoder Layers

encoded1 = Dense(300, activation = 'relu')(input_dim)

encoded2 = Dense(150, activation = 'relu')(encoded1)

encoded4 = Dense(50, activation = 'relu')(encoded2)

encoded5 = Dense(25, activation = 'relu')(encoded4)

encoded6 = Dense(n_comp, activation = 'relu')(encoded5)

# Decoder Layers

decoded1 = Dense(25, activation = 'relu')(encoded6)

decoded2 = Dense(50, activation = 'relu')(decoded1)

decoded4 = Dense(150, activation = 'relu')(decoded2)

decoded5 = Dense(300, activation='relu')(decoded4)

decoded6 = Dense(ncol, activation = 'sigmoid')(decoded5)

# Combine Encoder and Deocder layers

autoencoder = Model(inputs = input_dim, outputs = decoded6)

# Compile the Model

autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

autoencoder.fit(train_X[col], train_X[col], nb_epoch = 10, batch_size = 32, shuffle = False,

                validation_data = (test_X[col], test_X[col]))

encoder = Model(inputs = input_dim, outputs = encoded6)

autoencoder_results_train = encoder.predict(train_X[col])

autoencoder_results_test = encoder.predict(test_X[col])

for i in range(1, n_comp + 1):

        reduced_dim_cols.append('tsvd_' + str(i))

        train_X['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

        test_X['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

        reduced_dim_cols.append('pca_' + str(i))

        train_X['pca_' + str(i)] = pca2_results_train[:, i - 1]

        test_X['pca_' + str(i)] = pca2_results_test[:, i - 1]

        reduced_dim_cols.append('ica_' + str(i))

        train_X['ica_' + str(i)] = ica2_results_train[:, i - 1]

        test_X['ica_' + str(i)] = ica2_results_test[:, i - 1]

        reduced_dim_cols.append('grp_' + str(i))

        train_X['grp_' + str(i)] = grp_results_train[:, i - 1]

        test_X['grp_' + str(i)] = grp_results_test[:, i - 1]

        reduced_dim_cols.append('srp_' + str(i))

        train_X['srp_' + str(i)] = srp_results_train[:, i - 1]

        test_X['srp_' + str(i)] = srp_results_test[:, i - 1]

        reduced_dim_cols.append('ae_' + str(i))

        train_X['ae_' + str(i)] = autoencoder_results_train[:, i - 1]

        test_X['ae_' + str(i)] = autoencoder_results_test[:, i - 1]





def get_lgb_stack_data(params,rounds,train,col,label,test):

    ID = []

    train = train.reset_index(drop=True)

    kf = KFold(n_splits=5,shuffle=False)

    i=0

    R2_Score = []

    RMSE = []

    for train_index, test_index in kf.split(train):

        print("Training "+str(i+1)+' Fold')

        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]

        y_train, y_test = label.iloc[train_index],label.iloc[test_index]

        train_lgb=lgb.Dataset(X_train[col],y_train)

        model = lgb.train(params,train_lgb,num_boost_round=rounds)

        pred = model.predict(X_test[col])

        X_test['label'] = list(y_test)

        X_test['predicted'] = pred

        r2 = r2_score(y_test,pred)

        rmse = MSE(y_test,pred)**0.5

        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))

        R2_Score.append(r2)

        RMSE.append(rmse)

        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))

        if i==0:

            Final = X_test

        else:

            Final = Final.append(X_test,ignore_index=True)

        i+=1

    lgb_train_ = lgb.Dataset(train[col],label)

    print('Start Training')

    model_ = lgb.train(params,lgb_train_,num_boost_round=rounds)

    Final_pred = model_.predict(test[col])

    Final_pred = pd.DataFrame({'y':Final_pred})

    print('Calculating In-Bag R2 Score')

    print(r2_score(label, model.predict(train[col])))

    print('Calculating Out-Bag R2 Score')

    print(np.mean(R2_Score))

    print('Calculating In-Bag RMSE')

    print(MSE(label, model.predict(train[col]))**0.5)

    print('Calculating Out-Bag RMSE')

    print(np.mean(RMSE))

    return Final,Final_pred



def get_sklearn_stack_data(model,train,col,label,test):

    ID = []

    R2_Score = []

    RMSE = []

    train = train.reset_index(drop=True)

    kf = KFold(n_splits=5,shuffle=False)

    i=0

    for train_index, test_index in kf.split(train):

        print("Training "+str(i+1)+' Fold')

        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]

        y_train, y_test = label.iloc[train_index],label.iloc[test_index]

        model.fit(X_train[col],y_train)

        pred = model.predict(X_test[col])

        X_test['label'] = list(y_test)

        X_test['predicted'] = pred

        r2 = r2_score(y_test,pred)

        rmse = MSE(y_test,pred)**0.5

        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))

        R2_Score.append(r2)

        RMSE.append(rmse)

        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))

        if i==0:

            Final = X_test

        else:

            Final = Final.append(X_test,ignore_index=True)

        i+=1

    print('Start Training')

    model.fit(train[col],label)

    Final_pred = model.predict(test[col])

    Final_pred = pd.DataFrame({'y':Final_pred})

    print('Calculating In-Bag R2 Score')

    print(r2_score(label, model.predict(train[col])))

    print('Calculating Out-Bag R2 Score')

    print(np.mean(R2_Score))

    print('Calculating In-Bag RMSE')

    print(MSE(label, model.predict(train[col]))**0.5)

    print('Calculating Out-Bag RMSE')

    print(np.mean(RMSE))

    return Final,Final_pred





def get_xgb_stack_data(params,rounds,train,col,label,test):

    ID = []

    train = train.reset_index(drop=True)

    kf = KFold(n_splits=5,shuffle=False)

    i=0

    R2_Score = []

    RMSE = []

    for train_index, test_index in kf.split(train):

        print("Training "+str(i+1)+' Fold')

        X_train, X_test = train.iloc[train_index,:], train.iloc[test_index,:]

        y_train, y_test = label.iloc[train_index],label.iloc[test_index]

        dtrain = xgb.DMatrix(X_train[col],y_train)

        dtest = xgb.DMatrix(X_test[col])

        model = xgb.train(params,dtrain,num_boost_round=rounds)

        pred = model.predict(dtest)

        X_test['label'] = list(y_test)

        X_test['predicted'] = pred

        r2 = r2_score(y_test,pred)

        rmse = MSE(y_test,pred)**0.5

        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))

        R2_Score.append(r2)

        RMSE.append(rmse)

        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))

#         ID.append(X_test['ID'])

        if i==0:

            Final = X_test

        else:

            Final = Final.append(X_test,ignore_index=True)

        i+=1

    dtrain_ = xgb.DMatrix(train[col],label)

    dtest_ = xgb.DMatrix(test[col])

    print('Start Training')

    model_ = xgb.train(params,dtrain_,num_boost_round=rounds)

    Final_pred = model_.predict(dtest_)

    Final_pred = pd.DataFrame({ # 'ID':test['ID'],

        'y':Final_pred})

    print('Calculating In-Bag R2 Score')

    print(r2_score(dtrain_.get_label(), model.predict(dtrain_)))

    print('Calculating Out-Bag R2 Score')

    print(np.mean(R2_Score))

    print('Calculating In-Bag RMSE')

    print(MSE(dtrain_.get_label(), model.predict(dtrain_))**0.5)

    print('Calculating Out-Bag RMSE')

    print(np.mean(RMSE))

    return Final,Final_pred





# There are 3 models that we'd like to stack

col = list(test_X.columns)

# only with reduced additional features

# col = reduced_dim_cols

# col.remove('eval_set')

## Input 1: GBDT

# n_estimators is The number of boosting stages to perform.

# Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.

gb1 = GradientBoostingRegressor(n_estimators=1000,max_features=0.95,learning_rate=0.005,max_depth=4)

gb1_train,gb1_test = get_sklearn_stack_data(gb1,train_X,col,train_y,test_X)

## Input2: Lasso

# las1 = Lasso(alpha=5,random_state=42)

# las1_train,las1_test = get_sklearn_stack_data(las1,train_,col,train_['y'],test_)

y_mean = np.mean(train_y)

params = {

    'eta': 0.005,

    'max_depth': 2,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}

xgb_train, xgb_test = get_xgb_stack_data(params,800,train_X,col,train_y,test_X)

## Input 3: LGB

params = {

            'objective': 'regression',

            'metric': 'rmse',

            'boosting': 'gbdt',

            'learning_rate': 0.0045 , #small learn rate, large number of iterations

            'verbose': 0,

            'num_iterations': 500,

            'bagging_fraction': 0.95,

            'bagging_freq': 1,

            'bagging_seed': 42,

            'feature_fraction': 0.95,

            'feature_fraction_seed': 42,

            'max_bin': 100,

            'max_depth': 3,

            'num_rounds': 800,

            'sparse_threshold': 1.0,

#             'device': 'gpu',

#             'gpu_platform_id': 0,

#             'gpu_device_id': 0

        }

lgb_train, lgb_test = get_lgb_stack_data(params,800,train_X,col,train_y,test_X)



# Now we use xgboost to stack them



stack_train = gb1_train[['label','predicted']]

stack_train.columns=['label','gbdt']

stack_train['lgb']=lgb_train['predicted']

# stack_train['las'] = las1_train['predicted']

stack_train['xgb'] = xgb_train['predicted']



stack_test = gb1_test[['y']]

stack_test.columns=['gbdt']

stack_test['lgb']=lgb_test['y']

# stack_test['las'] = las1_test['y']

stack_test['xgb'] = xgb_test['y']



## Meta Model: xgb

y_mean = np.mean(train_y)



col = list(stack_test.columns)



params = {

    'eta': 0.005,

    'max_depth': 2,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}



print(col)

dtrain = xgb.DMatrix(stack_train[col], stack_train['label'])

dtest = xgb.DMatrix(stack_test[col])



model = xgb.train(params,dtrain, num_boost_round=900)

pred_1 = model.predict(dtest)



# Original XGBoost

train_ = train_X

train_['y'] = train_y

test_ = test_X

# 打乱顺序

train_ = train_.sample(frac=1, random_state=420)

col = list(test_X.columns)

# only with reduced additional features

# col = reduced_dim_cols



xgb_params = {

        'n_trees': 520, 

        'eta': 0.0045,

        'max_depth': 4,

        'subsample': 0.93,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'base_score': y_mean, # base prediction = mean(target)

        'silent': True,

        'seed': 42,

#         'tree_method': 'gpu_hist',

    }

dtrain = xgb.DMatrix(train_.drop('y', axis=1)[col], train_.y)

dtest = xgb.DMatrix(test_[col])

    

num_boost_rounds = 1250

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)



# Average Two Solutions



pred_test_average = 0.70*y_pred + 0.30*pred_1



# Calculate the results

print('Test results of Stacked Models: R2 socre:', r2_score(test_Y, pred_test_average))