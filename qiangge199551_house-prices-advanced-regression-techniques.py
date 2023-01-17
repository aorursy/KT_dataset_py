import os

import warnings

warnings.filterwarnings("ignore")

import gc



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as msno

from scipy import stats

from scipy.stats import norm



import pickle
def show_the_date_file(path):

    for root, dirnames, filenames in os.walk(path):

        for file in filenames:

            print(os.path.join(root, file))

    return



show_the_date_file('../input')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

matrix = train.iloc[:, :-1].append(test, ignore_index=True)

target = train['SalePrice']

target.index = train.Id

# matrix = pd.concat([train, test])

train.head(5)
# 所有变量的相关系数热力图

fig, ax = plt.subplots(figsize=(12,9))

train_corr = train.corr()

sns.heatmap(train_corr, vmax=1, square=True, cmap='BuPu')

# fig, ax = plt.subplots(figsize=(12,9))

# mask = np.zeros_like(train_corr, dtype = np.bool)

# mask[np.triu_indices_from(mask)] = True

# sns.heatmap(train_corr, mask = mask,

#             square = True, linewidths = .5, ax = ax, cmap = "BuPu")      

# plt.show()
# 筛选相关性前20的feature

feature_corr = train_corr['SalePrice'].sort_values(ascending=False).head(20).to_frame()

cm = sns.light_palette('grey', as_cmap = True)

feature_corr.style.background_gradient(cmap = cm)

feature_corr
# feature相关系数热力图

plt.figure(figsize=(8,8))

features = train_corr.nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[features].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 

                 annot_kws={'size': 10}, yticklabels=features.values, xticklabels=features.values)
sns.set()

sns.pairplot(train[features], size=2.5)
sns.set(style = "ticks")

msno.matrix(matrix)

msno.heatmap(matrix, cmap = 'binary')



missing_data = pd.DataFrame(matrix.isnull().sum())

missing_data.columns = ['MissingCount']

missing_data['PercentMissing'] = round(missing_data['MissingCount'] / matrix.shape[0], 3)

missing_data = missing_data.sort_values(by = 'MissingCount',ascending = False)

missing_data = missing_data[missing_data['PercentMissing'] > 0]

print(missing_data)



plt.figure(figsize=(16,12))

plt.subplot(2,1,1)

sns.barplot(x=np.arange(len(missing_data)), y=missing_data['MissingCount'])

for row in range(len(missing_data)):

    plt.text(row-0.3, missing_data.ix[row, 'MissingCount'] + 5,'%i' % missing_data.ix[row, 'MissingCount'])

plt.xticks(ticks=np.arange(19), labels=missing_data.index, rotation=60);

plt.title('Missing data count by feature', fontsize=15);



plt.subplots_adjust(hspace=0.5)

plt.subplot(2,1,2)

plt.hlines(0.15,-1,100000, color='r')

sns.barplot(x=missing_data.index, y=missing_data['PercentMissing'])

plt.xticks(ticks=np.arange(19), labels=missing_data.index, rotation=60);

plt.xlabel('Features', fontsize=15);

plt.ylabel('Percent of missing values', fontsize=15);

plt.title('Percent missing data by feature', fontsize=15);
# 初步进行数据删除和填充

# 当缺失值达到15%时，我们应当删除该变量。数据缺失标明没有收集到这个数据（或者因其他原因丢失），这表明这个变量本身就不是很重要。

matrix.drop(missing_data[missing_data.PercentMissing >= 0.15].index, axis=1, inplace=True)

for col in matrix.columns:

    if matrix[col].dtypes == 'O':

        # 众数填充

        matrix[col].fillna(matrix[col].mode()[0], inplace=True)

    else:

        matrix[col].fillna(method='ffill', inplace=True)

# matrix.drop(missing_data.where(missing_data['PercentMissing'] > 0.1)['ColumnName'], axis=1, inplace=True)

# matrix[missing_data.where(missing_data['PercentMissing'] <= 0.1)['ColumnName']].fillna(menthod='ffill')



# 进一步观察

# GarageX类变量缺失的数量相等，可以推测是来自相同记录，此外我们还有GarageCars这个变量，它与GarageX变量相关

# question：

# 1、非文本数据（GarageX）如何分析相关性
from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



feature_num =[key for key in dict(matrix.dtypes) if dict(matrix.dtypes)[key] in 

              ['float64', 'int64', 'float32', 'int32']]

# feature_cat = [key for key in dict(matrix.dtypes) if dict(matrix.dtypes)[key] in ['object']]

# print(len(feature_cat))

print('count of num-dtype variate:', len(feature_num))



matrix_skew = matrix[feature_num].apply(lambda x: x.skew()).sort_values(ascending=False)

high_skew_cols = matrix_skew[matrix_skew > 0.5].index

for col in high_skew_cols:

    matrix[col] = boxcox1p(matrix[col], boxcox_normmax(matrix[col] + 1))
# test，target预防行缺失，数据不匹配,IQR容易删除较多数据，不好把握

def IQR_Outlier(df,cols=None,whiskers=3,q1=0.01,q3=0.99):

    if cols == None:

        num_var=[key for key in dict(df.dtypes) if dict(df.dtypes)[key] in 

                 [ 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',  'float16', 'float32', 'float64']]

        df = df[num_var]

    for col in cols:

        Q1 = df[col].quantile(q1)

        Q3 = df[col].quantile(q3)

        IQR = Q3 - Q1

        if IQR == 0:

            IQR = 1

        down_bound = Q1 - whiskers * IQR

        up_bound = Q3 + whiskers * IQR

        df = df[df[col].between(down_bound, up_bound)]

    return df



# matrix = matrix.join(target, on='Id', how='left')

# matrix = IQR_Outlier(matrix, feature_num, whiskers=3)

# target = matrix[matrix.Id <= train.Id.max()]['SalePrice']

# matrix = matrix.drop('SalePrice', axis=1)



matrix[feature_num] = matrix[feature_num].apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)) )
# matrix = pd.get_dummies(matrix)

# matrix.columns= [var.strip().replace(r'[\.|&|\s]', '_') for var in matrix.columns]
from matplotlib.ticker import MultipleLocator



print("Skewness: %f" % target.skew())

print("Kurtosis: %f" % target.kurt())

ax = plt.subplot()

sns.distplot(target, fit=norm);

(mu, sigma) = norm.fit(target);

# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

ax.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

           loc='best', fontsize=10, framealpha=0)

ax.set_ylim(0, 0.00001);

# ax.set_xscale('log')

ax.xaxis.set_major_locator(MultipleLocator(200000))



plt.figure(figsize=(12,6))

plt.subplot(1,2,1);

target = np.log1p(target)

sns.distplot(target, fit=norm);

(mu, sigma) = norm.fit(target)

# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

           loc='best', fontsize=13, framealpha=0);

plt.ylabel('Frequence', fontsize=13);

plt.ylim(0, 1.5);

# plt.figure();

plt.subplot(1,2,2);

stats.probplot(target, plot=plt);

plt.figure();
# if not os.path.isdir('../input/cleaned_data'):

#     os.mkdir('../input/cleaned_data/')

matrix.to_pickle('matrix.pkl')

target.to_pickle('target.pkl')




# # %% [code] {"scrolled":false}

# plt.figure(figsize=(18,4))

# plt.subplot(1,2,1)

# plt.scatter(train['GrLivArea'], train['SalePrice'], c='g', edgecolor='k')

# plt.xlabel('GrLivArea')

# plt.ylabel('SalePrice')

# plt.subplot(1,2,2)

# plt.scatter(train['TotalBsmtSF'], train['SalePrice'], c='r', edgecolor='k')

# plt.xlabel('TotalBsmtSF')

# plt.ylabel('SalePrice')

# plt.subplots_adjust(wspace=0.2)

# # plt.figure(constrained_layout=True)



# # %% [code] {"scrolled":true}

# # 从图中可知，GrLivArea和TotalBsmtSF都存在较大的异常值，接下来我们看一下异常值对应的记录**

# # train.sort_values(by='GrLivArea', ascending=True).head(4)

# train.drop(index=train[(train['GrLivArea'] > 4000) | (train['TotalBsmtSF'] > 3000)].index, axis=0, inplace=True)

# train.reset_index(drop=True, inplace=True)



# # %% [code] {"scrolled":false}

# # OverallQual: Rates the overall material and finish of the house

# data = pd.concat([train['SalePrice_log'], train['OverallQual']], axis=1)

# sns.boxplot(x='OverallQual', y='SalePrice_log', data=data);



# # %% [code] {"scrolled":false}

# from matplotlib.ticker import MultipleLocator, FormatStrFormatter  



# fig, ax = plt.subplots(figsize=(16,6))

# data = pd.concat([train['SalePrice_log'], train['YearBuilt']], axis=1)

# sns.boxplot(x='YearBuilt', y='SalePrice_log', data=data)

# plt.xticks(rotation='90');

# xmajorLocator = MultipleLocator(10)

# ax.xaxis.set_major_locator(xmajorLocator)



# # %% [code]

# correlation_matrix = train.corr()

# fig, ax = plt.subplots(figsize=(12,9))

# sns.heatmap(correlation_matrix, vmax=8, square=True, cmap=plt.cm.Paired);



# # %% [code]

# plt.figure(figsize=(8,8))

# cols = correlation_matrix.nlargest(10, 'SalePrice')['SalePrice'].index

# cm = np.corrcoef(train[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)



# # %% [markdown]

# # 与SalePrice关系最密切的前十个因素



# # %% [code]

# sns.pairplot(train[cols])



# # %% [code] {"scrolled":false}

# from sklearn.preprocessing import StandardScaler



# scaler = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis])

# low_range = scaler[scaler[:, 0].argsort()][: 10]

# high_range = scaler[scaler[:, 0].argsort()][-10:]

# # pd.concat([pd.DataFrame(low_range), pd.DataFrame(high_range)],axis=1)

# pd.DataFrame(np.c_[low_range, high_range], columns=['low_range', 'high_range'])



# # %% [code] {"scrolled":false}

# pd.concat([train['SalePrice'],train['GrLivArea']], axis=1).plot.scatter(x='GrLivArea', y='SalePrice')

# # plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])

# # plt.xlabel('GrLivArea')

# # plt.ylabel('SalePrice')



# # %% [markdown]

# # 从图中可知，存在少量离群值，若分析GrLivArea和SalePrice的关系，需要剔除离群值



# # %% [code] {"scrolled":true}

# train.drop(train.sort_values(by='GrLivArea', ascending=False)[: 2].index, inplace=True)



# # %% [code] {"scrolled":false}

# pd.concat([train['SalePrice'],train['TotalBsmtSF']], axis=1).plot.scatter(x='TotalBsmtSF', y='SalePrice');



# # %% [code] {"scrolled":false}

# # TotalBsmtSF列存在0值，生成的新列TotalBsmtSF_log会有NaN

# sns.distplot(np.log(train[train['TotalBsmtSF']>0]['TotalBsmtSF']), fit=norm);

# plt.figure()

# stats.probplot(np.log(train[train['TotalBsmtSF']>0]['TotalBsmtSF']), plot=plt);



# # %% [code]

# train['GrLivArea_log'] = np.log(train['GrLivArea'])

# sns.distplot(train['GrLivArea_log'], fit=norm);

# plt.figure()

# stats.probplot(train['GrLivArea_log'], plot=plt);



# # %% [code]

# plt.scatter(train['GrLivArea_log'], train['SalePrice_log'])



# # %% [code] {"scrolled":true}

# plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice_log']);



# # %% [code]

# # 使用决策树简单的取出较为重要的属性

# enc = LabelEncoder()

# train_enc = train.apply(lambda x: x if x.dtype == 'int64' else enc.fit_transform(x))

# tree = DecisionTreeClassifier(random_state=38)

# tree.fit(train_enc[train_enc.columns.drop(['Id','SalePrice', 'SalePrice_log'])], train_enc['SalePrice_log'])

# COL = [col[1] for col in sorted(zip(tree.feature_importances_, train_enc.columns), reverse=True) if train[col[1]].dtype == int]

# col = COL[:15]



# # %% [code] {"scrolled":true}

# X_train, X_test, y_train, y_test = train_test_split(train_enc[col], train['SalePrice_log'], random_state=42)

# tree_normal = DecisionTreeRegressor(random_state=38)

# tree_normal.fit(X_train, y_train)

# print('默认参数的决策树回归分数：',tree_normal.score(X_test, y_test))

# scores = cross_val_score(tree_normal, X_test, y_test, cv=10)

# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # 获取置信区间。（也就是均值和方差）



# # export_graphviz(tree, out_file='../input/4_var_forest.dot',

# #                 feature_names=['TotalBsmtSF', 'GrLivArea', 'YearBuilt', 'LotArea'],

# #                impurity=False, filled=True)

# # with open('../input/4_var_forest.dot') as f:

# #     graph = f.read()

# #     graphviz.Sourc(graph).view()



# # 数据预处理，修改决策树参数，



# scaler = StandardScaler()

# scaler.fit(X_train)

# X_train_scaled = scaler.transform(X_train)

# X_test_scaled = scaler.transform(X_test)

# tree_scaled = DecisionTreeRegressor(random_state=38)

# tree_scaled.fit(X_train_scaled, y_train)

# print('正则化预处理后的得分：',tree_scaled.score(X_test_scaled, y_test))



# # scaler = MinMaxScaler()

# # scaler.fit(X_train)

# # X_train_scaled = scaler.transform(X_train)

# # X_test_scaled = scaler.transform(X_test)

# # tree.fit(X_train_scaled, y_train)

# # print('数据归一化预处理后的得分：',tree.score(X_test_scaled, y_test))



# # %% [code]

# lr = LinearRegression()

# lr.fit(X_train, y_train)

# print(lr.score(X_test, y_test))



# # %% [code]

# def plot_learning_curve(estimator, X, y, title="Learning Curves",

#                         ylim=(0.6,1), cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

#     plt.figure()

#     plt.title(title)

#     if ylim is not None:

#         plt.ylim(*ylim)

#     plt.xlabel("Training examples")

#     plt.ylabel("Score")

#     train_sizes, train_scores, test_scores = learning_curve(

#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

#     train_scores_mean = np.mean(train_scores, axis=1)

#     train_scores_std = np.std(train_scores, axis=1)

#     test_scores_mean = np.mean(test_scores, axis=1)

#     test_scores_std = np.std(test_scores, axis=1)

#     plt.grid()

 

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 

#                      train_scores_mean + train_scores_std, alpha=0.1, color="r")

#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")

    

#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

 

#     plt.legend(loc="best")

#     return

 

 

 

# cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)

# plot_learning_curve(estimator=LinearRegression(), X=X_train, y=y_train, cv=cv, n_jobs=1)



# # %% [code]

# submission = pd.DataFrame()

# submission['Id'] = test['Id']

# submission['SalePrice'] = np.exp(lr.predict(test[col].fillna(method='pad')))

# submission.to_csv('submission.csv', index=False)



# # %% [code]

# train = pd.read_csv('../input/train.csv', index_col=0)

# test = pd.read_csv('../input/test.csv', index_col=0)

# all_df = pd.concat([train, test], axis=0)

# y_train = np.log1p(train.pop('SalePrice'))



# # %% [code]

# # MSSubClass是category

# all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

# all_df['MSSubClass'].value_counts()



# # %% [code]

# pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()



# # %% [code]

# all_dummy_df = pd.get_dummies(all_df)

# all_dummy_df.head()



# # %% [code]

# all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)



# # %% [code]

# mean_cols = all_dummy_df.mean()

# mean_cols.head()



# # %% [code]

# all_dummy_df = all_dummy_df.fillna(mean_cols)



# # %% [code]

# all_dummy_df.isnull().sum().sum()



# # %% [code]

# numeric_cols = all_df.columns[all_df.dtypes != 'object']

# numeric_cols



# # %% [code]

# numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()

# numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()

# all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std



# # %% [code]

# dummy_trian_df = all_dummy_df.loc[train.index]

# dummy_test_df = all_dummy_df.loc[test.index]



# # %% [code]

# from sklearn.linear_model import Ridge

# from sklearn.model_selection import cross_val_score



# X_train = dummy_trian_df.values

# X_test = dummy_test_df.values



# alphas = np.logspace(-3, 2, 50

# test_scores = []

# for alpha in alphas:

#     clf = Ridge(alpha)

#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))

#     test_scores.append(np.mean(test_score))



# # %% [code]

# import matplotlib.pyplot as plt

# %matplotlib inline



# plt.plot(alphas, test_scores)



# # %% [code]