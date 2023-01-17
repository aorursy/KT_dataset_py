import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
df = pd.read_csv('../input/train.csv', header=0)
df.head()
df_s = pd.read_csv('../input/test.csv', header=0)
df_s.head()
df_null = df.loc[:, df.isnull().any()].isnull().apply(lambda x: x.value_counts())\
.reindex([True, False], axis=0).T
df_null.columns = ['null', 'non_null']

df_null
fig, axis = plt.subplots(figsize=(5, 8))
df_null[::-1].plot.barh(ax=axis, color=['r', 'blue'], alpha=.5)
df[df_null.T.columns].describe()
df_null.loc[df[df_null.T.columns].describe().columns, :]
fig, axes = plt.subplots(3, 1, figsize=(10,4))
for i, col in enumerate(df[df_null.T.columns].describe().columns):
    ax=axes.ravel()[i]
    df[col].plot.hist(ax=ax, bins=20, alpha=.5, title=col)
fig.tight_layout()
df.corr()
fig, axes = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), cmap='Blues', )
fig, axis = plt.subplots(3, 1, figsize=(10, 4))
for i, col in enumerate(df[df_null.T.columns].describe().columns):
    ax = axis.ravel()[i]
    df.corr()[col].plot.bar(ax=ax, alpha=.5)
    ax.set_title(col)
    ax.set_xticklabels('')
#     print(f"{col}:\n{df.corr()[col]}\n")
axis.ravel()[2].set_xticklabels(df.corr().index)
fig.tight_layout()

df[df_null.T.columns].describe(exclude='number')
null_obj_col = [col for col in df_null.T.columns if df[col].dtypes == object]
print(null_obj_col)
df['MasVnrType'].value_counts(dropna=False, sort=False)
# value_counts で (dropna=False, sort=False) にすると NaN が最初に来るのがデフォルトみたい
# print(f"{len(null_obj_col)}")
fig, axes = plt.subplots(4, 4, figsize=(12, 6))
for i, col in enumerate(null_obj_col):
    ax = axes.ravel()[i]
#     df_plot = df.astype(str).groupby(col).size()
    df_plot = df[col].value_counts(sort=False, dropna=False)
    df_plot.plot.bar(ax=ax, alpha=.5, color='b')
    ax.set_xlabel(''), ax.set_title(col)
    ax.set_xticklabels([indx[:3] for indx in df_plot.index.astype(str)])
#     print(f"{col}: {[type(indx) for indx in df_plot.index]}, {type(df_plot.index)}")
fig.tight_layout()
print("df.shape:   {}".format(df.shape))
# print(": {}, {}".format(list(df.columns.values), df.columns.size))
ID = df.iloc[:, [0]]
X = df.iloc[:, 1: -1]
y = df.iloc[:, [-1]]
print("X.shape:   {}\n".format(X.shape))

print("df_s.shape: {}".format(df_s.shape))
# print(": {}, {}".format(list(df_s.columns.values), df_s.columns.size))
ID_s = df_s.iloc[:, [0]]
X_s = df_s.iloc[:, 1:]
print("X_s.shape: {}".format(X_s.shape))
ctg_col = [c for c in df.describe(exclude='number').columns]
ctg_col = [col for col in df.columns if df[col].dtypes == object]
# print(f"{ctg_col}\n")
for i, col in enumerate(ctg_col):
    print(f"{i}: {col}, {df[col].nunique()}, {df[col].unique()}")
print(X.columns[X.dtypes ==object].isin(X_s.columns[X_s.dtypes ==object]).all())
print(X_s.columns[X_s.dtypes ==object].isin(X.columns[X.dtypes ==object]).all())
print(X.columns[X.dtypes !=object].isin(X_s.columns[X_s.dtypes !=object]).all())
print(X_s.columns[X_s.dtypes !=object].isin(X.columns[X.dtypes !=object]).all())
print("{}".format(list(X.columns[X.dtypes != object])))
type(X.describe())
X.describe().loc[['std', 'mean'], :].T
non_object_colmuns = X.columns[X.dtypes != object]
cv_dic = {}
print("{:>13}: {:>9} {:>11} {:>6}".format("col", "std()", "mean()", "cv"))
for col in non_object_colmuns:
    cv = X[col].std() / X[col].mean()
    print("{:>13}: {:>9.3f}, {:>10.3f}, {:>6.3f}".format(col, X[col].std(), X[col].mean(), cv))
    cv_dic[col] = cv
for i, elem in enumerate(sorted(cv_dic.items(), key=lambda item: -item[1])):
    print("{:>2}: {:>13}: {:>6.3f}".format(i, elem[0], elem[1]))
target = "OverallQual"
print(": \n{}\n".format(X[target].value_counts()))
# print(": \n{}\n".format(sorted(X[target].items(), key=lambda item: -item[1])))
X[target].hist()
fig, axes = plt.subplots(figsize=(12,3))
x1 = np.arange(len(cv_dic))
y1 = cv_dic.values()
axes.bar(x1, y1, label='CV')
axes.set_xticks(x1)
axes.set_xticklabels(cv_dic.keys(), rotation=90)
axes.set_xlabel('Features'); axes.set_ylabel('CV'); axes.set_title('Coefficient of Variance')

axes.legend()
fig, axes = plt.subplots(6, 6, figsize=(18,12))
for i, col in enumerate(non_object_colmuns):
    ax = axes.ravel()[i]
    X[col].plot.hist(ax=ax)
    ax.set_title(col, fontsize=14); ax.set_ylabel('')
fig.tight_layout()
corr_coef_dic = {}
for col_name in non_object_colmuns:
    corr_coef_matrix = np.corrcoef(X[col_name], y['SalePrice'])
    print("{:>13}: {: .3f}".format(col_name, corr_coef_matrix[0][1]))
    corr_coef_dic[col_name] = corr_coef_matrix[0][1]

np.corrcoef(X['MSSubClass'], y['SalePrice'])
# print(": \n{}\n".format(corr_coef_dic))
x1 = np.arange(len(corr_coef_dic))
y1 = corr_coef_dic.values()
fig, axes = plt.subplots(figsize=(15, 2))
axes.bar(x1, y1, label='Correlation Coefficient')
f_size = 12
axes.set_xticks(x1); axes.set_xticklabels(corr_coef_dic.keys(), rotation=90, fontsize=f_size)
axes.set_title("Correlation Coefficient to 'SalePrice'", fontsize=f_size)

axes.legend()
for i, elem in enumerate(sorted(corr_coef_dic.items(), key=lambda item: -item[1])):
    print("{:>2}: {:>13}: {: .4f}".format(i, elem[0], elem[1]))
ohe_columns = X.columns[X.dtypes == object]
X_ohe = pd.get_dummies(X, dummy_na=True, columns=ohe_columns)
print("X_ohe.shape: {}".format(X_ohe.shape))

X_ohe.head()
nullrow, nullcol = X_ohe.isnull().any(axis=1), X_ohe.columns[X_ohe.isnull().any()]
print("X_ohe.loc[nullrow, nullcol].head(7): \n{}\n".format(X_ohe.loc[nullrow, nullcol].head(7)))
MasVnrArea_nullrow = X_ohe['MasVnrArea'].isnull()
print("X_ohe[MasVnrArea_nullrow][['MasVnrArea']]: \n{}\n".format(X_ohe[MasVnrArea_nullrow][['MasVnrArea']]))
print(": \n{}\n"
      .format(X_ohe.query('MasVnrArea == "NaN"')[['MasVnrArea']])) # これでもオーケー
X_ohe.loc[:, nullcol].mean()
from sklearn.preprocessing import Imputer
X_ohe_columns = X_ohe.columns
print(": {}".format(X_ohe_columns))
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_ohe)
X_ohe = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe_columns)
print("X_ohe.loc[nullrow, nullcol].head(7): \n{}\n".format(X_ohe.loc[nullrow, nullcol].head(7)))
print("X_ohe[MasVnrArea_nullrow][['MasVnrArea']]: \n{}".format(X_ohe[MasVnrArea_nullrow][['MasVnrArea']]))
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
pipe_ols = Pipeline([('scl', StandardScaler()), ('est', LinearRegression())])
pipe_ridge = Pipeline([('scl', StandardScaler()), ('est', Ridge())])
# pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestRegressor(random_state=0))])
pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestRegressor())])
# pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingRegressor(random_state=0))])
pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingRegressor())])
from sklearn.feature_selection import RFE
n_best = 181
best_feature_n = 181
best_feature_n
selector = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=int(best_feature_n), step=.05)  # best_feature_n: 181
selector.fit(X_ohe, y.values.ravel())
X_fin = pd.DataFrame(selector.transform(X_ohe), columns=X_ohe_columns[selector.support_])
print("X_fin.shape: {}".format(X_fin.shape))
rfe_columns_set = set(X_ohe_columns[selector.support_])
print("\nrfe_columns_set = set(X_ohe_columns[selector.support_]: {}".format(len(rfe_columns_set)))
# print(f"{rfe_columns_set}")

no_rfe_set = set(X_ohe_columns) - rfe_columns_set
print("no_rfe_set = set(X_ohe_columns) - rfe_columns_set: {}".format(len(no_rfe_set)))
# print(f"{no_rfe_set}")
# とりあえずカテゴリ変数の指定なしで読み込んで pd.get_dummies してみる

df_s = pd.read_csv("../input/test.csv")
print("df.shape, df_s.shape: {}, {}".format(df.shape, df_s.shape))
ID_s = df_s.iloc[:, [0]]
X_s = df_s.iloc[:, 1:]
print("X.shape, X_s.shape:   {}, {}".format(X.shape, X_s.shape))

set_X_col = set(X.columns)
set_X_s_col = set(X_s.columns)
print("{}, {} ### カラムの不一致なし".format(set_X_col - set_X_s_col, set_X_s_col - set_X_col))
# 念のため ohe_columns もチェック
ohe_columns_s = X_s.columns[X_s.dtypes == object]
print(": {}, {}".format(set(ohe_columns) - set(ohe_columns_s), set(ohe_columns_s) - set(ohe_columns)))
X_ohe_s = pd.get_dummies(X_s, dummy_na=True, columns=ohe_columns_s)
print(": {}, {}  ## OMG あれれー".format(X_ohe.shape, X_ohe_s.shape))  ## OMG あれれー
X_ohe_s_columns = X_ohe_s.columns
print("set(X_ohe_columns) - set(X_ohe_s_columns): {}\n{}\n"
      .format(len(set(X_ohe_columns) - set(X_ohe_s_columns)), set(X_ohe_columns) - set(X_ohe_s_columns)))
print("set(X_ohe_s_columns) - set(X_ohe_columns): {}\n{}"
      .format(len(set(X_ohe_s_columns) - set(X_ohe_columns)), set(X_ohe_s_columns) - set(X_ohe_columns)))
# カラム構成が同じデータフレームの結合
df1 = pd.DataFrame([[1,2,3]], columns=['c1','c2','c3']); print("df1: \n{}".format(df1))
df2 = pd.DataFrame([[3,2,1]], columns=['c1','c2','c3']); print("df2: \n{}".format(df2))
df_all = pd.concat([df1, df2])
df_all
# カラム構成が異なるデータフレームの結合
df3 = pd.DataFrame([[0,1,2,3]], columns=['c0','c1','c3','c4']); print("df3: \n{}".format(df3))
df_all_2 = pd.concat([df_all, df3], sort=True)
df_all_2
# FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
# of pandas will change to not sort by default.
# To accept the future behavior, pass 'sort=False'.
# To retain the current behavior and silence the warning, pass 'sort=True'.
print("{}".format(X_ohe_columns))
df_col_model = pd.DataFrame(None, columns=X_ohe_columns, dtype=float) ### dtype=float
print("df_col_model.shape: {}".format(df_col_model.shape))
df_col_model
X_ohe_s_02 = pd.concat([df_col_model, X_ohe_s], sort=True)
print("X_ohe.shape, X_ohe_s_02.shape:{}, {}".format(X_ohe.shape, X_ohe_s_02.shape))
X_ohe_s_02.head()
print("set(X_ohe_s_02.columns) - set(X_ohe.columns): {}".format(set(X_ohe_s_02.columns) - set(X_ohe.columns)))
print(": \n{}\n{}".format(X_ohe.columns.values[:10], X_ohe_s_02.columns.values[:10]))
list(set(X_ohe_s.columns) - set(X_ohe.columns))
print("X_ohe_s_02.shape: {}  # before drop".format(X_ohe_s_02.shape))
X_ohe_s_02 = X_ohe_s_02.drop(list(set(X_ohe_s.columns) - set(X_ohe.columns)), axis=1)
print("X_ohe_s_02.shape: {}  # after drop".format(X_ohe_s_02.shape))

print(": \n{}\n".format(list(set(X_ohe_columns) - set(X_ohe_s_columns))))
X_ohe_s_02.loc[:, list(set(X_ohe_columns) - set(X_ohe_s_columns))].head()
X_ohe_s_02.loc[:, list(set(X_ohe_columns) - set(X_ohe_s_columns))] = \
    X_ohe_s_02.loc[:, list(set(X_ohe_columns) - set(X_ohe_s_columns))].fillna(0, axis=1)
X_ohe_s_02.loc[:, list(set(X_ohe_columns) - set(X_ohe_s_columns))].head()
test = pd.DataFrame([[1,2,3]], columns=['c1','c2','c3'])
print (test)
test = test.reindex(['c2','c3','c1'], axis=1)
test
X_ohe.head(3)
X_ohe_s_02.head(3)
X_ohe_s_02 = X_ohe_s_02.reindex(X_ohe_columns, axis=1)
print(": {}".format(X_ohe_columns))
X_ohe_s_02.head(3)
nullrow = X_ohe_s_02.isnull().any(axis=1)
nullcol = X_ohe_s_02.columns[X_ohe_s_02.isnull().any()]
X_ohe_s_02.loc[nullrow, nullcol].head()
X_ohe_s_02['BsmtFinSF1'][X_ohe_s_02['BsmtFinSF1'].isnull()]
for i, col in enumerate(nullcol):
    print("{}: {}: {}".format(i, col, X_ohe_s_02[col].isnull().any()))
imp.fit(X_ohe_s_02)
X_ohe_s_03 = pd.DataFrame(imp.transform(X_ohe_s_02), columns=X_ohe_columns)
X_ohe_s_03.head()
for i, col in enumerate(nullcol):
#     print("{}: {}\n{}\n".format(i, col, X_ohe_s_03[col][X_ohe_s_03[col].isnull()]))
    print("{:>2}: {:<12}: {}".format(i, col, X_ohe_s_03[col].isnull().any()))
    
X_ohe_s_03.loc[nullrow, nullcol].head()
X_fin_s = pd.DataFrame(selector.transform(X_ohe_s_03), columns=X_ohe_columns[selector.support_])
print("X_fin_s.shape: {}".format(X_fin_s.shape))
X_fin_s.head()
print("X_fin.shape, X_fin_s.shape: {}, {}".format(X_fin.shape, X_fin_s.shape))
print(": {}, {}".format(set(X_fin.columns) - set(X_fin_s.columns), set(X_fin_s.columns) - set(X_fin.columns)))
n_to_count = 0
for i, elem in enumerate(zip(X_fin.columns.values, X_fin_s.columns.values)):
    if elem[0] != elem[1]:
        print("{:>2}: {}, {}, {}".format(i, elem[0], elem[1], elem[0] == elem[1]))
        n_to_count += 1
print("n_to_count: {}".format(n_to_count))
[elem[0] == elem[1] for elem in zip(X_fin.columns.values, X_fin_s.columns.values) if elem[0] != elem[1]]
print("X_fin.shape, y.shape: {}, {}".format(X_fin.shape, y.shape))
# Holdout
indices = np.arange(X_fin.shape[0])
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_fin, y, indices)
# X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_fin, y, indices, random_state=0)
print(": {}, {}, {}, {}, {}, {}\n".format(
    X_train.shape, X_test.shape, y_train.shape, y_test.shape, indices_train.shape, indices_test.shape))
print("X_train.index.values[:5], y_train.index.values[:5], indices_train[:5]:\n{} \n{} \n{}"
      .format(X_train.index.values[:5], y_train.index.values[:5], indices_train[:5]))
pipe_gb.named_steps['est']  ## FOR CHECK
param_grid_ols = {
                  'est__normalize': ["True", "False"],
                  'est__fit_intercept': ["True", "False"]
                  }
param_grid_ridge = {
                    'est__alpha': [0.01, 0.1, 1, 2, 4, 10, 25, 50],
                    }
param_grid_rf = {
                 'est__n_estimators': [50, 100, 150],
                 'est__max_features': np.arange(0.05, 1.01, 0.05)
                 }
param_grid_gb = {
                 'est__n_estimators': [100, 150, 200],
#                  'est__learning_rate': [1e-2, 1e-1, 0.5, 1.],
                 'est__learning_rate': [1e-1, 0.5, 1.],
                 'est__max_depth': range(1, 11),
                 }
pipe_ols = Pipeline([('scl', StandardScaler()), ('est', LinearRegression())])
pipe_ridge = Pipeline([('scl', StandardScaler()), ('est', Ridge())])
# pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestRegressor(random_state=0))])
pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestRegressor())])
# pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingRegressor(random_state=0))])
pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingRegressor())])
pipes = [pipe_ols, pipe_ridge, pipe_rf, pipe_gb]
# pipes = [pipe_ols]
pipe_names = ['LinearRegression', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor']
params = [param_grid_ols, param_grid_ridge, param_grid_rf, param_grid_gb]
from sklearn.model_selection import GridSearchCV

import time
best_estimator = []
best_score_dic = {}
start = time.time()

best_r2_test = 0
for i, pipe in enumerate(pipes):
    start_for = time.time()
#     print("{}: {}\n{}\n[]\n".format(i, pipe_names[i], pipe.named_steps['est'], params[i]))
    param = params[i]
    gs = GridSearchCV(pipe, param_grid=param, scoring='r2', cv=3)
    gs.fit(X_train, y_train.values.ravel())
    best_estimator.append(gs.best_estimator_)
    train_r2 = r2_score(y_train.values.ravel(), gs.predict(X_train))
    test_r2 = r2_score(y_test.values.ravel(), gs.predict(X_test))
    best_score_dic[pipe_names[i]] = (gs.best_score_, train_r2, test_r2) # 辞書の値としてタプル追加
    if best_r2_test < test_r2:
        best_r2_test = test_r2
        best_n = i
        best_est_test = gs.best_estimator_
    
    print("{}, {}: {}".format(i, pipe_names[i], param))
    print("gs.best_params_: {}".format(gs.best_params_))
    print("gs.best_estimator_.named_steps['est']: \n{}".format(gs.best_estimator_.named_steps['est']))
    print("gs.best_score_: {:.5f}".format(gs.best_score_))
    print("r2_score_train: {:.5f}".format(train_r2))
    print("r2_score_test : {:.5f}".format(test_r2))
    
    print("{:.2f} sec. in total {:.2f} sec.\n".format(time.time() - start_for, time.time() - start))
    
for i, v in enumerate(best_estimator):
    print("{}, {}: \n{}\n".format(i, pipe_names[i], v))
print("{}: {}: {:.4f}\n{}".format(best_n, pipe_names[best_n], best_r2_test, best_est_test))
# これでももちろんいいんだけれども
# 辞書から best_estimator を指定してみる
best_score_dic
for i, v in enumerate(best_score_dic.items()):
    print("{}: {}".format(i, v[0]))
    for score in v[1]:
        print("  {: .4f}".format(score))
sorted(best_score_dic.items(), key=lambda item: -item[1][2])
max(best_score_dic.items(), key=lambda x: x[1][0])
max(best_score_dic.items(), key=lambda x: x[1][2])
b_test_n = max(best_score_dic.items(), key=lambda item: item[1][2])[0] # これで best_estimator 名を取得し、
print("b_test_n: {}".format(b_test_n))
b_test_index = pipe_names.index(b_test_n) # インデクスを取得
print(": {}".format(b_test_index))
print(": \n{}".format(best_estimator[b_test_index]))  # いま一つカッコ悪い
# b_test_n = max(best_score_dic.items(), key=lambda item: item[1][2])[0] # 上記と同様これで best_estimator 名を取得し、
b_test_n = max(best_score_dic.items(), key=lambda item: item[1][0])[0] # kaggle 用に gs.best_score_ のベストモデルを選択してみる
print("b_test_n: {}".format(b_test_n))
best_pipe = {k: v for k, v in zip(pipe_names, best_estimator)}[b_test_n]  # 辞書を作ってキーで指定。こっちがちょっとスマート
best_pipe # 学習済みベストモデル決定
from sklearn.externals import joblib
for i, pipe in enumerate(best_estimator):
#     print("{}: {}\n{}\n".format(i, pipe_names[i], pipe))
    joblib.dump(pipe, pipe_names[i] + '.pkl')
sorted(best_score_dic.items(), key=lambda item: -item[1][2]) # 辞書内の要素の取得方法を sorted で確認
target_pipe_name = max(best_score_dic.items(), key=lambda item: item[1][2])[0] # max で取得した最大値タプルの[0]で名前を取得
print("{}".format(target_pipe_name))

load_best_pipe = joblib.load(target_pipe_name + '.pkl')
load_best_pipe
joblib.dump(best_score_dic, 'best_score_dic.pkl')
load_best_score_dic = joblib.load('best_score_dic.pkl')
load_best_score_dic
best_pipe
best_pipe.predict(X_fin_s).shape
ID_s.shape
submission = pd.DataFrame({'Id': ID_s.values.ravel(), 'SalePrice': best_pipe.predict(X_fin_s)})
submission.head()
submission.to_csv('kaggle_submission.csv', index=False)
for i, pipe in enumerate(best_estimator):
    print(f"{i}: {pipe.named_steps['est']}")
from sklearn.ensemble.partial_dependence import plot_partial_dependence
pdp_pipe = best_estimator[3] # select Pipeline of GradientBoostingRegressor
print(f"{pdp_pipe.named_steps['est']}\n")
df_imp = pd.DataFrame(pdp_pipe.named_steps['est'].feature_importances_, index=X_train.columns, columns=['importance']).reset_index()
print(f"{df_imp.iloc[:5, :]}")
fig, axes = plt.subplots(figsize=(20, 2))
df_imp.plot.bar(ax=axes)
axes.set_xticklabels([c.upper()[:5] for c in df_imp['index']])
fig, axes = plt.subplots(figsize=(10, 2)) # 特徴量を少なくして再表示
df_imp_selected = df_imp.iloc[:40, :]
df_imp_selected.plot.bar(ax=axes); axes.set_xticklabels([i[:7] for i in df_imp_selected['index']])
fig.tight_layout()
plot_partial_dependence(pdp_pipe.named_steps['est'], X=pdp_pipe.named_steps['scl'].transform(X_train), 
                        features=df_imp.index, feature_names=df_imp['importance']) 
# features で特徴量を選択。これだと全表示
print(f"{range(X_train.shape[1])}")
n_to_select = 9 # 特徴量上位数の設定
df_sort = df_imp.sort_values(by=['importance'], ascending=False).iloc[: n_to_select, :] # 降順でインデクスごとソート。上位選択
# plot_partial_dependence 実行。df_sort.index と df_sort['index'] を下で使う
print(f"{df_sort}")
fig, axes = plt.subplots(figsize=(12,5))
plot_partial_dependence(pdp_pipe.named_steps['est'], X=pdp_pipe.named_steps['scl'].transform(X_train), 
                        features=df_sort.index, feature_names=df_sort['index'], ax=axes)
fig.tight_layout()
