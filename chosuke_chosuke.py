# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 100)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#データ取り込み。
train = pd.read_csv("/kaggle/input/exam-for-students20200923/train.csv")
test = pd.read_csv("/kaggle/input/exam-for-students20200923/test.csv")
country = pd.read_csv("/kaggle/input/exam-for-students20200923/country_info.csv")
#countryデータをjoin
train = pd.merge(train, country, on='Country', how='left')
test = pd.merge(test, country, on='Country', how='left')
train.shape
test.shape
train.head ()
#列名の確認
train.columns.to_list()
#列名の確認
test.columns.to_list()
#基礎統計量
train.describe()
#欠損状況の確認
train.isnull().sum()/train.shape[0]
#欠損状況の確認
test.isnull().sum()/train.shape[0]
#型の確認
train.dtypes
#型の確認
test.dtypes
test['MilitaryUS']
train['MilitaryUS']
#カテゴリ変数の区分値の数を確認する
cat_cols = train.select_dtypes(include=object).columns.to_list()
for c in cat_cols:
    print(c, len(train[c].unique())  )
#カテゴリ変数の区分値の種類を確認する
cat_cols = train.select_dtypes(include=object).columns.to_list()
for c in cat_cols:
    print(c, list(train[c].unique()), len(train[c].unique())  )
list(train['MilitaryUS'].unique())
list(test['MilitaryUS'].unique())
#以下は区分値が多い（テキスト系）のでとりあえず外す
#DevType 5696 object
#CommunicationTools 996 object
#CurrencySymbol 111 object
#FrameworkWorkedWith 812 object

#MilitaryUdはtrainしか含まれないので外す
train = train.drop(['DevType', 'CommunicationTools', 'FrameworkWorkedWith','MilitaryUS'], axis=1)
test = test.drop(['DevType', 'CommunicationTools', 'FrameworkWorkedWith','MilitaryUS'], axis=1)
train.shape
test.shape
#目的変数を確認する
train["ConvertedSalary"]
#目的変数の分布を確認する
hist = plt.hist(train["ConvertedSalary"], bins=100)
plt.show()
#log(x+1)をとる
hist = plt.hist(np.log1p(train["ConvertedSalary"]), bins=100)
plt.show()
#説明変数を数値列とカテゴリ列で分割
train_x_not_ob = train.drop(['Respondent', 'ConvertedSalary'], axis=1).select_dtypes(exclude=['object'])
train_x_ob = train.drop(['Respondent', 'ConvertedSalary'], axis=1).select_dtypes(include=['object'])
test_x_not_ob = test.drop(['Respondent'], axis=1).select_dtypes(exclude=['object'])
test_x_ob = test.drop(['Respondent'], axis=1).select_dtypes(include=['object'])
#数値列の相関確認
df = train.drop(['Respondent'], axis=1).select_dtypes(exclude=['object'])
corr = df.corr()
corr
plt.figure(figsize=(50,30),dpi=200)
sns.heatmap(corr, square=True, annot=True)
plt.show
#カテゴリ列をxgb用にLabelEncodingする
train_test_x_ob = pd.concat([train_x_ob, test_x_ob]) #ラベルはtestも含めて検索
cat_cols = train_x_ob.columns.to_list()
for c in cat_cols:
    le = LabelEncoder()
    train_test_x_ob[c].fillna('NULL', inplace=True)
    train_x_ob[c].fillna('NULL', inplace=True)
    test_x_ob[c].fillna('NULL', inplace=True)
    le.fit(train_test_x_ob[c])
    train_x_ob[c] = le.transform(train_x_ob[c])
    test_x_ob[c] = le.transform(test_x_ob[c])
#concattしてdfに戻し、説明変数にする
train_x = pd.merge(train_x_not_ob, train_x_ob, left_index=True, right_index=True)
test_x = pd.merge(test_x_not_ob, test_x_ob, left_index=True, right_index=True)
train_x.shape
test_x.shape
#ターゲットのログをとる（ゼロを含むのでlog1p）
train['ConvertedSalary'] = np.log1p(train['ConvertedSalary'])
#目的変数を作成
train_y = train.ConvertedSalary
#最終チェック
train_x.head()
test_x.head()
train_y
'''
#train,testに分割
tr_x, val_x, tr_y, val_y  = train_test_split(train_x, train_y, random_state=123, test_size=0.2)
'''
#xgbモデル作成
reg = xgb.XGBRegressor(objective = 'reg:squarederror')
'''
#デフォルトparamで学習
eval_set = [(val_x, val_y)]
reg.fit(tr_x, tr_y, eval_metric='rmse', eval_set = eval_set, early_stopping_rounds=10, verbose=False)
'''
'''
#tarinとvalidの予測を出力
pred_tr = reg.predict(tr_x)
pred_val = reg.predict(val_x)
'''
'''
#ベースライン（ハイパーパラメーターデフォルト）モデルのRMSEを確認
print(np.sqrt(mean_squared_error(tr_y, pred_tr)))
print(np.sqrt(mean_squared_error(val_y, pred_val)))
print(np.sqrt(mean_squared_error(np.exp(val_y), np.exp(pred_val))))
print(mean_absolute_error(np.exp(val_y), np.exp(pred_val)))
'''
'''
#①学習率とその他をデフォルトに固定して木の数を決定する
param1 = {'learning_rate': [0.1], 'n_estimators':[x for x in range(200,400,100)]}
grid1 = GridSearchCV(estimator=reg, param_grid=param1, scoring='neg_root_mean_squared_error', cv=5, refit=True)
#grid1.fit(tr_x, tr_y)
grid1.fit(train_x, train_y, eval_metric='rmse')
'''
'''
# 各パラメータのスコア・標準偏差をDataFrame化  
means = grid1.cv_results_['mean_test_score']  
stds = grid1.cv_results_['std_test_score']  
params = grid1.cv_results_['params']  
df = pd.DataFrame(data=zip(means, stds, params), columns=['mean', 'std', 'params'])  

# スコアの降順に並び替え  
df = df.sort_values('std', ascending=True)  
df = df.sort_values('mean', ascending=False)  

# スコア・標準偏差・パラメータを表示  
for index, row in df.iterrows():  
    print("mean: %.3f +/-%.4f, params: %r" %   
          (row['mean'], row['std']*2, row['params']))  
'''
'''
print("Best score: %.4f" % (grid1.best_score_))  
print(grid1.best_params_) 
'''
'''
#②学習率、木の数を固定し、木の形状パラメーターをきめる。

param2 = {'learning_rate': [0.1],
          'n_estimators': [300],
          'max_depth': [3],
          'min_child_weight': [1],
          'gamma': [0,1],
          'subsample':[0.5, 0.7],
          'colsample_bytree': [0.5, 0.7],}
grid2 = GridSearchCV(estimator = reg, param_grid = param2, scoring='neg_root_mean_squared_error', cv=5, refit=True)
grid2.fit(train_x, train_y, eval_metric='rmse')
'''
'''
# 各パラメータのスコア・標準偏差をDataFrame化  
means = grid2.cv_results_['mean_test_score']  
stds = grid2.cv_results_['std_test_score']  
params = grid2.cv_results_['params']  
df = pd.DataFrame(data=zip(means, stds, params), columns=['mean', 'std', 'params'])  

# スコアの降順に並び替え  
df = df.sort_values('std', ascending=True)  
df = df.sort_values('mean', ascending=False)  

# スコア・標準偏差・パラメータを表示  
for index, row in df.iterrows():  
    print("mean: %.3f +/-%.4f, params: %r" %   
          (row['mean'], row['std']*2, row['params']))  
'''
'''
print("Best score: %.4f" % (grid2.best_score_))  
print(grid2.best_params_)  
'''
'''
#③この状態で学習率を下げ、最後に100%で学習する
param3 = {'learning_rate': [x for x in np.arange(0.1,0.09,-0.01)],
          'n_estimators': [300],
          'max_depth': [3],
          'min_child_weight': [1],
          'gamma': [0],
          'subsample':[0.7],
          'colsample_bytree': [0.5],}
grid3 = GridSearchCV(estimator = reg, param_grid = param3, scoring='neg_root_mean_squared_error', cv=5, refit=True)
#grid3.fit(tr_x, tr_y)
grid3.fit(train_x, train_y, eval_metric='rmse')
'''
'''
# 各パラメータのスコア・標準偏差をDataFrame化  
means = grid3.cv_results_['mean_test_score']  
stds = grid3.cv_results_['std_test_score']  
params = grid3.cv_results_['params']  
df = pd.DataFrame(data=zip(means, stds, params), columns=['mean', 'std', 'params'])  

# スコアの降順に並び替え  
df = df.sort_values('std', ascending=True)  
df = df.sort_values('mean', ascending=False)  

# スコア・標準偏差・パラメータを表示  
for index, row in df.iterrows():  
    print("mean: %.3f +/-%.4f, params: %r" %   
          (row['mean'], row['std']*2, row['params']))  
'''
'''
print("Best score: %.4f" % (grid3.best_score_))  
print(grid3.best_params_)  
'''
#④確定版。最後に100%で学習する
param4 = {'learning_rate': [0.1],
          'n_estimators': [800],
          'max_depth': [3],
          'min_child_weight': [1],
          'gamma': [0],
          'subsample':[0.7],
          'colsample_bytree': [0.5],}
grid4 = GridSearchCV(estimator = reg, param_grid = param4, scoring='neg_root_mean_squared_error', cv=5, refit=True)
#grid4.fit(tr_x, tr_y)
grid4.fit(train_x, train_y, eval_metric='rmse')
# 各パラメータのスコア・標準偏差をDataFrame化  
means = grid4.cv_results_['mean_test_score']  
stds = grid4.cv_results_['std_test_score']  
params = grid4.cv_results_['params']  
df = pd.DataFrame(data=zip(means, stds, params), columns=['mean', 'std', 'params'])  

# スコアの降順に並び替え  
df = df.sort_values('std', ascending=True)  
df = df.sort_values('mean', ascending=False)  

# スコア・標準偏差・パラメータを表示  
for index, row in df.iterrows():  
    print("mean: %.3f +/-%.4f, params: %r" %   
          (row['mean'], row['std']*2, row['params']))  
print("Best score: %.4f" % (grid4.best_score_))  
print(grid4.best_params_)  
#ベストモデルを取得（100%学習済み）
best_model = grid4.best_estimator_
#Importanceの可視化
plt.figure(figsize=(5,15),dpi=150)
plt.barh(range(train_x.shape[1]), grid4.best_estimator_.feature_importances_, align='center')
plt.yticks(np.arange(train_x.shape[1]), train_x.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()
#テストデータの予測
pred = best_model.predict(test_x)
submission = pd.read_csv("/kaggle/input/exam-for-students20200923/sample_submission.csv")
submission.columns
submission['ConvertedSalary'] = list(np.exp(pred))
submission.head()
#提出
submission.to_csv('/kaggle/working/submission.csv', index=False)
