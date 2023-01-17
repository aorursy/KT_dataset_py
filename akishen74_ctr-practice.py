# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from matplotlib import pyplot as plt
import seaborn as sns
import gzip

test_file = '../input/avazu-ctr-prediction/test.gz'
samplesubmision_file = '../input/avazu-ctr-prediction/sampleSubmission.gz'
chunksize = 10 ** 6
num_of_chunk = 0
train = pd.DataFrame()
    
for chunk in pd.read_csv('../input/avazu-ctr-train/train.csv', chunksize=chunksize):
    num_of_chunk += 1
    train = pd.concat([train, chunk.sample(frac=.05, replace=False, random_state=123)], axis=0)
    print('Processing Chunk No. ' + str(num_of_chunk))     
    
train.reset_index(inplace=True)

# 備份train 資料長度，以便稍後df重新分割索引用途
train_len = len(train)
train_len
df = pd.concat([train, pd.read_csv(test_file, compression='gzip')]).drop(['index', 'id'], axis=1)
# 建立一個將hour資料轉換為日期格式的function
def get_date(hour):
    y = '20'+str(hour)[:2]
    m = str(hour)[2:4]
    d = str(hour)[4:6]
    return y+'-'+m+'-'+d

# 建立weekday欄位，將hour轉換後填入
df['weekday'] = pd.to_datetime(df.hour.apply(get_date)).dt.dayofweek.astype(str)

# 建立一個將hour資料轉換為時段的function
def tran_hour(x):
    x = x % 100
    while x in [23,0]:
        return '23-01'
    while x in [1,2]:
        return '01-03'
    while x in [3,4]:
        return '03-05'
    while x in [5,6]:
        return '05-07'
    while x in [7,8]:
        return '07-09'
    while x in [9,10]:
        return '09-11'
    while x in [11,12]:
        return '11-13'
    while x in [13,14]:
        return '13-15'
    while x in [15,16]:
        return '15-17'
    while x in [17,18]:
        return '17-19'
    while x in [19,20]:
        return '19-21'
    while x in [21,22]:
        return '21-23'

# 將hour轉換為時段
df['hour'] = df.hour.apply(tran_hour)
# 確認資料型別
df.info()
len_of_feature_count = []
for i in df.columns[2:23].tolist():
    print(i, ':', len(df[i].astype(str).value_counts()))
    len_of_feature_count.append(len(df[i].astype(str).value_counts()))
# 建立一個list，將需要轉換行別的特徵名稱存入該list
need_tran_feature = df.columns[2:4].tolist() + df.columns[13:23].tolist()

# 依序將變數轉換為object型別
for i in need_tran_feature:
    df[i] = df[i].astype(str)
obj_features = []

for i in range(len(len_of_feature_count)):
    if len_of_feature_count[i] > 10:
        obj_features.append(df.columns[2:23].tolist()[i])
obj_features
df_describe = df.describe()
df_describe
def obj_clean(X):
    # 定義一個縮減資料值的function，每次處理一個特徵向量

    def get_click_rate(x):
        # 定義一個取得點擊率的function
        temp = train[train[X.columns[0]] == x]
        res = round((temp.click.sum() / temp.click.count()),3)
        return res

    def get_type(V, str):
        # 定義一個取得新資料值之級距判斷的function
        very_high = df_describe.loc['mean','click'] + 0.04
        higher = df_describe.loc['mean','click'] + 0.02
        lower = df_describe.loc['mean','click'] - 0.02
        very_low = df_describe.loc['mean','click'] - 0.04

        vh_type = V[V[str] > very_high].index.tolist()
        hr_type = V[(V[str] > higher) & (V[str] < very_high)].index.tolist()
        vl_type = V[V[str] < very_low].index.tolist()
        lr_type = V[(V[str] < lower) & (V[str] > very_low)].index.tolist()

        return vh_type, hr_type, vl_type, lr_type

    def clean_function(x):
        # 定義一個依據級距轉換資料值的function
        # 判斷之依據為：總平均點擊率的正負  4% 為very_high(low), 總平均點擊率的正負 2％為higher (lower)
        while x in type_[0]:
            return 'very_high'
        while x in type_[1]:
            return 'higher'
        while x in type_[2]:
            return 'very_low'
        while x in type_[3]:
            return 'lower'
        return 'mid'
        
    print('Run: ', X.columns[0])
    fq = X[X.columns[0]].value_counts()
    # 建立一個暫存的資料值頻率列表
    # 理論上，將全部的資料值都進行分類轉換，可得到最佳效果；實務上為了執行時間效能，將捨去頻率低於排名前1000 row以後的資料值。
    if len(fq) > 1000:
        fq = fq[:1000]

    # 將頻率列表轉換為dataframe，並將index填入一個新的欄位。
    fq = pd.DataFrame(fq)
    fq['new_column'] = fq.index    

    # 使用index叫用get_click_rate function，取得每個資料值的點擊率
    fq['click_rate'] = fq.new_column.apply(get_click_rate)

    # 叫用 get_type function取得分類級距，並儲存為一個list，以便提供給下一個clean_function使用
    type_ = get_type(fq, 'click_rate')

    # 叫用 clean_funtion funtion，回傳轉換後的特徵向量
    return X[X.columns[0]].apply(clean_function)

# 使用for 迴圈將需轉換的特徵輸入到 obj_clean function
for i in obj_features:    
    df[[i]] = obj_clean(df[[i]])

df
# 確認所有特徵的資料值狀況
for i in df.columns:
    sns.countplot(x = i, hue = "click", data = df)
    plt.show()
df.drop(['device_id', 'C14', 'C17', 'C19', 'C20', 'C21'], axis=1, inplace=True)
# 對所有變數進行 one-hot 編碼
df = pd.get_dummies(df)

# 依據處理過得df資料表，重新將train, test分割出來
train = df[:train_len]
test = df[train_len:]
# # 將處理過的train, test 資料集匯出，避免每次重新的冗長處理時間。

# train.to_csv('new_train.csv', index=False)
# test.to_csv('new_test.csv', index=False)
# # 讀取處理過的train, test 資料集，跳過冗長的重新執行處理時間。
# train = pd.read_csv('new_train.csv')
# test = pd.read_csv('new_test.csv')
del df
# 從train資料集中，標籤為0的資料中，隨機抽樣與標籤為1一樣多的數量，並將其結合成正反標籤佔筆各佔50％的資料集
pre_X = train[train['click'] == 0].sample(n=len(train[train['click'] == 1]), random_state=111)
pre_X = pd.concat([pre_X, train[train['click'] == 1]]).sample(frac=1)
pre_y = pre_X[['click']]
pre_X.drop(['click'], axis=1, inplace=True)
test.drop(['click'], axis=1, inplace=True)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 將新的資料集分割為訓練集與驗證集
pre_X_train, pre_X_test, pre_y_train, pre_y_test = train_test_split(pre_X, pre_y, test_size=0.20, stratify=pre_y, random_state=1)
# 執行Grid Search調參，建立100棵樹來取得最佳參數
params = {"criterion":["gini", "entropy"], "max_depth":range(1,20)}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=params, scoring='roc_auc', cv=100, verbose=1, n_jobs=-1)
grid_search.fit(pre_X_train, pre_y_train)
grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_
# 依據Grid Search的結果建立一個決策樹模型，並fit完整資料 (前置資料)
tree = grid_search.best_estimator_
tree.fit(pre_X,pre_y)

# 輸出重要特徵，並依特徵之重要性排序
feature_importances = pd.DataFrame(tree.feature_importances_)
feature_importances.index = pre_X_train.columns
feature_importances = feature_importances.sort_values(0,ascending=False)
feature_importances
# 調整前置作業訓練集與驗證集，將特徵依特徵重要性縮減為重要性排名之1/3
pre_X_train = pre_X_train[feature_importances.index[:int(len(feature_importances)/3)]]
pre_X_test = pre_X_test[feature_importances.index[:int(len(feature_importances)/3)]]
# 使用33％的重要特徵重新進行Grid Search調參
params = {"criterion":["gini", "entropy"], "max_depth":range(1,12)}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=params, scoring='roc_auc', cv=100, verbose=1, n_jobs=-1)
grid_search.fit(pre_X_train, pre_y_train)
grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_
# 調整前置作業完整資料集，將特徵依特徵重要性縮減為重要性排名之1/3
pre_X = pre_X[feature_importances.index[:int(len(feature_importances)/3)]]

# 依據Grid Search的結果建立一個決策樹模型，並fit完整資料 (前置資料)
tree = grid_search.best_estimator_
tree.fit(pre_X,pre_y)

# 輸出重要特徵，並依特徵之重要性排序
feature_importances = pd.DataFrame(tree.feature_importances_)
feature_importances.index = pre_X_train.columns
feature_importances = feature_importances.sort_values(0,ascending=False)
feature_importances
# 最終預測模型之特徵，將採用特徵值 .005以上的變數
feature_len = len(feature_importances[feature_importances[feature_importances.columns[0]] > 0.005])

# 調整最終完整Train Set 與 Test set之特徵
y = train[['click']]
X = train[feature_importances[:feature_len].index]
test = test[feature_importances[:feature_len].index]
from xgboost import XGBClassifier

# 使用xgboost 建模，並指定先前調參得到的節點深度限制使用xgboost 建模，並指定先前調參得到的節點深度限制
model = XGBClassifier(tree_method = 'gpu_hist', n_jobs=-1, n_estimators=500, max_depth=11)
model.fit(X,y.values.ravel())
y_pred = model.predict(X)
print("Roc_auc_score: ",roc_auc_score(y,y_pred)*100,"%")

# 繪出混淆矩陣，查看預測結果
confmat = confusion_matrix(y_true=y, y_pred=y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

# 匯出submission並進行提交
submission = pd.read_csv(samplesubmision_file, compression='gzip', index_col='id')
submission[submission.columns[0]] = model.predict_proba(test)[:,1]
submission.to_csv('submission.csv')