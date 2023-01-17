import pandas as pd
import numpy as np
pd.set_option('display.max_columns',300)

from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore', UserWarning)
app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()
app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();
def missing_values_table(df):
        # 総欠損値
        mis_val = df.isnull().sum()
        
        # 欠損割合
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # 欠損値とその割合を示すテーブルを作る
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # 表の各列に名前を付ける
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # 欠落割合を降順でソート
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # 概要の追加
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # 欠損
        return mis_val_table_ren_columns
# 欠損地の表
missing_values = missing_values_table(app_train)
missing_values.head(20)
def drop_missing_values(df):

        # 欠損割合
        mis_val_percent =df.isnull().sum() / len(df)
        dfv2=df.loc[:,mis_val_percent < 0.5]
        return dfv2
new_train = drop_missing_values(app_train)
new_test = drop_missing_values(app_test)
new_train
# ラベルエンコーダーオブジェクト
le = LabelEncoder()
le_count = 0

# 全カラムへ適応
for col in new_train:
    if new_train[col].dtype == 'object':
        #カテゴリ数２以下ならやらない
        if len(list(new_train[col].unique())) <= 2:
            # 変更したい部分の選択
            le.fit(new_train[col])
            # 訓練、テストデータの両方に行う
            new_train[col] = le.transform(new_train[col])
            new_test[col] = le.transform(new_test[col])
            
            # ラベルがエンコードされた列の数を数える
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables
new_train = pd.get_dummies(new_train)
new_test = pd.get_dummies(new_test)

print('Training Features shape: ', new_train.shape)
print('Testing Features shape: ', new_test.shape)
for column in new_train:
    new_train[column]=new_train[column].fillna(new_train[column].median())
new_train
ext_data = new_train
ext_data_corrs = ext_data.corr()
ext_data_corrs
def drop_unrelation(df,cor):

        dfv2=df.loc[:'NAME_CONTRACT_TYPE',abs(cor['TARGET']) >0.05]
        return dfv2
cov_f1 = drop_unrelation(new_train,ext_data_corrs)
cov_f1
ext_data_corrsv2 = cov_f1.corr()
ext_data_corrsv2
plt.figure(figsize = (12, 10))

# Heatmap of correlations
sns.heatmap(ext_data_corrsv2, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');