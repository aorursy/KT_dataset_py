# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#使用するライブラリのインポート

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns; sns.set(style="ticks", color_codes=True)
#zipファイルを解凍

import zipfile

file_list = [

    '/kaggle/input/restaurant-revenue-prediction/test.csv.zip',

    '/kaggle/input/restaurant-revenue-prediction/train.csv.zip']

for file_name in file_list:

    with zipfile.ZipFile(file=file_name) as target_zip:

        target_zip.extractall()
#学習データとテストデータをDataframeとして読み込む

train = pd.read_csv("/kaggle/working/train.csv")

test =  pd.read_csv("/kaggle/working/test.csv")



train.head()
test.head()
#データ探索()

numerical_features = train.select_dtypes([np.number]).columns.tolist()

categorical_features = train.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()





print('numerical:',numerical_features)



print('categorical:',categorical_features)
# ターゲット列の分布の確認

print(train['revenue'].describe())

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})

sns.distplot(

    train['revenue'], norm_hist=False, kde=True

).set(xlabel='revenue', ylabel='P(revenue)');
#売り上げが10000000以上のデータを落とす

train[train['revenue'] > 10000000 ]

train = train[train['revenue'] < 10000000 ]

train.reset_index(drop=True).head()
# one hot encodingで、カテゴリデータをダミー変数化する



train = pd.get_dummies(train, columns=["City Group","Type"])

test = pd.get_dummies(test, columns=["City Group","Type"])





###参考：以下はカラムの値を数値に置き換える処理である。数値の大小などが影響するため、あまり使われない。###

# train.loc[train['City Group'] == "Big Cities", 'City Group'] = '1'

# train.loc[train['City Group'] == "Other", 'City Group'] = '0'



# train.loc[train['Type'] == "FC", 'Type'] = '0'

# train.loc[train['Type'] == "IL", 'Type'] = '1'

# train.loc[train['Type'] == "DT", 'Type'] = '2'

# train.loc[train['Type'] == "MB", 'Type'] = '3'



# test.loc[test['City Group'] == "Big Cities", 'City Group'] = '1'

# test.loc[test['City Group'] == "Other", 'City Group'] = '0'



# test.loc[test['Type'] == "FC", 'Type'] = '0'

# test.loc[test['Type'] == "IL", 'Type'] = '1'

# test.loc[test['Type'] == "DT", 'Type'] = '2'

# test.loc[test['Type'] == "MB", 'Type'] = '3'
train.head()
test.head()



#Type_MBはtestデータにしか存在しない
# 各値を標準化する



import scipy.stats

train2 = train.copy()

for i in range(36):

    train2['P'+str(i+1)] = scipy.stats.zscore(train2['P'+str(i+1)])



train2.head()

from sklearn.model_selection import train_test_split





#モデル学習に使用しないカラムを削除する

#Type_MBは、今回は使用しないこととして削除する。本来は削除してよいか検討が必要。

train2 = train.drop(['Id','Open Date','City'],axis = 1)

test2 = test.drop(['Id','Open Date','City','Type_MB'],axis = 1)





train_x = train2.drop('revenue',axis = 1)

train_y = train2.revenue



train_x.head()

train_y.head()



(x_train, y_train) = train_test_split(train_x, test_size = 0.3 , random_state = 0)

x_train.head()

y_train.head()

from sklearn.ensemble import RandomForestClassifier





#モデルの構築

rfc1 = RandomForestClassifier(random_state=0)



#学習データにて学習

rfc1.fit(train_x, train_y)



#テストデータを予測

y_pred = rfc1.predict(test2)

print(y_pred)



#ファイルsubmission_RF.csvに出力

submission = pd.DataFrame({"Id":test.Id,"Prediction":y_pred})

submission.to_csv("submission_RF.csv",index=False)
#k最近傍法（KNeighborsRegressor）にて学習



knn=KNeighborsRegressor(n_neighbors=5)

knn.fit(train_x, train_y)

predicted_test_values2 = knn.predict(test2)



#submission_KNN.csvに出力

submission2 = pd.DataFrame(columns=['Id','Prediction'])

submission2['Id'] = test['Id']

submission2['Prediction'] = predicted_test_values2

submission2.to_csv('submission_KNN.csv',index=False)