# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler # Used for scaling of data

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import metrics

import seaborn as sns

import matplotlib.pyplot as plt

from keras import backend as K

from keras.wrappers.scikit_learn import KerasRegressor

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
#descriptive statistics summary

train['SalePrice'].describe()
#histogram

#EDA

# 求めたいSalePriceの基本統計量を調べる

(mu, sigma) = stats.norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

print( '\n skew = {:.2f}\n'.format(stats.skew(train['SalePrice'])))



sns.distplot(train['SalePrice']);



#Now plot the distribution

sns.distplot(train['SalePrice'] , fit=stats.norm);

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# QQプロットが曲がっているので正規分布から外れている

# Log変換して正規分布に偏りを合わせる

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=stats.norm);



# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# EDAのの2

# データの素性を調べる

train.info()
# 欠損値を調べる

np.set_printoptions(threshold=np.inf)        # 全件表示設定

pd.set_option('display.max_rows',10000)      # 10000件表示設定



train.isnull().sum()
#数値データ一括log変換

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: stats.skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])





#結局、数値データのみ扱う

np.set_printoptions(threshold=10)        # 全件表示設定

pd.set_option('display.max_rows',10)      # 10件表示設定



#all_data = all_data[numeric_feats] 

all_data.info()
#correlation matrix

corrmat = all_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
# 基本統計量

train.describe()
# カテゴリカルデータを一括ダミー変数に変換(One-hot)

all_data = pd.get_dummies(all_data)



# 欠損値は平均値に変換

all_data = all_data.fillna(all_data.mean())
# trainデータとtestデータに分ける

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train['SalePrice']
scale_x = StandardScaler()

scale_y = StandardScaler()

X_train = scale_x.fit_transform(X_train)

X_test = scale_x.fit_transform(X_test)



y = np.array(y).reshape(len(y),1)

y = scale_y.fit_transform(y)

#seed = 7

#np.random.seed(seed)

# split into 67% for train and 33% for test

#X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.25, random_state=seed)

train_X, test_X, train_y, test_y = train_test_split(X_train, y, test_size=0.25)
from keras.layers.advanced_activations import LeakyReLU

from keras import regularizers

leaky_relu = LeakyReLU()



def create_model():

    # create model

    model = Sequential()

    model.add(Dense(40, input_dim=train_X.shape[1], activation=leaky_relu,kernel_regularizer=regularizers.l1(0.01)))

    model.add(Dropout(0.2))

    model.add(Dense(80, activation=leaky_relu,kernel_regularizer=regularizers.l1(0.01)))

    model.add(Dropout(0.2))

    model.add(Dense(40, activation=leaky_relu,kernel_regularizer=regularizers.l1(0.01)))

    model.add(Dense(1))

    # Compile model

    model.compile(optimizer ='adam', loss = 'mean_squared_error', 

              metrics =[metrics.mae])

    return model
model = create_model()

model.summary()
history = model.fit(train_X, train_y, validation_data=(test_X,test_y), epochs=150, batch_size=32)
# summarize history for accuracy

plt.plot(history.history['mean_absolute_error'])

plt.plot(history.history['val_mean_absolute_error'])

plt.title('model absolute error')

plt.ylabel('error')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.metrics import mean_squared_error,r2_score



def model_Eval(testX,testY):



    #np.expm1で予測結果をLog逆変換すること！！

    predicition = np.expm1(scale_y.inverse_transform(model.predict(testX)))

    true = np.expm1(scale_y.inverse_transform(testY))



    # 予測と結果

    preds = pd.DataFrame({"preds":predicition.flatten(), "true":true.flatten()})

    preds

    

    # 残差プロット

    preds["residuals"] = preds["true"] - preds["preds"]

    preds.plot(x = "preds", y = "residuals",kind = "scatter") 

    

    #モデルの当てはめ

    fig, ax = plt.subplots(figsize=(50, 40))

    plt.style.use('ggplot')

    plt.plot(preds["preds"], preds["true"], 'ro')

    plt.xlabel('Predictions', fontsize = 30)

    plt.ylabel('Reality', fontsize = 30)

    plt.title('Predictions x Reality on dataset Test', fontsize = 30)

    ax.plot([preds["true"].min(), preds["true"].max()], [preds["true"].min(), preds["true"].max()], 'k--', lw=4)

    plt.show()

    

    # モデルのmseとr2を求める

    mse = mean_squared_error(preds["true"], preds["preds"])

    print('mse:',mse)

    r2 = r2_score(preds["true"], preds["preds"])

    print('r2:',r2)
# 訓練データの確認

model_Eval(train_X,train_y)
# 過学習の確認

# バリデーションデータで確認

model_Eval(test_X,test_y)
# 予測

predicition_result = np.expm1(scale_y.inverse_transform(model.predict(X_test)))
pd.options.display.float_format = '{:.2f}'.format

pred_df = pd.DataFrame(predicition_result, index=test["Id"], columns=["SalePrice"])

pred_df
pred_df.to_csv('output.csv', header=True, index_label='Id')