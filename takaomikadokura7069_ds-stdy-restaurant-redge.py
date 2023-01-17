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
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd



train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')



# Idは不要なので、削除して別に変数化し、スコア提出時に使用

train_Id = train.Id

test_Id = test.Id



# Id列削除

train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# import seaborn as sns

import seaborn as sns; sns.set(style="ticks", color_codes=True)



from datetime import datetime

from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import lightgbm as lgb



# 最大カラム数を100に拡張(デフォルトだと省略されてしまうので)

# 常に全ての列（カラム）を表示

pd.options.display.max_columns = None

pd.options.display.max_rows = 80



# 小数点2桁で表示(指数表記しないように)

pd.options.display.float_format = '{:.2f}'.format

%matplotlib inline

#ワーニングを抑止

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
print('Size of train data', train.shape)

print('Size of test data', test.shape)
train.shape
train.info()
train.describe()
train.describe(include='O')
# import pandas_profiling as pdp

# pdp.ProfileReport(train)
train["revenue"].describe()
#目的変数であるrevenueのヒストグラムとQ-Qプロットを表示する

# 分布確認

fig = plt.figure(figsize=(10, 4))

plt.subplots_adjust(wspace=0.4)



# ヒストグラム

ax = fig.add_subplot(1, 2, 1)

sns.distplot(train['revenue'], ax=ax)



# QQプロット

ax2 = fig.add_subplot(1, 2, 2)

stats.probplot(train['revenue'], plot=ax2)



plt.show()



# 変換後の要約統計量表示

print(train['revenue'].describe())

print("------------------------------")

print("歪度: %f" % train['revenue'].skew())

print("尖度: %f" % train['revenue'].kurt())
# 学習データをコピーし、新たなdataframeで検証

df = train.copy()



#目的変数の対数log(x+1)をとる

df['revenue'] = np.log1p(df['revenue'])



# 標準化(平均0, 分散1)

scaler=StandardScaler()

df['revenue']=scaler.fit_transform(df[['revenue']])



# 分布確認

fig = plt.figure(figsize=(10, 4))

plt.subplots_adjust(wspace=0.4)



# ヒストグラム

ax = fig.add_subplot(1, 2, 1)

sns.distplot(df['revenue'], ax=ax)



# QQプロット

ax2 = fig.add_subplot(1, 2, 2)

stats.probplot(df['revenue'], plot=ax2)



plt.show()



# 変換後の要約統計量表示

print(df['revenue'].describe())

print("------------------------------")

print("歪度: %f" % df['revenue'].skew())

print("尖度: %f" % df['revenue'].kurt())
# 学習データをコピーし、新たなdataframeで検証

df = train.copy()



# 標準化(平均0, 分散1)

scaler=StandardScaler()

df['revenue']=scaler.fit_transform(df[['revenue']])





# 分布確認

fig = plt.figure(figsize=(10, 4))

plt.subplots_adjust(wspace=0.4)



# ヒストグラム

ax = fig.add_subplot(1, 2, 1)

sns.distplot(df['revenue'], ax=ax)



# QQプロット

ax2 = fig.add_subplot(1, 2, 2)

stats.probplot(df['revenue'], plot=ax2)



plt.show()



# 変換後の要約統計量表示

print(df['revenue'].describe())

print("------------------------------")

print("歪度: %f" % df['revenue'].skew())

print("尖度: %f" % df['revenue'].kurt())
# 学習データをコピーし、新たなdataframeで検証

df = train.copy()



# Min-Max変換(正規化(最大1, 最小0))

scaler=MinMaxScaler()

df['revenue']=scaler.fit_transform(df[['revenue']])



# 分布確認

fig = plt.figure(figsize=(10, 4))

plt.subplots_adjust(wspace=0.4)



# ヒストグラム

ax = fig.add_subplot(1, 2, 1)

sns.distplot(df['revenue'], ax=ax)



# QQプロット

ax2 = fig.add_subplot(1, 2, 2)

stats.probplot(df['revenue'], plot=ax2)



plt.show()



# 変換後の要約統計量表示

print(df['revenue'].describe())

print("------------------------------")

print("歪度: %f" % df['revenue'].skew())

print("尖度: %f" % df['revenue'].kurt())
# 学習データ

# Open Dateを日付型に変換

train['pd_date'] = pd.to_datetime(train['Open Date'], format='%m/%d/%Y')

# 年のみを抽出

train['Open_Year'] = train['pd_date'].dt.strftime('%Y')

# 月のみを抽出

train['Open_Month'] = train['pd_date'].dt.strftime('%m')



train = train.drop('pd_date',axis=1)

train = train.drop('Open Date',axis=1)
# テストデータ

# Open Dateを日付型に変換

test['pd_date'] = pd.to_datetime(test['Open Date'], format='%m/%d/%Y')

# 年のみを抽出

test['Open_Year'] = test['pd_date'].dt.strftime('%Y')

# 月のみを抽出

test['Open_Month'] = test['pd_date'].dt.strftime('%m')



test = test.drop('pd_date',axis=1)

test = test.drop('Open Date',axis=1)
train.dtypes.value_counts()
#カテゴリ変数と数値変数に分ける

cats = list(train.select_dtypes(include=['object']).columns)

nums = list(train.select_dtypes(exclude=['object']).columns)

print(f'categorical variables:  {cats}')

print(f'numerical variables:  {nums}')
train.nunique(axis=0)
# 値の追加

# cats.extend([''])



# 値の削除

# nums.remove('')



print(f'categorical variables:  {cats}')

print(f'numerical variables:  {nums}')
# 名義変数

nominal_list =cats

               

# 順序変数

# ordinal_list = []



# 数値変数

num_list = nums
columns = len(nominal_list)/2+1



fig = plt.figure(figsize=(30, 20))

plt.subplots_adjust(hspace=0.6, wspace=0.4)



for i in range(len(nominal_list)):

    ax = fig.add_subplot(columns, 2, i+1)

    sns.countplot(x=nominal_list[i], data=train, ax=ax)

    plt.xticks(rotation=45)

plt.show()
columns = len(num_list)/3+1



fig = plt.figure(figsize=(30, 40))

plt.subplots_adjust(hspace=0.6, wspace=0.4)



for i in range(len(num_list)):

    ax = fig.add_subplot(columns, 3, i+1)



    train[num_list[i]].hist(ax=ax)

    ax2 = train[num_list[i]].plot.kde(ax=ax, secondary_y=True,title=num_list[i])

    ax2.set_ylim(0)

    

plt.show()
columns = len(nominal_list)/2+1



fig = plt.figure(figsize=(20, 10))

plt.subplots_adjust(hspace=0.6, wspace=0.4)



for i in range(len(nominal_list)):

    ax = fig.add_subplot(columns, 2, i+1)



    # 回帰の場合    

    sns.boxplot(x=nominal_list[i], y=train.revenue, data=train, ax=ax)

    plt.xticks(rotation=45)

    # 分類の場合

#     sns.barplot(x = nominal_list[i], y = train.revenue, data=train, ax=ax)

plt.show()

train = train.drop('Open_Month',axis=1)

test= test.drop('Open_Month',axis=1)

nominal_list.remove('Open_Month')
columns = len(num_list)/4+1



fig = plt.figure(figsize=(30, 35))

plt.subplots_adjust(hspace=0.6, wspace=0.4)



for i in range(len(num_list)):

    ax = fig.add_subplot(columns, 4, i+1)



    # 回帰の場合    

    sns.regplot(x=num_list[i],y='revenue',data=train, ax=ax)

    plt.xticks(rotation=45)

    # 分類の場合

#     sns.barplot(x = nominal_list[i], y = train.revenue, data=train, ax=ax)

plt.show()

train[['City','revenue']].groupby('City').mean().plot(kind='bar')

plt.title('Mean Revenue Generated vs City')

plt.xlabel('City')

plt.ylabel('Mean Revenue Generated')
# Cityごとのrevenue平均値を1000000単位とする

mean_revenue_per_city = train[['City', 'revenue']].groupby('City', as_index=False).mean()

mean_revenue_per_city.head()

mean_revenue_per_city['revenue'] = mean_revenue_per_city['revenue'].apply(lambda x: int(x/1e6)) 



mean_revenue_per_city



mean_dict = dict(zip(mean_revenue_per_city.City, mean_revenue_per_city.revenue))

mean_dict
# city_rev = []



# for i in train['City']:

#     for key, value in mean_dict.items():

#         if i == key:

#             city_rev.append(value)

            

# df_city_rev = pd.DataFrame({'city_rev':city_rev})

# train = pd.concat([train,df_city_rev],axis=1)

# train.head()
# train.replace({"City":mean_dict}, inplace=True)

# test.replace({"City":mean_dict}, inplace=True)

# test['City'] = test['City'].apply(lambda x: 6 if isinstance(x,str) else x)



# train['City_rev'] = train['City']

# test['City_rev'] = test['City']


print(train['City'].sort_values().unique())
test['City'].sort_values().unique()

# Cityについて、学習データとテストデータにて重複削除し、リスト化

city_train_list = list(train['City'].unique())

city_test_list = list(test['City'].unique())
l1_l2_and = set(city_train_list) & set(city_test_list)

print(l1_l2_and)

print(len(l1_l2_and))
# どちらかにしかないCityを抽出

l1_l2_sym_diff = set(city_test_list) ^ set(city_train_list)

print(l1_l2_sym_diff)

print(len(l1_l2_sym_diff))
# テストデータのみ存在するCityの件数

len(set(city_test_list).difference(city_train_list))

# 学習データのみ存在するCityの件数

len(set(city_train_list).difference(city_test_list))
# P変数の1つのクラスは地理的属性であると指定されているため

# 各都市のP変数の平均をプロットすると、どのP変数が都市と関連性が高いかが分かる

distinct_cities = train.loc[:, "City"].unique()



# P変数のcityごとの平均値を取得

means = []

for i in range(len(num_list)):

    temp = []

    for city in distinct_cities:

        temp.append(train.loc[train.City == city, num_list[i]].mean())  

    means.append(temp)

    

city_pvars = pd.DataFrame(columns=["city_var", "means"])

for i in range(37):

    for j in range(len(distinct_cities)):

        city_pvars.loc[i+37*j] = ["P"+str(i+1), means[i][j]]



print(city_pvars)            

# 箱ひげ図を表示

plt.rcParams['figure.figsize'] = (18.0, 6.0)

sns.boxplot(x="city_var", y="means", data=city_pvars)



# From this we observe that P1, P2, P11, P19, P20, P23, and P30 are approximately a good

# proxy for geographical location.
from sklearn import cluster



def adjust_cities(full_full_data, train, k):

    

    # As found by box plot of each city's mean over each p-var

    relevant_pvars =  ["P1", "P2", "P11", "P19", "P20", "P23","P30"]

    train = train.loc[:, relevant_pvars]

    

    # Optimal k is 20 as found by DB-Index plot    

    kmeans = cluster.KMeans(n_clusters=k)

    kmeans.fit(train)

    

    # Get the cluster centers and classify city of each full_data instance to one of the centers

    full_data['City_Cluster'] = kmeans.predict(full_data.loc[:, relevant_pvars])

    

    return full_data
num_train = train.shape[0]

num_test = test.shape[0]

print(num_train, num_test)



full_data = pd.concat([train, test], ignore_index=True)                
# 学習データを使用しクラスタリングを行い、その学習結果を全データに適用させる

full_data = adjust_cities(full_data, train, 20)

full_data



# City項目は不要なので削除

full_data = full_data.drop(['City'], axis=1)
# Split into train and test datasets

train = full_data[:num_train]

test = full_data[num_train:]

# check the shapes 

print("Train :",train.shape)

print("Test:",test.shape)

test
train[['City_Cluster','revenue']].groupby('City_Cluster').mean().plot(kind='bar')

plt.title('Mean Revenue Generated vs City Cluster')

plt.xlabel('City Cluster')

plt.ylabel('Mean Revenue Generated')
mean_revenue_per_city = train[['City_Cluster', 'revenue']].groupby('City_Cluster', as_index=False).mean()

mean_revenue_per_city.head()

mean_revenue_per_city['revenue'] = mean_revenue_per_city['revenue'].apply(lambda x: int(x/1e6)) 



mean_revenue_per_city



mean_dict = dict(zip(mean_revenue_per_city.City_Cluster, mean_revenue_per_city.revenue))

mean_dict
city_rev = []



for i in full_data['City_Cluster']:

    for key, value in mean_dict.items():

        if i == key:

            city_rev.append(value)

            

df_city_rev = pd.DataFrame({'city_rev':city_rev})

full_data = pd.concat([full_data,df_city_rev],axis=1)

full_data.head



# 値の追加

nominal_list.extend(['City_Cluster'])

# 値の削除

nominal_list.remove('City')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le_count = 0



# Iterate through the columns

# for col in application_full_data:

for i in range(len(nominal_list)):    

    

#     if application_full_data[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(full_data[nominal_list[i]].unique())) <= 2:

            # full_data on the full_dataing data

            le.fit(full_data[nominal_list[i]])

            # Transform both full_dataing and testing data

            full_data[nominal_list[i]] = le.transform(full_data[nominal_list[i]])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables

full_data = pd.get_dummies(full_data)

print('full_dataing Features shape: ', full_data.shape)
def tukey_outliers(x):

    q1 = np.percentile(x,25)

    q3 = np.percentile(x,75)

    

    iqr = q3-q1

    

    min_range = q1 - iqr*1.5

    max_range = q3 + iqr*1.5

    

    outliers = x[(x<min_range) | (x>max_range)]

    return outliers
# 外れ値の詳細レコードを表示

# for col in num_list:

#     outliers = tukey_outliers(train[col])

#     if len(outliers):

#         print(f"* {col} has these tukey outliers,\n{outliers}\n")

#     else:

#         print(f"* {col} doesn't have any tukey outliers.\n")
# train.iloc[list(tukey_outliers(df_num.acceleration).index)]
columns = len(num_list)/4+1



# boxplot

fig = plt.figure(figsize=(15,20))

plt.subplots_adjust(hspace=0.2, wspace=0.8)

for i in range(len(num_list)):

    ax = fig.add_subplot(columns, 4, i+1)

    sns.boxplot(y=full_data[num_list[i]], data=full_data, ax=ax)

plt.show()
# 学習データを置き換え

# for i in range(len(num_list)):

#      # 置き換え値

#     upper_lim = full_data[num_list[i]].quantile(.95)

#     lower_lim = full_data[num_list[i]].quantile(.05)

    

#     # IQR

#     Q1 = full_data[num_list[i]].quantile(.25)

#     Q3 = full_data[num_list[i]].quantile(.75)

#     IQR = Q3 - Q1

#     outlier_step = 1.5 * IQR

    

#     # 1.5IQR超える数値は95%tile値で埋める、下回る数値は5%tile値で埋める

#     full_data.loc[(full_data[num_list[i]] > (Q3 + outlier_step)), num_list[i]] =upper_lim

#     full_data.loc[(full_data[num_list[i]] < (Q1 - outlier_step)), num_list[i]] = lower_lim
# columns = len(num_list)/4+1



# # boxplot

# fig = plt.figure(figsize=(15,20))

# plt.subplots_adjust(hspace=0.2, wspace=0.8)

# for i in range(len(num_list)):

#     ax = fig.add_subplot(columns, 4, i+1)

#     sns.boxplot(y=full_data[num_list[i]], data=full_data, ax=ax)

# plt.show()
skewed_data = train[num_list].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_data[:10]
# skew_col = skewed_data[skewed_data > 10].index



# # 可視化

# fig = plt.figure(figsize=(10, 8))

# for i in range(len(skew_col)):

#     ax = fig.add_subplot(2, 3, i+1)

#     try:

#         sns.distplot(combined_df[skew_col[i]], fit=norm, ax=ax)

#     except:

#         # kde計算できない時は、kde=False

#         sns.distplot(combined_df[skew_col[i]], fit=norm, kde=False, ax=ax)

# plt.show()



# # 対数変換

# for i in range(len(skew_col)):

#     combined_df[skew_col[i]] = np.log1p(combined_df[skew_col[i]])

    

#     # 可視化

# # 可視化

# fig = plt.figure(figsize=(10, 8))

# for i in range(len(skew_col)):

#     ax = fig.add_subplot(2, 3, i+1)

#     try:

#         sns.distplot(combined_df[skew_col[i]], fit=norm, ax=ax)

#     except:

#         # kde計算できない時は、kde=False

#         sns.distplot(combined_df[skew_col[i]], fit=norm, kde=False, ax=ax)

# plt.show()
# Split into train and test datasets

train = full_data[:num_train]

test = full_data[num_train:]

# check the shapes 

print("Train :",train.shape)

print("Test:",test.shape)
sns.set(font_scale=1.1)

correlation_train = train.corr()

mask = np.triu(correlation_train.corr())

fig = plt.figure(figsize=(50,50))

sns.heatmap(correlation_train,

            annot=True,

            fmt='.1f',

            cmap='coolwarm',

            square=True,

#             mask=mask,

            linewidths=1)



plt.show()
# Find correlations with the target and sort

correlations = train.corr()['revenue'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
# 相関が高い10項目のみ抽出

correlations = train.corr()

# 絶対値で取得

correlations = abs(correlations)



cols = correlations.nlargest(10,'revenue')['revenue'].index

cols
# 相関が高い10項目のみ抽出

train = train[cols]



#学習データを目的変数とそれ以外に分ける

train_X = train.drop("revenue",axis=1)

train_y = train["revenue"]



#revenueを対数変換する 

train_y = np.log1p(train_y)



#テストデータを学習データのカラムのみにする 

tmp_cols = train_X.columns

test_X = test[tmp_cols]



#それぞれのデータのサイズを確認

print("train_X: "+str(train_X.shape))

print("train_y: "+str(train_y.shape))

print("test_X: "+str(test_X.shape))
#訓練データとモデル評価用データに分けるライブラリ

from sklearn.model_selection import train_test_split



#フォールドアウト法により、学習データとテストデータに分割 

(X_train, X_test, y_train, y_test) = train_test_split(train_X, train_y , test_size = 0.3 , random_state = 0)



print("X_train: "+str(X_train.shape))

print("X_test: "+str(X_test.shape))

print("y_train: "+str(y_train.shape))

print("y_test: "+str(y_test.shape))
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
#機械学習モデルをリストに格納

random_state = 2

classifiers = []

classifiers.append(Lasso(random_state=random_state))

classifiers.append(LinearRegression())

classifiers.append(Ridge(random_state=random_state))

classifiers.append(ElasticNet(random_state=random_state))

classifiers.append(KNeighborsRegressor())

classifiers.append(SVR())

classifiers.append(RandomForestRegressor(random_state=random_state))

classifiers.append(GradientBoostingRegressor())

classifiers.append(AdaBoostRegressor(random_state = random_state))

classifiers.append(DecisionTreeRegressor())

classifiers.append(XGBRegressor())
#複数のclassifier の適用

cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y_train, scoring='neg_mean_squared_error', cv =10, n_jobs=4))



#適用したclassifierのスコアを取得    

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["Lasso","LinearRegression","Ridge",

"ElasticNet","KNeighborsRegressor","SVR","RandomForestRegressor","GradientBoostingRegressor","AdaBoostRegressor","DecisionTreeRegressor", "XGBRegressor"]})
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res.sort_values(ascending=False, by='CrossValMeans')
from sklearn import datasets

from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import optuna

 

def objective(trial):

    params = {

        'alpha': trial.suggest_loguniform("alpha", 0.1, 5), 

        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),

        'normalize': trial.suggest_categorical('normalize', [True, False]),

    }

 

    reg = Ridge(**params)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

 

    mae = mean_absolute_error(y_test, y_pred)

    return mae

 
# optuna によるハイパーパラメータ最適化

study = optuna.create_study()

study.optimize(objective, n_trials=100)



# 結果を表示

print(f'best score: {study.best_value:.4f}, best params: {study.best_params}')
params = {'alpha': 1.9510706324753746, 'fit_intercept': True, 'normalize': True}



reg = Ridge(**params)

reg.fit(X_train, y_train)

prediction_log = reg.predict(test_X)

prediction =np.exp(prediction_log) 

print(prediction)
# 予測した値を提出用CSVファイル(submissionファイル)に書き出し

submission = pd.DataFrame({"Id":test_Id, "Prediction":prediction})

submission.to_csv("submission.csv", index=False)
# #LightGBMライブラリ

# import lightgbm as lgb

# #ハイパーパラメータチューニング自動化ライブラリ

# import optuna



# lgb_train = lgb.Dataset(X_train, y_train)

# lgb_eval = lgb.Dataset(X_test, y_test)
# def objective(trial):

#     params = {'metric': {'rmse'},

#               'max_depth' : trial.suggest_int('max_depth', 1, 10),

#               'subsumple' : trial.suggest_uniform('subsumple', 0.0, 1.0),

#               'subsample_freq' : trial.suggest_int('subsample_freq', 0, 1),

#               'leaning_rate' : trial.suggest_loguniform('leaning_rate', 1e-5, 1),

#               'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.0, 1.0),

#               'lambda_l1' : trial.suggest_uniform('lambda_l1' , 0.0, 1.0),

#               'lambda_l2' : trial.suggest_uniform('lambda_l2' , 0.0, 1.0)}

 

#     gbm = lgb.train(params,

#                     lgb_train,

#                     valid_sets=(lgb_train, lgb_eval),

#                     num_boost_round=10000,

#                     early_stopping_rounds=100,

#                     verbose_eval=50)

#     predicted = gbm.predict(X_test)

#     RMSE = np.sqrt(mean_squared_error(y_test, predicted))

    

#     pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'rmse')

#     return RMSE
# study = optuna.create_study()

# study.optimize(objective, timeout=360)
# print('Best trial:')

# trial = study.best_trial

# print('Value:{}'.format(trial.value))

# print('Params:')

# for key, value in trial.params.items():

#     print('"{}" : {}'.format(key, value))


# #Optunaで最適化されたパラメータ

# params = {"metric": {'rmse'},

#           "max_depth" : 7,

#           "subsumple" : 0.0527053286950852,

#           "subsample_freq" : 0,

#           "leaning_rate" : 0.00012337315517641352,

#           "feature_fraction" : 0.27094712699951107,

#           "lambda_l1" : 0.4567708349707908,

#           "lambda_l2" :6.452511288039886e-07

#          }

 

# #LightGBMのモデル構築

# gbm = lgb.train(params,

#                 lgb_train,

#                 valid_sets=(lgb_train, lgb_eval),

#                 num_boost_round=10000,

#                 early_stopping_rounds=100,

#                 verbose_eval=50)
# #特徴量の重要度

# lgb.plot_importance(gbm, height=0.5, figsize=(8,16))
# テストデータにて予測

# prediction_log = gbm.predict(test_X)

# print(prediction_log)

# prediction =np.exp(prediction_log) 

# print(prediction)