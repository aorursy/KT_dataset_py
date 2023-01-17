# Gerekli kütüphaneler geliştirme ortamına dahil ediliyor



#Veri işleme kütüphanleri

import numpy as np 

import pandas as pd 



#Görselleştirme kütüphaneleri

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns



#istatistik kütüphaneleri

from scipy import stats

from scipy.stats import norm, skew 



#Makine öğrenmesi kütüphaneleri

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold,cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Eğitim ve test veri setleri yükleniyor

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



#Gönderi hazırlanırken lazım olacağı için veri setlerindeki örneklerin

# id numaralarını tutuyoruz

train_ids=train['Id']

test_ids=test['Id']



# Makine öğrenemsi modelleri için id numarasının anlamı olmadığı için

# veri setinden çıkartıyoruz

train.drop('Id',axis=1, inplace=True)

test.drop('Id',axis=1, inplace=True)
train.columns.values
print("train shape:",train.shape)

print("test shape:",test.shape)
train.info()
#Veri setindeki özelliklerin eksik değerlerini sayısal ve

#görsel olarak verir.

def show_missing_values(function_data):

#Veri setindeki eksik değerleri bulalım

    number_of_sample=function_data.shape[0]

    check_isnull=function_data.isnull().sum()

    

    check_isnull=check_isnull[check_isnull!=0].sort_values(ascending=False)



    if check_isnull.shape[0]==0:

        print("Veri setinde eksik bilgi yoktur")

        print(check_isnull)

    else:

        print(check_isnull)

        f, ax = plt.subplots(figsize=(15, 6))

        plt.xticks(rotation='90')

        sns.barplot(x=check_isnull.index, y=check_isnull)

        plt.title("Eksik veri içeren özellilere ait eksik veri sayısı")
#train veri seti için eksik bilgiler gösterilsin

show_missing_values(train)
#veri setindeki özelliklerin birbirleriyle olan korelasyonunu elde edilir.

corr=train.corr().abs()

n_most_correlated=12

#'SalePrice' ile en yüksek korelasyona sahip özellikler elde edilir.

most_correlated_feature=corr['SalePrice'].sort_values(ascending=False)[:n_most_correlated].drop('SalePrice')

#En yüksek korelasyona sahip özelliklerin adları elde edilr. 

most_correlated_feature_name=most_correlated_feature.index.values
#En yüksek korelasyona sahip özellikler barplot ile gösteririlir

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=most_correlated_feature_name, y=most_correlated_feature)

plt.title("SalePrice ile en fazla korelasyona sahip özellikler")
def draw_scatter_pairs(data,cols=4, rows=3):

    feature_names=data.columns.values



    counter=0

    fig, axarr = plt.subplots(rows,cols,figsize=(22,16))

    for i in range(rows):

        for j in range(cols):

            if counter>=len(feature_names):

                break



            name=feature_names[counter]

            axarr[i][j].scatter(x = data[name], y = data['SalePrice'])

            axarr[i][j].set(xlabel=name, ylabel='SalePrice')



            counter+=1





    plt.show()
#'SalePrice' ile en yüksek korelasyona sahip özelliklerin 

#grafikle gösterimi

feature_names =list(most_correlated_feature_name) + ['SalePrice']

draw_scatter_pairs(train[feature_names], rows=4, cols=3)
print("Aykırı değerler çıkarılmadan önce train.shape:",train.shape)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
print("Aykırı değerler çıkarıldıktan sonra train.shape:",train.shape)
draw_scatter_pairs(train[feature_names], rows=4, cols=3)
ntrain = train.shape[0]

ntest = test.shape[0]
y_train=train['SalePrice']

X_train=train.drop('SalePrice', axis='columns')



#İki veri seti satırlar üst üste gelecek biçimde birleştiriliyor

datasets=pd.concat((X_train, test),axis='index')



print(datasets.shape)
show_missing_values(datasets)
#Eksik değerlerin doldurulması için stratejiler belirleniyor

staretegies={}

staretegies['PoolQC']='None'

staretegies['MiscFeature']='None'

staretegies['Alley']='None'

staretegies['Fence']='None'

staretegies['FireplaceQu']='None'



#özel doldurma işleme gerektiren özelliklere kendi adını atıyoruz

staretegies['LotFrontage']='LotFrontage'



staretegies['GarageType']='None'

staretegies['GarageFinish']='None'

staretegies['GarageQual']='None'

staretegies['GarageCond']='None'



staretegies['GarageYrBlt']='Zero'

staretegies['GarageArea']='Zero'

staretegies['GarageCars']='Zero'



staretegies['BsmtFinSF1']='Zero'

staretegies['BsmtFinSF2']='Zero'

staretegies['BsmtUnfSF']='Zero'

staretegies['TotalBsmtSF']='Zero'

staretegies['BsmtFullBath']='Zero'

staretegies['BsmtHalfBath']='Zero'



staretegies['BsmtQual']='None'

staretegies['BsmtCond']='None'

staretegies['BsmtExposure']='None'

staretegies['BsmtFinType1']='None'

staretegies['BsmtFinType2']='None'



staretegies['MasVnrType']='None'

staretegies['MasVnrArea']='Zero'



staretegies['MSZoning']='Mode'



staretegies['Utilities']='Drop'



#özel doldurma işleme gerektiren özelliklere kendi adını atıyoruz

staretegies['Functional']='Functional'



staretegies['Electrical']='Mode'

staretegies['KitchenQual']='Mode'

staretegies['Exterior1st']='Mode'

staretegies['Exterior2nd']='Mode'

staretegies['SaleType']='Mode'



staretegies['MSSubClass']='None'

def fill_missing_values(fill_data, mystaretegies):

    

    for column, strategy in mystaretegies.items():

        if strategy=='None':

            fill_data[column]=fill_data[column].fillna('None')

        elif strategy=='Zero':

            fill_data[column]=fill_data[column].fillna(0)

        elif strategy=='Mode':

            fill_data[column]=fill_data[column].fillna(fill_data[column].mode()[0])   

        elif strategy=='LotFrontage':

            #temp=fill_data.groupby("Neighborhood")

            fill_data[column]=fill_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

        elif strategy=='Drop':

            fill_data=fill_data.drop([column],axis=1)

        elif strategy=='Functional':

            fill_data[column]=fill_data[column].fillna('Typ')

    

    return fill_data
datasets_no_missing=fill_missing_values(datasets, staretegies)
#Eksik veri kalmadığından emin olalım

show_missing_values(datasets_no_missing)

print(datasets_no_missing.shape)
#String tipinde değer içeren ancak numerik değerlere sahip sütünları

# str yani object tipine dönüştürüyoruz

for name in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:

    datasets_no_missing[name]= datasets_no_missing[name].astype(str)
col_names=datasets_no_missing.columns.values

col_types=datasets_no_missing.dtypes

object_cols=[]

numeric_cols=[]

for col_name, col_type in zip(col_names, col_types):

    if col_type=='object':

        object_cols.append(col_name)

    else:

        numeric_cols.append(col_name)

print("String değerler içeren özellikler:")

print(object_cols)

print("\nSayıal değerler içeren özellikler:")

print(numeric_cols)


label_encoder_col_names = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

#for ocn in object_cols:

    #if ocn not in label_encoder_cols:

        #print("{}: {}".format(ocn,len(datasets_no_missing[ocn].unique())))
for col_name in label_encoder_col_names:

    labelEncoder=LabelEncoder()

    labelEncoder.fit(datasets_no_missing[col_name].values)

    datasets_no_missing[col_name]=labelEncoder.transform(datasets_no_missing[col_name].values)
print(datasets_no_missing.shape)
#Toplam sofa alanı için yeni bir özellik ekleyelim

datasets_no_missing['TotalSF'] = datasets_no_missing['TotalBsmtSF'] + datasets_no_missing['1stFlrSF'] + datasets_no_missing['2ndFlrSF']
numeric_feats = datasets_no_missing.dtypes[datasets_no_missing.dtypes != "object"].index



# Özelliklerin çarpıklıkları belirleniyor

skewed_feats = datasets_no_missing[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nNormal dağılımdan uzak değerler: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("Box Cox transform uygulanan özelliklerin sayısı:{}".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    datasets_no_missing[feat] = boxcox1p(datasets_no_missing[feat], lam)
datasets_no_missing_dummies=pd.get_dummies(datasets_no_missing)

print(datasets_no_missing_dummies.shape)
from yellowbrick.features import Rank1D



X_yellow=datasets_no_missing_dummies[:ntrain]

nfeature_name=datasets_no_missing_dummies.columns.values[:-1]

rank1D=Rank1D(features=nfeature_name, algorithm="shapiro")

rank1D.fit(X_yellow[nfeature_name], y_train)

rank1D.transform(X_yellow[nfeature_name])

 

#rank1D.poof()

print("önem hesaplandı")


df=pd.DataFrame()

df['feature_name']=nfeature_name

df['ranks']=rank1D.ranks_





df.sort_values(by=['ranks'],ascending=False, inplace=True)

df.set_index('feature_name', inplace=True)

df.head()



fig, ax=plt.subplots(1, figsize=(12,20))

df[:30].plot.barh(ax=ax)
print(datasets_no_missing_dummies.shape)
n=30

#En önemli özellikler alınıyor

n_most_important=df.index.values[:n]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

#en önemli özellikler ölçeklendirilmesi için eğitiliyot

scaler.fit(datasets_no_missing_dummies[n_most_important])



#En önemli özellikler ölçeklendiriliyor

datasets_no_missing_dummies_scaled=scaler.transform(datasets_no_missing_dummies[n_most_important])
#Eğitim veri seti

preprocessed_train = datasets_no_missing_dummies_scaled[:ntrain]



#Test veri seti

preprocessed_test = datasets_no_missing_dummies_scaled[ntrain:]



#preprocessed_train.head(10)
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(preprocessed_train)

    rmse= np.sqrt(-cross_val_score(model, preprocessed_train, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)

print("LGMRboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("XGBRboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(preprocessed_train, y_train)

xgb_train_pred = model_xgb.predict(preprocessed_train)

xgb_pred = model_xgb.predict(preprocessed_test)

print(rmsle(y_train, xgb_train_pred))
print("Gönderi hazırlanıyor")

submision = pd.DataFrame()

submision['Id'] = test_ids

submision['SalePrice'] = xgb_pred

submision.to_csv('n_most_xgb_submission.csv',index=False)

print("Gönderi kaydedildi")

submision.head(20)

print(xgb_train_pred[:20])

print(y_train.values[:20])

print(submision.values[:20])