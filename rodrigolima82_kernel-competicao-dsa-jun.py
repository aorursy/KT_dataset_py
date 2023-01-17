# Importando bibliotecas que serao utilizadas neste projeto

import pandas as pd

import numpy as np

import seaborn as sns

import itertools

import pickle

import matplotlib.pyplot as plt

%matplotlib inline



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance





# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn import preprocessing

from sklearn import utils



# Keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

pd.set_option('display.max_columns', None)



import gc 

import pickle

import datetime

import os

#print(os.listdir("data"))

print(os.listdir("../input"))
print(datetime.datetime.now().time())
#'../input/datadsa/transacoes_historicas/transacoes_historicas.csv'

# limitei em 10.000.000 para poder simular aqui no Kernel (no proprio notebook processei completo)

hist = pd.read_csv('../input/datadsa/transacoes_historicas/transacoes_historicas.csv', nrows=10000000

                            ,parse_dates=['purchase_date']

                            ,dtype = {

                                'city_id': np.int16

                                ,'installments': np.int16

                                ,'merchant_category_id': np.int16

                                ,'month_lag': np.int8

                                ,'purchase_amount': np.float32

                                ,'state_id': np.int8

                                ,'subsector_id': np.int8

                            }) 



#'../input/competicao-dsa-machine-learning-jun-2019/novas_transacoes_comerciantes.csv'

novas = pd.read_csv('../input/datadsa/novas_transacoes_comerciantes/novas_transacoes_comerciantes.csv'

                            ,parse_dates=['purchase_date']

                            ,dtype = {

                                'city_id': np.int16

                                ,'installments': np.int16

                                ,'merchant_category_id': np.int16

                                ,'month_lag': np.int8

                                ,'purchase_amount': np.float32

                                ,'state_id': np.int8

                                ,'subsector_id': np.int8

                            })   



#'../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv'

train = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv'

                       ,parse_dates=['first_active_month']

                       ,dtype = {

                                'feature_1': np.int8

                                ,'feature_2': np.int8

                                ,'feature_3': np.int8

                            })



#'../input/competicao-dsa-machine-learning-jun-2019/comerciantes.csv'

com = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/comerciantes.csv')



#'../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv'

test = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv'

                        ,parse_dates=['first_active_month']

                        ,dtype = {

                                'feature_1': np.int8

                               ,'feature_2': np.int8

                               ,'feature_3': np.int8

                            })
# Criar um index para o dataframe de treino

train = train.reset_index()

test = test.reset_index()
tmp = pd.concat( [hist, novas],axis=0,ignore_index=True)
novas.shape, hist.shape, tmp.shape
# Uniao dos dataset de treino com transacoes novas e historicas

df = pd.merge(train, tmp, on='card_id', how='left')

dfTest = pd.merge(test, tmp, on='card_id', how='left')



del train, hist, tmp, novas

gc.collect()



# Uniao dos dataset de treino e teste com comerciantes

df = pd.merge(df, com, on='merchant_id', how='left')

dfTest = pd.merge(dfTest, com, on='merchant_id', how='left')



del com

gc.collect()


df.drop(columns = ["merchant_category_id_y",

                   "merchant_category_id_x", 

                   "subsector_id_y", 

                   "subsector_id_x", 

                   "city_id_y", 

                   "city_id_x", 

                   "state_id_y", 

                   "state_id_x", 

                   "category_1_y", 

                   "category_1_x", 

                   "category_2_y",

                   "category_2_x",

                   "category_3", 

                   "category_4", 

                   "authorized_flag", 

                   "installments", 

                   "merchant_id", 

                   "merchant_group_id", 

                   "numerical_1", 

                   "numerical_2", 

                   "most_recent_sales_range", 

                   "most_recent_purchases_range"

                  ], inplace = True) 



dfTest.drop(columns = ["merchant_category_id_y",

                   "merchant_category_id_x", 

                   "subsector_id_y", 

                   "subsector_id_x", 

                   "city_id_y", 

                   "city_id_x", 

                   "state_id_y", 

                   "state_id_x", 

                   "category_1_y", 

                   "category_1_x", 

                   "category_2_y",

                   "category_2_x",

                   "category_3", 

                   "category_4", 

                   "authorized_flag", 

                   "installments", 

                   "merchant_id", 

                   "merchant_group_id", 

                   "numerical_1", 

                   "numerical_2", 

                   "most_recent_sales_range", 

                   "most_recent_purchases_range"

                  ], inplace = True)  
df.describe()
dfTest.describe()
def percent_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    return dict_x



missing = percent_missing(df)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:50]
# Setup do plot

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

sns.set_color_codes(palette='deep')



# Identificando os valores missing

missing = round(df.isnull().mean()*100,2)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(color="b")



# Visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Percent of missing values")

ax.set(xlabel="Features")

ax.set(title="Percent missing data by feature")

sns.despine(trim=True, left=True)
df = df.replace([np.inf, -np.inf], np.nan)

df.update(df.fillna(0))
dfSample = df.copy()

dfSample = dfSample.sample(n=100000)

dfSample.shape
dfSample.head()
# Verificando os tipos de dados do dataset

dfSample.dtypes
# Dataset de Treino

dfSample['first_active_month'] = pd.to_datetime(dfSample['first_active_month'])

dfSample['active_dayofweek'] = dfSample.first_active_month.apply(lambda dt: dt.dayofweek)

dfSample['active_year'] = dfSample.first_active_month.apply(lambda dt: dt.year)

dfSample['active_month'] = dfSample.first_active_month.apply(lambda dt: dt.month)

dfSample.drop(columns =["first_active_month"], inplace = True) 



# Codigo abaixo comentado pois nao funcionou no Kernel (somente na maquina local)

#dfSample['purchase_date'] = pd.to_datetime(dfSample['purchase_date'])

#dfSample['purchase_date_day'] = dfSample.purchase_date.apply(lambda dt: dt.day)

#dfSample['purchase_date_dayofweek'] = dfSample.purchase_date.apply(lambda dt: dt.dayofweek)

#dfSample['purchase_date_month'] = dfSample.purchase_date.apply(lambda dt: dt.month)

#dfSample['purchase_date_year'] = dfSample.purchase_date.apply(lambda dt: dt.year)

#dfSample['purchase_date_hour'] = dfSample.purchase_date.apply(lambda dt: dt.hour)

dfSample.drop(columns =["purchase_date"], inplace = True)  



# Dataset de Test

dfTest['first_active_month'] = pd.to_datetime(dfTest['first_active_month'])

dfTest['active_dayofweek'] = dfTest.first_active_month.apply(lambda dt: dt.dayofweek)

dfTest['active_year'] = dfTest.first_active_month.apply(lambda dt: dt.year)

dfTest['active_month'] = dfTest.first_active_month.apply(lambda dt: dt.month)

dfTest.drop(columns =["first_active_month"], inplace = True) 



# Codigo abaixo comentado pois nao funcionou no Kernel (somente na maquina local)

#dfTest['purchase_date'] = pd.to_datetime(dfTest['purchase_date'])

#dfTest['purchase_date_day'] = dfTest.purchase_date.apply(lambda dt: dt.day)

#dfTest['purchase_date_dayofweek'] = dfTest.purchase_date.apply(lambda dt: dt.dayofweek)

#dfTest['purchase_date_month'] = dfTest.purchase_date.apply(lambda dt: dt.month)

#dfTest['purchase_date_year'] = dfTest.purchase_date.apply(lambda dt: dt.year)

#dfTest['purchase_date_hour'] = dfTest.purchase_date.apply(lambda dt: dt.hour)

dfTest.drop(columns =["purchase_date"], inplace = True)  
def plot_feature_scatter(df1, df2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5,4,figsize=(24,24))



    for feature in features:

        i += 1

        plt.subplot(5,4,i)

        plt.scatter(df2[feature], df1['target'], marker='+')

        plt.xlabel(feature, fontsize=10)

    plt.show();
features = ['feature_1', 'feature_2','feature_3','month_lag', 'purchase_amount', 

            'avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12', 'avg_purchases_lag3', 'avg_purchases_lag6',

            'avg_purchases_lag12','active_months_lag3', 'active_months_lag6', 'active_months_lag12', 

            'active_dayofweek','active_month','active_year'

           ]

plot_feature_scatter(dfSample,dfSample, features)
from scipy import stats



sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))



# Fit a normal distribution

mu, std = norm.fit(dfSample['target'])



# Verificando a distribuicao de frequencia da variavel TARGET

sns.distplot(dfSample['target'], color="b", fit = stats.norm);

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="target")

ax.set(title="Target distribution: mu = %.2f,  std = %.2f" % (mu, std))

sns.despine(trim=True, left=True)



# Skewness: It is the degree of distortion from the symmetrical bell curve or the normal distribution

# If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.

# If the skewness is between -1 and -0.5(negatively skewed) or between 0.5 and 1(positively skewed), the data are moderately skewed.

# If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed.



# Kurtosis: It is actually the measure of outliers present in the distribution.

# High kurtosis in a data set is an indicator that data has heavy tails or outliers. 

# Low kurtosis in a data set is an indicator that data has light tails or lack of outliers



ax.text(x=1.1, y=1, transform=ax.transAxes, s="Skewness: %f" % dfSample['target'].skew(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:poo brown')

ax.text(x=1.1, y=0.95, transform=ax.transAxes, s="Kurtosis: %f" % dfSample['target'].kurt(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:dried blood')



plt.show()
dfSample[dfSample.columns.drop('target')].corrwith(dfSample.target)
fig = plt.subplots(figsize = (30,30))

sns.set(font_scale=1.5)

sns.heatmap(dfSample.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

plt.show()
# Fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in dfSample.columns:

    if dfSample[i].dtype in numeric_dtypes:

        numeric.append(i)

        

        

# Create box plots for all numeric features

sns.set_style("white")

f, ax = plt.subplots(figsize=(14, 11))

ax.set_xscale("log")

ax = sns.boxplot(data=dfSample[numeric] , orient="h", palette="Set1")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
# Setup do plot

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))



# Verificando a distribuicao

sns.distplot(dfSample['target'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="Target")

ax.set(title="Target distribution")

sns.despine(trim=True, left=True)

plt.show()
# Verificando mais de perto a variavel target

dfSample['target'].describe()
# Removendo outliers da variavei target (abaixo de -10 e acima de 10)

dfSample.drop(dfSample[(dfSample['target'] < -10)].index, inplace=True)

dfSample.drop(dfSample[(dfSample['target'] > 10)].index, inplace=True)
# Realizando uma transformacao logaritma

# log(1+x) transform

dfSample["target"] = np.log1p(dfSample["target"])
dfSample = dfSample.replace([np.inf, -np.inf], np.nan)

dfSample.update(dfSample["target"].fillna(0))
from scipy import stats



sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))



# Fit a normal distribution

mu, std = norm.fit(dfSample['target'])



# Verificando a distribuicao de frequencia da variavel TARGET

sns.distplot(dfSample['target'], color="b", fit = stats.norm);

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="target")

ax.set(title="Target distribution: mu = %.2f,  std = %.2f" % (mu, std))

sns.despine(trim=True, left=True)



ax.text(x=1.1, y=1, transform=ax.transAxes, s="Skewness: %f" % dfSample['target'].skew(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:poo brown')

ax.text(x=1.1, y=0.95, transform=ax.transAxes, s="Kurtosis: %f" % dfSample['target'].kurt(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:dried blood')



plt.show()
# Setup do plot

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))



# Verificando a distribuicao

sns.distplot(dfSample['purchase_amount'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="purchase_amount")

ax.set(title="purchase_amount distribution")

sns.despine(trim=True, left=True)

plt.show()
dfSample['purchase_amount'].describe()
dfSample['purchase_amount'] = dfSample['purchase_amount'].apply(lambda x: 0.01 if x <= 0 else x)

dfSample['purchase_amount'] = dfSample['purchase_amount'].apply(lambda x: 1 if x > 1 else x)
# Setup do plot

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))



# Verificando a distribuicao

sns.distplot(dfSample['purchase_amount'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="purchase_amount")

ax.set(title="purchase_amount distribution")

sns.despine(trim=True, left=True)

plt.show()
def plot_feature_dist(df1, features):

    i = 0

    sns.set_style('whitegrid')

    sns.set_color_codes(palette='deep')

    plt.figure()

    fig, ax = plt.subplots(3,2,figsize=(24,12))



    for feature in features:

        i += 1

        plt.subplot(3,2,i)

        sns.distplot(df1[feature], color="b");

        plt.xlabel(feature, fontsize=10)

    plt.show();
features = ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12', 

            'avg_purchases_lag3', 'avg_purchases_lag6','avg_purchases_lag12'

           ]

plot_feature_dist(dfSample, features)
dfSample['avg_sales_lag3'].describe()
dfSample['avg_sales_lag6'].describe()
dfSample['avg_sales_lag12'].describe()
# Dataset de Treino

dfSample['var_lag3'] = dfSample['avg_sales_lag3'] * dfSample['avg_purchases_lag3']

dfSample['var_lag6'] = dfSample['avg_sales_lag6'] * dfSample['avg_purchases_lag6']

dfSample['var_lag12'] = dfSample['avg_sales_lag12'] * dfSample['avg_purchases_lag12']



# Dataset de Teste

dfTest['var_lag3'] = dfTest['avg_sales_lag3'] * dfTest['avg_purchases_lag3']

dfTest['var_lag6'] = dfTest['avg_sales_lag6'] * dfTest['avg_purchases_lag6']

dfTest['var_lag12'] = dfTest['avg_sales_lag12'] * dfTest['avg_purchases_lag12']

# Remover as variaveis originais

dfSample.drop(columns = ['avg_sales_lag3', 'avg_purchases_lag3',

                       'avg_sales_lag6', 'avg_purchases_lag6',

                       'avg_sales_lag12', 'avg_purchases_lag12'

                  ], inplace = True)



dfTest.drop(columns = ['avg_sales_lag3', 'avg_purchases_lag3',

                       'avg_sales_lag6', 'avg_purchases_lag6',

                       'avg_sales_lag12', 'avg_purchases_lag12'

                  ], inplace = True)

def logs(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   

        res.columns.values[m] = l + '_log'

        m += 1

    return res



log_features = ['var_lag3','var_lag6','var_lag12']



dfSample = logs(dfSample, log_features)

dfTest = logs(dfTest, log_features)
def squares(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   

        res.columns.values[m] = l + '_sq'

        m += 1

    return res 



squared_features = ['var_lag3','var_lag6','var_lag12']



dfSample = squares(dfSample, squared_features)

dfTest = squares(dfTest, squared_features)
dfSample[dfSample.columns.drop('target')].corrwith(dfSample.target)
dfSample.head()
all_features = dfSample.copy()

all_features.shape
all_features.head()
all_features = pd.DataFrame(all_features.groupby( ['card_id'] ).mean().to_dict())
# Split features and labels

X = all_features.drop(['target'], axis=1)

y = all_features['target']



# Aplicando a mesma escala nos dados

X = MinMaxScaler().fit_transform(X)



# Padronizando os dados (0 para a média, 1 para o desvio padrão)

X = StandardScaler().fit_transform(X)
X.shape, y.shape, dfTest.shape
# Setup cross validation folds

kf = KFold(n_splits=2, random_state=123, shuffle=True)
# Defini a metrica de validacao (RMSL)

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=200,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       #verbose=0,

                       random_state=123)



# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=200,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:squarederror',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       #verbosity=3,

                       random_state=123)



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 5, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=200,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                #verbose=True,

                                random_state=123)  



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          #verbose=True,

                          random_state=123)



# KerasRegressor

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(13, input_dim=21, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



keras = KerasRegressor(build_fn=baseline_model, 

                       epochs=10, 

                       batch_size=5)
print(datetime.datetime.now().time())
scores = {}



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lgb'] = (score.mean(), score.std())
score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgb'] = (score.mean(), score.std())
score = cv_rmse(svr)

print("svr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['svr'] = (score.mean(), score.std())
score = cv_rmse(rf)

print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['rf'] = (score.mean(), score.std())
score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gbr'] = (score.mean(), score.std())
score = cv_rmse(keras)

print("keras: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['kr_norm'] = (score.mean(), score.std())
print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)
print('Svr')

svr_model_full_data = svr.fit(X, y)
print('RandomForest')

rf_model_full_data = rf.fit(X, y)
print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)
print('KerasRegressor')

keras_model_full_data = keras.fit(X, y)

# Fazendo as previsoes finais

# Nao consegui colocar o keras pois ele grava um History (estudando como fazer)

def blended_predictions(X):

    return ((svr_model_full_data.predict(X)) + \

            (gbr_model_full_data.predict(X)) + \

            (xgb_model_full_data.predict(X)) + \

            (lgb_model_full_data.predict(X)) + \

            (rf_model_full_data.predict(X)))
# Verificando as predictions dos modelos

blended_score = rmsle(y, blended_predictions(X))

scores['blended'] = (blended_score, 0)

print('RMSLE score no dataset de Treino:')

print(blended_score)
# Plot com a previsao de cada modelo

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] , '{:.6f}'.format(score[0]), horizontalalignment='left', size='22', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
# Usando o split para separar dados de treino e dados de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 10)

predictions = blended_predictions(X_test)

predictions
sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



plt.plot(range(y_test.shape[0]),y_test,label="Dados Originais")

plt.plot(range(y_test.shape[0]),predictions,label="Dados Previstos")

plt.legend(loc='best')

plt.ylabel('target')

plt.title('Comparacao com dados de teste')

plt.show()
print(datetime.datetime.now().time())
sub_final = pd.DataFrame(dfTest.groupby( ['card_id'] ).mean().to_dict())
sub_final = sub_final.replace([np.inf, -np.inf], np.nan)

sub_final.update(sub_final.fillna(0))
# Aplicando a mesma escala nos dados

X_final = MinMaxScaler().fit_transform(sub_final)



# Padronizando os dados (0 para a média, 1 para o desvio padrão)

X_final = StandardScaler().fit_transform(X_final)
X_final.shape
predictions = blended_predictions(X_final)

predictions
#Gerando Arquivo de Submissao

submission = pd.DataFrame({

    "card_id": sub_final.index, 

    "target": predictions

})
submission.head(30)
submission.to_csv('./submission_file.csv', index=False)