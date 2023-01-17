# https://www.kaggle.com/c/squadra-ml-junho-2020/
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from pylab import rcParams
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
import warnings
from pandas.errors import DtypeWarning
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)
warnings.simplefilter("ignore", DtypeWarning)
%matplotlib inline
%config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='darkgrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10
# from google.colab import drive
# drive.mount('/content/drive')
types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'PromoInterval': np.dtype(str)}
# Carregamento dos datasets.

filepath = '/kaggle/input/squadra-ml-junho-2020/'
# filepath = '/content/drive/My Drive/Colab Notebooks/squadra-ml-junho-2020/datasets/'

stores = pd.read_csv(filepath + 'lojas.csv')
test = pd.read_csv(filepath + 'dataset_teste.csv', dtype=types)
train = pd.read_csv(filepath + 'dataset_treino.csv', dtype=types)
train2 = pd.read_csv(filepath + 'dataset_treino.csv', parse_dates=['Date'], index_col='Date')
train.shape
stores.head(1)
train.head()
train['Id'] = 0
train = train.drop(['Customers'], axis=1)
train.head()
test['Sales'] = 0
test.head()
test.fillna(0, inplace=True)
train.fillna(0, inplace=True)
train2.fillna(0, inplace=True)
stores.fillna(0, inplace=True)
test.shape
train = pd.concat([train, test])
train = train.reset_index()
# Ordenando as lojas e as datas
train = train.sort_values(by=['Store', 'Date'])
# Definindo a data como 'index'
train = train.set_index(['Date'])
# Convertendo o 'index' para 'datetime'
train.index = pd.to_datetime(train.index)
train
# Adicionando informações complementares relacionadas à Data
def add_date_info(df):
  df['Day'] = df.index.day
  df['Month'] = df.index.month
  df['Year'] = df.index.year
  df['Week'] = df.index.week
  df['Weekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
  return df
train = add_date_info(train)
train2 = add_date_info(train2)
PREVIOUS_DAYS = 7#7

train_prev_sales = pd.DataFrame()

for i in train['Store'].unique().tolist():

  store_x = train[train['Store'] == i]
  store_x_train2 = train2[train2['Store'] == i]

  salesMeanStdMedianMonthStore = store_x_train2.groupby(by=['Month']).agg({'Sales': ['mean', 'std', 'median']}).values
  salesMeanStdMedianDayOfWeekStore = store_x_train2.groupby(by=['DayOfWeek']).agg({'Sales': ['mean', 'std', 'median']}).values
  #salesMeanStdMedianDayStore = store_x_train2.groupby(by=['Day']).agg({'Sales': ['mean', 'std', 'median']}).values
  
  store_x['SalesMeanMonth'] = store_x['Month'].apply(lambda x: salesMeanStdMedianMonthStore[x-1][0])
  store_x['SalesStdMonth'] = store_x['Month'].apply(lambda x: salesMeanStdMedianMonthStore[x-1][1])
  store_x['SalesMedianMonth'] = store_x['Month'].apply(lambda x: salesMeanStdMedianMonthStore[x-1][2])

  store_x['SalesMeanDayOfWeek'] = store_x['DayOfWeek'].apply(lambda x: salesMeanStdMedianDayOfWeekStore[x-1][0])
  store_x['SalesStdDayOfWeek'] = store_x['DayOfWeek'].apply(lambda x: salesMeanStdMedianDayOfWeekStore[x-1][1])
  store_x['SalesMedianDayOfWeek'] = store_x['DayOfWeek'].apply(lambda x: salesMeanStdMedianDayOfWeekStore[x-1][2])

  #store_x['SalesMeanDay'] = store_x['Day'].apply(lambda x: salesMeanStdMedianDayStore[x-1][0])
  #store_x['SalesStdDay'] = store_x['Day'].apply(lambda x: salesMeanStdMedianDayStore[x-1][1])
  #store_x['SalesMedianDay'] = store_x['Day'].apply(lambda x: salesMeanStdMedianDayStore[x-1][2])

  # Adicionando as vendas anteriories dos últimos X dias
  # for inc in range(PREVIOUS_DAYS):
    # field_name = 'PrevSales' + str(inc+1)
    # store_x[field_name] = store_x['Sales'].shift(inc+1)

  train_prev_sales = pd.concat([train_prev_sales, store_x])
  # break
train_prev_sales = train_prev_sales.dropna()
train_prev_sales.head()
train, test = train_prev_sales[train_prev_sales['Id'] == 0], train_prev_sales[train_prev_sales['Id'] > 0]
train = train.drop(['Id'], axis=1)
test = test.drop(['Sales'], axis=1)
test.shape
train[train['Store'] == 1].tail()
test[test['Store'] == 1].head()
print(train.shape)

# Considerando apenas lojas abertas para treinamento. Lojas fechadas não contam para a pontuação
train = train[train['Open'] != 0]
# Usando apenas vendas maiores que zero. Simplifica o cálculo do RMSPE
train = train[train['Sales'] > 0]

print(train.shape)
train['Sales'].describe()
# Identificando Outliers
train.boxplot(column='Sales')
plt.show()
sns.boxplot(train['DayOfWeek'], train['Sales'])
plt.show()
# Removendo Outliers
train_without_outliers = pd.DataFrame()

for day in np.sort(train['DayOfWeek'].unique()):
    
    train_day_of_week = train[train['DayOfWeek'] == day]
    outliers_min = train_day_of_week['Sales'].quantile(.05) # 0
    outliers_max = train_day_of_week['Sales'].quantile(.95)
    
    # print(day, outliers_min, outliers_max)
    
    train_day_of_week = train_day_of_week[(train_day_of_week['Sales'] >= outliers_min) & (train_day_of_week['Sales'] <= outliers_max)]
    
    train_without_outliers = pd.concat([train_without_outliers, train_day_of_week])

# 1ª Versão
# train = train_without_outliers

# 2ª Versão
outliers_min = train['Sales'].quantile(0) #.05
outliers_max = train['Sales'].quantile(.95)
# print(outliers_min, outliers_max)
# train = train[(train['Sales'] >= outliers_min) & (train['Sales'] <= outliers_max)]

# 3ª Versão
train = train[train['Sales'] <= 13300]

train['Sales'].describe()
train.boxplot(column='Sales')
plt.show()
sns.boxplot(train['DayOfWeek'], train['Sales'])
plt.show()
# Acrescentando as informações complementares das lojas
def add_store_info(df):
  df_stores = pd.merge(df, stores, on='Store')
  df_stores.index = df.index
  return df_stores
from sklearn.preprocessing import LabelEncoder

# Transformando todos os valores (categóricos) em valores numéricos
def transform_categ_info(df):
  # df.loc[df['StateHoliday'] != '0', 'StateHoliday'] = '1'

  # df['StateHoliday'] = df['StateHoliday'].astype(int)
  df['StoreType'] = LabelEncoder().fit_transform(df['StoreType'])
  df['Assortment'] = LabelEncoder().fit_transform(df['Assortment'])
  return df
# Calcula o tempo de abertura do concorrente em meses
def add_months_competition_open(df):
  df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
  # Removendo colunas previamente utilizadas
  df = df.drop(['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], axis=1)
  return df
# Tempo de abertura da promoção em meses
def add_months_promo_open(df):
  df['PromoOpen'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['Week'] - df['Promo2SinceWeek']) / 4.0
  df['PromoOpen'] = df['PromoOpen'].apply(lambda x: x if x > 0 else 0)
  df.loc[df['Promo2SinceYear'] == 0, 'PromoOpen'] = 0
  # Removendo colunas previamente utilizadas
  df = df.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis=1)
  return df
# Indica que as vendas nesse dia estão no intervalo da promoção
def add_is_promo_month(df):
  month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
  df['MonthStr'] = df['Month'].map(month2str)
  df.loc[df['PromoInterval'] == 0, 'PromoInterval'] = ''
  df['IsPromoMonth'] = 0
  for interval in df['PromoInterval'].unique():
    if interval != '':
      for month in interval.split(','):
        df.loc[(df['MonthStr'] == month) & (df['PromoInterval'] == interval), 'IsPromoMonth'] = 1
  # Removendo colunas previamente utilizadas
  df = df.drop(['PromoInterval', 'MonthStr'], axis=1)
  return df
# Adiciona recursos ao dataset
def add_features(df):
  df = add_store_info(df)
  df = transform_categ_info(df)
  df = add_months_competition_open(df)
  df = add_months_promo_open(df)
  df = add_is_promo_month(df)
  return df
test = add_features(test)
train = add_features(train)
test = test.drop(['index'], axis=1)
train = train.drop(['index'], axis=1)
train.head()
test.head()
# sns.lineplot(x=train.index, y='Sales', data=train)
train.columns
# Import statsmodels.formula.api
import statsmodels.formula.api as smf
# Define the regression formula
# model = smf.ols(formula='Sales ~ PrevSales1 + PrevSales2 + PrevSales3 + PrevSales4 + PrevSales5 + PrevSales6 + PrevSales7', data=train)
model = smf.ols(formula='Sales ~ DayOfWeek + Promo + SchoolHoliday + SalesStdMonth + SalesStdDayOfWeek + SalesMeanMonth + SalesMeanDayOfWeek + SalesMedianMonth + SalesMedianDayOfWeek'\
                # ' + PrevSales1 + PrevSales2 + PrevSales3 + PrevSales4 + PrevSales5 + PrevSales6 + PrevSales7'\
                ' + Day + Month + Year + Week + Weekend + StoreType + Assortment + CompetitionDistance + Promo2 + CompetitionOpen + PromoOpen + IsPromoMonth', data=train)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)
# Removendo colunas inúteis
train.drop(['Open', 'StateHoliday'], axis=1, inplace =True)
test.drop(['Open', 'StateHoliday'], axis=1, inplace =True)
train.corr()
# plt.subplots(figsize=(24,20))
# sns.heatmap(train.corr(), annot=True, vmin=-0.1, vmax=0.1,center=0)
train
# !pip install xgboost
# !pip install xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl
import xgboost as xgb
def rmspe(y, yhat):
    return np.sqrt(np.mean(((y - yhat) / y) ** 2))
def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)
# from sklearn.model_selection import train_test_split

SIX_WEEKS = 6*7*1115
train = train.sort_index(ascending = False)
X_train, X_valid = train[SIX_WEEKS:], train[:SIX_WEEKS]

# Testando o desempenho do modelo, sem algumas features.
# X_train.drop(['SalesMedianDayOfWeek'], axis=1, inplace =True)
# X_valid.drop(['SalesMedianDayOfWeek'], axis=1, inplace =True)

Y_train, Y_valid = np.log1p(X_train['Sales']), np.log1p(X_valid['Sales'])

dtrain = xgb.DMatrix(X_train.drop(['Sales'], axis=1), Y_train)
dvalid = xgb.DMatrix(X_valid.drop(['Sales'], axis=1), Y_valid)
params = {
  'objective': 'reg:squarederror', # 'reg:linear'
  'booster' : 'gbtree',
  'eta': 0.03,
  'max_depth': 10,
  'subsample': 0.9,
  'colsample_bytree': 0.3,
  'silent': 1, 'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor',
  'seed': 10
}
num_boost_round = 1500#1000
early_stopping_rounds = 150#50
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

model = xgb.train(
  params,
  dtrain,
  num_boost_round,
  evals=watchlist,
  early_stopping_rounds=early_stopping_rounds,
  feval=rmspe_xg,
  verbose_eval=True
)
yhat = model.predict(
  xgb.DMatrix(X_valid.drop(['Sales'], axis=1))
)
error = rmspe(X_valid['Sales'].values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))
test = test.sort_values(by=['Id'])

dtest = xgb.DMatrix(test.drop(['Id'], axis=1))
test_probs = model.predict(dtest)

submission = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(test_probs * 0.995)}) # 0.985
submission.to_csv('/kaggle/working/submission_xbgregressor_new_202007152013.csv', index=False)

# from google.colab import files
# files.download('submission_xbgregressor_new_t.csv')
fig, ax = plt.subplots(1,1,figsize=(10,14))
_ = xgb.plot_importance(booster=model, ax=ax)
