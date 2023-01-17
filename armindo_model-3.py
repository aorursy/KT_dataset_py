import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn import model_selection

from sklearn import metrics

warnings.filterwarnings('ignore')
#print(os.listdir('../input'))
df_lojas = pd.read_csv('../input/lojas.csv')

df_treino = pd.read_csv('../input/dataset_treino.csv', low_memory=False)

df_teste = pd.read_csv('../input/dataset_teste.csv')
display(df_treino.shape)

display(df_lojas.shape)
df_treino.head()
df_lojas.head()
df_total = df_treino.merge(df_lojas, how='left', on='Store')
df_total.Date = pd.to_datetime(df_total.Date)

df_total['day'] = df_total.Date.apply(lambda x: x.day)

df_total['month'] = df_total.Date.apply(lambda x: x.month)

df_total['year'] = df_total.Date.apply(lambda x: x.year)
# Quando a loja está fechada, não há vendas

df_total = df_total[df_total.Open!=0]
#coletando as variáveis com valores na

def imprime_colunas_com_valores_missing(df):

    list_na = []

    for i in range(0,len(df.columns)):

        if df.isna().any()[i] == True:

            list_na.append(df.isna().any().index[i])

            print(df.isna().any().index[i])

imprime_colunas_com_valores_missing(df_total)
#Variável PromoInterval quando não há promoção é na

df_total[(df_total.Promo==0)].PromoInterval.value_counts()
df_total[(df_total.Promo2==0)].PromoInterval.value_counts()
df_total.PromoInterval = np.where(df_total.Promo2==0, 'no promo consec', df_total.PromoInterval)
df_total.PromoInterval.isna().any()
df_total[df_total.Sales == 0].shape
for i in df_total.columns:

    print(i)

    print(df_total[i].value_counts().head(15))

    print('--------\n')
df_total.CompetitionDistance.fillna(np.mean(df_total.CompetitionDistance), inplace=True)

df_total.CompetitionOpenSinceMonth.fillna(np.mean(df_total.CompetitionOpenSinceMonth), inplace=True)

df_total.CompetitionOpenSinceYear.fillna(np.mean(df_total.CompetitionOpenSinceYear), inplace=True)
df_total.Promo2SinceYear = np.where(df_total.Promo2==0,0,df_total.Promo2SinceYear)

df_total.Promo2SinceWeek = np.where(df_total.Promo2==0,0,df_total.Promo2SinceWeek)
df_total.isna().any()
df_total.info()
list_ = ['Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'Promo2']
for i in list(df_total.columns):

    print(i)

    print(df_total[i].value_counts().head(15))

    print('-------------\n')
def convert_to_int(x):

    if x == 0:

        return 6

    if x == 1:

        return 1    

    elif x == 'a':

        return 1

    if x == 'b':

        return 2

    elif x == 'c':

        return 3    

    elif x == 'd':

        return 4 

    else:

        return 5 
for i in list_:

    df_total[i] = df_total[i].apply(lambda x: convert_to_int(x))
df_total.PromoInterval.value_counts()
def convert_to_int_promoInterval(x):

    if x == 'no promo consec':

        return 1 

    elif x == 'Jan,Apr,Jul,Oct':

        return 2 

    elif x == 'Feb,May,Aug,Nov':

        return 3 

    elif x == 'Mar,Jun,Sept,Dec':

        return 4
# tratando promoInterval

df_total.PromoInterval = df_total.PromoInterval.apply(lambda x: convert_to_int_promoInterval(x))
df_total.PromoInterval.value_counts()
fig1, ax = plt.subplots()

ax.boxplot(df_total.Sales)
mean = np.mean(df_total.Sales, axis=0)

median = np.median(df_total.Sales, axis=0)

sd = np.std(df_total.Sales, axis=0)

outliers = mean+(2*sd)

outliers
# % de Outliers

df_total[df_total.Sales>outliers].shape[0]/df_total.shape[0]*100
df_total_sem_outliers = df_total[df_total.Sales<outliers]
fig1, ax = plt.subplots()

ax.boxplot(df_total_sem_outliers.Sales)
plt.hist(df_total_sem_outliers.Sales)

plt.show()
df_total_sem_outliers.columns
# Compute the correlation matrix

corr = df_total_sem_outliers.loc[:,['DayOfWeek', 'Promo',

       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

       'CompetitionDistance', 'CompetitionOpenSinceMonth',

       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',

       'Promo2SinceYear', 'PromoInterval', 'day', 'month', 'year']].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,

            square=True, linewidths=.9, cbar_kws={"shrink": .9})



plt.show()
# Compute the correlation matrix

corr = df_total_sem_outliers.loc[:,['DayOfWeek', 'Promo',

       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

       'CompetitionDistance', 'CompetitionOpenSinceMonth',

       'CompetitionOpenSinceYear', 'Promo2', 'day', 'month', 'year']].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,

            square=True, linewidths=.9, cbar_kws={"shrink": .9})



plt.show()
corr
X = df_total_sem_outliers.loc[:,['DayOfWeek', 'Promo',

       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

       'CompetitionDistance', 'CompetitionOpenSinceMonth',

       'CompetitionOpenSinceYear', 'Promo2', 'day', 'month', 'year']]



Y = df_total_sem_outliers.loc[:,['Sales']]
scaler = MinMaxScaler(feature_range = (0, 1))

X_scaled = scaler.fit_transform(X)
teste_size = 0.3

seed = 7



X_treino, X_teste, Y_treino, Y_teste = model_selection.train_test_split(X_scaled, Y, 

                                                                         test_size = teste_size, 

                                                                         random_state = seed)
def evaluate(model, X_teste, Y_teste):

    predictions = model.predict(X_teste)

    errors = abs(predictions - Y_teste)

    mape = 100 * np.mean(errors / Y_teste)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))
def rmspe(y_true, y_pred):



    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_pred)), axis=0))
#xgb1 = XGBRegressor()

#parameters = {'nthread':[4], 

#              'objective':['reg:linear'],

#              'learning_rate': [.03, 0.05, .07], 

#              'max_depth': [5, 6, 7],

#              'min_child_weight': [4],

#              'silent': [1],

#              'subsample': [0.7],

#              'colsample_bytree': [0.7],

#              'n_estimators': [500]}



#xgb_grid = GridSearchCV(xgb1, parameters, cv = 2, n_jobs = 5, verbose=True)

#xgb_grid.fit(X_treino,Y_treino)
#print(xgb_grid.best_score_)

#print(xgb_grid.best_params_)
xgboost_model = XGBRegressor(colsample_bytree = 0.7, 

                             learning_rate = 0.07,

                             max_depth = 7,

                             min_child_weight = 4,

                             n_estimators = 500,

                             nthread = 4,

                             objective = 'reg:linear',

                             silent = 1,

                             subsample = 0.7)
xgboost_model.fit(X_treino,Y_treino)
df_teste_total = df_teste.merge(df_lojas, how='left', on='Store')
df_teste_total.Date = pd.to_datetime(df_teste_total.Date)

df_teste_total['day'] = df_teste_total.Date.apply(lambda x: x.day)

df_teste_total['month'] = df_teste_total.Date.apply(lambda x: x.month)

df_teste_total['year'] = df_teste_total.Date.apply(lambda x: x.year)
for i in list_:

    df_teste_total[i] = df_teste_total[i].apply(lambda x: convert_to_int(x))
df_teste_total.CompetitionDistance.fillna(np.mean(df_total.CompetitionDistance), inplace=True)

df_teste_total.CompetitionOpenSinceMonth.fillna(np.mean(df_total.CompetitionOpenSinceMonth), inplace=True)

df_teste_total.CompetitionOpenSinceYear.fillna(np.mean(df_total.CompetitionOpenSinceYear), inplace=True)
#df_total.Promo2SinceYear = np.where(df_total.Promo2==0,0,df_total.Promo2SinceYear)

#df_total.Promo2SinceWeek = np.where(df_total.Promo2==0,0,df_total.Promo2SinceWeek)
df_teste_sem_zero = df_teste_total[df_teste_total.Open != 0]

df_teste_com_zero = df_teste_total[df_teste_total.Open == 0]
X_ = df_teste_sem_zero.loc[:,['DayOfWeek', 'Promo',

       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',

       'CompetitionDistance', 'CompetitionOpenSinceMonth',

       'CompetitionOpenSinceYear', 'Promo2', 'day', 'month', 'year']]
scaler = MinMaxScaler(feature_range = (0, 1))

X_scaled_teste = scaler.fit_transform(X_)
pred = xgboost_model.predict(X_scaled_teste)
df_teste_sem_zero['Sales'] = pred
sub1 = df_teste_sem_zero.loc[:,['Id', 'Sales']]
sub2 = df_teste_com_zero.loc[:,['Id']]
sub2['Sales'] = 0
sub_final = pd.concat([sub1, sub2])
sub_final.shape
sub_final.to_csv('sub_xgboost.csv', index=False)