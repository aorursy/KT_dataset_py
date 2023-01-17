# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt

import datetime 

import sklearn # ML
from kaggle.competitions import twosigmanews

# Any results you write to the current directory are saved as output.
# Retreive the environment of the competition
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Data loaded!')
# Retrieve all training data
(market_train_df, news_train_df) = env.get_training_data()
print("Fetching training data finished... ")
print('Data obtained!')
# Market data analysis
# Types of the columns
print(market_train_df.dtypes)
market_train_df.head()
# Correlation between the numericals (except universe)
# Note that this removes the null values from the computation
market_train_df.iloc[:, 3:].corr(method='pearson')
# Lets analyze further the target variable
# Very big outliers, lets see their number and distribution
fig, axes = plt.subplots(3,2, figsize=(20, 12)) # create figure and axes
print("# Rows with |value| > 1 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>1].shape[0])
print("# Rows with |value| > 0.5 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>0.5].shape[0])
print("# Rows with |value| > 0.25 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>0.25].shape[0])
print("# Rows with |value| > 0.1 =", market_train_df[market_train_df["returnsOpenNextMktres10"].abs()>0.1].shape[0])

# Boxplot with all values
market_train_df.boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[0])
axes.flatten()[0].set_xlabel('Boxplot with all values', fontsize=18)
# Removing rows with outliers (bigger or smaller than 1)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<1].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[1])
axes.flatten()[1].set_xlabel('Boxplot with values such that |val| < 1', fontsize=18)
# Removing rows with outliers (bigger or smaller than 0.5)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.5].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[2])
axes.flatten()[2].set_xlabel('Boxplot with values such that |val| < 0.5', fontsize=18)
# Removing rows with outliers (bigger or smaller than 0.25)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.25].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[3])
axes.flatten()[3].set_xlabel('Boxplot with values such that |val| < 0.25', fontsize=18)
# Removing rows with outliers (bigger or smaller than 0.1)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.1].boxplot(column="returnsOpenNextMktres10", ax=axes.flatten()[4])
axes.flatten()[4].set_xlabel('Boxplot with values such that |val| < 0.1', fontsize=18)
# Distribution of the target value (not including values bigger or smaller than 1)
market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.25].hist(column="returnsOpenNextMktres10", bins=100, ax=axes.flatten()[5])
axes.flatten()[5].set_xlabel('Histogram for values such that |val| < 0.25', fontsize=18)
print("The variable is actually centered in 0 and only a few outliers higher than 0.1. This makes sense considering that the returns of the \
market for 10 days are really small. Our goal then should be to detect those times in which the wins or loses are really high by making \
use of the news. A good approach for this could be an algorithm to control the small temporal oscilation of the market and then use the news \
to detect those imprevisible changes.")
# Let's analyze null values 
#print(market_train_df.isnull().sum())
nuls = market_train_df.loc[:,['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']]
nuls
#print (nuls)
#tol = 0.10
#nuls_r1 = market_train_df.loc[market_train_df['returnsClosePrevMktres1'] <= tol,'returnsClosePrevMktres1']
#print(nuls_r1, len(nuls_r1), 'Media--> ', nuls_r1.mean())
# Lets change null values by means without taking into account the possible outliers. 
tol = 0.15
lista_valores_rellenar = list()
for col in nuls:
    print (nuls[col], "Media--> ", nuls[col].loc[nuls[col]<= tol].mean())
    lista_valores_rellenar += [nuls[col].loc[nuls[col]<= tol].mean()]
lista_valores_rellenar
# And now we fill the nulls
market_train_df.loc[pd.Series(market_train_df['returnsClosePrevMktres1'].isnull()), 'returnsClosePrevMktres1'] = lista_valores_rellenar[0]
market_train_df.loc[pd.Series(market_train_df['returnsOpenPrevMktres1'].isnull()), 'returnsOpenPrevMktres1'] = lista_valores_rellenar[1]
market_train_df.loc[pd.Series(market_train_df['returnsClosePrevMktres10'].isnull()), 'returnsClosePrevMktres10'] = lista_valores_rellenar[2]
market_train_df.loc[pd.Series(market_train_df['returnsOpenPrevMktres10'].isnull()), 'returnsOpenPrevMktres10'] = lista_valores_rellenar[3]
print(lista_valores_rellenar)

# Did it work?
print(market_train_df.isnull().sum())

# Let us remove some not important columns.

news_train_df = news_train_df.drop(['sourceTimestamp','firstCreated'], axis = 1)
#market_train_df = market_train_df.drop(['daily_diff'], axis = 1)
#Para dataset market
#creamos un nuevo campo con el formato que queremos usando strtime
market_train_df['date']=market_train_df['time'].dt.strftime('%Y-%m-%d')
#esto nos crea un nuevo campo date con el formato que queremos pero con tipo object
#market_train_df['date'].dtypes
#cambiamos el tipo de dato object a datetime, manteniendo el formato que queremos
market_train_df['date']=pd.to_datetime(market_train_df['date'],  format='%Y/%m/%d')
#comprobamos qe hemos cambiado el tipo de dato
print(market_train_df.dtypes)
#mostramos resultados del nuevo DF
market_train_df.head()
#Para dataset news
#creamos un nuevo campo con el formato que queremos usando strtime
news_train_df['date']=news_train_df['time'].dt.strftime('%Y-%m-%d')
#esto nos crea un nuevo campo date con el formato que queremos pero con tipo object
#market_train_df['date'].dtypes
#cambiamos el tipo de dato object a datetime, manteniendo el formato que queremos
news_train_df['date']=pd.to_datetime(news_train_df['date'],  format='%Y/%m/%d')
#comprobamos qe hemos cambiado el tipo de dato
news_train_df.dtypes
#mostramos resultados del nuevo DF
news_train_df.head()
# Lets join both datsets. Firstly, we need to do some minor changes on dates. We know that there might be more than 1 new for a given asset in one day.
#news_train_df['time2'] = news_train_df['time'].astype(str).str.slice(0,10)
#market_train_df['time2'] = market_train_df['time'].astype(str).str.slice(0,10)

print(len(news_train_df), len(market_train_df))
# As after when we are training we run out of RAM, I make a random sampling of the market dataset. There is no need 
# to do that on the news, since when we do the merge we use the keys of the markets. It would be nice to be sure that
# we have same type of distributions. 
valores_mercado = np.random.choice(int(len(market_train_df)/4), int(len(market_train_df)/4), replace=False)
valores_news = np.random.choice(int(len(news_train_df)/1), int(len(news_train_df)/1), replace=False)
# We select those rows of the dataframes
print(len(market_train_df))
news_train_df = news_train_df.iloc[valores_news,:]
market_train_df = market_train_df.iloc[valores_mercado,:]


# We do the final merge by 2 keys
df_final = market_train_df.merge(news_train_df, how = 'left', on = ['assetName', 'date'])
print(len(df_final))
# Posterior analysis say that the dataset may have many outliers. Most of the dataset is below 0.5 so the rest might be outliers
print(len(df_final), len(df_final.loc[abs(df_final['returnsOpenNextMktres10']) <0.5]))
df_final.head(20)
# Lets study a little bit more the target variable. It is a sharp gaussian!!

df_final.loc[abs(df_final['returnsOpenNextMktres10']) <0.5, 'returnsOpenNextMktres10'].hist(bins = 1000)
# We are happy with that number

df_final = df_final.loc[abs(df_final['returnsOpenNextMktres10']) <0.5]

print('KURTOSIS --> ', df_final['returnsOpenNextMktres10'].kurtosis(), 'SKEWNESS --> ', df_final['returnsOpenNextMktres10'].skew())
# Lets make a list out of the columns related to novelty and counts.
df_final['list_novelties'] = df_final.reset_index()[['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D']].values.tolist()
df_final = df_final.drop(['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D'], axis = 1)
df_final['list_counts']= df_final.reset_index()[['volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']].values.tolist()
df_final = df_final.drop(['volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D'], axis = 1)
df_final.head(10)
len(df_final)
# Lets do the splitting for ML !
from sklearn.model_selection import train_test_split
df_final = df_final.dropna()
X = df_final.loc[:, df_final.columns != 'returnsOpenNextMktres10']
y = df_final.loc[:, 'returnsOpenNextMktres10']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
# We need to delete datafrmes to avoid memory problems
del ( market_train_df, news_train_df, df_final)
print(len(X_train),X_train.isnull().sum(), type(X_train))
# We delete nans from our datasets. THey appear after the join
print(len(X_train), len(X_test),len(y_train),len(y_test))
X_train, X_test, y_train, y_test = X_train.dropna(), X_test.dropna(), y_train.dropna(), y_test.dropna() 
print(len(X_train), len(X_test),len(y_train),len(y_test))
y_train.head()
print(len(y_train))
print('KURTOSIS NO OUT--> ', y_train.loc[abs(y_train) <0.5].kurtosis(), 'SKEWNESS NO OUT--> ', y_train.loc[abs(y_train) <0.5].skew())
print('KURTOSIS --> ', y_train.kurtosis(), 'SKEWNESS--> ', y_train.skew())
print(X_train.dtypes)
X_train.head(6)
# Pandas already does this for us!!!!
#cols_cod = []
#for i in X_train:
 #   print (type(X_train[i]))
    #if type(i) != 'float64':
     #   cols_cod += [i]
#print(cols_cod)
#from sklearn.preprocessing import OneHotEncoder
#enc =  OneHotEncoder(handle_unknown='ignore')
#enc.fit(X)

# For some first train tests we just use the numerical data, we need to think about what we should do with the strings.

x_train = X_train.select_dtypes(['float32', 'float64'])
x_train
x_test = X_test.select_dtypes(['float32', 'float64'])
x_train.head(5)
# Let's make a function than standarizes numerical values.
def normalizer(df):
    for col in df:
        df[col] = df[col] / df[col].max()
    return df

print(len(x_train), len(y_train))
x_train.head(160)
# x_train and y_Train podrian no tener l misma longitud
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_predicted=reg.predict(x_test)
print("Acabó")
mean_absolute_error(y_test, y_predicted)

# Let us try with the standarize dataframes
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
reg1 = linear_model.LinearRegression()
reg1.fit(normalizer(x_train), y_train)
y_predicted=reg1.predict(normalizer(x_test))
print("Acabó")
mean_absolute_error(y_test, y_predicted)

# WE NEED TO ENCODE DATA. 
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder(handle_unknown='ignore')
#enc.fit(X)
# Now ML TIME!!
# PODRIA HABER PROBLEMAS DE QUE LAS LISTAS NO MIDAN LO MISMO!!!


from sklearn import svm
clf = svm.SVR()
clf.fit(X_train, y_train)
# Predict
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=0, n_estimators=20)
regr.fit(x_train, y_train)
y_predicted=regr.predict(x_test)
print("Acabó")
mean_absolute_error(y_test, y_predicted)
from sklearn.ensemble import RandomForestRegressor
regr1 = RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=0, n_estimators=20)
regr1.fit(normalizer(x_train), y_train)
y_predicted=regr1.predict(normalizer(x_test))
print("Acabó")
mean_absolute_error(y_test, y_predicted)
#from sklearn.naive_bayes import GaussianNB
#gnb = GaussianNB()
#gnb.fit(X_train, y_train)
df_results = X_test
df_results.insert(loc=df_results.shape[1], column="y_real", value=y_test)
df_results.insert(loc=df_results.shape[1], column="y_pred", value=y_predicted)

days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
predictions_template_df.head()
days.head()
