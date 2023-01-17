# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.simplefilter("ignore", UserWarning)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split as split

from sklearn import  linear_model as lr

from sklearn.metrics import mean_squared_error, r2_score

import pickle

from sklearn.ensemble import RandomForestRegressor

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

from sklearn.preprocessing import StandardScaler as ss

from scipy import stats

import tensorflow as tf

import keras
import pandas as pd

df = pd.read_csv("../input/Sales_prediction_Colombia.csv")
def faltante(df, p):

    cols=[]

    missing=df.isnull().sum()

    for i in range(len(df.columns)):

        if(missing[i]>p*len(df)):

            cols.append(missing.index[i])

    return cols
lis=['areabalcon','areaterraza']

for a in lis:

    df[a].fillna(0, inplace=True)



subset=faltante(df,0.05)



for c in subset:

    if(len(df[c].unique())==2):

        df.loc[df[c]!="Si",c]="No"

        df.loc[df[c]=="Si",c]=1

        df.loc[df[c]=="No",c]=0

c="balcon"

df.loc[df[c].isnull() ,c]="Ninguno"

df.loc[df[c]=="Si" ,c]="Balcón"

c='depositoocuartoutil'

df.loc[df[c].isnull() ,c]="Ninguno"

df.loc[df[c]=="Si" ,c]="1"

c='depositos'

df.loc[df[c].isnull() ,c]="Ninguno"

c='instalaciondegas'

df.loc[df[c].isnull() ,c]="Ninguno"

c='numeroascensores'

df.loc[df[c].isnull() ,c]=0

c="porteriaovigilancia"

df.loc[df[c].isnull() ,c]="Ninguno"

c='terraza'

df.loc[df[c].isnull() ,c]="Ninguno"

df.loc[df[c]=="Si" ,c]="Terraza"

c='tipodegaraje'

df.loc[df[c].isnull() ,c]="Ninguno"

c='vigilancia'

df.loc[df[c].isnull() ,c]="Ninguno"

c='vista'

df.loc[df[c].isnull() ,c]="Ninguno"

c='garajes'

df.loc[df[c].isnull() ,c]=0

strings=[]

for i in range(len(df.dtypes)):

    if df.dtypes[i]=='O':

        strings.append(df.dtypes.index[i])

for s in strings:

    print(s)

    df= pd.get_dummies(df, prefix="Dummy_"+str(s), columns=[s])
sns.heatmap(df.corr())
corr_matrix = df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.97)]

df=df.drop(df[to_drop], axis=1)



sns.heatmap(df.corr())
df=df.dropna()

df_log=df.copy()

df_log["log_area"]=np.log(df_log['area']+1)

df_log["log_area_bal"]=np.log(df_log['areabalcon']+1)

df_log["log_valor"]=np.log(df_log['valor']+1)
sns.boxplot(df_log.log_valor)
Q1 = df_log.log_valor.quantile(0.25)

Q3 = df_log.log_valor.quantile(0.75)

IQR = Q3 - Q1
lower=Q1 - 1.5 * IQR

upper=Q3 + 1.5 * IQR
df_log = df_log[(df_log.log_valor <upper)|(df_log.log_valor<lower)]
df_log.columns
def boxcox(df, columna):   

    price,fitted_lambda = stats.boxcox((df[columna]+1))

    print(columna, round(fitted_lambda,2))
boxcox(df, "valor")
stats.probplot(df['valor'], dist = "norm", plot = plt)

plt.title("QQ Plot for Prices")

plt.show()
sns.distplot(df['valor'],fit=stats.norm, kde=False)

sns.set(color_codes=True)

plt.xticks(rotation=90)

plt.title("Histogram of prices")
sns.lmplot(x='area',y='valor',data=df, 

           line_kws = {'color': "red"} ,aspect= 2)

plt.title("Valor vs. Area")
plt.subplot(2,1,1)

stats.probplot(df_log['log_valor'], plot = plt)

plt.title("QQ Plot para log(valor+1)")

plt.show()

plt.subplot(2,1,2)

sns.distplot(df_log["log_valor"],fit=stats.norm, kde=False)

sns.set(color_codes=True)

plt.xticks(rotation=90)

plt.title("Histograma de precios")
df_log["log_area"]=np.log(df_log['area']+1)

df_log["log_latitud"]=np.log(df_log['latitud']+1)

df_log["log_longitud"]=np.log(abs(df_log['longitud']+1))
sns.lmplot(x='area',y='log_valor',data = df_log, 

           line_kws = {'color': "red"} ,aspect= 2, scatter_kws={"s": 5})

plt.title("Log_Valor vs. area")
sns.lmplot(x='log_area',y='log_valor',data = df_log, 

           line_kws = {'color': "red"} ,aspect= 2, scatter_kws={"s": 5})

plt.title("log_valor vs. log_area")
sns.lmplot(x='latitud',y='log_valor',data = df_log, 

           line_kws = {'color': "red"} ,aspect= 2, scatter_kws={"s": 5})

plt.title("latitud vs. log_area")
sns.lmplot(x='longitud',y='log_valor',data = df_log, 

           line_kws = {'color': "red"} ,aspect= 2, scatter_kws={"s": 5})

plt.title("longitud vs. log_area")
sns.lmplot(x = 'log_area', y= 'log_valor',data = df_log, 

           hue="estrato", height=8, scatter_kws={"s": 10})

plt.title("log_valor vs. Latitud")
sns.lmplot(x = 'latitud', y= 'log_valor',data = df_log, 

           hue="estrato", height=8, scatter_kws={"s": 10})

plt.title("log_valor vs. Latitud")
sns.pairplot(df_log,vars=["log_area","latitud","longitud","log_valor"])
df_log_clean = df_log.drop(["valor","area","log_latitud","log_longitud"], axis=1)
def mape(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import keras.backend as K

def r2_keras(y_true, y_pred):

    SS_res =  K.sum(K.square(y_true - y_pred)) 

    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 

    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
xn_train, xn_test, yn_train, yn_test= split(x_s,y_s, test_size=0.2, random_state=72)
x=df_log_clean.drop("log_valor", axis=1)

y=df_log_clean[["log_valor"]]
x_train, x_test, y_train, y_test= split(x,y, test_size=0.2, random_state=72)
reg= lr.Ridge(alpha=0.9)

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

print("RIDGE")

print("MSE ",mean_squared_error(y_test, y_pred))

print("R2 SCORE ",r2_score(y_test, y_pred))
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

rf.fit(x_train, y_train)

y_pred=rf.predict(x_test)

print("RANDOM FOREST")

print("MSE ",mean_squared_error(y_test, y_pred))

print("R2 SCORE ",r2_score(y_test, y_pred))
scaler_x = ss()

scaler_y = ss()

scaler_y.fit(y.values)

scaler_x.fit(x.values)

y_s = scaler_y.transform(y)

x_s = scaler_x.transform(x)
xn_train, xn_test, yn_train, yn_test= split(x_s,y_s, test_size=0.2, random_state=72)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def dnn_log():

    model = Sequential()

    model.add(Dense(128, input_dim = x.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    model.compile(loss='mean_squared_error', optimizer=adam,metrics=[r2_keras])

    return model

dnn_log().fit(xn_train,yn_train,epochs=5, batch_size=8,validation_data=[xn_test, yn_test])

print("NEURAL NETWORK")