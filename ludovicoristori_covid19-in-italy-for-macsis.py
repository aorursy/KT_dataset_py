import numpy as np
import pandas as pd
import datetime as datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
from random import random
from datetime import date
from datetime import datetime
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(precision=3)
import os
os.listdir("../input/")
path = "../input/covid19-in-italy/"
df_regions = pd.read_csv(path+'covid19_italy_region.csv',decimal=",")
df_regions.head()
cols = ['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement',
'CurrentPositiveCases','NewPositiveCases','Recovered','Deaths','TotalPositiveCases','TestsPerformed']
df_italy=df_regions.groupby(by=['Date'], as_index=False)[cols].sum()
df_italy.head()
denominazione_regione = 'RegionName'
codice_regione = 'RegionCode'
campo_data = 'Date'
variabile = 'TotalPositiveCases'
df_regions[denominazione_regione].unique()
df_trentino = df_regions[df_regions[codice_regione]==4].copy()
df_trentino.columns
df_trentino = df_trentino.groupby(by=[codice_regione,campo_data],as_index=False).sum()
df_trentino[denominazione_regione]='Trentino Alto-Adige'
df_regions = df_regions[df_regions[codice_regione]!=4]
df_regions = pd.concat([df_regions,df_trentino],axis=0)
df_regions.reset_index()
def stringToDatetime(col, fmt):
    return pd.to_datetime(col, format=fmt)
def dfNorm(df):
    df['timestp']=stringToDatetime(df[campo_data],'%Y-%m-%d %H:%M:%S')
    df['timestp']=df['timestp'].dt.normalize()
    print('min date = ',min(df['timestp']),'max date = ',max(df['timestp']))
    return df
df_italy = dfNorm(df_italy)
df_regions = dfNorm(df_regions)
y_true  = df_italy[variabile]
#y_true = df_regions[df_regions[denominazione_regione]=='Lombardia'][variabile]
y_true = y_true.values
y_true
N_train = len(y_true)
N_train
N_test = 15
X_train = np.arange(0,N_train).reshape(-1, 1)
X_train
X_test = np.arange(N_train,N_train+N_test+1).reshape(-1, 1)
X_test
lrm = LinearRegression()
lrm.fit(X_train,y_true)
y_lrm_val = lrm.predict(X_train)
y_lrm_pre = lrm.predict(X_test)
def plotGraph(X_train, y_true, y_valid, X_test, y_pred):
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(X_train, y_true, color='blue')
    plt.plot(X_train, y_valid, color='red',linestyle='dashed')
    plt.plot(X_test, y_pred, color='red')
    x_conn = [X_train[len(X_train)-1],X_test[0]]
    y_conn = [y_true[len(X_train)-1],y_pred[0]]
    plt.plot(x_conn, y_conn, color='gray')
    y_conv = [y_valid[len(X_train)-1],y_pred[0]]
    plt.plot(x_conn, y_conv, color='gray',linestyle='dashed')
plotGraph(X_train, y_true, y_lrm_val, X_test, y_lrm_pre)
regs = df_regions[denominazione_regione].unique()
nc = 3
nr = int(len(regs)/nc)+1
fig, ax = plt.subplots(nrows=nr,ncols=nc,figsize=(18,30))
i = 0
j = 0
while i<nr:
    for j in range(nc):
        k = nc*i+j
        if (k<len(regs)):
            reg = regs[k]
            y_true_reg = df_regions[df_regions[denominazione_regione]==reg][variabile].values
            lrr = LinearRegression()
            lrr.fit(X_train,y_true_reg)
            y_lrr_val = lrr.predict(X_train)
            y_lrr_pre = lrr.predict(X_test)
            ax[i,j].plot(X_train, y_true_reg, color='blue')
            ax[i,j].plot(X_train, y_lrr_val, color='red', linestyle='dashed')
            ax[i,j].plot(X_test, y_lrr_pre, color='red')
            ax[i,j].set_title(reg)            
    i = i + 1
# the time practically is ignored in further steps, it's just useful for plots
t_all = np.append(X_train,X_test)
t_all
y_all = np.append(y_true,np.zeros(N_test+1))
y_all
N_trend = 10
if (N_trend>N_test) :
    print('N_trend too large !!!')
X_all = pd.DataFrame()
for k in np.arange(N_test+1,N_trend+N_test+1,1):
    X_all['y_'+str(k).zfill(2)] = pd.Series(np.roll(y_all,k).flatten())
ord_cols = X_all.columns.sort_values(ascending=False)
X_all = X_all[ord_cols]
X_all
X_all[0:N_trend] = -1
X_all['y_tr'] = y_all
X_all
X_train = X_all[N_test+1:N_train]
X_train
X_test = X_all[N_train:]
X_test
tr_cols = [c for c in X_train.columns if c not in ['y_tr']]
X_tr = X_train[tr_cols]
X_tr
y_tr = X_train['y_tr']
y_tr
X_te = X_test[tr_cols]
X_te
def plotGraph2(t_all, y_all, y_valid, t_test, y_pred):
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(t_all[0:N_train], y_all[0:N_train], color='blue')
    plt.plot(t_all[0:N_train], y_valid, color='red',linestyle='dashed')
    plt.plot(t_test, y_pred, color='red')
    x_conn = [t_all[N_train-1],t_test[0]]
    y_conn = [y_all[N_train-1],y_pred[0]]
    plt.plot(x_conn, y_conn, color='gray')
    y_conv = [y_valid[N_train-1],y_pred[0]]
    plt.plot(x_conn, y_conv, color='gray',linestyle='dashed')
t_test = t_all[len(t_all)-N_test-1:]
t_test
md1 = LinearRegression()
md1.fit(X_tr,y_tr)
y_val = md1.predict(X_tr)
y_val
y_pre = md1.predict(X_te)
y_pre
y_valid = np.append(y_all[0:N_test+1],y_val)
y_valid
plotGraph2(t_all, y_all, y_valid, t_test, y_pre)
pd.Series(y_val).describe()