import numpy as np
import pandas as pd
import datetime as datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from random import random
from datetime import date
from datetime import datetime
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(precision=3)
# -----------------------------
# settings for local use - 1/2
# -----------------------------
# path = "c:\\temp\\"
# df_italy = pd.read_csv(path+'dpc-covid19-ita-andamento-nazionale.csv',decimal=",")
# df_italy.head()
# -----------------------------
# settings for local use - 2/2
# -----------------------------
# df_regions = pd.read_csv(path+'dpc-covid19-ita-regioni.csv',decimal=",")
# df_regions.head()
# path_reg = path
# ------------------------------
# settings for Kaggle use - 1/2
# ------------------------------
path = "../input/covid19-in-italy/"
df_regions = pd.read_csv(path+'covid19_italy_region.csv',decimal=",")
df_regions.head()
path_reg = "../input/italian-regions/"
# ------------------------------
# settings for Kaggle use - 2/2
# ------------------------------
cols = ['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement',
'CurrentPositiveCases','NewPositiveCases','Recovered','Deaths','TotalPositiveCases','TestsPerformed']
df_italy=df_regions.groupby(by=['Date'], as_index=False)[cols].sum()
df_italy.head()
denominazione_regione = 'RegionName'
codice_regione = 'RegionCode'
campo_data = 'Date'
variabile = 'TotalPositiveCases'
df_regions_fix_data=pd.read_csv(path_reg+'ita_reg_ann_data.csv',decimal=".")
df_regions_fix_data.head(10)
df_regions_monthly_data=pd.read_csv(path_reg+'ita_reg_mens_clima.csv',decimal=".")
df_regions_monthly_data.head()
df_regions[denominazione_regione].unique()
df_trentino = df_regions[df_regions[codice_regione]==4].copy()
df_trentino.columns
df_trentino = df_trentino.groupby(by=[codice_regione,campo_data],as_index=False).sum()
df_trentino[denominazione_regione]='Trentino Alto-Adige'
df_regions = df_regions[df_regions[codice_regione]!=4]
df_regions = pd.concat([df_regions,df_trentino],axis=0)
df_regions.columns
# remove Country, Latidude, Longitude and TestsPerformed to kill Nan values
df_regions = df_regions[['Date', 'RegionCode', 'RegionName', 'HospitalizedPatients', 'IntensiveCarePatients',
       'TotalHospitalizedPatients', 'HomeConfinement', 'CurrentPositiveCases',
       'NewPositiveCases', 'Recovered', 'Deaths', 'TotalPositiveCases']]
df_regions.reset_index(drop=True)
df_regions.info()
def stringToDatetime(col, fmt):
    return pd.to_datetime(col, format=fmt)
def dfNorm(df):
    df['timestp'] = stringToDatetime(df[campo_data],'%Y-%m-%d %H:%M:%S')
    df['timestp'] = df['timestp'].dt.normalize()
    df['month']   = df['timestp'].apply(lambda s : s.month)
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
N_test = 10
X_train = np.arange(0,N_train).reshape(-1, 1)
X_train
X_test = np.arange(N_train,N_train+N_test+1).reshape(-1, 1)
X_test
def doLinearRegression(X_train,y_true,X_test):
    lrm = LinearRegression()
    lrm.fit(X_train,y_true)
    y_lrm_val = lrm.predict(X_train)
    y_lrm_pre = lrm.predict(X_test)
    y_lrm_res = y_true - y_lrm_val
    return y_lrm_val, y_lrm_pre, y_lrm_res  
y_lrm_val, y_lrm_pre, y_lrm_res = doLinearRegression(X_train,y_true,X_test)
sns.lineplot(x=X_train.flatten(),y=y_lrm_res)
sns.distplot(y_lrm_res,kde=True)
pd.Series(y_lrm_res).describe()
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
p = 1
d = 1
q = 1
def doARIMA(X_train,y_true,X_test):
    N_train = len(X_train)
    N_test  = len(X_test)
    arm = SARIMAX(y_true, order=(p,d,q), enforce_stationarity=False)
    arm = arm.fit()
    y_arm_pre = y_true[N_train-1]+arm.predict(start=0,end=N_test-1)
    y_arm_res = arm.resid
    y_arm_val = y_true - y_arm_res
    return y_arm_val, y_arm_pre, y_arm_res  
y_arm_val, y_arm_pre, y_arm_res = doARIMA(X_train,y_true,X_test)
sns.lineplot(x=X_train.flatten(),y=y_arm_res)
sns.distplot(y_arm_res,kde=True)
pd.Series(y_arm_res).describe()
plotGraph(X_train, y_true, y_arm_val, X_test, y_arm_pre)
def plotARIMARegions():
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
                y_true_cyc = df_regions[df_regions[denominazione_regione]==reg][variabile].values
                y_val, y_pre, y_res = doARIMA(X_train,y_true_cyc,X_test)
                ax[i,j].plot(X_train, y_true_cyc, color='blue')
                ax[i,j].plot(X_train, y_val, color='red', linestyle='dashed')
                ax[i,j].plot(X_test, y_pre, color='red')
                ax[i,j].set_title(reg)            
        i = i + 1
plotARIMARegions()
t_all = np.append(X_train,X_test)
t_all
def to_supervised(y_true,N_train,N_test,N_trend):
    y_all = np.append(y_true,np.zeros(N_test+1))
    if (N_trend>N_train) :
        print('N_trend too large !!!')
        X_tr = None
        y_tr = None
        X_te = None
    else :
        X_all = pd.DataFrame()
        for k in np.arange(N_test+1,N_trend+N_test+1,1):
            X_all['y_'+str(k).zfill(2)] = pd.Series(np.roll(y_all,k).flatten())
        ord_cols = X_all.columns.sort_values(ascending=False)
        X_all = X_all[ord_cols]
        X_all[0:N_trend] = -1
        X_all['y_tr'] = y_all
        X_train_red = X_all[N_test+1:N_train]
        X_test = X_all[N_train:]
        tr_cols = [c for c in X_train_red.columns if c not in ['y_tr']]
        X_tr = X_train_red[tr_cols]
        y_tr = X_train_red['y_tr']
        X_te = X_test[tr_cols]
    return X_tr,y_tr,X_te,y_all
N_trend = 10
X_tr,y_tr,X_te,y_all = to_supervised(y_true,N_train,N_test,N_trend)
X_tr
y_tr
X_te
y_all
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
y_lrm_val, y_lrm_pre, y_lrm_res = doLinearRegression(X_tr,y_tr,X_te)
y_lrm_plt = np.concatenate([np.zeros(len(t_test)),y_lrm_val])
plotGraph2(t_all, y_all, y_lrm_plt, t_test, y_lrm_pre)
sns.lineplot(x=X_train[N_test+1:].flatten(),y=y_lrm_res)
sns.distplot(y_lrm_res,kde=True)
pd.Series(y_lrm_res).describe()
X_tr
df_regions_tr = df_regions.merge(df_regions_fix_data,left_on=codice_regione,right_on='cod_reg')
df_regions_tr.columns
cols = ['RegionCode', variabile, 'timestp', 'month', 'gdp_tot', 'gdp_procap', 'pop_resid', 'superf_kmq', 'dens_ab',
       'num_com', 'num_prov', 'num_teams']
df_regions_tr = df_regions_tr[cols]
df_regions_tr.info()
df_regions_te = pd.DataFrame(columns=[codice_regione,'timestp','month',variabile])
regs = df_regions[codice_regione].unique()
date_start = df_regions['timestp'].min()
for reg in regs:
    print('region = ',reg)
    for xt in X_test:
        date_row = date_start + timedelta(days=int(xt))
        print(date_row)
        df_regions_te=df_regions_te.append({codice_regione: reg,
                                            'timestp': date_row, 
                                            'month': date_row.month, 
                                             variabile:0}, ignore_index=True)
df_regions_te = df_regions_te.merge(df_regions_fix_data,left_on=codice_regione,right_on='cod_reg')
df_regions_te['month']=df_regions_te['month'].apply(int)
df_regions_te[codice_regione]=df_regions_te[codice_regione].apply(int)
df_regions_te[variabile] = -1
df_regions_te = df_regions_te[cols]
df_regions_te.info()
df_regions_te
df_regions_tr = df_regions_tr.merge(df_regions_monthly_data,left_on=[codice_regione,'month'],right_on=['cod_reg','month'])
df_regions_tr.columns
df_regions_te = df_regions_te.merge(df_regions_monthly_data,left_on=[codice_regione,'month'],right_on=['cod_reg','month'])
df_regions_te.columns
X_all = pd.concat([df_regions_tr,df_regions_te],axis=0)
X_all = X_all.sort_values(by=['RegionCode','timestp']).reset_index(drop=True)
X_all.tail()
for k in np.arange(N_test+1,N_trend+N_test+1,1):
    X_all['y_'+str(k).zfill(2)] = 0
X_all.info()
regs = X_all[codice_regione].unique()
X_tr = pd.DataFrame()
for reg in regs:
    y_all_reg = X_all[X_all[codice_regione]==reg][variabile].copy()
    print('region = ',reg, np.mean(y_all_reg[0:N_train-N_test-1]))
    for k in np.arange(N_test+1,N_trend+N_test+1,1):
        X_all['y_'+str(k).zfill(2)][X_all[codice_regione]==reg] = np.roll(y_all_reg,k)
ord_cols = X_all.columns.sort_values(ascending=False)
X_all = X_all[ord_cols]
X_all.info()
def foundMinusOne(row):
    found = False
    for k in np.arange(N_test+1,N_trend+N_test+1,1):
        if row['y_'+str(k).zfill(2)]==-1 :
            found = found | True
    return found
X_all_clean = X_all[X_all.apply(foundMinusOne,axis=1)==False]
X_all_clean
X_all_clean['RegionCode'].value_counts()
X_all_clean[X_all_clean['RegionCode']==8]
X_train = X_all_clean[X_all_clean[variabile]!=-1]
y_tr    = X_train[variabile].copy()
X_tr    = X_train.drop([variabile,'timestp','month'],axis=1)
X_tr
X_test  = X_all_clean[X_all_clean[variabile]==-1]
X_te    = X_test.drop([variabile,'timestp','month'],axis=1)
X_tr.info()
X_te.info()
y_lrm_val, y_lrm_pre, y_lrm_res = doLinearRegression(X_tr,y_tr,X_te)
X_te = X_te.reset_index()
sns.distplot(y_lrm_res,kde=True)
pd.Series(y_lrm_res).describe()
def rebTrainTest(X_tr,y_tr,X_te,y_pr):
    df_train = pd.concat([X_tr,pd.Series(y_tr,name='y')],axis=1).copy()
    df_test = pd.concat([X_te,pd.Series(y_pr,name='y')],axis=1).copy()
    return df_train,df_test
df_train,df_test = rebTrainTest(X_tr,y_tr,X_te,y_lrm_pre)
df_train.head()
valid = df_train[df_train['RegionCode']==1]['y']
valid.index = np.arange(N_trend,N_train-N_test)
valid
prev = df_test[df_test['RegionCode']==1]['y']
prev.index = np.arange(N_train-N_test,N_train+1)
prev
plt.plot(valid)
plt.plot(prev)
def plotRegions(X_tr,y_tr,X_te,y_xgb_pre):
    df_tr,df_te = rebTrainTest(X_tr,y_tr,X_te,y_lrm_pre)
    regs = df_tr[codice_regione].unique()
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
                y_ac_values = df_tr[df_tr[codice_regione]==reg]['y']
                y_ac_index  = np.arange(N_trend,N_train-N_test) 
                y_pr_values = df_te[df_te[codice_regione]==reg]['y']
                y_pr_index  = np.arange(N_train-N_test,N_train+1)
                ax[i,j].plot(y_ac_index,y_ac_values, color='blue')
                ax[i,j].plot(y_pr_index,y_pr_values, color='red')
                ax[i,j].set_title(reg)            
        i = i + 1
plotRegions(X_tr,y_tr,X_te,y_lrm_pre)
def doXGB(X_train,y_true,X_test):
    xgb = XGBRegressor()
    xgb.fit(X_train,y_true)
    y_xgb_val = xgb.predict(X_train)
    y_xgb_pre = xgb.predict(X_test)
    y_xgb_res = y_true - y_xgb_val
    feat_imp = pd.DataFrame({'Feature':X_train.columns,'Importance':xgb.feature_importances_,})
    print(feat_imp)
    return y_xgb_val, y_xgb_pre, y_xgb_res 
X_te = X_te.drop(['index'],axis=1)
X_te.info()
y_xgb_val, y_xgb_pre, y_xgb_res = doXGB(X_tr,y_tr,X_te)
sns.distplot(y_xgb_res,kde=True)
pd.Series(y_xgb_res).describe()
plotRegions(X_tr,y_tr,X_te,y_xgb_pre)
#learning_rate=0.001,max_depth=3,num_rounds=3000
def doLGBM(X_train,y_true,X_test):
    lgr = lgb.LGBMRegressor()
    lgr.fit(X_train,y_true)
    y_lgb_val = lgr.predict(X_train)
    y_lgb_pre = lgr.predict(X_test)
    y_lgb_res = y_true - y_lgb_val
    feat_imp = pd.DataFrame({'Feature':X_train.columns,'Importance':lgr.feature_importances_,})
    print(feat_imp)
    return y_lgb_val, y_lgb_pre, y_lgb_res 
y_lgb_val, y_lgb_pre, y_lgb_res = doLGBM(X_tr,y_tr,X_te)
y_lgb_plt = np.concatenate([np.zeros(len(t_test)),y_lgb_val])
sns.distplot(y_lgb_res,kde=True)
pd.Series(y_lgb_res).describe()
plotRegions(X_tr,y_tr,X_te,y_lgb_pre)