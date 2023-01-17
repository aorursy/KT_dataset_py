import numpy as np

import pandas as pd

import datetime as datetime

from sklearn import preprocessing
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
training_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

training_set.shape
training_set.tail()
test_set = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test_set.shape
test_set.head()
my_country = 'Italy'

training_set[training_set['Country_Region']==my_country].tail()
N_train=len(training_set[training_set['Country_Region']==my_country])

N_train
N_test=len(test_set[test_set['Country_Region']==my_country])

N_test
df_train = training_set.copy()

df_train['DsType']='train'

df_train.rename({'Id': 'ForecastId'}, axis=1, inplace=True)

df_train.info()
df_test  = test_set.copy()

df_test['DsType']='test'

df_test['ConfirmedCases']=0

df_test['Fatalities']=0

df_test.info()
df_union=pd.concat([df_train,df_test],sort=False).copy()

df_union.fillna('ND', inplace = True)
df_union['Month']=df_union['Date'].apply(lambda s : int(s.replace('-','')[4:6]))

df_union['Day']=df_union['Date'].apply(lambda s : int(s.replace('-','')[6:9]))

df_union['Date']=df_union['Date'].apply(lambda s : datetime.datetime.strptime(s, '%Y-%m-%d'))

df_union['DateOrd']=df_union['Date'].apply(lambda s : s.toordinal())

df_union['Province_Norm']=df_union['Country_Region']+'-'+df_union['Province_State']

df_union.tail()
le1 = preprocessing.LabelEncoder()

le1.fit(df_union['Country_Region'])

df_union['Country'] = le1.transform(df_union['Country_Region'])
le2 = preprocessing.LabelEncoder()

le2.fit(df_union['Province_Norm'])

df_union['Province'] = le2.transform(df_union['Province_Norm'])

df_union.drop('Province_Norm',axis=1,inplace=True)
t_max_tr = max(df_union['Date'][df_union['DsType']=='train'])

t_min_te = min(df_union['Date'][df_union['DsType']=='test'])

Noverlap=(t_max_tr-t_min_te).days+1

Noverlap
df_union.head()
def cut_neg(y):

    m = 0

    for i in range(0,len(y)):

        if y[i]<m : y[i]=m

    return y
def polynomial_trend(deg,X_train,y_true,X_test):

    pf = PolynomialFeatures(degree=deg)

    pr = pf.fit_transform(X_train)

    lrm = LinearRegression()

    lrm.fit(pr, y_true)

    y_valid = lrm.predict(pf.fit_transform(X_train))

    y_pred  = lrm.predict(pf.fit_transform(X_test))

    y_valid = cut_neg(y_valid)

    y_pred  = cut_neg(y_pred)

    return y_valid,y_pred
Nlim = 5

delta_max = 0.1

def calc_trends(X_train,y_true,X_test,deg):

    pct_hist = 100*sum(1 for x in y_true if x > 0)/N_train

    y=np.array(y_true).flatten()

    y_last  = y[len(y)-1]

    y_check = y[len(y)-1-Nlim]

    if (y_check==0) :

        den = 1 

    else :

        den = y_check

    diff = np.abs(y_last-y_check)/den

    lim  = np.mean(y_true[len(y_true)-Nlim:len(y_true)])

    if (diff<=delta_max):

        y_valid = np.full([len(y_true)],lim)

        y_pred  = np.full([len(X_test)],lim)

        msg1    = 'cost'

        msg2    = 'cost'

    if (diff>delta_max):

        y_valid,y_pred = polynomial_trend(deg,X_train,y_true,X_test)

        msg1    = 'poly'

        msg2    = 'poly' 

    return y_valid, y_pred, msg1, msg2
df_train = df_union[df_union['DsType']=='train'].drop('DsType',axis=1)

df_test  = df_union[df_union['DsType']=='test'].drop('DsType',axis=1)

df_train['ConfirmedCasesValid'] = 0

df_train['FatalitiesValid'] = 0

df_train['ConfirmedCasesTrend'] = 0

df_train['FatalitiesTrend'] = 0

df_train['ConfirmedCasesResid'] = 0

df_train['FatalitiesResid'] = 0

df_test['ConfirmedCases'] = 0

df_test['Fatalities'] = 0

df_test['ConfirmedCasesTrend'] = 0

df_test['FatalitiesTrend'] = 0

df_test['ConfirmedCasesResid'] = 0

df_test['FatalitiesResid'] = 0
y1_train = df_train['ConfirmedCases'].astype(float)

y2_train = df_train['Fatalities'].astype(float)
NPT = 4

md1 = XGBRegressor(n_estimators=2000, random_state=1234) # RandomForestRegressor(random_state=1234) 

md2 = XGBRegressor(n_estimators=1000, random_state=1234) # RandomForestRegressor(random_state=1234) 

for country in df_train['Country_Region'].unique():

    df_train_cy = df_train[df_train['Country_Region']==country].copy()

    df_test_cy  = df_test[df_test['Country_Region']==country].copy()

    for province in df_train_cy['Province_State'].unique():

        df_train_pr = df_train_cy[df_train_cy['Province_State']==province].copy()

        df_test_pr  = df_test_cy[df_test_cy['Province_State']==province].copy()

        X_train_pr  = df_train_pr[['DateOrd']]

        y1_train_pr = df_train_pr['ConfirmedCases']

        y2_train_pr = df_train_pr['Fatalities']

        df_test_pr  = df_test_pr[df_test_pr['Province_State']==province].copy()

        X_test_pr   = df_test_pr[['DateOrd']]

        # trend

        X_train_pr1  = X_train_pr[len(X_train_pr)-Noverlap:len(X_train_pr)]

        y1_train_pr1 = y1_train_pr[len(y1_train_pr)-Noverlap:len(y1_train_pr)]

        y2_train_pr1 = y2_train_pr[len(y2_train_pr)-Noverlap:len(y2_train_pr)]

        y1_check_pr_trend, y1_pred_pr_trend, msg1, msg2 = calc_trends(X_train_pr1,y1_train_pr1,X_test_pr,NPT)

        y2_check_pr_trend, y2_pred_pr_trend, msg3, msg4 = calc_trends(X_train_pr1,y2_train_pr1,X_test_pr,NPT)

        y1_check_pr_trend = np.append(np.zeros(len(df_train_pr)-Noverlap),y1_check_pr_trend)

        y2_check_pr_trend = np.append(np.zeros(len(df_train_pr)-Noverlap),y2_check_pr_trend)

        df_train.loc[((df_train['Country_Region']==country) & (df_train['Province_State']==province)),'ConfirmedCasesTrend'] = y1_check_pr_trend

        df_train.loc[((df_train['Country_Region']==country) & (df_train['Province_State']==province)),'FatalitiesTrend'] = y2_check_pr_trend

        df_test.loc[((df_test['Country_Region']==country) & (df_test['Province_State']==province)),'ConfirmedCasesTrend'] = y1_pred_pr_trend

        df_test.loc[((df_test['Country_Region']==country) & (df_test['Province_State']==province)),'FatalitiesTrend'] = y2_pred_pr_trend

        # residuals

        y1_train_pr_resid = y1_train_pr - y1_check_pr_trend

        y2_train_pr_resid = y2_train_pr - y2_check_pr_trend

        md1.fit(X_train_pr,y1_train_pr_resid)

        md2.fit(X_train_pr,y2_train_pr_resid)

        y1_check_pr_resid = md1.predict(X_train_pr)

        y2_check_pr_resid = md2.predict(X_train_pr)

        y1_pred_pr_resid  = md1.predict(X_test_pr)

        y2_pred_pr_resid  = md2.predict(X_test_pr)

        df_train.loc[((df_train['Country_Region']==country) & (df_train['Province_State']==province)),'ConfirmedCasesResid'] = y1_train_pr_resid

        df_train.loc[((df_train['Country_Region']==country) & (df_train['Province_State']==province)),'FatalitiesResid'] = y2_train_pr_resid

        df_test.loc[((df_test['Country_Region']==country) & (df_test['Province_State']==province)),'ConfirmedCasesResid'] = y1_pred_pr_resid

        df_test.loc[((df_test['Country_Region']==country) & (df_test['Province_State']==province)),'FatalitiesResid'] = y2_pred_pr_resid

        # sum

        df_train.loc[((df_train['Country_Region']==country) & (df_train['Province_State']==province)),'ConfirmedCasesValid'] = y1_check_pr_trend + y1_check_pr_resid 

        df_train.loc[((df_train['Country_Region']==country) & (df_train['Province_State']==province)),'FatalitiesValid'] = y2_check_pr_trend + y2_check_pr_resid

        df_test.loc[((df_test['Country_Region']==country) & (df_test['Province_State']==province)),'ConfirmedCases'] = y1_pred_pr_trend + y1_pred_pr_resid

        df_test.loc[((df_test['Country_Region']==country) & (df_test['Province_State']==province)),'Fatalities'] = y2_pred_pr_trend + y2_pred_pr_resid

        print(f'Finished > country = {country} : province = {province} > used trends = [{msg1},{msg2},{msg3},{msg4}]')
def isPos(x):

    if (x>0):

        return 1

    else :

        return 0
df_train['ConfirmedCasesPos']=df_train['ConfirmedCases'].apply(isPos)

df_train['FatalitiesPos']=df_train['Fatalities'].apply(isPos)

df_test['ConfirmedCasesPos']=df_test['ConfirmedCases'].apply(isPos)

df_test['FatalitiesPos']=df_test['Fatalities'].apply(isPos)
sel_cols = ['Country_Region','Province_State','ConfirmedCasesPos','FatalitiesPos']

df_tr_cnt = df_train[sel_cols].groupby(by=['Country_Region','Province_State']).sum()

df_te_cnt = df_test[sel_cols].groupby(by=['Country_Region','Province_State']).sum()
sel_cols = ['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']

df_tr_last = df_train[df_train['Date']==max(df_train['Date'])][sel_cols]

df_te_last = df_test[df_test['Date']==max(df_test['Date'])][sel_cols]
df_tr_last = df_tr_last.merge(df_tr_cnt,on=['Country_Region','Province_State'])

df_te_last = df_te_last.merge(df_te_cnt,on=['Country_Region','Province_State'])
# this delta is different from the one used in trend prediction but they are related (both grows or )

df_last = df_tr_last.merge(df_te_last,on=['Country_Region','Province_State'])

df_last['DeltaConfirmedCases']=100*(df_last['ConfirmedCases_y']-df_last['ConfirmedCases_x'])/df_last['ConfirmedCases_x']

df_last['DeltaFatalities']=100*(df_last['Fatalities_y']-df_last['Fatalities_x'])/df_last['Fatalities_x']

df_last['DeltaConfirmedCases']=df_last['DeltaConfirmedCases'].apply(np.abs)

df_last['DeltaFatalities']=df_last['DeltaFatalities'].apply(np.abs)

df_last.fillna(0,inplace=True)

df_last = df_last.replace([np.inf, -np.inf], 0)
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(df_last['ConfirmedCasesPos_x'],kde=False,ax=ax)
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(df_last['FatalitiesPos_x'],kde=False,ax=ax,color='red')
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(df_last['DeltaConfirmedCases'],kde=False,ax=ax)
n_head = 5

d_lim1 = 120

d_lim2 = 200

sel_var   ='ConfirmedCases'

sel_cols  = ['Country_Region','Province_State',sel_var+'_x','Delta'+sel_var,sel_var+'Pos_x']

condition = (df_last['Delta'+sel_var]>d_lim1)&(df_last['Delta'+sel_var]<d_lim2)

df_last[condition][sel_cols].sort_values(by=['Delta'+sel_var], ascending=True).head(n_head)
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(df_last['DeltaFatalities'],kde=False,ax=ax,color='red')
n_head = 5

d_lim1 = 0

d_lim2 = 10

sel_var   = 'Fatalities'

sel_cols  = ['Country_Region','Province_State',sel_var+'_x','Delta'+sel_var,sel_var+'Pos_x']

condition = (df_last['Delta'+sel_var]>d_lim1)&(df_last['Delta'+sel_var]<d_lim2)

df_last[condition][sel_cols].sort_values(by=['Delta'+sel_var], ascending=True).head(n_head)
def plotExample():

    train_cond = ((df_train['Country_Region']==my_country) & (df_train['Province_State']==my_province))

    test_cond  = ((df_test['Country_Region']==my_country) & (df_test['Province_State']==my_province))

    x_train_plt = df_train[train_cond]['Date']

    y_train_plt = df_train[train_cond][my_variable]

    y_valid_plt = df_train[train_cond][my_variable+'Valid']

    y_trend_plt = df_train[train_cond][my_variable+'Trend']

    y_resid_plt = df_train[train_cond][my_variable+'Resid']

    x_test_plt  = df_test[test_cond]['Date']

    y_test_plt  = df_test[test_cond][my_variable]



    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()

    ax.plot(x_train_plt,y_train_plt,'o',color='orange', label='y_true')

    ax.plot(x_train_plt,y_valid_plt,'x',color='gray' , label ='y_valid')

    ax.plot(x_train_plt,y_trend_plt,'.',color='lightblue', label='y_trend')

    ax.plot(x_train_plt,y_resid_plt,'-',color='green', label='y_resid')

    ax.plot(x_test_plt,y_test_plt,'*',color='red', label='y_test')

    ax.set_xticks([])

    ax.legend()
# country with delta=low and history=high

my_country='China'

my_province='Hubei'

my_variable='ConfirmedCases'

plotExample()
sel_cols  = ['Country_Region','Province_State','ConfirmedCases_x','ConfirmedCases_y','DeltaConfirmedCases','ConfirmedCasesPos_x']

df_last[sel_cols][((df_last['Country_Region']==my_country) & (df_last['Province_State']==my_province))]
# country with delta=low and history=low

my_country='Brunei'

my_province='ND'

my_variable='ConfirmedCases'

plotExample()
sel_cols  = ['Country_Region','Province_State','ConfirmedCases_x','ConfirmedCases_y','DeltaConfirmedCases','ConfirmedCasesPos_x']

df_last[sel_cols][((df_last['Country_Region']==my_country) & (df_last['Province_State']==my_province))]
# country with delta=high and history=low

my_country='Italy'

my_province='ND'

my_variable='ConfirmedCases'

plotExample()
sel_cols  = ['Country_Region','Province_State','ConfirmedCases_x','ConfirmedCases_y','DeltaConfirmedCases','ConfirmedCasesPos_x']

df_last[sel_cols][((df_last['Country_Region']==my_country) & (df_last['Province_State']==my_province))]
# country with delta=high and history=high

my_country='US'

my_province='Iowa'

my_variable='ConfirmedCases'

plotExample()
sel_cols  = ['Country_Region','Province_State','ConfirmedCases_x','ConfirmedCases_y','DeltaConfirmedCases','ConfirmedCasesPos_x']

df_last[sel_cols][((df_last['Country_Region']==my_country) & (df_last['Province_State']==my_province))]
y1_valid = cut_neg(df_train['ConfirmedCasesValid'].copy())

y2_valid = cut_neg(df_train['FatalitiesValid'].copy())

y1_pred  = cut_neg(df_test['ConfirmedCases'].copy())

y2_pred  = cut_neg(df_test['Fatalities'].copy())
y1_valid.describe()
y2_valid.describe()
y1_pred.describe()
y2_pred.describe()
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(y1_train-y1_valid,kde=False,ax=ax)
fig, ax = plt.subplots(figsize=(8,6))

sns.distplot(y2_train-y2_valid,kde=False,ax=ax,color='red')
from sklearn.metrics import mean_squared_log_error

mse = (mean_squared_log_error(y1_train, y1_valid)+mean_squared_log_error(y2_train, y2_valid))/2

mse
submission = pd.DataFrame({'ForecastId': df_test['ForecastId'],'ConfirmedCases': y1_pred,'Fatalities': y2_pred})

submission.head()
submission.to_csv('submission.csv',index = False)