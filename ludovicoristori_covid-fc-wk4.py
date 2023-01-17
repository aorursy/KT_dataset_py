import numpy as np

import pandas as pd

import datetime as datetime

from sklearn import preprocessing
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib.pyplot as plt
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
# For the examples I take my country

my_country='Italy'

my_variable='ConfirmedCases'
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
df_union.head()
def cut_neg(y):

    m = max(0,np.amin(y))

    for i in range(0,len(y)):

        if y[i]<m : y[i]=m

    return y
def linear_trend(X_train,y_true,X_test):

    lrm = LinearRegression()

    lrm.fit(X_train,y_true)

    y_valid = lrm.predict(X_train)

    y_pred  = lrm.predict(X_test)

    y_valid = cut_neg(y_valid)

    y_pred  = cut_neg(y_pred)

    return y_valid,y_pred
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

md1 = RandomForestRegressor(random_state=1234) #XGBRegressor(n_estimators=2000, random_state=1234) 

md2 = RandomForestRegressor(random_state=1234) #XGBRegressor(n_estimators=1000, random_state=1234)

for country in df_train['Country_Region'].unique():

    df_train_cy = df_train[df_train['Country_Region']==country].copy()

    df_test_cy  = df_test[df_test['Country_Region']==country].copy()

    for province in df_train_cy['Province_State'].unique():

        print(f'Analizing > country = {country} : province = {province}')

        df_train_pr = df_train_cy[df_train_cy['Province_State']==province].copy()

        df_test_pr  = df_test_cy[df_test_cy['Province_State']==province].copy()

        X_train_pr  = df_train_pr[['DateOrd']]

        y1_train_pr = df_train_pr['ConfirmedCases']

        y2_train_pr = df_train_pr['Fatalities']

        df_test_pr  = df_test_pr[df_test_pr['Province_State']==province].copy()

        X_test_pr   = df_test_pr[['DateOrd']]

        # trend

        y1_check_pr_trend, y1_pred_pr_trend = polynomial_trend(NPT,X_train_pr,y1_train_pr,X_test_pr)

        y2_check_pr_trend, y2_pred_pr_trend = polynomial_trend(NPT,X_train_pr,y2_train_pr,X_test_pr)

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
from sklearn.metrics import mean_squared_log_error
def plotExample():

    train_cond = ((df_train['Country_Region']==my_country) & (df_train['Province_State']==my_province))

    test_cond  = ((df_test['Country_Region']==my_country) & (df_test['Province_State']==my_province))

    x_test_plt  = df_test[test_cond]['Date']

    y_test_plt  = df_test[test_cond][my_variable]

    x_train_plt = df_train[train_cond]['Date']

    y_train_plt = df_train[train_cond][my_variable]

    y_valid_plt = df_train[train_cond][my_variable+'Valid']



    plt.rcParams["figure.figsize"] = (12,6)

    fig, ax = plt.subplots()

    ax.plot(x_train_plt,y_train_plt,'o')

    ax.plot(x_train_plt,y_valid_plt,'x')

    ax.plot(x_test_plt,y_test_plt,'*')

    ax.set_xticks([])
my_country='Italy'

my_province='ND'

my_variable='ConfirmedCases'

plotExample()
print_columns = ['Date','ConfirmedCases','Fatalities','ConfirmedCasesTrend','FatalitiesTrend','ConfirmedCasesResid','FatalitiesResid']

df_test[df_test['Country_Region']==my_country][print_columns].tail()
y1_valid = cut_neg(df_train['ConfirmedCasesValid'].copy())

y2_valid = cut_neg(df_train['FatalitiesValid'].copy())

y1_pred  = cut_neg(df_test['ConfirmedCases'].copy())

y2_pred  = cut_neg(df_test['Fatalities'].copy())
y1_valid.describe()
y2_valid.describe()
y1_pred.describe()
y2_pred.describe()
mse = (mean_squared_log_error(y1_train, y1_valid)+mean_squared_log_error(y2_train, y2_valid))/2

mse
submission = pd.DataFrame({'ForecastId': df_test['ForecastId'],'ConfirmedCases': y1_pred,'Fatalities': y2_pred})

submission.head()
submission.to_csv('submission.csv',index = False)