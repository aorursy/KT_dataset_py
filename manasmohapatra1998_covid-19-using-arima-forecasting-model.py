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
!pip install pyramid-arima
import pyramid.arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

train.head()
test.head()
confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})
fig, ax = plt.subplots(1, figsize=(17,7))
confirmed_total_date.plot(ax=ax, color='orange')
fatalities_total_date.plot(ax=ax,  color='red')
ax.set_title("Cases and fatalities", size=13)
ax.set_ylabel("Number of cases/fatalities", size=13)
ax.set_xlabel("Date", size=13)
no_countr = train['Country_Region'].nunique()
no_province = train['Province_State'].nunique()
no_countr_with_prov = len(train[train['Province_State'].isna()==False]['Country_Region'].unique())
total_forecasting_number = no_province + no_countr - no_countr_with_prov+2
no_days = train['Date'].nunique()
print('there are ', no_countr, 'unique Countries_Region, each with ', no_days, 'days of data, all of them having the same dates. There are also ',no_province, 'Provinces_States which can be found on ',
no_countr_with_prov, 'countries_region.' )
plt.plot([i for i in range(no_days)], train['ConfirmedCases'].iloc[[i for i in range(0,no_days)]].values)
plt.xlabel('No. of days since 2020-01-22')
plt.ylabel('Cases')
plt.title('Plotting cases for Afghanistan')
plt.show()
df = confirmed_total_date.copy()
df = pd.DataFrame({'date': [df.index[i] for i in range(len(df))] , 'cases': df['ConfirmedCases'].values.reshape(1,-1)[0].tolist()})
dfog = df.copy()
def l_regr(x,y):
    model = LinearRegression().fit(x, y)
    return model

x = df['cases']
x = x.drop(x.index[-1]).values.reshape((-1, 1))
y = df['cases']
y = y.drop(y.index[0])
ex_slope = l_regr(x,y).coef_

d = 0

for i in range(1,5):
    plt.plot(df['cases'])
    plt.show()
    plt.close()
    df['prev_cases'] = df['cases'].shift(1)
    df['cases'] = (df['cases'] - df['prev_cases'])
    df = df.drop(['prev_cases'],axis=1)
    df = df.drop(df.index[0])
    x = df['cases']
    x = x.drop(y.index[-1]).values.reshape((-1, 1))
    y = df['cases']
    y = y.drop(y.index[0])
    model = l_regr(x,y)
    if( abs(model.coef_) > ex_slope):
        print('this is it! ', ex_slope)
        break
    d += 1
    ex_slope = model.coef_
    print(model.coef_)

index = 1
cases_pred= []
fatalities_pred = []
pbar = tqdm(total=total_forecasting_number)
while index < total_forecasting_number+1:
    x = train['ConfirmedCases'].iloc[[i for i in range(no_days*(index-1),no_days*index)]].values
    z = train['Fatalities'].iloc[[i for i in range(no_days*(index-1),no_days*index)]].values
    
    index += 1
    
    no_nul_cases = pd.DataFrame(x)
    no_nul_cases = no_nul_cases[no_nul_cases.values != 0]
    if(not no_nul_cases.empty):
        X = [xi for xi in no_nul_cases.values]
        try:
            model = pyramid.arima.auto_arima(X,seasonal=True, m=12)
            pred = model.predict(31)
            pred = pred.astype(int)
            pred = pred.tolist()
        except:
            model = l_regr(np.array([i for i in range(len(X))]).reshape(-1, 1),X)
            pred = [(model.coef_*(len(X)+i) + model.intercept_).astype('int')[0][0] for i in range(1,32)]
                
    else:
        pred = [0] * 31
    pred = x[-12:].astype(int).tolist() + pred
    cases_pred+=pred
    
    no_nul_fatalities = pd.DataFrame(z)
    no_nul_fatalities = no_nul_fatalities[no_nul_fatalities.values != 0]
    if(not no_nul_fatalities.empty):
        Z = [zi for zi in no_nul_fatalities.values]
        try:
            model = pyramid.arima.auto_arima(Z, seasonal=False, m=12)
            pred = model.predict(31)
            pred = pred.astype(int)
            pred = pred.tolist()
        except:
            model = l_regr(np.array([i for i in range(len(Z))]).reshape(-1, 1),Z)
            pred = [(model.coef_*(len(Z)+i) + model.intercept_).astype('int')[0][0] for i in range(1,32)]
    else:
        pred = [0] * 31
    pred = z[-12:].astype(int).tolist() + pred
    fatalities_pred+=pred
    pbar.update(1)
pbar.close()
submission = pd.DataFrame({'ForecastId': [i for i in range(1,len(cases_pred)+1)] ,'ConfirmedCases': cases_pred, 'Fatalities': fatalities_pred})
filename = 'submission.csv'
submission.to_csv(filename,index=False)
