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
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
data.tail()
data.info()
data.set_index('Date',inplace=True)
data.index = pd.to_datetime(data.index)
data.asfreq = "D"
PositiveCase =  data[data['ConfirmedCases']>0]


PositiveCase['ConfirmedCases'].plot()


PositiveCase['Fatalities'].plot()
PositiveCase["ConfirmedCases"][0]
PositiveCase['FirstDerivative']= 0

for i in range(len(PositiveCase)):

    if i==0:

        PositiveCase["FirstDerivative"][i]= 0

    else:

        PositiveCase['FirstDerivative'][i]= (PositiveCase['ConfirmedCases'][i]-PositiveCase['ConfirmedCases'][0])/i

    
PositiveCase.head()
PositiveCase['FirstDerivative'].plot()
def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
Positiveseries = PositiveCase[['ConfirmedCases','Fatalities']]
Positiveseries.head()
!pip install pmdarima
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults

from statsmodels.tsa.stattools import adfuller

# from pmdarima import auto_arima

from statsmodels.tools.eval_measures import rmse
adf_test(Positiveseries['ConfirmedCases'],title="Confirmed Cases")
Positiveseries_tr = Positiveseries.diff()
Positiveseries_tred = Positiveseries_tr.dropna()

adf_test(Positiveseries_tred['ConfirmedCases'],title="Confirmed Cases")
Positive = Positiveseries_tred.diff()
Positived =  Positive.dropna()

adf_test(Positived['ConfirmedCases'],title="Confirmed Cases")
Positived1 = Positived.diff()
Positived1 =  Positived1.dropna()

adf_test(Positived1['ConfirmedCases'],title="Confirmed Cases")
Positived1.head()
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults

# Trying varius ARIMA model for 3 differencing level
ARIMA(1,3,1)
model = ARIMA(Positiveseries['ConfirmedCases'],order=(1,0,1))

results = model.fit()

results.summary()
Positiveseries['ConfirmedCasesDiff1']= Positiveseries['ConfirmedCases'].diff()
Positiveseries.head()
Positiveseries['Fatalitiesdiff1']= Positiveseries['Fatalities'].diff()

Positiveseries['ConfirmedCasesDiff2'] = Positiveseries['ConfirmedCasesDiff1'].diff()

Positiveseries['Fatalitiesdiff2']= Positiveseries['Fatalitiesdiff1'].diff()

Positiveseries['ConfirmedCasesDiff3'] = Positiveseries['ConfirmedCasesDiff2'].diff()

Positiveseries['Fatalitiesdiff3']= Positiveseries['Fatalitiesdiff2'].diff()
Positiveseries
Positiveseries.plot()
Positiveseries[['ConfirmedCasesDiff3','Fatalitiesdiff3']].plot()
#Adding rows for testing set

test = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv",index_col='Date',parse_dates=True)
test.head()
len(test)
test = test.loc['25-03-2020':]
for i in list(Positiveseries.columns):

    test[i] = 0
test.drop(['ForecastId','Province/State','Lat','Long'],inplace=True,axis=1)
test.drop('Country/Region',axis=1,inplace=True)
test.head()
Positiveseries = Positiveseries.append(test)
Positiveseries.tail()
Positiveseries
# Positiveseries.asfreq ="D"

for i in range(15,len(Positiveseries)):

    Positiveseries.iloc[i]['ConfirmedCasesDiff3'] = (Positiveseries.iloc[i-1]['ConfirmedCasesDiff3'] + Positiveseries.iloc[i-2]['ConfirmedCasesDiff3']+Positiveseries.iloc[i-3]['ConfirmedCasesDiff3']+Positiveseries.iloc[i-4]['ConfirmedCasesDiff3'])/4

    Positiveseries.iloc[i]['Fatalitiesdiff3'] = (Positiveseries.iloc[i-1]['Fatalitiesdiff3'] + Positiveseries.iloc[i-2]['Fatalitiesdiff3'])/2

    Positiveseries.iloc[i]['ConfirmedCasesDiff2'] = Positiveseries.iloc[i-1]['ConfirmedCasesDiff2']+ Positiveseries.iloc[i]['ConfirmedCasesDiff3']

# #     Positiveseries.iloc[i]['ConfirmedCasesDiff2'] = Positiveseries.iloc[i-1]['ConfirmedCasesDiff2']+ Positiveseries.iloc[i]['ConfirmedCasesDiff3']

    Positiveseries.iloc[i]['ConfirmedCasesDiff1'] = Positiveseries.iloc[i-1]['ConfirmedCasesDiff1']+ Positiveseries.iloc[i]['ConfirmedCasesDiff2']

    Positiveseries.iloc[i]['ConfirmedCases'] = Positiveseries.iloc[i-1]['ConfirmedCases']+ Positiveseries.iloc[i]['ConfirmedCasesDiff1']

    Positiveseries.iloc[i]['Fatalitiesdiff2']= Positiveseries.iloc[i-1]['Fatalitiesdiff2'] + Positiveseries.iloc[i]['Fatalitiesdiff3']

    Positiveseries.iloc[i]['Fatalitiesdiff1']= Positiveseries.iloc[i-1]['Fatalitiesdiff1'] + Positiveseries.iloc[i]['Fatalitiesdiff2']

    Positiveseries.iloc[i]['Fatalities']= Positiveseries.iloc[i-1]['Fatalities'] + Positiveseries.iloc[i]['Fatalitiesdiff1']

    

# Positiveseries['ConfirmedCasesDiff3']['24-03-2020']
Output = Positiveseries[['ConfirmedCases','Fatalities']]
Output.loc['2020-04-03' :] =0
Output.head()
submission = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")
submission.head()
submission['ConfirmedCases']= list(Output.loc['2020-03-12':]['ConfirmedCases'])
submission['Fatalities']= list(Output.loc['2020-03-12':]['Fatalities'])
submission.to_csv("submission.csv",index=False)