from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

import numpy as np

%matplotlib inline

from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import r2_score
df=pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True)
df.head()
df.index
df.index.freq='MS'
df.info()
df.plot(color='green',figsize=(10,7))
from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(df['S4248SM144NCEN'])

decompose.plot();
decompose.seasonal.plot(figsize=(10,7))
s_test=adfuller(df['S4248SM144NCEN'])

print("p-value :",s_test[1])
span=12

alpha=2/13
ses=SimpleExpSmoothing(df['S4248SM144NCEN']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1).rename('simpleexpsmoothing')

desa=ExponentialSmoothing(df['S4248SM144NCEN'],trend='add').fit().fittedvalues.shift(-1).rename('double-expo add')

desm=ExponentialSmoothing(df['S4248SM144NCEN'],trend='add').fit().fittedvalues.shift(-1).rename('double-expo mul')

df['S4248SM144NCEN'].plot(figsize=(10,7),legend=True)

ses.plot(legend=True)

desa.plot(legend=True)

desm.plot(legend=True)
tesa=ExponentialSmoothing(df['S4248SM144NCEN'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues.rename('tripple-expo add')

tesm=ExponentialSmoothing(df['S4248SM144NCEN'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues.rename('tripple-expo mul')
df['S4248SM144NCEN'].plot(figsize=(10,7),legend=True)

tesa.plot(legend=True)

tesm.plot(legend=True)
df['S4248SM144NCEN'].iloc[:24].plot(figsize=(10,7),legend=True,)

tesa[:24].plot(legend=True)

tesm[:24].plot(legend=True)
print('rmse tesa:',r2_score(df['S4248SM144NCEN'],tesa))

print('rmse tesm:',r2_score(df['S4248SM144NCEN'],tesm))
## since rmse of tesm is more near to 1
len(df)
325-36

train=df.iloc[:289]

test=df.iloc[289:]
model=ExponentialSmoothing(train['S4248SM144NCEN'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

model_predict=model.forecast(36)
train['S4248SM144NCEN'].plot(figsize=(12,7),label='Train',legend=True)

test['S4248SM144NCEN'].plot(legend=True,label='test')

model_predict.plot(legend=True,label='test-predictions')
test['S4248SM144NCEN'].plot(figsize=(10,7),legend=True,label='test')

model_predict.plot(legend=True,label='test-predictions')
print('rmse:',r2_score(test,model_predict))
final_model=ExponentialSmoothing(df['S4248SM144NCEN'],trend='mul',seasonal='mul',seasonal_periods=12).fit()

final_prediction=final_model.forecast(36)
df['S4248SM144NCEN'].plot(figsize=(15,10),legend=True)

final_prediction.plot(label='prediction',legend=True)
final_prediction
date=pd.date_range('2019-02-01',periods=36,freq='MS')
predicted_df=pd.DataFrame(data=list(zip(date,final_prediction)),columns=['date','prediction sale'])
predicted_df=predicted_df.set_index('date')
predicted_df.head()