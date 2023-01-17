import pandas as pd

import numpy as np

from fbprophet import Prophet

from tqdm import tqdm
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train =  pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
for i in tqdm(range(sales_train.shape[0])):

    #print('We are at product {} out of {}'.format(i,sales_train.shape[0]))

    temp_series = sales_train.iloc[i,600:]

    temp_series.index = calendar_df['date'][600:600+len(temp_series)]

    temp_series =  pd.DataFrame(temp_series)

    temp_series = temp_series.reset_index()

    temp_series.columns = ['ds', 'y']



    m1 = Prophet(uncertainty_samples=False)

    m1.fit(temp_series)



    future1 = m1.make_future_dataframe(periods=28).tail(28)

    forecast1 = m1.predict(future1)



    submission.iloc[i,1:] = forecast1['yhat'].iloc[-28:].values

    submission.iloc[:,1:][submission.iloc[:,1:]<0]=0
submission.to_csv('submission.csv', index=False)