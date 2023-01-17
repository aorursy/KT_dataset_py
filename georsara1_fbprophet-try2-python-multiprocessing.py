import pandas as pd

import numpy as np

from fbprophet import Prophet

from tqdm import tqdm, tnrange

from multiprocessing import Pool, cpu_count
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train =  pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
def run_prophet(timeserie):

    model = Prophet(uncertainty_samples=False)

    model.fit(timeserie)

    future = model.make_future_dataframe(periods=28, include_history=False)

    forecast = model.predict(future)

    return forecast
start_from_ob = 800

for i in tnrange(sales_train.shape[0]):

    temp_series = sales_train.iloc[i,start_from_ob:]

    temp_series.index = calendar_df['date'][start_from_ob:start_from_ob+len(temp_series)]

    temp_series =  pd.DataFrame(temp_series)

    temp_series = temp_series.reset_index()

    temp_series.columns = ['ds', 'y']



    with Pool(cpu_count()) as p:

        forecast1 = p.map(run_prophet, [temp_series])



    submission.iloc[i,1:] = forecast1[0]['yhat'].values



submission.iloc[:,1:][submission.iloc[:,1:]<0]=0
submission.to_csv('submission.csv', index=False)