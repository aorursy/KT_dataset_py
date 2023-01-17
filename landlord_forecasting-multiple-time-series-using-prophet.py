import pandas as pd

import numpy as np

from matplotlib import pyplot
def timeseries(min_date, max_date):

    time_index = pd.date_range(min_date, max_date)

    dates = (pd.DataFrame({'ds': pd.to_datetime(time_index.values)},

                          index=range(len(time_index))))

    y = np.random.random_sample(len(dates))*10

    dates['y'] = y

    return dates
s = timeseries('2019-01-01','2020-12-30')
series = [timeseries('2019-01-01','2020-12-30') for x in range(0,500)]
series[1].plot(x = 'ds', y = 'y')
from fbprophet import Prophet
def run_prophet(timeserie):

    model = Prophet(yearly_seasonality=False,daily_seasonality=False)

    model.fit(timeserie)

    forecast = model.make_future_dataframe(periods=200, include_history=False)

    forecast = model.predict(forecast)

    return forecast
x = run_prophet(series[0])
x.head()
import time

from tqdm import tqdm
start_time = time.time()

result = list(map(lambda timeserie: run_prophet(timeserie), tqdm(series)))

print("--- %s seconds ---" % (time.time() - start_time))
from multiprocessing import Pool, cpu_count
start_time2 = time.time()

p = Pool(cpu_count())

predictions = list(tqdm(p.imap(run_prophet, series), total=len(series)))

p.close()

p.join()

print("--- %s seconds ---" % (time.time() - start_time2))