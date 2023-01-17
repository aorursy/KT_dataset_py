import pandas as pd
from darts import TimeSeries
df = pd.read_csv('../input/air-passengers/AirPassengers.csv')

Series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')

train, val = Series.split_before(pd.Timestamp('19580101'))
from darts.models import ExponentialSmoothing

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))
import matplotlib.pyplot as plt

Series.plot(label = 'actual')
prediction.plot(label = 'forecast', lw = 3)
plt.legend()
from darts.models import AutoARIMA

model_aarima = AutoARIMA()
model_aarima.fit(train)
prediction_aarima = model_aarima.predict(len(val))
Series.plot(label = 'actual')
prediction_aarima.plot(label = 'forecast_aarima', lw = 3)
plt.legend()
from darts.models import FFT

model_fft = FFT()
model_fft.fit(train)
prediction_fft = model_fft.predict(len(val))
Series.plot(label = 'actual')
prediction_fft.plot(label = 'forecast-fft', lw = 3)
plt.legend()
# facebook prophet model
from darts.models import Prophet

model_prophet = Prophet()
model_prophet.fit(train)
prediction_prophet = model_prophet.predict(len(val))
Series.plot(label = 'actual')
prediction_prophet.plot(label = 'forecast-prophet', lw = 3)
plt.legend()
from darts.backtesting import backtest_forecasting

models = [ExponentialSmoothing(), Prophet()]

backtests = [backtest_forecasting(Series,
                                 model,
                                 pd.Timestamp('19550101'),
                                 fcast_horizon_n=3)
            for model in models]
from darts.metrics import mape

Series.plot(label='actual')
for i, m in enumerate(models):
    err = mape(backtests[i], Series)
    backtests[i].plot(lw = 3, label = '{}, MAPE = {:.2f}%'.format(m, err))
    
plt.title('Backtest with 3-months forecast horizon')
plt.legend()