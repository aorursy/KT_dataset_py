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
import pandas as pd
corna_rok = pd.read_csv("/kaggle/input/covid19-in-south-korea/corona_rok.csv")
mers_rok = pd.read_csv("/kaggle/input/covid19-in-south-korea/mers_rok.csv")
Sars_rok = pd.read_csv("/kaggle/input/covid19-in-south-korea/2003-SARS_rok.csv")
print(corna_rok)
print(corna_rok.info())
corna_rok
print(mers_rok.head())
print(Sars_rok.head())
print(Sars_rok.info())
mres_rok_pre = mers_rok[['reported','number','death']].groupby(['reported'],as_index = False).count()
mres_rok_pre
corna_rok['index'] = pd.Series(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65'], index = corna_rok.index)
corna_rok
Sars_rok['index'] = pd.Series(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59'], index = Sars_rok.index)
Sars_rok
mres_rok_pre['index'] = pd.Series(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35'], index = mres_rok_pre.index)
mres_rok_pre
# import module to draw plots

import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go
trace1 = go.Scatter(x = corna_rok['index'], y = corna_rok['confirmed'], mode = 'lines', name = 'corna_rok')
trace2 = go.Scatter(x = Sars_rok['index'], y = Sars_rok['confirmed'], mode = 'lines', name = 'Sars_rok')
trace3 = go.Scatter(x = mres_rok_pre['index'], y = mres_rok_pre['number'], mode = 'lines', name = 'mres_rok_pre')

data = [trace1, trace2, trace3]
pyo.iplot(data)
from sklearn.preprocessing import MinMaxScaler
print(corna_rok['confirmed'].tail())
print(Sars_rok['confirmed'].head())
print(mres_rok_pre['number'].head())
scaler = MinMaxScaler()
corna_rok['confirmed_scaled'] = scaler.fit_transform(corna_rok['confirmed'].values.reshape(-1,1))
mres_rok_pre['confirmed_scaled'] = scaler.fit_transform(mres_rok_pre['number'].values.reshape(-1,1))
Sars_rok['confirmed_scaled'] = scaler.fit_transform(Sars_rok['confirmed'].values.reshape(-1,1))
corna_rok.head()
mres_rok_pre.head()
Sars_rok.head()
trace1 = go.Scatter(x = corna_rok['index'], y = corna_rok['confirmed_scaled'], mode = 'lines', name = 'corna_rok')
trace2 = go.Scatter(x = Sars_rok['index'], y = Sars_rok['confirmed_scaled'], mode = 'lines', name = 'Sars_rok')
trace3 = go.Scatter(x = mres_rok_pre['index'], y = mres_rok_pre['confirmed_scaled'], mode = 'lines', name = 'mres_rok_pre')

data = [trace1, trace2, trace3]
pyo.iplot(data)
corna_rok.head()
corna_arima = corna_rok[['dates','confirmed']]
corna_arima.head()
corna_arima.dtypes
corna_arima['dates'] = pd.DatetimeIndex(corna_arima['dates'])
corna_arima = corna_arima.rename(columns = {'dates':'ds', 'confirmed':'y'})
corna_arima
corna_arima['y'] = corna_arima['y'].values.astype(float)
corna_arima.dtypes
corna_arima = corna_arima.set_index("ds")
corna_arima.head()
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

corna_arima.plot()
plt.show()
plot_acf(corna_arima)
plot_pacf(corna_arima)
plt.figure(figsize=(20,4))
plt.show()
diff_1 = corna_arima.diff(periods=1).iloc[1:]
diff_1.plot()
plot_acf(diff_1)
plot_pacf(diff_1)
plt.show()
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
model = ARIMA(corna_arima, order=(2,1,0))
model_fit = model.fit(trend = 'nc', full_output = True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()
fore = model_fit.forecast(steps = 1)
print(fore)
# ARIMA의 p,d,q를 변경
model = ARIMA(corna_arima, order=(0,1,1))
model_fit = model.fit(trend = 'nc', full_output = True, disp=1)
print(model_fit.summary())
model_fit.plot_predict()

# 모델에 돌린걸 예측해서 보여줘라
fore = model_fit.forecast(steps = 1)
print(fore)
corna_arima.tail()
