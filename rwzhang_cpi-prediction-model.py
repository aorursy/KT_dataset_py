import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd 
import matplotlib.pyplot as plt
import fbprophet
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/SG_CPI_v2017.csv',header=0)
data['Month_Year'] =  pd.to_datetime(data['Month_Year'], format='%m/%Y')
df = data.set_index('Month_Year')
plt.plot(df.index, df.CPI)
plt.plot(df.index, df.Transport,'r')
plt.legend()
plt.show()
import fbprophet 
# Prophet requires columns ds (Date) and y (value)
df_=pd.read_csv('../input/SG_CPI_v2017.csv',header=0)
df_1 = df_.rename(columns={'Month_Year': 'ds', 'CPI': 'y'}).loc[:,['ds','y']]
df_1.head()
#df_1['y'] = np.log(df_1['y'])
model =fbprophet.Prophet() #instantiate Prophet
model.fit(df_1)
future_data = model.make_future_dataframe(periods=24, freq = 'm')
forecast_data  = model.predict(future_data)
forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast_data)
model.plot_components(forecast_data)