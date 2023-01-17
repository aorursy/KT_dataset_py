# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
%matplotlib inline
from fbprophet import Prophet


df = pd.read_csv('../input/avocado.csv')
df['Date'] = pd.to_datetime(df['Date'])
regions = df.groupby(df.region)
Predicting_for="Atlanta"
atlanta = regions.get_group(Predicting_for)[['Date','AveragePrice']].reset_index(drop=True)
atlanta.plot(x='Date',y='AveragePrice',kind='line')
atlanta = atlanta.rename(columns={'Date':'ds','AveragePrice':'y'})
len(atlanta)
m = Prophet()
m.fit(atlanta)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

