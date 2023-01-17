# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from fbprophet import Prophet
df = pd.read_csv('/kaggle/input/estpiv2/estestprirostivanovo2.csv')
predictions=4
train_df=df[:-predictions]

df.tail()
m = Prophet(seasonality_mode='multiplicative').fit(train_df)
future = m.make_future_dataframe(periods=6, freq='M')
fcst = m.predict(future)
fig = m.plot(fcst)
fcst.tail()
print(', '.join(fcst.columns))
cmp_df= fcst.set_index ('ds') [['yhat','yhat_lower','yhat_upper']].join(df.set_index('ds'))
cmp_df.tail()
import numpy as np
cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']
cmp_df.tail()


#MAPE
print ('MAPE=', np.mean(abs(cmp_df['p'])))

#MAE
print ('MAE=', np.mean(abs(cmp_df['e'])))
print ('Точность прогноза =', 100-np.mean(abs(cmp_df['p'])),'%')