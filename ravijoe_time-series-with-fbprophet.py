# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

from fbprophet import Prophet

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !pip install pystan

!pip install fbprophet
df=pd.read_csv('../input/monthly-milk-production-pounds.csv')

df.head(10)
df.tail()
df.drop(168,axis=0,inplace=True)
df.columns=["ds","y"]
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,9)



df.plot()
df['ds']=pd.to_datetime(df['ds'])
df['y'].plot()
### intiialize the Model

model=Prophet()

model.fit(df)
model
model.seasonalities
model.component_modes
future_dates=model.make_future_dataframe(periods=365)
df.tail()
future_dates
prediction=model.predict(future_dates)

prediction.head()
prediction[['ds','yhat','yhat_lower','yhat_upper']].tail()
prediction[['ds','yhat','yhat_lower','yhat_upper']].head()
model.plot(prediction)