import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline

import numpy as np

img=np.array(Image.open('../input/butter-fruit/Butter_fruit.jpg'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.show()
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
df=pd.read_csv('../input/avocado-prices/avocado.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
df.info()
df.describe().T
df.select_dtypes(exclude=['int','float']).columns
import random 

import seaborn as sns

from fbprophet import Prophet
df.tail()
df=df.sort_values('Date')

df.tail()
df.plot(x='Date', y='AveragePrice',legend=True,figsize=(20,4));

plt.ioff()
plt.figure(figsize=[25,12])

sns.countplot(x='region',data=df);

plt.xticks(rotation=45)

plt.ioff()
sns.countplot(x='year',data=df);
df_prophet=df[['Date','AveragePrice']]

df_prophet
df_prophet=df_prophet.rename(columns={'Date':'ds','AveragePrice':'y'})

df_prophet
m=Prophet()

m.fit(df_prophet)
future=m.make_future_dataframe(periods=365)

forecast=m.predict(future)

forecast
figure=m.plot(forecast,xlabel='Date',ylabel='Price')
figure=m.plot_components(forecast)
df.columns
df['region'].unique()
region_sample=df[df['region']=='California']

region_sample.head()
region_sample=region_sample.sort_values('Date')
region_sample.plot(x='Date', y='AveragePrice',legend=True,figsize=(20,4));

plt.ioff()
region_sample=region_sample[['Date','AveragePrice']]

region_sample
region_sample=region_sample.rename(columns={'Date':'ds','AveragePrice':'y'})

region_sample
m=Prophet()

m.fit(region_sample)

future=m.make_future_dataframe(periods=365)

forecast=m.predict(future)

forecast
figure=m.plot(forecast,xlabel='Date',ylabel='Price')
figure=m.plot_components(forecast)