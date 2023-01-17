# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp

import seaborn as sns



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.font_manager as fm



from fbprophet import Prophet

from fbprophet.plot import plot_plotly, add_changepoints_to_plot



import plotly.offline as py

import plotly.express as px

import plotly.graph_objects as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
deal_df = '/kaggle/input/korean-real-estate-transaction-data/Apart Deal.csv'

deal_df = pd.read_csv(deal_df)

deal_df = deal_df.rename({'지역코드':'Area code', '법정동':'Dong', '거래일':'Trading day', '아파트':'Apartment', '지번':'Lot number', '전용면적':'Exclusive area', '층':'Floor', '건축년도':'Year of construction', '거래금액':'Transaction amount'}, axis='columns')

deal_df.head(3)
deal_df.dtypes
deal_df = deal_df.astype({'Area code':'int',

                          'Dong':'category', 

                          'Trading day':'datetime64', 

                          'Apartment':'category', 

                          'Exclusive area':'float', 

                          'Floor':'int', 

                          'Year of construction':'int', 

                          'Transaction amount':'int'})

deal_df.dtypes
deal_df.info()
deal_df.describe()
profiling_report = pp.ProfileReport(deal_df.describe(), title="Deal data Profiling Report")

profiling_report
# sns.pairplot(deal_df)

# plt.show()
# deal_df.hist(bins=50, figsize=(20, 15))
trading_group_pyeong = deal_df

trading_group_pyeong['Exclusive area'] = pd.DataFrame(np.true_divide(deal_df['Exclusive area'], 3.305785))



trading_group_pyeong['Trans per Pyeong'] = np.true_divide(trading_group_pyeong['Transaction amount'], trading_group_pyeong['Exclusive area'])



trading_group_pyeong = trading_group_pyeong.groupby('Trading day', as_index=False).mean()



trading_group_pyeong = trading_group_pyeong.drop(['Area code', 

                                                  'Exclusive area', 

                                                  'Floor', 

                                                  'Year of construction',

                                                  'Transaction amount'], axis=1)



trading_group_pyeong
trading_group_mean = deal_df.groupby('Trading day', as_index=False).mean()

trading_group_mean = trading_group_mean.drop(['Exclusive area',

                                              'Floor',

                                              'Year of construction',

                                              'Area code'], axis=1)

trading_group_mean.head(3)
trading_group_count = deal_df.groupby('Trading day', as_index=False).count()

trading_group_count = trading_group_count.drop(['Area code', 

                                                'Apartment', 

                                                'Lot number', 

                                                'Dong', 

                                                'Exclusive area', 

                                                'Floor', 

                                                'Year of construction'], axis=1)

trading_group_count.head(3)
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=trading_group_mean['Trading day'], y=trading_group_mean['Transaction amount'],

                    mode='lines',

                    name='Price'))

fig.add_trace(go.Scatter(x=trading_group_count['Trading day'], y=trading_group_count['Transaction amount'],

                    mode='lines',

                    name='Volume'))

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=trading_group_pyeong['Trading day'], y=trading_group_pyeong['Trans per Pyeong'],

                    mode='lines',

                    name='Trans per Pyeong'))

fig.show()
fig = go.Figure()

fig = px.scatter(deal_df, x="Trading day", y="Transaction amount", color="Area code",

                 size='Exclusive area')

fig.show()
dong_group_mean = deal_df.groupby(['Dong', 'Trading day'], as_index=False).mean()

dong_group_mean.head(3)
convers_data = deal_df[{'Area code'}]

y = deal_df['Transaction amount']



print(y.head(3))

print(convers_data.head(3))