import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import seaborn as sns
import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()
sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sales.head()
sales.shape
sales.describe()
sales['d_1'].iplot()
sales['d_1'].value_counts().iplot()
sales['dept_id'].value_counts().iplot(kind='bar')
# plt.figure(figsize=[20,7])
# sns.countplot(sales['cat_id'])
sales['cat_id'].value_counts().iplot(kind='bar')
sales['store_id'].value_counts().iplot(kind='bar')
sales['state_id'].value_counts().iplot(kind='bar')
sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
sell_prices.head()
sell_prices.describe()
sell_prices.shape
px.line(data_frame=sell_prices[sell_prices['item_id'] == 'HOBBIES_1_001'], x='wm_yr_wk', y='sell_price', color='store_id')
px.line(data_frame=sell_prices[sell_prices['item_id'] == 'HOBBIES_1_002'], x='wm_yr_wk', y='sell_price', color='store_id')
px.line(data_frame=sell_prices[sell_prices['item_id'] == 'HOBBIES_1_003'], x='wm_yr_wk', y='sell_price', color='store_id')
calender = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
calender
calender.describe()
calender['event_type_1'].value_counts().iplot(kind='bar')
calender['event_type_2'].value_counts().iplot(kind='bar')
calender['event_name_1'].value_counts().iplot(kind='bar')
calender['event_name_1'].value_counts()
sample_op = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
sample_op