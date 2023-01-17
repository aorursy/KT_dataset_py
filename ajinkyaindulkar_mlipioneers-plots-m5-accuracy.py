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
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(style='whitegrid', palette="deep", font_scale=0.7, rc={"figure.figsize": [16, 8]})
# declare global variables

ROOT="/kaggle/input/m5-forecasting-accuracy/"
sell_prices_df = pd.read_csv(ROOT+"sell_prices.csv")

calendar_df = pd.read_csv(ROOT+"calendar.csv")

sales_train_eval_df = pd.read_csv(ROOT+"sales_train_validation.csv")



predicted_df = pd.read_csv("/kaggle/input/m5-final-models/submission_LGBM.csv")
d_cols = [c for c in sales_train_eval_df.columns if 'd_' in c]

p_cols = []

for i in range(1914, 1942):

    p_cols.append('d_{}'.format(i))

    

states = sales_train_eval_df['state_id'].unique()
sales_train_eval_df.head()
# df transform

past_sales = sales_train_eval_df.set_index('id')[d_cols].T.merge(calendar_df.set_index('d')['date'],

                                                                 left_index=True,right_index=True,

                                                                 validate='1:1').set_index('date')

past_sales.head()
# consider only validation data and rename columns

predicted_df = predicted_df[predicted_df['id'].str.contains("validation")]

predicted_df.columns = ['id'] + p_cols
# df transform

predicted_sales = predicted_df.set_index('id')[p_cols].T.merge(calendar_df.set_index('d')['date'],

                                                                 left_index=True,right_index=True,

                                                                 validate='1:1').set_index('date')

predicted_sales.head()
xt = ['2011-10-06', '2012-06-12', '2013-02-17', '2013-10-25',  '2014-07-02', '2015-03-09', '2015-11-14']
items_col = [c for c in past_sales.columns if states[0] in c]

past_ca = past_sales[items_col].sum(axis=1).tolist()

past_ca

items_col = [c for c in predicted_sales.columns if states[0] in c]

pred_ca = predicted_sales[items_col].sum(axis=1).tolist()

past_dates = past_sales.index.tolist()

pred_dates = predicted_sales.index.tolist()



plt.plot(past_dates, past_ca, label='ground truth')

plt.plot(pred_dates, pred_ca, label='predicted forecast')

plt.title('Predicted Forecast for last 28 days: CA')

plt.xticks(xt)

plt.legend()

plt.show()
items_col = [c for c in past_sales.columns if states[1] in c]

past_ca = past_sales[items_col].sum(axis=1).tolist()

past_ca

items_col = [c for c in predicted_sales.columns if states[1] in c]

pred_ca = predicted_sales[items_col].sum(axis=1).tolist()

past_dates = past_sales.index.tolist()

pred_dates = predicted_sales.index.tolist()



plt.plot(past_dates, past_ca, label='ground truth')

plt.plot(pred_dates, pred_ca, label='predicted forecast')

plt.title('Predicted Forecast for last 28 days: TX')

plt.xticks(xt)

plt.legend()

plt.show()
items_col = [c for c in past_sales.columns if states[2] in c]

past_ca = past_sales[items_col].sum(axis=1).tolist()

past_ca

items_col = [c for c in predicted_sales.columns if states[2] in c]

pred_ca = predicted_sales[items_col].sum(axis=1).tolist()

past_dates = past_sales.index.tolist()

pred_dates = predicted_sales.index.tolist()



plt.plot(past_dates, past_ca, label='ground truth')

plt.plot(pred_dates, pred_ca, label='predicted forecast')

plt.title('Predicted Forecast for last 28 days: WI')

plt.xticks(xt)

plt.legend()

plt.show()