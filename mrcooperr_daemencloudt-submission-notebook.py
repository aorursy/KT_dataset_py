# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#submission = pd.read_csv('/kaggle/input/mlip-daemencloudt-lightgbm-notebook/submission.csv')

submission = pd.read_csv('/kaggle/input/local-submission-files-m5/submission_46.csv')
sales = pd.read_csv(f'/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')# sales train validation

ids = sorted(list(set(sales['id'])))

#d_cols = [c for c in sales.columns if 'd_' in c]

d_cols = [f'd_{x}' for x in range(1258,1914)]
def calc_coef(x,y, alpha):

    y = np.mean(y, axis = 0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    line = slope*x+intercept

    return 1 + (slope*y.shape[0]/alpha)
def get_plot_indices(n):

    ''' Returns row, column for the plotly subplot '''

    if(n>5):

        return (2,n-5)

    

    return (1, n)
def plot_trend_data(data, s, fig, i):

    x = np.arange(len(d_cols))

    y = np.mean(data, axis = 0)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    line = slope*x+intercept

    print(f'store {s} with slope: {slope}')

    

    r, c = get_plot_indices(i)

    



    fig.add_trace(go.Scatter(

        x=x,

        y=y

    ),row = r, col=c)



    fig.add_trace(go.Scatter(

        x=x,

        y=line

        

    ), row =r, col=c)

    

    if c == 3:

        fig.update_xaxes(title_text=f"Number of days after 09-07-2014", row=r, col=c)

    if c == 1:

        fig.update_yaxes(title_text="Average sales", row=r, col=c)



    return fig

    
stores = sales['store_id'].unique()

fig = make_subplots(cols=5,rows=2, subplot_titles=(stores))

coefs = {}

i = 1

for s in stores:

    x = np.arange(len(d_cols))

    store_data = sales[sales['store_id'] == s].set_index('id')[d_cols]

    fig = plot_trend_data(store_data, s, fig, i)

    coefs[s] = calc_coef(x,store_data, 6)

    i += 1  

    

fig.update_layout( height=700, showlegend = False)

fig.show()
np.mean([value for key, value in coefs.items()])

    
stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

for store in stores:

    submission.loc[submission['id'].str.contains(store), [f'F{x}' for x in range(1,29)]]   *= coefs[store]
submission.to_csv('submission.csv', index = False)