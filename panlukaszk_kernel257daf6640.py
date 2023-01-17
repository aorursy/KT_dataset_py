import pandas as pd

import cufflinks as cf

cf.go_offline()

import numpy as np

import plotly.graph_objects as go

import numpy as np

import plotly

print(plotly.__version__)

from plotly.offline import iplot





# plotly.offline.init_notebook_mode(connected=True)

# import plotly.io as pio

# pio.renderers.default = 'colab'
df = pd.read_csv("../input/anomaly_guys.csv", parse_dates=['created_at'], index_col='created_at')
def calc_percentage(row):

    if row['prev_trend'] == 0:

        return row['delta']

    else:

        return row['delta'] / row['prev_trend']
def plot_user_activity_and_anomalies(df_user, avg):

    print("calling plot, avg=" + str(avg))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_user.index, y=df_user.cnt,

                        mode='lines', name='cnt'))

    fig.add_trace(go.Scatter(x=df_user.index, y=df_user.anomaly_points,

                        mode='markers', name='anomaly_points', marker={'size':15}))

#     pio.show(fig)

    display(fig)

# plot_user_activity_and_anomalies(df_user)
def user_draw_params2(average_period, username, normalization_trigger, delta_percentage_anomaly):

#     df = pd.read_csv("/data/learnamp/active_users_daily.csv", parse_dates=['created_at'], index_col='created_at')

    print(average_period+normalization_trigger+delta_percentage_anomaly)

    df_user = df[[username]].copy().rename(columns={username: 'cnt'})

    df_user['trend'] = df_user.cnt.rolling(average_period).agg(np.mean)

    df_user['prev_trend'] = df_user['trend'].shift(1)

    df_user['delta'] = df_user['cnt'] - df_user['prev_trend'] 

    df_user['delta_percentage'] = df_user.apply(calc_percentage, axis=1)

#     df_user['delta_percentage'] = df_user['delta_percentage'].replace(np.inf, np.nan)

    df_user['delta_norm'] = np.tanh(df_user.cnt / normalization_trigger) * df_user['delta_percentage']

#     df_user['delta_norm'].iplot()

    df_user['anomaly_points'] = df_user.cnt.where(df_user['delta_norm'] > delta_percentage_anomaly / 100)

#     print(df_user['anomaly_points'].dropna())

#     df_user[['delta_percentage','delta_norm']].iplot()

    plot_user_activity_and_anomalies(df_user, average_period)

    return

    # return df_user[df_user['anomaly_points'] > 0]

import ipywidgets

ipywidgets.__version__
from ipywidgets import interact

from ipywidgets import IntSlider, Dropdown



usernames = df.columns



interact(user_draw_params2, 

         average_period=IntSlider(min=7, max=31), 

         username=Dropdown(options=usernames),

         normalization_trigger = IntSlider(min=20, max=200, step=10),

         delta_percentage_anomaly = IntSlider(min=100, max=3000, step=100)

        )