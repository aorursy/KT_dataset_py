import numpy as np

import pandas as pd

import plotly.graph_objects as go

import warnings



pd.set_option('display.max_columns', 300)

warnings.filterwarnings('ignore')
multiple_choice_df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

multiple_choice_df.head()
def plot_bivariant(df, x1, x2, mode='group', percentage=False, legend_orientation='h'):

    temp = df.groupby([x1, x2]).agg('size').reset_index()

    data = []

    for i in temp[x2].unique():

        temp_df = temp[temp[x2].isin([i])]

        col = 0

        if percentage:

            temp_df.loc[:, 'percentage'] = temp_df[0].apply(lambda x: (x*100)/temp_df[0].sum())

            col = 'percentage'

        data.append(

            go.Bar(name=i, x=temp_df[x1].unique(), y=temp_df[col].values)

        )

    fig = go.Figure(data=data)

    # Change the bar mode

    fig.update_layout(barmode=mode, legend_orientation=legend_orientation)

    fig.show()
plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q1', x2='Q14', mode='group', percentage=True)
plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q2', x2='Q14', percentage=True)
plot_bivariant(

    multiple_choice_df[multiple_choice_df['Q3'].isin(['India', 'United States of America'])].iloc[1:, :],

    x1='Q3', x2='Q14')
plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q5', x2='Q14', percentage=True, legend_orientation='h')
plot_bivariant(multiple_choice_df.iloc[1:, :], x1='Q8', x2='Q14', legend_orientation='h')