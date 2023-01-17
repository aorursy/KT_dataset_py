# Load standard libraries



import numpy as np

RANDOM_SEED = 1337

np.random.seed(RANDOM_SEED)

import pandas as pd

pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 70)



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode()



%time df_train = pd.read_csv('../input/train.csv', na_values=-1)



print('Loaded data and libraries.')
cols = ['ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',

        'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']



groupped = df_train.groupby(cols)['target'].agg(['sum', 'count', 'mean']).reset_index()

groupped
py.iplot([go.Scatter(y=groupped['count'])])

py.iplot([go.Scatter(y=groupped['sum'])])

py.iplot([go.Scatter(y=groupped['mean'])])
groupped['ps_calc_15_to_20']  = groupped['ps_calc_15_bin']*32 + groupped['ps_calc_16_bin']*16

groupped['ps_calc_15_to_20'] += groupped['ps_calc_17_bin']*8  + groupped['ps_calc_18_bin']*4

groupped['ps_calc_15_to_20'] += groupped['ps_calc_19_bin']*2  + groupped['ps_calc_20_bin']*1

groupped['ps_calc_15_to_20'].astype(np.uint8, inplace=True)

groupped
groupped = groupped.sort_values(by='count', ascending=False).reset_index()

groupped = groupped.drop('index', axis=1)

groupped
# Draw Chord Graph

dots = groupped['ps_calc_15_to_20'].copy()

n = 64.0

# Dots start at 12 o'clock, and rotates clockwise.

x = np.sin((dots+0.5)/n*2*np.pi)

y = np.cos((dots+0.5)/n*2*np.pi)

edges = list(zip(x,y))

data = [go.Scattergl(

    x=x,

    y=y,

    text=['Dot %d'%c for c in dots],

    mode='lines+markers',

    marker={

        'color':'rgb(30,30,30)',

        'size':5

    },

    line={

        'color':'rgba(46, 147, 219, 0.5)',

        'width':5

    },

    hoverinfo='text'

)]

layout = go.Layout(autosize=False, width=500, height=500, showlegend=False)

py.iplot(go.Figure(data=data, layout=layout))
dots = pd.DataFrame(dots)

dots['ps_calc_15_to_20_reversed'] = dots['ps_calc_15_to_20'].iloc[::-1].values

dots['sum_of_them'] = dots['ps_calc_15_to_20'] + dots['ps_calc_15_to_20_reversed']

dots
dots[dots['sum_of_them']!=63]
new_index = list(range(4)) + [5,4] + list(range(6,12)) + [13,12] + list(range(14,64))

#new_index += [51,50] + list(range(52,58)) + [59,58] + list(range(60,64))

print(len(new_index))

np.array(new_index) # used np.array to display the list nicer.
dots2 = dots.copy()

dots2 = dots2.reindex(index=new_index).reset_index(drop=True)

dots2['ps_calc_15_to_20_reversed'] = dots2['ps_calc_15_to_20'].iloc[::-1].values

dots2['sum_of_them'] = dots2['ps_calc_15_to_20'] + dots2['ps_calc_15_to_20_reversed']



dots2
dots2[dots2['sum_of_them']!=63]
groupped = groupped.reindex(index=new_index).reset_index(drop=True)

groupped
py.iplot([go.Scatter(y=groupped['count'])])

py.iplot([go.Scatter(y=groupped['sum'])])

py.iplot([go.Scatter(y=groupped['mean'])])
# Columns -> binary decoded.



tmp  = df_train['ps_calc_15_bin'] * 32 + df_train['ps_calc_16_bin'] * 16 + df_train['ps_calc_17_bin'] * 8

tmp += df_train['ps_calc_18_bin'] * 4 + df_train['ps_calc_19_bin'] * 2 + df_train['ps_calc_20_bin'] * 1



tmp2 = [5, 22, 9, 32, 13, 38, 20, 47, 2, 19, 8, 30, 10, 35, 17, 45, 1,

        15, 4, 24, 7, 29, 14, 40, 0, 12, 3, 21, 6, 26, 11, 36, 27, 52,

        37, 57, 42, 60, 51, 63, 23, 49, 34, 56, 39, 59, 48, 62, 18, 46,

        28, 53, 33, 55, 44, 61, 16, 43, 25, 50, 31, 54, 41, 58]

tmp2 = pd.Series(tmp2)



df_train['ps_calc_15_16_17_18_19_20'] = tmp.map(tmp2)



# You may now drop the others peacefully.

#df_train.drop(['ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',

#               'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1, inplace=True)