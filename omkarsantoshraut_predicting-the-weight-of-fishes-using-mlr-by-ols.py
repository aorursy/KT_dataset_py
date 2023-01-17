import pandas as pd

import numpy as np



df = pd.read_csv('../input/fish-dataset/Fish.csv')

df
new = {'Bream':0, 'Roach':1, 'Whitefish':2, 'Parkki':3, 'Perch':4, 'Pike':5, 'Smelt':6}

df['Species'] = df['Species'].map(new)

df
import statsmodels.api as sm

target = df.iloc[:, 1]

input = sm.add_constant(df[['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']])

print(input, target)
B_ols = np.linalg.inv(input.T.dot(input)).dot(input.T.dot(target))

B_ols
def pre_weight(x1,x2,x3,x4,x5, x6):

    target_pre = B_ols[1]*x1 + B_ols[2]*x2 + B_ols[3]*x3 + B_ols[4]*x4 + B_ols[5]*x5 + B_ols[6]*x6 + B_ols[0]

    return target_pre
ex1 = pre_weight(1, 21.1, 22.5, 25, 6.4, 3.8)

ex1
ex2 = pre_weight(2, 33.7, 36.4, 39.6, 11.7612, 6.5736)

ex2
y_hat = pre_weight(df['Species'], df['Length1'], df['Length2'], df['Length3'], df['Height'], df['Width'])

y_hat
r_square = 1 - sum((target - y_hat)*(target - y_hat))/sum((target - target.mean())*(target - target.mean()))

r_square
import plotly.figure_factory as ff

df_data = df.loc[:,['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]

df_data['index'] = np.arange(1, len(df_data)+1)

fig = ff.create_scatterplotmatrix(df_data, diag = 'box', index = 'index')

import plotly.offline as pyo

pyo.plot(fig)

fig