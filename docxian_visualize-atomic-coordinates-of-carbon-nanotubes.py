# packages

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
# import

df = pd.read_csv('../input/carbon-naonotubes/carbon_nanotubes.csv', sep=';', decimal=',')

df.shape
df.head()
df.describe()
# add index combination n/m

df['Chiral_nm'] = df['Chiral indice n'].map(str) + '/' + df['Chiral indice m'].map(str)
# and show frequency plot

plt.figure(figsize=(16,4))

df['Chiral_nm'].value_counts().plot(kind='bar')

plt.grid()

plt.show()
# "shortcut" for column names

cu = "Calculated atomic coordinates u'"

cv = "Calculated atomic coordinates v'"

cw = "Calculated atomic coordinates w'"



iu = "Initial atomic coordinate u"

iv = "Initial atomic coordinate v"

iw = "Initial atomic coordinate w"
# interactive scatter plot

fig = px.scatter_3d(df, x=cu, y=cv, z=cw,

                    color='Chiral indice n', # display n via color

                    size='Chiral indice m',  # display m via size of sphere

                    hover_data=[],

                    opacity=0.5)

fig.update_layout(title='Calculated atomic coordinates')

fig.show()
# select a specific subset

sel = '11/2'

df_select = df[df.Chiral_nm==sel]



# interactive scatter plot of selection

fig = px.scatter_3d(df_select, x=cu, y=cv, z=cw,

                    color='Chiral indice n',

                    size='Chiral indice m',

                    hover_data=[],

                    opacity=0.5)

fig.update_layout(title='Calculated atomic coordinates - Index '+sel)

fig.show()
# another example

sel = '7/3'

df_select = df[df.Chiral_nm==sel]



# interactive scatter plot of selection

fig = px.scatter_3d(df_select, x=cu, y=cv, z=cw,

                    color='Chiral indice n',

                    size='Chiral indice m',

                    hover_data=[],

                    opacity=0.5)

fig.update_layout(title='Calculated atomic coordinates - Index '+sel)

fig.show()
# show also INITIAL coordinates

sel = '7/3'

df_select = df[df.Chiral_nm==sel]



# interactive scatter plot of selection

fig = px.scatter_3d(df_select, x=iu, y=iv, z=iw,

                    color='Chiral indice n',

                    size='Chiral indice m',

                    hover_data=[],

                    opacity=0.5)

fig.update_layout(title='Initial atomic coordinates - Index '+sel)

fig.show()
# select by chiral index n

sel_n = 12

df_select = df[df['Chiral indice n']==sel_n]



# interactive scatter plot of selection

fig = px.scatter_3d(df_select, x=cu, y=cv, z=cw,

                    color='Chiral indice m', # use color for displaying m in this case, n is fixed anyway

                    hover_data=[],

                    opacity=0.5)

fig.update_layout(title='Calculated atomic coordinates - n='+str(sel_n))

fig.show()