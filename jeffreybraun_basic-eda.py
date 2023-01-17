import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
init_notebook_mode()

df = pd.read_csv('/kaggle/input/chopped-10-years-of-episode-data/chopped.csv')
df.head()
df.info()
df_judge1 = pd.DataFrame(df['judge1'])
df_judge2 = pd.DataFrame(df['judge2'])
df_judge3 = pd.DataFrame(df['judge3'])

df_judge1.rename(columns={'judge1':'judge'}, inplace=True)
df_judge2.rename(columns={'judge2':'judge'}, inplace=True)
df_judge3.rename(columns={'judge3':'judge'}, inplace=True)

df_judge = pd.concat([df_judge1, df_judge2, df_judge3])
df_judge = pd.DataFrame(df_judge.judge.value_counts()).reset_index()
df_judge.rename(columns={'index':'judge', 'judge':'show appearances'}, inplace=True)

fig = px.pie(df_judge, values='show appearances', names='judge', title='Judge Frequency')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()
appetizer_list = []
appetizer_dict = {}


for i in range(df.shape[0]):
    food = df.loc[i, 'appetizer'].lower().split(',')
    food_clean = []
    food_new = 0
    food_old = 0
    for item in food:
        item = item.strip()
        food_clean.append(item)
        if item in appetizer_list:
            food_old += 1
        else:
            food_new += 1
    appetizer_list.extend(food_clean)
    appetizer_dict[i] = food_clean
    df.loc[i, 'appetizer_new'] = food_new
    df.loc[i, 'appetizer_old'] = food_old
    df.loc[i, 'appetizer_new_percentage'] = (food_new / (food_new + food_old))

df_app = pd.DataFrame(df.groupby('appetizer_new_percentage').size()).reset_index()
df_app.rename(columns = {0:'count'}, inplace=True)
fig = px.bar(df_app, x = 'appetizer_new_percentage', y = 'count', title='New Appetizer Ingredients each episode')
fig.show()

df['appetizer_new_percentage'].describe()