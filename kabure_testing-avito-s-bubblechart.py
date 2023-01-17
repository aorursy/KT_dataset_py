import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv("../input/train.csv", parse_dates=['activation_date'], nrows=50000)

print("Shape train: ", df_train.shape)
is_null = round((df_train.isnull().sum() / len(df_train) * 100),2)
print("NaN values in train Dataset")
print(is_null[is_null > 0].sort_values(ascending=False))
import plotly
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)
df_train['price_log'] = np.log(df_train['price'] + 1)

df_2007 = df_train[(df_train.user_type == 'Private') & (df_train.deal_probability > 0)].copy()

df_2007['deal_probability_rounded'] = round(df_2007['deal_probability'],2)
df_2007['index'] = df_2007['deal_probability'] / df_2007['price'] 
slope = 3.2121e-05
hover_text = []
bubble_size = []
import math

for index, row in df_2007.iterrows():
    hover_text.append(('Region: {region}<br>'+
                      'Parent Category: {par_cat}<br>'+
                      'Title: {title}<br>'+
                      'Price: {price}<br>'+
                      'Deal Probability: {deal_prob}').format(region=row['region'],
                                            par_cat=row['parent_category_name'],
                                            title=row['title'],
                                            price=row['price'],
                                            deal_prob=row['deal_probability']))
    bubble_size.append(math.sqrt(row['deal_probability']*slope))

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 14*max(df_2007['size'])/(40**2)

trace0 = go.Scatter(
    x=df_2007['price_log'][df_2007['parent_category_name'] == 'Личные вещи'],
    y=df_2007['deal_probability'][df_2007['parent_category_name'] == 'Личные вещи'],
    mode='markers', opacity=0.7,
    name='Личные вещи',
    text=df_2007['text'][df_2007['parent_category_name'] == 'Личные вещи'],
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['parent_category_name'] == 'Личные вещи'],
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df_2007['price_log'][df_2007['parent_category_name'] == 'Для дома и дачи'],
    y=df_2007['deal_probability'][df_2007['parent_category_name'] == 'Для дома и дачи'],
    mode='markers', opacity=0.7,
    name='Для дома и дачи',
    text=df_2007['text'][df_2007['parent_category_name'] == 'Для дома и дачи'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['parent_category_name'] == 'Для дома и дачи'],
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df_2007['price_log'][df_2007['parent_category_name'] == 'Бытовая электроника'],
    y=df_2007['deal_probability'][df_2007['parent_category_name'] == 'Бытовая электроника'],
    mode='markers', opacity=0.7,
    name='Бытовая электроника',
    text=df_2007['text'][df_2007['parent_category_name'] == 'Бытовая электроника'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['parent_category_name'] == 'Бытовая электроника'],
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df_2007['price_log'][df_2007['parent_category_name'] == 'Недвижимость'],
    y=df_2007['deal_probability'][df_2007['parent_category_name'] == 'Недвижимость'],
    mode='markers', opacity=0.7,
    name='Недвижимость', 
    text=df_2007['text'][df_2007['parent_category_name'] == 'Недвижимость'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['parent_category_name'] == 'Недвижимость'],
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df_2007['price_log'][df_2007['parent_category_name'] == 'Хобби и отдых'],
    y=df_2007['deal_probability'][df_2007['parent_category_name'] == 'Хобби и отдых'],
    mode='markers', opacity=0.7,
    name='Хобби и отдых', 
    text=df_2007['text'][df_2007['parent_category_name'] == 'Хобби и отдых'],
    marker=dict(
        sizemode='area',
        sizeref=sizeref,
        size=df_2007['size'][df_2007['parent_category_name'] == 'Хобби и отдых'],
        line=dict(
            width=2
        ),
    )
)

data = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(
    title='Price vs Deal Probability', showlegend=True,
    xaxis=dict(
        title="Price Logof Avito's Ads(US)",
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Deal Probability',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ), legend=dict(
        orientation="v")
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)

