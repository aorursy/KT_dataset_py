import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

#import matplotlib as mpl

import matplotlib.pyplot as plt

#import matplotlib.font_manager as fm

import plotly.offline as py

import colorlover as cl

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

import cufflinks as cf
py.offline.init_notebook_mode(connected=True)

pd.set_option('float_format', '{:.4f}'.format)

init_notebook_mode(connected=True)



cf.set_config_file(offline=True, world_readable=True, theme='ggplot')



# colab 환경에서 사용

#mpl.rc('font', family='nanumgothic')

#mpl.rc('axes', unicode_minus=False)
train = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')
train.info()
train[train['id'].duplicated()]
dataCorr_10 = abs(train.corr(method='spearman')).nlargest(10, 'price').index

cm = np.array(sp.stats.spearmanr(train[dataCorr_10].values))[0]

plt.figure(figsize=(15,10))

sns.heatmap(cm, annot=True, linewidths=.5, yticklabels=dataCorr_10.values, xticklabels=dataCorr_10.values)
price = train['price']
price.describe()
price.T.iplot(kind='histogram', bins=100, histnorm='percent')
seoul = pd.read_csv('../input/2014-seoul-house-price/2014.csv')
seoul.info()
seoul['물건금액'] = seoul['물건금액'].div(1000)
seoul['물건금액'].describe()
seoul['물건금액'].sort_values().tail(10)
pd.DataFrame({

    'King County': price,

    '서울특별시': seoul[seoul['물건금액'] < 8000000]['물건금액']

}).iplot(kind='histogram', bins=100, histnorm='percent')
train['grade'].value_counts().sort_index()
grade_df = []

grade = train[['grade', 'price']]

grade[grade['grade'] == 0]['price'].reset_index(drop=True).values



for i in range(1, 14):

    grade_df.append(pd.DataFrame({i: grade[grade['grade'] == i]['price'].values}))



grade_df=pd.concat(grade_df)
grade_df.iplot(kind='box', boxpoints='outliers')
sqft_living = train[['sqft_living', 'price']]

sqft_living.head()
sqft_living.describe()
f, ax = plt.subplots(figsize=(12, 10))

fig = sns.regplot(x='sqft_living', y="price", data=sqft_living)
asd = train[['sqft_living', 'sqft_lot', 'lat', 'long', 'price']]

asd.head()
trace1 = go.Scatter3d(

    x=asd.long,

    y=asd.lat,

    z=asd.price,

    mode='markers',

    marker=dict(

            size=1.8,

            cmax=asd.sqft_living.quantile(0.99),

            cmin=asd.sqft_living.quantile(0.01),

            color=asd.sqft_living,

            colorbar=dict(

                title='Colorbar'

            ),

            colorscale='Viridis'

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)



fig = dict(data=data, layout=layout)



py.iplot(fig)
asd['sqft_per_floors'] = train['sqft_living'] / train['floors']
trace1 = go.Scatter3d(

    x=asd.long,

    y=asd.lat,

    z=asd.price,

    mode='markers',

    marker=dict(

            size=1.8,

            cmax=asd.sqft_per_floors.quantile(0.99),

            cmin=asd.sqft_per_floors.quantile(0.01),

            color=asd.sqft_per_floors,

            colorbar=dict(

                title='Colorbar'

            ),

            colorscale='Viridis'

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)



fig = dict(data=data, layout=layout)



py.iplot(fig)
(train['sqft_lot']-train['sqft_living']).describe()
asd['without_living'] = train['sqft_lot'] - train['sqft_living']

asd.head()
asd.without_living.describe()
asd['wl_cut'] = pd.qcut(asd['without_living'], q=1000, labels=list(range(1000))).astype(int)
asd.plot(kind='scatter', x='lat', y='long', c='wl_cut', colormap='viridis', alpha=0.1)
trace1 = go.Scatter3d(

    x=asd.long,

    y=asd.lat,

    z=asd.price,

    mode='markers',

    text=asd.without_living,

    marker=dict(

            size=2,

            cmax=asd.wl_cut.quantile(0.99),

            cmin=asd.wl_cut.quantile(0.01),

            color=asd.wl_cut,

            colorbar=dict(

                title='Colorbar'

            ),

            colorscale='Viridis'

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)



fig = dict(data=data, layout=layout)



py.iplot(fig)
trace1 = go.Scatter3d(

    x=asd.long,

    y=asd.lat,

    z=asd.price,

    mode='markers',

    text=asd.without_living,

    marker=dict(

            size=2,

            cmax=1,

            cmin=0,

            color=(asd.sqft_per_floors / asd.sqft_lot),

            colorbar=dict(

                title='Colorbar'

            ),

            colorscale='Viridis'

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)



fig = dict(data=data, layout=layout)



py.iplot(fig)
train.sort_values('date')[['date', 'price']].head()
train[['date', 'price']].groupby(train['date'].str[:6]).size()
train_month = sorted(train['date'].str[:6].unique())

train_month
top5_zipcode = train.zipcode.value_counts().nlargest(5).index.tolist()

top5_zipcode
top5_monthly = []



for i in train_month: #월

    monthly_price = []

    for j in top5_zipcode: #집코드

        monthly_price.append(train[(train.date.str[:6] == i) & (train.zipcode == j)].price.mean())

    top5_monthly.append(monthly_price)
top5_monthly = pd.DataFrame(top5_monthly).T

top5_monthly
data = []

for i in range(5):

    data.append(go.Scatter(

        x = [i for i in range(12)], 

        y = top5_monthly.iloc[i], 

        name = top5_zipcode[i],

        mode = 'lines+markers'

    ))

layout= go.Layout(

    title= '거래량 상위 5개 지역 월별 평균거래가 변동',

    xaxis=go.layout.XAxis(

        ticktext=train_month,

        tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    ),

    yaxis=dict(

        title= '',

        ticklen= 5,

        gridwidth= 2,

    )

)

fig = go.Figure(data, layout)

py.iplot(fig)
loc_price = train[['lat', 'long', 'price']]
loc_price.head()
loc_price['price_cut'] = pd.qcut(loc_price['price'], q=10, labels=list(range(10))).astype(int)
loc_price.head()
loc_price.plot(kind='scatter', x='lat', y='long', c='price_cut', cmap=plt.get_cmap('viridis'), alpha=0.1)
floors = train[['price', 'floors', 'lat', 'long']]
floors.floors.value_counts()
trace1 = go.Scatter3d(

    x=floors.long,

    y=floors.lat,

    z=floors.price,

    mode='markers',

    marker=dict(

            size=2,

            cmax=4,

            cmin=0,

            color=floors.floors,

            colorbar=dict(

                title='Colorbar'

            ),

            colorscale='Viridis'

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)



fig = dict(data=data, layout=layout)



py.iplot(fig)
floors_list = []

data = []



floors_list.append(floors[floors.floors == 1])

floors_list.append(floors[floors.floors == 1.5])

floors_list.append(floors[floors.floors == 2])

floors_list.append(floors[floors.floors == 2.5])

floors_list.append(floors[floors.floors == 3])

floors_list.append(floors[floors.floors == 3.5])

for i in range(0,6):

    data.append(

        go.Scatter3d(

        x=floors_list[i].long,

        y=floors_list[i].lat,

        z=floors_list[i].floors,

        text=floors_list[i].price,

        name=['1', '1.5', '2', '2.5', '3', '3.5'][i],

        mode='markers',

        marker=dict(

                size=2,

                cmax=pd.concat(floors_list).price.quantile(0.99),

                cmin=pd.concat(floors_list).price.quantile(0.01),

                color=floors_list[i].price,

                colorbar=dict(

                    title='가격'

                ),

                colorscale='Viridis'

        )

    )

    )

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    legend=dict(

        x=0,

        y=1,

        traceorder='normal',

        font=dict(

            family='sans-serif',

            size=12,

            color='#000'

        ),

        bgcolor='#E2E2E2',

        bordercolor='#FFFFFF',

        borderwidth=2

    ),

    yaxis=dict(scaleanchor="x", scaleratio=1)

)



fig = dict(data=data, layout=layout)



py.iplot(fig)
top5_monthly.set_index(pd.Series(top5_zipcode), inplace=True, drop=True)

top5_monthly.columns = train_month

top5_monthly
top5_df = train[train.zipcode.isin(top5_monthly.index)]
data = []

for i in top5_monthly.index:

    data.append(

        go.Scatter(

        x=top5_df[top5_df.zipcode == i].long,

        y=top5_df[top5_df.zipcode == i].lat,

        text=top5_df[top5_df.zipcode == i].price,

        name=i,

        mode='markers',

        marker=dict(

                size=2,

                cmax=top5_df.price.quantile(0.99),

                cmin=top5_df.price.quantile(0.01),

                color=top5_df[top5_df.zipcode == i].price,

                colorbar=dict(

                    title='가격'

                ),

                colorscale='Viridis'

        )

    )

    )

    

layout = go.Layout(

    title='거래량 상위 5개 지역',

    legend=dict(

        x=0,

        y=1,

        traceorder='normal',

        font=dict(

            family='sans-serif',

            size=12,

            color='#000'

        ),

        bgcolor='#E2E2E2',

        bordercolor='#FFFFFF',

        borderwidth=2

    ),

    yaxis=dict(scaleanchor="x", scaleratio=1)

)

fig = go.Figure(data, layout)

py.iplot(fig)