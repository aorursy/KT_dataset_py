!pip install chart_studio
import math

import pandas as pd

import numpy as np

import seaborn as sns

sns.set(style="white")

import matplotlib.pyplot as plt

from sklearn import linear_model as lm

import statsmodels.api as sm



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from statsmodels.tools.eval_measures import mse, rmse



import warnings

warnings.filterwarnings('ignore')



# pd.options.display.max_columns = 999

%matplotlib inline



from matplotlib import style

style.use('fivethirtyeight')



import chart_studio

chart_studio.tools.set_credentials_file(username='d.kosarevsky', api_key='TY6xzvZdSXn0JiLotHNX')

import chart_studio.plotly as py

import plotly.graph_objects as go

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Для виз-ии https://streamlit.io ?
woh_data = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv')

woh_data.head()
woh_data.columns
woh_data.columns = ['Страна', 'Год', 'Статус', 'ОПЖ', 'Смертность(взрослые)', 

       'Смертность(младенцы)', 'Алкоголь(потребление)', 'Расходы на здравоохранение (на душу)', 'Гепатит B',

       'Корь', 'ИМТ', 'Смертность (до пяти лет)', 'Полиомиелит', 'Расходы на здравоохранение (общие)',

       'Дифтерия', 'ВИЧ/СПИД', 'ВВП','Население', 'Худоба (1-19 лет)', 'Худоба (5-9 лет)',

       'Доходы(индекс)', 'Образование(лет)']
woh_data.head()
wbank_data = pd.read_csv('/kaggle/input/world-bank-data-1960-to-2016/life_expectancy.csv')

wbank_data = wbank_data.rename(columns={'Country Name': 'Страна'}).drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
wbank_data.sample(5)
# rosstat_data = pd.read_excel('/kaggle/input/rosstat-all/rosstat_all.xls')

# # rosstat_data = rosstat_data.rename(columns={'Ожидаемая продолжительность жизни при рождении (2018, год, значение показателя за год)': 'Регион РФ'})

# rosstat_data.sample(3)
wbank_data_rus = wbank_data[wbank_data['Страна'] == 'Russian Federation']

wbank_data_rus = wbank_data_rus.set_index('Страна').T.reset_index()

wbank_data_rus.columns=['Год', 'ОПЖ']
wbank_data_chn = wbank_data[wbank_data['Страна'] == 'China']

wbank_data_chn = wbank_data_chn.set_index('Страна').T.reset_index()

wbank_data_chn.columns=['Год', 'ОПЖ']
wbank_data_rus_chn = pd.merge(wbank_data_rus, wbank_data_chn, how='left', left_on='Год', right_on='Год', suffixes=('_rus', '_chn'))

wbank_data_rus_chn.head(3)
china = go.Scatter(

    x=wbank_data_rus_chn['Год'].tolist(),

    y=wbank_data_rus_chn['ОПЖ_chn'].tolist(), name='Китай',

)

russia = go.Scatter(

    x=wbank_data_rus_chn['Год'].tolist(),

    y=wbank_data_rus_chn['ОПЖ_rus'].tolist(), name='Россия',

)



data = [china, russia]



py.iplot(data)
wbank_data_usa = wbank_data[wbank_data['Страна'] == 'United States']

wbank_data_usa = wbank_data_usa.set_index('Страна').T.reset_index()

wbank_data_usa.columns=['Год', 'ОПЖ']
wbank_data_rus_usa = pd.merge(wbank_data_rus, wbank_data_usa, how='left', left_on='Год', right_on='Год', suffixes=('_rus', '_usa'))

wbank_data_rus_usa.head(3)
usa = go.Scatter(

    x=wbank_data_rus_usa['Год'].tolist(),

    y=wbank_data_rus_usa['ОПЖ_usa'].tolist(), name='США',

)

russia = go.Scatter(

    x=wbank_data_rus_usa['Год'].tolist(),

    y=wbank_data_rus_usa['ОПЖ_rus'].tolist(), name='Россия',

)

data = [usa, russia]



py.iplot(data)
fig = px.line(wbank_data_rus, x='Год', y='ОПЖ')

fig.update_layout(

    title=go.layout.Title(

        text="Ожидаемая продолжительность жизни в России начиная с 1960 года (Данные всемирного банка)",

        xref="paper",

        x=0

    ),

)

fig.show()
woh_data_jpn = woh_data[woh_data['Страна'] == 'Japan']

woh_data_jpn.shape
woh_data_can = woh_data[woh_data['Страна'] == 'Canada']

woh_data_can.shape
woh_data_rus = woh_data[woh_data['Страна'] == 'Russian Federation']

woh_data_rus.shape
years = woh_data_rus['Год'].tolist()



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=woh_data_rus['ОПЖ'].tolist(),

                name='Россия',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=woh_data_jpn['ОПЖ'].tolist(),

                name='Япония',

                marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='ВОЗ, сравнение ожидаемой продолжительности жизни России и Японии',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='ОПЖ',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=.8,

        y=1.1,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
years = woh_data_rus['Год'].tolist()



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=woh_data_rus['ОПЖ'].tolist(),

                name='Россия',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=woh_data_can['ОПЖ'].tolist(),

                name='Канада',

                marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='ВОЗ, сравнение ожидаемой продолжительности жизни России и Канады',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='ОПЖ',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
woh_data_rus.sample(3)
woh_data_rus.columns
fig = px.bar(woh_data_rus, x='Год', y='ВИЧ/СПИД',

             hover_data=['ОПЖ', 'ВВП'], color='ОПЖ',

             labels={'ВИЧ/СПИД':'ВИЧ/СПИД России'}, height=400)

fig.update_layout(

    title='Корреляция ВИЧ/СПИД и ВВП на ожидание продолжительности жизни',)

fig.show()
# fig = go.Figure()

# fig.add_trace(go.Bar(

#     x=woh_data_rus['Год'].tolist(),

#     y=woh_data_rus['ОПЖ'].tolist(),

#     name='Смертность(младенцы)',

#     marker_color='indianred'

# ))

# fig.add_trace(go.Bar(

#     x=woh_data_rus['Год'].tolist(),

#     y=woh_data_rus['ОПЖ'].tolist(),

#     name='Смертность (взрослые)',

#     marker_color='lightsalmon'

# ))



# # Here we modify the tickangle of the xaxis, resulting in rotated labels.

# fig.update_layout(barmode='group', xaxis_tickangle=-45)

# fig.show()
# plt.figure(figsize=(5,5))

# sns.distplot(woh_data_rus[woh_data_rus['ОПЖ'] != np.nan]['Смертность(младенцы)'].dropna(), axlabel='Корреляция ОПЖ и младенческой смертности, Россия')

# plt.show()
plt.figure(figsize=(18,18))

sns.heatmap(woh_data_rus.corr(), annot=True, cmap='RdBu_r')

plt.title('Матрица корреляции факторов влияющих на ожидаемую продолжительность жизни в России');
woh_data_rus.head(2)
woh_data_rus_ill = woh_data_rus[['Гепатит B','Корь', 'Полиомиелит','Дифтерия','ВИЧ/СПИД', 'Худоба (1-19 лет)',

                                      'Худоба (5-9 лет)', 'ИМТ', 'ОПЖ']]



plt.figure(figsize=(8,8))

sns.heatmap(woh_data_rus_ill.corr(), annot=True, cmap='RdBu_r')

plt.title('Матрица корреляции между заболеваниями и ожидаемой продолжительностью жизни в России');
fig = px.line(woh_data_rus, x='Год', y='ОПЖ')

fig.update_layout(

    title=go.layout.Title(

        text="Ожидаемая продолжительность жизни в России начиная с 2000 года (Данные ВОЗ)",

        xref="paper",

        x=0

    ),

)

fig.show()
fig = px.scatter_matrix(woh_data, dimensions=['Худоба (1-19 лет)', 'Худоба (5-9 лет)', 'Доходы(индекс)', 'ВИЧ/СПИД', 'Смертность(младенцы)'], color='ОПЖ')

fig.update_layout(

    title='Корреляция Худобы, ВИЧ/СПИД, Доходов и Смертности среди младенцев на ожидание продолжительности жизни',)

fig.show()
fig = px.parallel_categories(woh_data, color="Год", color_continuous_scale=px.colors.sequential.Inferno)

fig.update_layout(

    title='Зависимость между годом и статусом страны',)

fig.show()
x0 = woh_data['Худоба (5-9 лет)'].tolist()

# Add 1 to shift the mean of the Gaussian distribution

x1 = woh_data['Доходы(индекс)'].tolist()



fig = go.Figure()

fig.add_trace(go.Histogram(x=x0, name='Худоба'))

fig.add_trace(go.Histogram(x=x1, name='Доходы'))



# Overlay both histograms

fig.update_layout(barmode='overlay')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
woh_data.columns
fig = px.box(woh_data, x="Год", y="ОПЖ",

             notched=True, # used notched shape

             title="Обзор ожидания продолжительности жизни в мире",

#              hover_data=["day"] # add day column to hover data

            )

fig.show()
regions = pd.read_csv('/kaggle/input/countryinfo/country-continent.csv')

regions = regions[['name', 'region', 'sub-region']]

regions.columns = ['Страна', 'Регион', 'Субрегион']
# # woh_data['Страна'].nunique()

# woh_cntrs = woh_data['Страна'].unique().tolist()

# woh_cntrs.sort()

# woh_cntrs[:5]



# regions_cntrs = regions['Страна'].unique().tolist()

# regions_cntrs.sort()

# regions_cntrs[:5]



# result_cntrs1 = list(set(woh_cntrs) - set(regions_cntrs))

# result_cntrs1
regions['Страна'] = regions['Страна'].replace('Congo (Democratic Republic of the)', 'Democratic Republic of the Congo')

regions['Страна'] = regions['Страна'].replace('Moldova (Republic of)', 'Republic of Moldova')

regions['Страна'] = regions['Страна'].replace('Korea (Republic of)', 'Republic of Korea')

regions['Страна'] = regions['Страна'].replace('Macedonia (the former Yugoslav Republic of)', 'The former Yugoslav republic of Macedonia')

regions['Страна'] = regions['Страна'].replace("Korea (Democratic People's Republic of)", "Democratic People's Republic of Korea")

regions['Страна'] = regions['Страна'].replace('Tanzania, United Republic of', 'United Republic of Tanzania')
# woh_cntrs = woh_data['Страна'].unique().tolist()

# woh_cntrs.sort()

# regions_cntrs = regions['Страна'].unique().tolist()

# regions_cntrs.sort()



# result_cntrs1 = list(set(woh_cntrs) - set(regions_cntrs))

# result_cntrs1
woh_data_regions = pd.merge(woh_data, regions, how='left', left_on='Страна', right_on='Страна')

woh_data_regions.sample(3)
fig = px.box(woh_data_regions.loc[woh_data_regions['Год'] > 2009], x="Год", y="ОПЖ", color="Регион",

             notched=True, # used notched shape

             title="Значения продолжительности жизни по континентальным регионам с 2010 года",

            )

fig.show()
woh_data_regions.columns
fig = px.scatter(woh_data_regions, x="ВВП", y="ОПЖ",

	         size="Смертность(младенцы)", color="Регион",

                 hover_name="Страна", log_x=True, size_max=60)

fig.show()
fig = px.scatter(woh_data_regions, x="ВВП", y="ОПЖ",

	         size="ВИЧ/СПИД", color="Регион",

                 hover_name="Страна", log_x=True, size_max=60)

fig.show()
fig = px.scatter(woh_data_regions, x="ВВП", y="ОПЖ",

	         size="Расходы на здравоохранение (на душу)", color="Регион",

                 hover_name="Страна", log_x=True, size_max=60)

fig.show()
happines_2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

happines_2015.sample(3)
happines_2015.columns=(['Страна', 'Регион', 'Индекс счастья', 'Оценка счастья', 'Стандартная ошибка', 'ВВП на душу населения', 

'Семья', 'ОПЖ', 'Свобода', 'Доверие правительству', 'Щедрость', 'Остаточная дистопия'])

happines_2015 = happines_2015.drop(columns=['Регион', 'ВВП на душу населения', 'ОПЖ'])
happines_2015.sample(3)
# happ_cntrs = happines_2015['Страна'].unique().tolist()

# happ_cntrs.sort()



# regions_cntrs = woh_data_regions['Страна'].unique().tolist()

# regions_cntrs.sort()
happines_2015['Страна'] = happines_2015['Страна'].replace('Bolivia', 'Bolivia (Plurinational State of)')

happines_2015['Страна'] = happines_2015['Страна'].replace('Somaliland region', 'Somalia')

happines_2015['Страна'] = happines_2015['Страна'].replace('Moldova', 'Republic of Moldova')

happines_2015['Страна'] = happines_2015['Страна'].replace('Congo (Kinshasa)', 'Democratic Republic of the Congo')

happines_2015['Страна'] = happines_2015['Страна'].replace('United States', 'United States of America')

happines_2015['Страна'] = happines_2015['Страна'].replace('Macedonia', 'The former Yugoslav republic of Macedonia')

happines_2015['Страна'] = happines_2015['Страна'].replace('United Kingdom', 'United Kingdom of Great Britain and Northern Ireland')

happines_2015['Страна'] = happines_2015['Страна'].replace('Venezuela', 'Venezuela (Bolivarian Republic of)')

happines_2015['Страна'] = happines_2015['Страна'].replace('Iran', 'Iran (Islamic Republic of)')

happines_2015['Страна'] = happines_2015['Страна'].replace('Congo (Brazzaville)', 'Congo')

happines_2015['Страна'] = happines_2015['Страна'].replace('Syria', 'Syrian Arab Republic')

happines_2015['Страна'] = happines_2015['Страна'].replace('Russia', 'Russian Federation')

happines_2015['Страна'] = happines_2015['Страна'].replace('Tanzania', 'United Republic of Tanzania')
# result_cntrs1 = list(set(happ_cntrs) - set(regions_cntrs))

# result_cntrs1
# result_cntrs2 = list(set(regions_cntrs) - set(happ_cntrs))

# result_cntrs2
woh_data_regions_happ = pd.merge(woh_data_regions, happines_2015, how='left', left_on='Страна', right_on='Страна')

woh_data_regions_happ.sample(3)
woh_data_regions_happ.columns
woh_data_regions_happ_corr = woh_data_regions_happ[['Страна', 'Регион', 'Оценка счастья', 'Стандартная ошибка', 'ВВП', 

'Семья', 'ОПЖ', 'Свобода', 'Доверие правительству', 'Щедрость', 'Остаточная дистопия']]



plt.figure(figsize=(8,8))

sns.heatmap(woh_data_regions_happ_corr.corr(), annot=True, cmap='RdBu_r')

plt.title('Матрица корреляции с новыми признаками в разрезе всех стран');
# ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania']
fig = px.line(woh_data_regions_happ[woh_data_regions_happ['Регион'] == 'Oceania'], x='Год', y='ОПЖ', color='Страна')

fig.show()
fig = px.line(woh_data_regions_happ[woh_data_regions_happ['Регион'] == 'Americas'], x='Год', y='ОПЖ', color='Страна')

fig.show()
fig = px.line(woh_data_regions_happ[woh_data_regions_happ['Регион'] == 'Africa'], x='Год', y='ОПЖ', color='Страна')

fig.show()
fig = px.line(woh_data_regions_happ[woh_data_regions_happ['Регион'] == 'Asia'], x='Год', y='ОПЖ', color='Страна')

fig.show()
fig = px.line(woh_data_regions_happ[woh_data_regions_happ['Регион'] == 'Europe'], x='Год', y='ОПЖ', color='Страна')

fig.show()