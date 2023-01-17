import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.subplots import make_subplots

import plotly.express as px

import plotly.graph_objs as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

%config InlineBackend.figure_format = 'retina'

plt.rc('axes', unicode_minus = False)
df = pd.read_csv('/kaggle/input/volcanic-eruptions/database.csv')

df.head()
df.info()
df.isnull().sum()
# BCE -> -

# CE -> +



# 함수 생성



def BCE_CE(x):

    if x.split()[-1] == 'BCE':

        out = int(x.split()[0]) * -1

    elif x.split()[-1] == 'CE':

        out = int(x.split()[0])

    else:

        out = np.nan

    return out
df['Last Known Eruption(year)'] = df['Last Known Eruption'].map(lambda x : BCE_CE(x))

df.head()
# 나머지 column의 null값 처리

df['Activity Evidence'].fillna('Unknown', inplace = True)

df['Dominant Rock Type'].fillna('Unknown', inplace = True)

df['Tectonic Setting'].fillna('Unknown', inplace = True)



df.isnull().sum()
fig = sns.pairplot(df)
col_list = df.columns.tolist()

col_list
# 화산 분포도

fig = px.scatter_geo(data_frame= df, lat = 'Latitude', lon = 'Longitude', color= 'Region')

fig.show()
# 화산 분포 지역

values = df['Region'].value_counts()

labels = df['Region'].value_counts().index



fig = go.Figure(go.Pie(values= values, labels= labels))

fig.update_traces(textinfo = 'percent + label', textfont_size= 12

                  ,marker=dict(line=dict(color='#000000', width=1)))

fig.show()
df['Type'].value_counts()
# 상위 6개의 화산종류 분포 현황

type_top6 = df['Type'].value_counts().head(6).index.tolist()

type_top6
fig = px.scatter_geo(data_frame= df[df['Type'].isin(type_top6)], 

                     lat = 'Latitude', lon = 'Longitude', color = 'Type',

                    title = 'Top 6 of Volcano type')

fig.show()
fig = px.bar(df['Country'].value_counts().head(10), title = 'Top 10 of Country')

fig.show()
# Top 10의 화산 분포현황

top10_list = df['Country'].value_counts().head(10).index.tolist()

fig = px.scatter_geo(data_frame= df[df['Country'].isin(top10_list)], 

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Country',

                    title = 'Top 10 of Country with Volcano')

fig.show()
fig = px.scatter_geo(data_frame= df[df['Country'].isin(['United States'])], 

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Type',

                    title = 'Volcano types in USA')

fig.show()
fig = px.scatter_geo(data_frame= df[df['Country'].isin(['Russia'])], 

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Type',

                    title = 'Volcano types in USA')

fig.show()
fig = px.histogram(data_frame = df, x = 'Last Known Eruption(year)', marginal= 'rug', nbins=100)

fig.show()
# 1D 로 표현

y_val = df['Last Known Eruption(year)'].shape[0]



fig = go.Figure(go.Scatter( x = df['Last Known Eruption(year)'], 

                           y = np.zeros(y_val),

                           mode = 'markers' ,marker_size = 10))



fig.add_annotation(x = df['Last Known Eruption(year)'].min(), y= 0, 

                   text = 'First eruption' )



fig.add_annotation(x = df['Last Known Eruption(year)'].max(), y= 0, 

                   text = 'Recent eruption' )



fig.update_xaxes(showgrid=False)

fig.update_yaxes(showgrid=False, 

                 zeroline=True, zerolinecolor='black', 

                 zerolinewidth=5,

                 showticklabels=False)

fig.update_layout(height=300, plot_bgcolor='white')



fig.show()
fig = px.scatter_geo(data_frame = df.sort_values('Last Known Eruption(year)').head(100),

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Region',

                    title = 'Top 100 of Volcano erupted a long time ago')

fig.show()
fig = px.scatter_geo(data_frame = df[df['Last Known Eruption(year)'].isin([2016])],

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Region',

                    title = 'Volcano eruption in 2016')

fig.show()
#가장 높은 화산은?

df_highest = df.sort_values('Elevation (Meters)', 

               ascending = False).head(10)[['Name', 

                                            'Country', 

                                            'Region', 

                                            'Type',

                                            'Elevation (Meters)' , 

                                            'Latitude','Longitude',

                                           'Last Known Eruption']]



df_highest.style.background_gradient('Reds')
#가장 낮은 화산은?

df_lowest = df.sort_values('Elevation (Meters)', 

               ascending = True).head(10)[['Name', 

                                            'Country', 

                                            'Region', 

                                            'Type',

                                            'Elevation (Meters)',

                                           'Latitude','Longitude',

                                          'Last Known Eruption']]

df_lowest.style.background_gradient('Reds')
# highest + lowest

df_high_low = pd.concat([df_highest, df_lowest], axis = 0)



fig = px.scatter_geo(df_high_low, lat= 'Latitude', lon= 'Longitude', 

                     color = df_high_low['Elevation (Meters)'] > 0)

fig.update_layout(title = 'Top 10 of Highest & Lowest, Blue : Highest, Red : Lowest')

fig.show()

values = df['Dominant Rock Type'].value_counts()

names = df['Dominant Rock Type'].value_counts().index





fig = px.pie(values = values, names= names)

fig.update_traces(textinfo = 'percent + label', textfont_size= 12

                  ,marker=dict(line=dict(color='#000000', width=1)))

fig.show()
fig = px.scatter_geo(df[df['Dominant Rock Type'].isin(labels[:2])],

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Dominant Rock Type',

                    title = 'Dominant Rock Type')

fig.show()
df['Tectonic Setting'].value_counts()
fig = px.scatter_geo(df,

                     lat = 'Latitude', lon = 'Longitude',

                     color = 'Tectonic Setting',

                    title = 'Tectonic Setting')

fig.show()