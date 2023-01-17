# EDA and plotting libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots
df = pd.read_csv('/kaggle/input/fertilizers-by-product-fao/FertilizersProduct.csv',encoding='ISO-8859-1')

df.head()
data_nepal = df[(df['Area']=='Nepal')].reset_index(drop=True)

data_nepal.head()
fertilizer_by_item = df.groupby(["Item"])["Value"].sum().reset_index().sort_values("Value",ascending=False).reset_index(drop=True)
fig = px.pie(fertilizer_by_item, values=fertilizer_by_item['Value'], 

             names=fertilizer_by_item['Item'],

             title='Amount of Fertilizer used in Nepal',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_dark'

)

fig.show()
imported_urea = data_nepal.loc[(data_nepal['Item'] == 'Urea') & (data_nepal['Element'] == 'Import Value')]

imported_urea
fig = go.Figure()



fig.add_trace(go.Scatter(x=imported_urea['Year'], y=imported_urea['Value'],

                    mode='lines',

                    name='Urea',marker_color='green'))



fig.update_layout(

    title='Import of urea over the years in Nepal (US$1000)',

        template='plotly_dark'



)



fig.show()
cost_of_imported_urea = imported_urea['Value'].sum()

print('Total amount of money spend in Urea since 2002: Rs.{:.2f}'.format(cost_of_imported_urea*121.27))
imported_potassium = data_nepal.loc[(data_nepal['Item'] == 'Potassium chloride (muriate of potash) (MOP)') & (data_nepal['Element'] == 'Import Value')]

imported_potassium
fig = go.Figure()

fig.add_trace(go.Scatter(x=imported_urea['Year'], y=imported_urea['Value'],

                    mode='lines',

                    name='Urea'))



fig.add_trace(go.Scatter(x=imported_potassium['Year'], y=imported_potassium['Value'],

                    mode='lines',

                    name='Potassium Chloride',line=dict(dash='dot')))





fig.update_layout(

    title='Comparison between Urea and Potassium Chloride',

    template='plotly_dark',



)



fig.show()
fig = go.Figure(data=[go.Bar(

            x=fertilizer_by_item['Item'][0:10], y=fertilizer_by_item['Value'][0:10],

            text=fertilizer_by_item['Value'][0:10],

            textposition='auto',

            marker_color='red',

 

        )])

fig.update_layout(

    title='10 Most Used Fertilizer since 2002 in Nepal',

    xaxis_title="Items",

    yaxis_title="Value",

    template='plotly_dark'

)

fig.show()
plt.figure(figsize=(20,10))

sns.set_style('dark')

sns.countplot(x='Year',data=data_nepal);

plt.title('Import/Export of fertilizer over the years')
fig = px.pie(data_nepal,values=data_nepal['Element'].value_counts().values,

             names=df['Element'].unique(),

             title='Import and Export amount of fertilizer in Nepal',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(

    template='plotly_dark'

)

fig.show()
nepal_production = data_nepal.loc[data_nepal['Element'] == 'Production']

nepal_production.sort_values(by=['Value'], ascending=False)
fig = px.area(nepal_production, x="Year", y="Value", color="Item", line_group="Item", title='Production of fertilizers in Nepal')

fig.show()