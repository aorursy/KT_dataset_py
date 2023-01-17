import IPython

url = "https://flo.uri.sh/visualisation/1953176/embed"

iframe = '<iframe src=' + url + ' width=800 height=600></iframe>'

IPython.display.HTML(iframe)
import numpy as np

import pandas as pd

import pycountry

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_white"

from plotly.subplots import make_subplots

%config IPCompleter.greedy=True



import os

os.chdir("../input/novel-corona-virus-2019-dataset")



cleaned_data = pd.read_csv('covid_19_data.csv', parse_dates=['ObservationDate']) 



cleaned_data.rename(columns={'ObservationDate': 'Дата', 

                     'Province/State':'Облус',

                     'Country/Region':'Мамлекет',

                     'Last Update':'Акыркы_өзгөрүүлөр',

                     'Confirmed': 'Ооругандар',

                     'Deaths':'Көз_жумгандар',

                     'Recovered':'Айыккандар'

                    }, inplace=True)



cases = ['Ооругандар', 'Көз_жумгандар', 'Айыккандар', 'Активдүү']



# Активдүү = Ооругандар - Көз жумгандар

cleaned_data['Активдүү'] = cleaned_data['Ооругандар'] - cleaned_data['Көз_жумгандар'] - cleaned_data['Айыккандар']



# Кытай = Кытай

cleaned_data['Мамлекет'] = cleaned_data['Мамлекет'].replace('Mainland China', 'China')



# Толуктоо 

cleaned_data[['Облус']] = cleaned_data[['Облус']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)

cleaned_data.rename(columns={'Date':'Дата'}, inplace=True)



data = cleaned_data



grouped = data.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()

fig = px.line(grouped, x="Дата", y="Ооругандар", 

              title="Коронавирустун дуйнөдө тарашы")

fig.show()





grouped_china = data[data['Мамлекет'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()



grouped_italy = data[data['Мамлекет'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()



grouped_us = data[data['Мамлекет'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()



grouped_kg = data[data['Мамлекет'] == "Kyrgyzstan"].reset_index()

grouped_kg_date = grouped_kg.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()



grouped_kz = data[data['Мамлекет'] == "Kazakhstan"].reset_index()

grouped_kz_date = grouped_kz.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()



grouped_rest = data[~data['Мамлекет'].isin(['China', 'Italy', 'US','Kyrgyzstan', 'Kazakhstan'])].reset_index()

grouped_rest_date = grouped_rest.groupby('Дата')['Дата', 'Ооругандар', 'Көз_жумгандар'].sum().reset_index()


fig = px.line(grouped_china_date, x="Дата", y="Ооругандар", 

              title=f"Кытайда ооругандардын саны", 

              color_discrete_sequence=['#F61067'],

              height=500

             )

fig.show()



fig = px.line(grouped_italy_date, x="Дата", y="Ооругандар", 

              title=f"Италияда ооругандардын саны", 

              color_discrete_sequence=['#91C4F2'],

              height=500

             )

fig.show()



fig = px.line(grouped_us_date, x="Дата", y="Ооругандар", 

              title=f"АКШда ооругандардын саны", 

              color_discrete_sequence=['#6F2DBD'],

              height=500

             )

fig.show()



fig = px.line(grouped_kg_date, x="Дата", y="Ооругандар", 

              title=f"Кыргызстанда ооругандардын саны", 

              color_discrete_sequence=['#6F2DBD'],

              height=500

             )

fig.show()



fig = px.line(grouped_kz_date, x="Дата", y="Ооругандар", 

              title=f"Казакстанда ооругандардын саны", 

              color_discrete_sequence=['#6F2DBD'],

              height=500

             )

fig.show()



fig = px.line(grouped_rest_date, x="Дата", y="Ооругандар", 

              title=f"Башка өлкөлөрдө ооругандардын саны", 

              color_discrete_sequence=['#FFDF64'],

              height=500

             )

fig.show()
data['Облус'] = data['Облус'].fillna('')

temp = data[[col for col in data.columns if col != 'Область']]



latest = temp[temp['Дата'] == max(temp['Дата'])].reset_index()

latest_grouped = latest.groupby('Мамлекет')['Ооругандар', 'Көз_жумгандар', 'Айыккандар'].sum().reset_index()

fig = px.choropleth(latest_grouped, locations="Мамлекет", 

                    locationmode='country names', color="Ооругандар", 

                    hover_name="Мамлекет", range_color=[1,5000], 

                    color_continuous_scale="peach", 

                    title='Оору тастыкталган мамлекеттер')

fig.show()
cis = list(['Armenia','Azerbaijan','Belarus','Kazakhstan',

            'Kyrgyzstan','Moldova','Russia','Tajikistan','Uzbekistan'])



cis_grouped_latest = latest_grouped[latest_grouped['Мамлекет'].isin(cis)]

fig = px.choropleth(cis_grouped_latest, locations="Мамлекет", 

                    locationmode='country names', color="Ооругандар", 

                    hover_name="Мамлекет", range_color=[1,500], 

                    color_continuous_scale='portland', 

                    title='КМШда коронавирус тастыкталган мамлекеттер', scope='world', height=800)

fig.show()
fig = px.bar(latest_grouped.sort_values('Ооругандар', ascending=False)[:20][::-1], 

             x='Ооругандар', y='Мамлекет',

             title='Коронавирус тастыкталган мамлекеттер', text='Ооругандар', height=1000, orientation='h')

fig.show()
fig = px.bar(cis_grouped_latest.sort_values('Ооругандар', ascending=False)[:10][::-1], 

             x='Ооругандар', y='Мамлекет', color_discrete_sequence=['#84DCC6'],

             title='КМШда тастыкталган мамлекеттер', text='Ооругандар', orientation='h')

fig.show()
fig = px.line(grouped, x="Дата", y="Көз_жумгандар", title="Дүйнө жүзүндө коронавирустан көз жумгандардын саны",

             color_discrete_sequence=['#F42272'])

fig.show()
fig = px.bar(latest_grouped.sort_values('Көз_жумгандар', ascending=False)[:10][::-1], 

             x='Көз_жумгандар', y='Мамлекет', color_discrete_sequence=['#F42272'],

             title="Дүйнө жүзүндө коронавирустан көз жумгандардын саны", text='Көз_жумгандар', orientation='h')

fig.show()
fig = px.bar(cis_grouped_latest.sort_values('Көз_жумгандар', ascending=False)[:9][::-1], 

             x='Көз_жумгандар', y='Мамлекет',

             title="КМШда коронавирустан көз жумгандардын саны", text='Көз_жумгандар', orientation='h')

fig.show()
fig = px.bar(latest_grouped.sort_values('Айыккандар', ascending=False)[:10][::-1], 

             x='Айыккандар', y='Мамлекет',

             title='Дүйнө жүзүндө коронавирустан айыккандардын саны', text='Айыккандар', orientation='h')

fig.show()
fig = px.bar(cis_grouped_latest.sort_values('Айыккандар', ascending=False)[:9][::-1], 

             x='Айыккандар', y='Мамлекет',

             title="КМШда коронавирустан айыккандардын саны", text='Айыккандар', orientation='h')

fig.show()
temp = cleaned_data.groupby('Дата')['Айыккандар', 'Көз_жумгандар', 'Ооругандар'].sum().reset_index()

temp = temp.melt(id_vars="Дата", value_vars=['Айыккандар', 'Көз_жумгандар', 'Ооругандар'],

                 var_name='Кырдаал', value_name='Адамдардын саны')





fig = px.line(temp, x="Дата", y="Адамдардын саны", color='Кырдаал',

             title='Коронавирустун тарашы сызык графигинде', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()





fig = px.area(temp, x="Дата", y="Адамдардын саны", color='Кырдаал',

             title='Коронавирустун тарашы аймак графигинде', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.show()
cleaned_latest = cleaned_data[cleaned_data['Дата'] == max(cleaned_data['Дата'])]

flg = cleaned_latest.groupby('Мамлекет')['Ооругандар', 'Көз_жумгандар', 'Айыккандар', 'Активдүү'].sum().reset_index()



flg['Өлүм'] = round((flg['Көз_жумгандар']/flg['Ооругандар'])*100, 2)

temp = flg[flg['Ооругандар']>100]

temp = temp.sort_values('Өлүм', ascending=False)



fig = px.bar(temp.sort_values(by="Өлүм", ascending=False)[:10][::-1],

             x = 'Өлүм', y = 'Мамлекет', 

             title='Ар бир 100 ооруган адамга көз жумгандардын саны', text='Өлүм', height=800, orientation='h',

             color_discrete_sequence=['darkred']

            )

fig.show()
flg['Айыккандардын_саны'] = round((flg['Айыккандар']/flg['Ооругандар'])*100, 2)

temp = flg[flg['Ооругандар']>100]

temp = temp.sort_values('Айыккандардын_саны', ascending=False)



fig = px.bar(temp.sort_values(by="Айыккандардын_саны", ascending=False)[:10][::-1],

             x = 'Айыккандардын_саны', y = 'Мамлекет', 

             title='Ар бир 100 ооруган адамга айыккандардын саны', text='Айыккандардын_саны', height=800, orientation='h',

             color_discrete_sequence=['#2ca02c']

            )

fig.show()
formated_gdf = data.groupby(['Дата', 'Мамлекет'])['Ооругандар', 'Көз_жумгандар'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Дата'] = pd.to_datetime(formated_gdf['Дата'])

formated_gdf['Дата'] = formated_gdf['Дата'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Ооругандар'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Мамлекет", locationmode='country names', 

                     color="Ооругандар", size='size', hover_name="Мамлекет", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Дата", 

                     title='Дүйнөдө коронавирустун тарашы', color_continuous_scale="portland")

fig.show()