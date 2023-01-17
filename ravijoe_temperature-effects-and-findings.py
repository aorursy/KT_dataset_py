

import seaborn as sns

import numpy as np

import pandas as pd

from os import path

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



path="../input/CORD-19-research-challenge"

metadata=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

# import os

# os.listdir(path)

metadata.head()
metadata.columns
metadata.shape
word_cloud_text = ''.join(str(i) for i in metadata['title'])

word_cloud_text


plt.figure( figsize=(20,10) )



wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(word_cloud_text)

# plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
for i in metadata['title'].values:

    if 'incubation' in str(i):

        print(i)

        print('*******')
cure_text=''

for i in metadata['abstract'].values:

    if 'cure' in str(i):

        cure_text+=i

import pprint

pprint.pprint(cure_text)
plt.figure( figsize=(20,10) )

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="black").generate(cure_text)

# plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
medicine_text=''

for i in metadata['abstract'].values:

    if 'medicine' in str(i):

        medicine_text+=i

        print(i)

        print('********************')

medicine_text

        
import spacy 

  

nlp = spacy.load('en_core_web_sm') 

  

sentence = cure_text

  

doc = nlp(sentence ) 

  

for ent in doc.ents: 

    if ent.label_ not in['DATE','GPE']:

        print(ent.text,  ent.label_) 
# temperature_text=''

# for i in metadata['abstract'].values:

#     if 'temperature' or'weather' or 'warm' or 'cool' or 'cold' in str(i):

#         temperature_text+=str(i)

#         print(i)

#         print('********************')

# temperature_text
global_temp_country = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")

df_temp=global_temp_country.groupby('Country')['AverageTemperature'].mean()

# Series({'avg_temp_by_country' : global_temp_country.groupby( "Country")['AverageTemperature'].mean()}).reset_index()

df_temp=df_temp.apply(pd.Series).reset_index()

df_temp.columns=['Country/Region','avg_temp_by_country']

df_temp
full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

full_table.head()
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Defining Active Case: Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# Renaming Mainland china as China in the data table

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[cases] = full_table[cases].fillna(0)



# cases in the ships

ship = full_table[full_table['Province/State'].str.contains('Grand Princess')|full_table['Country/Region'].str.contains('Cruise Ship')]



# china and the row

china = full_table[full_table['Country/Region']=='China']

row = full_table[full_table['Country/Region']!='China']



# latest

full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

china_latest = full_latest[full_latest['Country/Region']=='China']

row_latest = full_latest[full_latest['Country/Region']!='China']



# latest condensed

full_latest_grouped = full_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

china_latest_grouped = china_latest.groupby('Province/State')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

row_latest_grouped = row_latest.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = full_table.groupby(['Country/Region', 'Province/State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].max()
temp_f = full_latest_grouped.sort_values(by='Confirmed', ascending=False)

temp_f
df_final = pd.merge(df_temp,temp_f,how='left',on=['Country/Region'])

df_final

import seaborn as sns

import matplotlib.pyplot as plt

corr = df_final.corr()

# ax = sns.heatmap(

#     corr, 

#     vmin=-1, vmax=1, center=0,

#     cmap=sns.diverging_palette(20, 220, n=200),

#     square=True

# )



plt.figure(figsize = (16,5))

sns.heatmap(corr, annot=True, linewidths=.5)
plt.figure(figsize=(30,15))

from scipy import stats

sns.jointplot(x="avg_temp_by_country", y="Deaths",data=df_final,kind='reg',stat_func=stats.pearsonr)

# plt.figure(figsize=(30,15))

plt.show()