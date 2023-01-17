import re

import math

import time

import numpy as np

import pandas as pd
import requests

import urllib.request

from bs4 import BeautifulSoup

from urllib.request import urlopen



url = 'https://en.wikipedia.org/wiki/List_of_countries_by_sex_ratio'

html = urlopen(url) 

soup = BeautifulSoup(html, 'html.parser')
tables = soup.find_all('table')



# ---- Validate input propriety ---- #

def process_num(num):

    if not math.isnan(float(num)):

        res = float(re.sub(r'[^\w\s.]','',num))

    else:

        res = num

    return res



# ----- Validate cell propriety ---- #

def is_cell_valid(cells):

    for i in range(len(cells)):

        if cells[i].text.strip() == 'N/A':

            return False

    return True



# ---- Validate string propriety --- #

def make_str_valid( str ):

    ind_delim = str.find('(')



    if ind_delim != -1 :

        wrd = str[:ind_delim-1]

    else:

        wrd = str

        

    return wrd
countries, Sex_R = [], []



for table in tables:

    rows = table.find_all('tr')

    

    for row in rows:

        cells = row.find_all('td')

        

        if ( len(cells) > 1 and is_cell_valid(cells) ):

            # Col_1 :: country

            country = cells[0]

            country_strip = country.text.strip()

            countries.append( make_str_valid( country_strip ))

            

            # Col_2 :: sex-ratio

            col_last = len(cells)-1

            S_R = cells[col_last]

            Sex_R.append(process_num(S_R.text.strip()))
# Instantiate data frame

df = pd.DataFrame({'Country':countries, 'Sex-Ratio':Sex_R})



# Clean data-frame ( Duplicates & NaNs )

df[df.duplicated()]

df = df.drop_duplicates(subset=['Country'], keep='last').dropna()

df.sample(15)
country_raw = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')

df_c = country_raw.iloc[:, [0,2]]

df_c = df_c.rename(columns={'name':'Country', 'alpha-3':'ISO-code'})

df_c = df_c.drop_duplicates(subset=['Country'], keep='last').dropna()



# Manual modification of the data

df_c.at[ df_c[df_c['Country']=='Viet Nam'].index.values[0], 'Country' ] = 'Vietnam'

df_c.at[ df_c[df_c['Country']=='United States of America'].index.values[0], 'Country' ] = 'United States'

df_c.at[ df_c[df_c['Country']=='Iran (Islamic Republic of)'].index.values[0], 'Country' ] = 'Iran'
df_c.at[ df_c[df_c['Country']=='Russian Federation'].index.values[0], 'Country' ] = 'Russia'

df_c.at[ df_c[df_c['Country']=='United Kingdom of Great Britain and Northern Ireland'].index.values[0], 'Country' ] = 'United Kingdom'

df_c.at[ df_c[df_c['Country']=='Venezuela (Bolivarian Republic of)'].index.values[0], 'Country' ] = 'Venezuela'

df_c.at[ df_c[df_c['Country']=='Korea (Democratic People\'s Republic of)'].index.values[0], 'Country' ] = 'Korea, North'

df_c.at[ df_c[df_c['Country']=='Korea, Republic of'].index.values[0], 'Country' ] = 'Korea, South'

df_c.at[ df_c[df_c['Country']=='Bolivia (Plurinational State of)'].index.values[0], 'Country' ] = 'Bolivia'

df_c.at[ df_c[df_c['Country']=='Côte d\'Ivoire'].index.values[0], 'Country' ] = 'Ivory Coast'

df_c.at[ df_c[df_c['Country']=='Congo'].index.values[0], 'Country' ] = 'Congo, Republic of the'

df_c.at[ df_c[df_c['Country']=='Tanzania, United Republic of'].index.values[0], 'Country' ] = 'Tanzania'



# Using the ISO-3166 coding standard to map countries

df['ISO-code'] = df['Country'].map(df_c.set_index('Country')['ISO-code'])

# Clean data-frame ( Duplicates & NaNs )

df.isna().sum()

df = df.dropna()
import plotly.express as px



thres = 1.3

df_th = df.drop(df[ df['Sex-Ratio'] > thres ].index)



# color pallete @ https://plotly.com/python/builtin-colorscales/

fig = px.choropleth(df_th, locations='ISO-code',

                color="Sex-Ratio", hover_name="Country",

                    color_continuous_scale=px.colors.sequential.Sunset, projection="natural earth")

fig.update_layout(title={'text':'Sex-Ratio per country', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'})

fig.show() # Sunset / Bluered / Electric
import plotly.express as px



thres = 1.3

df_th = df.drop(df[ df['Sex-Ratio'] > thres ].index)



# color pallete @ https://plotly.com/python/builtin-colorscales/

fig = px.choropleth(df_th, locations='ISO-code',

                color="Sex-Ratio", hover_name="Country",

                    color_continuous_scale=px.colors.sequential.Sunset, projection="orthographic")

fig.update_layout(title={'text':'Sex-Ratio per country', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'})

fig.show() # Sunset / Bluered / Electric