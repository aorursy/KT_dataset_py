import pandas as pd   
from pandasql import sqldf 
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import datetime
import re 
df_vil = pd.read_csv('../input/animal-crossing/villagers.csv')
df_vil.nunique()
df_vil = df_vil.drop(columns=['row_n','id','full_id','phrase','url'])
df_vil.head(10)
birth_list = df_vil['birthday'].to_list()
def horoscope_dates(start, end):
    horoscope_start = datetime.datetime.strptime(start, "%m-%d")
    horoscope_end = datetime.datetime.strptime(end, "%m-%d")
    horoscope_interval = [horoscope_start + datetime.timedelta(days=x) for x in range(0, (horoscope_end-horoscope_start).days)]
    
    string_interval = []
    final_interval = []
    for date in horoscope_interval:
        string_interval.append(date.strftime("%m-%d"))
        #we clean up the string here using regex and strip methods 
        string_interval = [i.lstrip("0") for i in string_interval]
        final_interval = [re.sub(r'(-0)', '-', i) for i in string_interval]
    return final_interval
aries = horoscope_dates("3-21", "4-20")
taurus = horoscope_dates("4-20", "5-21")
gemini = horoscope_dates("5-21", "6-22")
cancer = horoscope_dates("6-22", "7-23")
leo = horoscope_dates("7-23", "8-23")
virgo = horoscope_dates("8-23", "9-23")
libra = horoscope_dates("9-23", "10-23")
scorpio = horoscope_dates("10-23", "11-23")
sagittarius = horoscope_dates("11-23", "12-22")
capricorn = horoscope_dates("12-22", "1-20")
aquarius = horoscope_dates("1-20", "2-19")
pisces = horoscope_dates("2-19", "3-21")

star_signs = []
for birthday in birth_list: 
    if birthday in aries: 
        star_signs.append("Aries")
    elif birthday in taurus: 
        star_signs.append("Taurus")
    elif birthday in gemini: 
        star_signs.append("Gemini")
    elif birthday in cancer: 
        star_signs.append("Cancer")
    elif birthday in leo: 
        star_signs.append("Leo")
    elif birthday in virgo: 
        star_signs.append("Virgo")
    elif birthday in libra: 
        star_signs.append("Libra")
    elif birthday in scorpio: 
        star_signs.append("Scorpio")
    elif birthday in sagittarius: 
        star_signs.append("Sagittarius")
    elif birthday in aquarius: 
        star_signs.append("Aquarius")
    elif birthday in pisces: 
        star_signs.append("Pisces")
    else: 
        #Since it's at the end of the year, the function doesn't work on Capricorn 
        #You can leave it as the else statement 
        star_signs.append("Capricorn")
df_vil['sign'] = star_signs
sign = sqldf("SELECT sign, COUNT(sign) AS count FROM df_vil GROUP BY sign")
pie = px.pie(sign, values='count', names='sign', title='Horoscope Signs')
pie.show()
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

quote_page = 'https://gamewith.net/animal-crossing-new-horizons/article/show/18171'

# query the website and return the html to the variable ‘page’
page = urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')
tier_list = []
for i in soup.find_all('tr', class_ = 'w-idb-element'):
    tier_list.append(i.previous_element)
tier_list.remove(' PASTE DATA HERE↓\u3000')
# Creating new dataframe to merge with original one
df_rank = pd.DataFrame({"name":tier_list,'rank':list(range(1,len(tier_list)+1))})
df_villager = pd.merge(df_vil,df_rank,how='inner', on=['name'])
top_50 = df_villager.sort_values('rank').head(50)
top_50.head()
popular_species = top_50['species'].value_counts()
popular_species.plot(kind='pie', subplots=True, figsize=(16,8),autopct='%1.1f%%')
plt.title('Popular Species')
plt.show()
popular_sign = top_50['sign'].value_counts()
popular_sign.plot(kind='pie', subplots=True, figsize=(16,8),autopct='%1.1f%%')
plt.title('Popular Horoscope')
plt.show()
popular_personality = top_50['personality'].value_counts()
popular_personality.plot(kind='pie', subplots=True, figsize=(16,8),autopct='%1.1f%%')
plt.title('Popular Personality')
plt.show()
popular_gender = top_50['gender'].value_counts()
popular_gender.plot(kind='pie', subplots=True, figsize=(16,8),autopct='%1.1f%%')
plt.title('Popular Gender')
plt.show()