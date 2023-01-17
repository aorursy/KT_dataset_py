#import packages 

import pandas as pd

from pandasql import sqldf 

#We're using plotly to graph

import plotly.express as px

#This is for cleaning up the data later 

import datetime

import re 
#Load villager data 

df_villagers = pd.read_csv("../input/animal-crossing/villagers.csv", encoding='utf-8')
df_villagers.head()

df_villagers.drop(columns=['id', 'row_n', 'phrase', 'full_id', 'url'])
species = sqldf("SELECT species, COUNT(species) AS size FROM df_villagers GROUP BY species ORDER BY size DESC")

pie = px.pie(species, values='size', names='species', title='Villager Species', color_discrete_sequence=px.colors.qualitative.Dark24,)

pie.show()





barh = px.bar(species, x="size", y="species", orientation='h', color="size",  title='Villager Species')

barh.update_layout(

    

    autosize=False,

    height=800,

)

barh.show()

gender = sqldf("SELECT gender, COUNT(gender) AS count FROM df_villagers GROUP BY gender ORDER BY count DESC")

species_gender = sqldf("SELECT species, gender, COUNT(gender) AS count FROM df_villagers GROUP BY species, gender ORDER BY count ASC")



fig = px.pie(gender, values='count', names='gender', title='Gender Breakdown')

fig.show()



bar = px.bar(species_gender, x="species", y="count", color="gender", title="Gender and Species")

bar.show()
personality = sqldf("SELECT personality, COUNT(personality) AS count FROM df_villagers GROUP by personality ORDER BY count ASC")

species_personality = sqldf("SELECT species, personality, COUNT(personality) AS count FROM df_villagers GROUP BY species, personality")



fig = px.pie(personality, values='count', names='personality', title='Personality Types')

fig.show()



bar = px.bar(species_personality, x="count", y="species", color="personality", orientation='h', title='Personality and Species')

bar.update_layout(

    

        autosize=False,

    width=1000,

    height=1000,

)

bar.show()
df_villagers.fillna({'song': 'Not available'}, inplace = True)

df_villagers.isnull().sum()



song = sqldf("SELECT species, personality, song, COUNT(song) AS count FROM df_villagers GROUP by species, personality, song")



fig = px.scatter(song, x="species", y="personality", size="count", color="song")

fig.update_layout(

     autosize=False,

    width=1000,

    height=1000,

)

fig.show()
birth_list = df_villagers['birthday'].to_list()
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
df_villagers['sign'] = star_signs 
sign = sqldf("SELECT sign, COUNT(sign) AS count FROM df_villagers GROUP BY sign")

sign_personality = sqldf("SELECT sign, personality, COUNT(sign) AS count FROM df_villagers GROUP BY sign, personality")

sign_personality_species = sqldf("SELECT species, personality, sign, COUNT(sign) AS count FROM df_villagers GROUP by species, personality, sign")



bar = px.bar(sign_personality, x="count", y="sign", color="personality", orientation='h')

bar.show()

pie = px.pie(sign, values='count', names='sign', title='Horoscope Signs')

pie.show()



fig = px.scatter(sign_personality_species, x="species", y="sign", size="count", color="personality", title="Horoscope, Personality, Species")



fig.update_layout(autosize=False,width=1000, height=1000)

fig.show()
#Load items data 

df_items = pd.read_csv("../input/animal-crossing/items.csv", encoding='utf-8')
df_items.head()

#Let's drop the columns we don't need

df_items.drop(columns=['num_id', 'id', 'orderable', 'sources', 'customizable', 'recipe', 'recipe_id', 'games_id', 'id_full', 'image_url'])
categories = sqldf("SELECT category, COUNT(category) AS count FROM df_items GROUP BY category ORDER BY count DESC")



fig = px.pie(categories, values='count', names='category', title='Item Categories')

fig.update_layout(

    

    autosize=False,

    width=800,

    height=500,

)

fig.show()



barh = px.bar(categories, x="count", y="category", barmode='relative', orientation='h', title='Item Categories')

barh.show()
resale = sqldf("SELECT category,sell_value, buy_value from df_items")

#Since some items have missing values we drop those 

resale = resale.dropna()



#We define resale value as the percentage of sell value over buy value

resale['resale'] = (resale['sell_value'] / resale['buy_value'])*100





resale_categories = sqldf("SELECT category, AVG(resale) AS avg_resale from resale GROUP BY category ORDER BY avg_resale DESC")

resale_categories 
bar = px.bar(resale_categories, x="category", y="avg_resale", color="avg_resale", title='Average Resale Value by Item Category')

bar.update_layout(

    

        autosize=False,

    width=1000,

    height=1000,

)

bar.show()