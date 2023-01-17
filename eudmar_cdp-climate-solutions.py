import numpy as np

import pandas as pd

import nltk

import os



# Plot

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary
#Loading Cities response

df_ct_full_2018 = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2018_Full_Cities_Dataset.csv")

df_ct_full_2019 = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2019_Full_Cities_Dataset.csv")

df_ct_full_2020 = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")
# Dataset_Corporates

df_cl_2018 = pd.read_csv("../input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/2018_Corporates_Disclosing_to_CDP_Climate_Change.csv")

df_cl_2019 = pd.read_csv("../input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/2019_Corporates_Disclosing_to_CDP_Climate_Change.csv")

df_cl_2020 = pd.read_csv("../input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/2020_Corporates_Disclosing_to_CDP_Climate_Change.csv")
# Cities Responses

# concatenating

df = [df_ct_full_2018, df_ct_full_2019, df_ct_full_2020]

df_ct = pd.concat(df)
# resetting the index

df_ct = df_ct.reset_index()
resumetable(df_ct)
# Corporations Disclosing

# concatenating

df = [df_cl_2018, df_cl_2019, df_cl_2020]

df_cl = pd.concat(df)
# resetting the index

df_cl = df_cl.reset_index()
resumetable(df_cl)
group = df_ct.groupby('CDP Region').size()

group.sort_values(ascending = False)
plt.figure(figsize=(18, 8))



freq = len(df_ct)



sns.set_palette("pastel")



g = sns.countplot(df_ct['CDP Region'], order = df_ct['CDP Region'].value_counts().index)

g.set_xlabel('Region', fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
group = df_ct.groupby('Country').size()

group.sort_values(ascending = False)
city_count = df_ct['Country'].value_counts()

city_count_10 = city_count[:10,]

city_count_10
plt.figure(figsize=(18, 8))



freq = len(df_ct)



sns.set_palette("pastel")



g = sns.barplot(city_count_10.index, city_count_10.values)

g.set_title('Top 10 Country', fontsize = 15)

g.set_xlabel('Region', fontsize = 15)

g.set_ylabel("Count", fontsize = 15)

plt.xticks(rotation = 90)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
group_ct = df_ct.groupby('Organization').size()

group_ct.sort_values(ascending = False)
ct_count = df_ct['Organization'].value_counts()

ct_count_10 = ct_count[:10,]

ct_count_10
plt.figure(figsize=(18, 8))



freq = len(df_ct)



sns.set_palette("pastel")



g = sns.barplot(ct_count_10.index, ct_count_10.values)

g.set_title('Top 10 City', fontsize = 15)

g.set_xlabel('City', fontsize = 15)

g.set_ylabel("Count", fontsize = 15)

plt.xticks(rotation=90)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
cities_6_2 = df_ct[df_ct['Question Number'] == '6.2'].rename(columns={'Organization': 'City'})



cities_6_2['Response Answer'] = df_ct['Response Answer'].fillna('No Response')



cities_6_2.head()
df_import = pd.read_csv('../input/countries-iso-codes/country_codes.csv').rename(columns={'COUNTRY': 'country'})
df_import
countries = df_import['country'].unique().tolist()

Number_of_countries = len(countries)

print(countries)

print("\nTotal countries df_import present: ",Number_of_countries)
countries = cities_6_2['Country'].unique().tolist()

Number_of_countries = len(countries)

print(countries)

print("\nTotal countries CDP_6.2 present: ",Number_of_countries)
rename = {

    'United States of America': 'United States',

    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',

    'Venezuela (Bolivarian Republic of)': 'Venezuela',

    'Bolivia (Plurinational State of)': 'Bolivia',

    'China, Hong Kong Special Administrative Region': 'Hong Kong',

    'Taiwan, Greater China': 'Taiwan',

    'Viet Nam': 'Vietnam',

    'Democratic Republic of the Congo': 'Congo, Democratic Republic of the',

    'Russian Federation': 'Russia',

    'Republic of Korea': 'Korea, South',

    "Côte d'Ivoire": "Cote d'Ivoire",

    'United Republic of Tanzania': 'Tanzania',

    'Republic of Moldova': 'Moldova',

    'Chile': 'Chile'    

}



cities_6_2['re_country'] = cities_6_2['Country'].map(rename)

cities_6_2['re_country'] = cities_6_2['re_country'].fillna(cities_6_2['Country'])
countries_year = cities_6_2.groupby(['Year Reported to CDP',

                                     'CDP Region',

                                     're_country'])['City'].count().to_frame()

countries_year = countries_year.reset_index().rename(columns={'Year Reported to CDP': 'Year',

                                                              're_country': 'country'})
countries_year = pd.merge(countries_year, df_import, how='inner', on = 'country')
df = countries_year

fig = px.scatter_geo(df, locations="CODE", color="CDP Region", 

                     hover_name="country", size="City",

                     animation_frame="Year", projection="natural earth")

fig.show()
countries_total = cities_6_2.groupby(['CDP Region','re_country'])['City'].count().to_frame()

countries_total = countries_total.reset_index().rename(columns={'re_country': 'country'})
countries_total = pd.merge(countries_total, df_import, how='inner', on = 'country')
df = countries_total

fig = px.scatter_geo(df, locations="CODE", color="CDP Region",

                     hover_name="country", size="City",

                     projection="natural earth")

fig.show()
cities_6_2.dropna(subset=['Question Name'], axis=0, inplace = True)
Response = cities_6_2['Question Name']
Response_summary = " ".join(s for s in Response)
stopwords=nltk.corpus.stopwords.words('english')
wordcloud = WordCloud(stopwords=stopwords,

                      background_color='white', width=1600,                            

                      height=800).generate(Response_summary)
fig, ax = plt.subplots(figsize=(16,8))       



ax.imshow(wordcloud, interpolation='bilinear') 

ax.set_axis_off()



plt.imshow(wordcloud)              

wordcloud.to_file('rafael.png',);
Response = cities_6_2['Response Answer']
Response_summary = " ".join(s for s in Response)
stopwords=nltk.corpus.stopwords.words('portuguese', 'english')
wordcloud = WordCloud(stopwords=stopwords,

                      background_color='white', width=1600,                            

                      height=800).generate(Response_summary)
fig, ax = plt.subplots(figsize=(16,8))       



ax.imshow(wordcloud, interpolation='bilinear') 

ax.set_axis_off()



plt.imshow(wordcloud)              

wordcloud.to_file('rafael.png',);
df_cl.groupby('country').size()
plt.figure(figsize=(12, 5))



freq = len(df_cl)



sns.set_palette("pastel")



g = sns.countplot(df_cl['country'])

g.set_xlabel('Country', fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)
df_cl.groupby('survey_year').size()
plt.figure(figsize=(12, 5))



freq = len(df_cl)



sns.set_palette("pastel")



g = sns.countplot(df_cl['survey_year'])

g.set_xlabel('Survey year', fontsize = 15)

g.set_ylabel("Count", fontsize = 15)



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x() + p.get_width() / 2., height + 3,

          '{:1.2f}%'.format(height / freq * 100),

          ha = "center", fontsize = 18)