from google.cloud import bigquery

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import bq_helper

from tabulate import tabulate

client = bigquery.Client()

# Creating my own color pallette

my_colors =       ['#ff8000', # Indian saffron

                   '#0000ff',  # Indian Blue

                   '#008000', # Indian Green

                    '#78C850',  # Grass

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                    '#F08030',  # Fire

                   ]



query = """

SELECT *

FROM `bigquery-public-data.hacker_news.stories` 

WHERE REGEXP_CONTAINS(title, r"( [Ii][Oo][tT] | [Ii]nternet [Oo][Ff] [Tt]hings) ") 

OR REGEXP_CONTAINS(title, r"(^[Ii][Oo][tT]|^[Ii]nternet [Oo][Ff] [Tt]hings) ") 

OR REGEXP_CONTAINS(title, r"( [Ii][Oo][tT].|[Ii]nternet [Oo][Ff] [Tt]hings).") 

OR REGEXP_CONTAINS(title, r"( [Ii][Oo][tT]$|[Ii]nternet [Oo][Ff] [Tt]hings)$") 

ORDER BY time

"""

#print("estimating size of the query:")

#hacker_news.estimate_query_size(query)

query_job = client.query(query)

iterator = query_job.result(timeout=30)

rows = list(iterator)

# Transform the rows into a nice pandas dataframe

df_iot = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines

df_first = df_iot[['title','time_ts','author']].copy().head(5) 
print(tabulate(df_first, headers='keys', tablefmt='simple', showindex=False))
df_iot['date'] = pd.to_datetime(df_iot.time_ts)

df_iot['year'] = df_iot.date.dt.year.astype('category')

sns.set(style="whitegrid", color_codes=True)

plt.figure(figsize=(12,8))

#ax = sns.countplot(x="year",palette=my_colors, data=df_iot)

sns.countplot(x="year",palette=my_colors, data=df_iot);
# Tagging and categorization of IOT Data obtained above

def sanitize_string(str_text):

    #str_text = "a dog is baby."

    str_text = str_text.lower()

    str_text = str_text.replace('.', '') # Take care of space.

    str_text = str_text.replace(',', '') # Take care of Comma.

    #str_text = str_text.replace("'", '') # Take care of Single Quote.

    return str_text



def get_microsoft(str_text):

    #str_check = str_text.str.contains('^microsoft| microsoft |microsoft.')

    str_return = 0

    s_new = str_text.split(' ')

    p = 'microsoft' in s_new

    q = 'azure' in s_new

    if (p | q):

       str_return = 1

    return str_return



def get_aws(str_text):

    #str_check = str_text.str.contains('^microsoft| microsoft |microsoft.')

    str_return = 0

    s_new = str_text.split(' ')

    p = 'aws' in s_new

    q = 'amazon' in s_new

    if (p | q):

       str_return = 1

    return str_return



def get_cisco(str_text):

    #str_check = str_text.str.contains('^microsoft| microsoft |microsoft.')

    str_return = 0

    s_new = str_text.split(' ')

    p = 'cisco' in s_new

    #q = 'amazon' in s_new

    if (p ):

       str_return = 1

    return str_return



def get_ibm(str_text):

    #str_check = str_text.str.contains('^microsoft| microsoft |microsoft.')

    str_return = 0

    s_new = str_text.split(' ')

    p = 'ibm' in s_new

    q = 'bluemix' in s_new

    if (p | q):

       str_return = 1

    return str_return



def get_google(str_text):

    #str_check = str_text.str.contains('^microsoft| microsoft |microsoft.')

    str_return = 0

    s_new = str_text.split(' ')

    p = 'google' in s_new

    q = 'gcp' in s_new

    if (p | q):

       str_return = 1

    return str_return



def get_provider(str_text):

    str_type = "Others"

    str_ms = get_microsoft(sanitize_string(str_text))

    str_aws = get_aws(sanitize_string(str_text))

    str_cisco = get_cisco(sanitize_string(str_text))

    str_google = get_google(sanitize_string(str_text))

    str_ibm = get_ibm(sanitize_string(str_text))

    if (str_ms == 1):

        str_type = "Microsoft"

    if (str_aws == 1):

        str_type = "Amazon"

    if (str_cisco == 1):

        str_type = "Cisco"

    if (str_google == 1):

        str_type = "Google"

    if (str_ibm == 1):

        str_type = "IBM"

    str_total = str_aws + str_ms + str_cisco + str_google

    if (str_total > 1) :

        str_type = "Mixed"

    return str_type



df_iot['Type'] = df_iot.title.apply(get_provider)

df_iot_providers_master = df_iot.copy()

df_iot_providers = df_iot[df_iot.Type != "Others"]

sns.set(style="whitegrid", color_codes=True)

plt.figure(figsize=(12,8))

#ax = sns.countplot(x="Type", data=df_iot_providers)

#ax = sns.countplot(x="Type", palette=my_colors, data=df_iot_providers).set(xlabel='Organization', ylabel='Number of appearances')

sns.countplot(x="Type", palette=my_colors, data=df_iot_providers).set(xlabel='Organization', ylabel='Number of appearances');
df_iot_providers = df_iot_providers.copy()

df_iot_providers['Mentions'] = 1

df_iot_providers_grouped_years_type = df_iot_providers['Mentions'].groupby([df_iot_providers['year'], df_iot_providers['Type']]).count()

df_iot_providers_year_wise_mentions = df_iot_providers_grouped_years_type.reset_index()

plt.figure(figsize=(11,8))

sns.pointplot(x="year", y="Mentions", palette=my_colors, hue="Type", data=df_iot_providers_year_wise_mentions);
print(tabulate(df_iot_providers_year_wise_mentions, headers='keys', tablefmt='simple', showindex=False))
df_words = df_iot_providers_master[[ 'Type','title']].copy()

df_words_google = df_words[df_words.Type == "Google"]

import wordcloud

words = ' '.join(df_words_google.title).lower()

words = words.replace('google','')

words = words.replace('internet','')

words = words.replace('things','')

words = words.replace('iot','')

#words

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=200,

                            width=1200,

                            height=600,

                           max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('google-wordcloud.png')

plt.imshow(cloud);
df_words = df_iot_providers_master[[ 'Type','title']].copy()

df_words_google = df_words[df_words.Type == "IBM"]

import wordcloud

words = ' '.join(df_words_google.title).lower()

words = words.replace('ibm','')

words = words.replace('bluemix','')

words = words.replace('internet','')

words = words.replace('things','')

words = words.replace('iot','')

#words

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=200,

                            width=1200,

                            height=600,

                           max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('ibm-wordcloud.png')

plt.imshow(cloud);
df_words = df_iot_providers_master[[ 'Type','title']].copy()

df_words_google = df_words[df_words.Type == "Microsoft"]

import wordcloud

words = ' '.join(df_words_google.title).lower()

words = words.replace('microsoft','')

words = words.replace('internet','')

words = words.replace('things','')

words = words.replace('iot','')

#words

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=200,

                            width=1200,

                            height=600,

                           max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('microsoft-wordcloud.png')

plt.imshow(cloud);
df_words = df_iot_providers_master[[ 'Type','title']].copy()

df_words_google = df_words[df_words.Type == "Amazon"]

#import wordcloud

words = ' '.join(df_words_google.title).lower()

words = words.replace('amazon','')

words = words.replace('internet','')

words = words.replace('things','')

words = words.replace('iot','')

words = words.replace('aws','')

#words

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=200,

                            width=1200,

                            height=600,

                           max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('amazon-wordcloud.png')

plt.imshow(cloud);
def top(df, n=5, column='score'):

     df = df.sort_values(by=column)[-n:]

     return df.sort_values(by = 'score',ascending = False)

 

df_iot_providers_master.to_csv('iot-data-final.csv')

df_iot_sorting = df_iot_providers_master[[ 'score','year','title']].copy()

df_top_articles = pd.DataFrame()



years = df_iot_sorting.year.unique().sort_values(ascending = False)

for yr in years:

    df_temp = df_iot_sorting[df_iot_sorting.year == yr]

    #print( "***************************Sorting Year ***********************")

    df_result = top(df_temp)

    df_top_articles = df_top_articles.append(df_result)

    #print(df_result)    

print(tabulate(df_top_articles, headers='keys', tablefmt='simple', showindex=False))