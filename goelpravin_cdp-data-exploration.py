# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# state abbreviation dictionary

us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cities_df = pd.read_csv("/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")

#Let's visualize Q5.4 most impactful city actions and Q3.5 vulnerable groups city action

usa_cities = cities_df[cities_df['Country'] == 'United States of America']

usa_SVI_action_3_5 = usa_cities[(usa_cities['Question Number'] == '3.5') & (usa_cities['Response Answer'] != 'Question not applicable')]



uk_cities = cities_df[cities_df['Country'] == 'United Kingdom of Great Britain and Northern Ireland']

uk_SVI_action_3_5 = uk_cities[(uk_cities['Question Number'] == '3.5') & (uk_cities['Response Answer'] != 'Question not applicable')]



aus_cities = cities_df[cities_df['Country'] == 'Australia']

aus_SVI_action_3_5 = aus_cities[(aus_cities['Question Number'] == '3.5') & (aus_cities['Response Answer'] != 'Question not applicable')]





from wordcloud import WordCloud, STOPWORDS

vul_stop_words = ["City", "city", "provides","nan","provided","helped","will","vulnerable","including","government","people","vulnerability","action","https","program","project","Implementation","plan","policy"] + list(STOPWORDS)

usa_svi_wordcloud = WordCloud(stopwords=vul_stop_words).generate(' '.join(map(str, usa_SVI_action_3_5['Response Answer'])))

uk_svi_wordcloud = WordCloud(stopwords=vul_stop_words).generate(' '.join(map(str, uk_SVI_action_3_5['Response Answer'])))

aus_svi_wordcloud = WordCloud(stopwords=vul_stop_words).generate(' '.join(map(str, aus_SVI_action_3_5['Response Answer'])))

plt.figure()

fig, (ax0, ax1,ax2) = plt.subplots(1, 3,figsize=(50,25))

fig.suptitle('City DEI Themes',fontsize=32)

ax0.set_title("USA",fontsize=32)

ax1.set_title("UK",fontsize=32)

ax2.set_title("Australia",fontsize=32)





ax0.imshow(usa_svi_wordcloud, interpolation="bilinear")

ax1.imshow(uk_svi_wordcloud, interpolation="bilinear")

ax2.imshow(aus_svi_wordcloud, interpolation="bilinear")



plt.axis("off")

plt.show()
usa_SVI_action_5_4 = usa_cities[(usa_cities['Question Number'] == '5.4') & (usa_cities['Response Answer'] != 'Question not applicable')]

uk_SVI_action_5_4 = uk_cities[(uk_cities['Question Number'] == '5.4') & (uk_cities['Response Answer'] != 'Question not applicable')]

aus_SVI_action_5_4 = aus_cities[(aus_cities['Question Number'] == '5.4') & (aus_cities['Response Answer'] != 'Question not applicable')]



impact_stop_words = ["City", "city", "provides","nan","provided","helped","will","vulnerable","including","government","people","vulnerability","action","https","program","programme","scheme","Improved","Reduced","reduced","improved","project","Implementation","plan","policy"] + list(STOPWORDS)

usa_svi_wordcloud = WordCloud(stopwords=impact_stop_words).generate(' '.join(map(str, usa_SVI_action_5_4['Response Answer'])))

uk_svi_wordcloud = WordCloud(stopwords=impact_stop_words).generate(' '.join(map(str, uk_SVI_action_5_4['Response Answer'])))

aus_svi_wordcloud = WordCloud(stopwords=impact_stop_words).generate(' '.join(map(str, aus_SVI_action_5_4['Response Answer'])))

plt.figure()

fig, (ax0, ax1,ax2) = plt.subplots(1, 3,figsize=(50,25))

fig.suptitle('City Actions towards Positive Impact',fontsize=32)

ax0.set_title("USA",fontsize=32)

ax1.set_title("UK",fontsize=32)

ax2.set_title("Australia",fontsize=32)





ax0.imshow(usa_svi_wordcloud, interpolation="bilinear")

ax1.imshow(uk_svi_wordcloud, interpolation="bilinear")

ax2.imshow(aus_svi_wordcloud, interpolation="bilinear")



plt.axis("off")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import matplotlib.pyplot as plt





def get_tf_idf_query_similarity(query,vectorizer,docs_tfidf):

    """

    vectorizer: TfIdfVectorizer model

    docs_tfidf: tfidf vectors for all docs

    query: query doc



    return: cosine similarity between query and all docs

    """

    query_joined = ' '.join(query)

    query_tfidf = vectorizer.transform([query_joined])

    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()

    return max(cosineSimilarities)

def get_corp_city_synergies(city_name, corp_city_state):

    

    #Need to setup TF-IDF Vectorizer to given city's 5.4 response

    given_city = cities_df[cities_df['Organization'] == city_name]

    city_5_4 = given_city[(given_city['Question Number'] == '5.4') & (given_city['Response Answer'] != " ")]

    city_5_4['Response Answer'].replace('', np.nan, inplace=True)

    city_5_4.dropna(subset=['Response Answer'], inplace=True)

    vectorizer = TfidfVectorizer()

    docs_tfidf = vectorizer.fit_transform(city_5_4['Response Answer'])

    

    city_orgs = cities_cdpmeta_df[cities_cdpmeta_df['city_state'] == corp_city_state]

    city_orgs_corp_data = pd.merge(left=city_orgs, right=corporates_df, how='left', on ='organization')

    city_orgs_2_4 = city_orgs_corp_data[(city_orgs_corp_data['question_number'] == 'C2.4a') & (city_orgs_corp_data['response_value'] != 'Question not applicable')]

    #for each org's 2_4a response, find TF-IDF cosine distance with City 5.4 response

    city_orgs_2_4['response_value'].replace('', np.nan, inplace=True)

    city_orgs_2_4.dropna(subset=['response_value'], inplace=True)



    city_orgs_2_4_grouped = city_orgs_2_4.groupby(by='organization')

    org_synergies = {}

    for org_name, org_data in city_orgs_2_4_grouped:

        #print(org_name)

        org_2_4_response = org_data['response_value'].values.astype(str).tolist()

        #print(org_2_4_response, type(org_2_4_response))

        tf_idf_similarity = int(100*get_tf_idf_query_similarity(org_2_4_response,vectorizer,docs_tfidf))

        #print(tf_idf_similarity)

        org_synergies[org_name] = tf_idf_similarity

    return org_synergies



def show_city_plot(sorted_list, synergies_dict, plot_title):

    labels = []

    numbers = []

    total = 0

    org_len = len(sorted_list)

    for org_name in sorted_list:

        labels.append(org_name)

        value = synergies_dict[org_name]

        numbers.append(value)

    index = np.arange(len(labels))

    plt.figure(figsize=(10,org_len))

    plt.barh(index,numbers)

    for i, v in enumerate(numbers):

        plt.text(v + 0.2, i , str(v), color='black', fontweight='bold',fontsize=20)

    plt.yticks(index, labels, fontsize=24, rotation=0)

    plt.xlabel(plot_title, fontsize=32)

    plt.title(plot_title, fontsize=32)

    plt.show()
#corporate question_number=C2.1 is about short/med/long-term horizons c2.4 is about climate-related strategic opportunities (details in C2.4a)



corporates_df =  pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')

# cities metadata - lat,lon locations for US cities

cities_meta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Simple Maps US Cities Data/uscities.csv")



# cities metadata - CDP metadata on organisation HQ cities

cities_cdpmeta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Locations of Corporations/NA_HQ_public_data.csv")

# map dict to clean full state names to abbreviations

cities_cdpmeta_df['state'] = cities_cdpmeta_df['address_state'].map(us_state_abbrev)



# infill non-matched from dict

cities_cdpmeta_df['state'] = cities_cdpmeta_df['state'].fillna(cities_cdpmeta_df['address_state'])

cities_cdpmeta_df['state'] = cities_cdpmeta_df['state'].replace({'ALBERTA':'AB'})

cities_cdpmeta_df['address_city'] = cities_cdpmeta_df['address_city'].replace({'CALGARY':'Calgary'})

cities_cdpmeta_df= cities_cdpmeta_df.drop(columns=['address_state'])



# create joint city state variable

cities_cdpmeta_df['city_state'] = cities_cdpmeta_df['address_city'].str.cat(cities_cdpmeta_df['state'],sep=", ")



org_synergies_nyc = get_corp_city_synergies('New York City','New York, NY')

org_synergies_sorted_nyc = sorted(org_synergies_nyc, key=org_synergies_nyc.get)

org_synergies_sf = get_corp_city_synergies('City of San Francisco','San Francisco, CA')

org_synergies_sorted_sf = sorted(org_synergies_sf, key=org_synergies_sf.get)



show_city_plot(org_synergies_sorted_nyc, org_synergies_nyc, "New York City")

show_city_plot(org_synergies_sorted_sf, org_synergies_sf, "City of San Francisco")
