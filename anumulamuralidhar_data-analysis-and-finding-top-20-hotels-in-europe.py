# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from collections import Counter

import re, nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

import folium

from matplotlib.colors import LinearSegmentedColormap

import missingno as msno

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#loading the dataset from the Hotel_reviews dataset

df = pd.read_csv("../input/Hotel_Reviews.csv")
#printing the columns names of the datset

df.columns
#printing the shape of the dataset

df.shape
print ('Number of data points : ', df.shape[0], \

       '\nNumber of features:', df.shape[1])

df.head()
#Removing duplicates from the dataset

print(sum(df.duplicated()))

df = df.drop_duplicates()

print('After removing Duplicates: {}'.format(df.shape))
msno.matrix(df)
nans = lambda df: df[df.isnull().any(axis=1)]

nans_df = nans(df)

nans_df = nans_df[['Hotel_Name','lat','lng']]

print('No of missing values in the dataset: {}'.format(len(nans_df)))
nans_df.Hotel_Name.describe()
# let's look at the reviews frequency of the missing Hotels.

nans_df.Hotel_Name.value_counts()
print('No of reviews in the dataset to that Hotel:')

print('Fleming s Selection Hotel Wien City: {}'.format(len(df.loc[df.Hotel_Name == 'Fleming s Selection Hotel Wien City'])))

print('Hotel City Central: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel City Central'])))

print('Hotel Atlanta: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Atlanta'])))

print('Maison Albar Hotel Paris Op ra Diamond: {}'.format(len(df.loc[df.Hotel_Name == 'Maison Albar Hotel Paris Op ra Diamond'])))

print('Hotel Daniel Vienna: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Daniel Vienna'])))

print('Hotel Pension Baron am Schottentor: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Pension Baron am Schottentor'])))

print('Austria Trend Hotel Schloss Wilhelminenberg Wien: {}'.format(len(df.loc[df.Hotel_Name == 'Austria Trend Hotel Schloss Wilhelminenberg Wien'])))

print('Derag Livinghotel Kaiser Franz Joseph Vienna: {}'.format(len(df.loc[df.Hotel_Name == 'Derag Livinghotel Kaiser Franz Joseph Vienna'])))

print('NH Collection Barcelona Podium: {}'.format(len(df.loc[df.Hotel_Name == 'NH Collection Barcelona Podium'])))

print('City Hotel Deutschmeister: {}'.format(len(df.loc[df.Hotel_Name == 'City Hotel Deutschmeister'])))

print('Hotel Park Villa: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Park Villa'])))

print('Cordial Theaterhotel Wien: {}'.format(len(df.loc[df.Hotel_Name == 'Cordial Theaterhotel Wien'])))

print('Holiday Inn Paris Montmartre: {}'.format(len(df.loc[df.Hotel_Name == 'Holiday Inn Paris Montmartre'])))

print('Roomz Vienna: {}'.format(len(df.loc[df.Hotel_Name == 'Roomz Vienna'])))

print('Mercure Paris Gare Montparnasse: {}'.format(len(df.loc[df.Hotel_Name == 'Mercure Paris Gare Montparnasse'])))

print('Renaissance Barcelona Hotel: {}'.format(len(df.loc[df.Hotel_Name == 'Renaissance Barcelona Hotel'])))

print('Hotel Advance: {}'.format(len(df.loc[df.Hotel_Name == 'Hotel Advance'])))
#latitude information of Hotels

loc_lat = {'Fleming s Selection Hotel Wien City':48.209270,

       'Hotel City Central':48.2136,

       'Hotel Atlanta':48.210033,

       'Maison Albar Hotel Paris Op ra Diamond':48.875343,

       'Hotel Daniel Vienna':48.1888,

       'Hotel Pension Baron am Schottentor':48.216701,

      'Austria Trend Hotel Schloss Wilhelminenberg Wien':48.2195,

      'Derag Livinghotel Kaiser Franz Joseph Vienna':48.245998,

      'NH Collection Barcelona Podium':41.3916,

      'City Hotel Deutschmeister':48.22088,

      'Hotel Park Villa':48.233577,

      'Cordial Theaterhotel Wien':48.209488,

      'Holiday Inn Paris Montmartre':48.888920,

      'Roomz Vienna':48.186605,

      'Mercure Paris Gare Montparnasse':48.840012,

      'Renaissance Barcelona Hotel':41.392673,

      'Hotel Advance':41.383308}
#longitude information of Hotels

loc_lng ={'Fleming s Selection Hotel Wien City':16.353479,

       'Hotel City Central':16.3799,

       'Hotel Atlanta':16.363449,

       'Maison Albar Hotel Paris Op ra Diamond':2.323358,

       'Hotel Daniel Vienna':16.3840,

       'Hotel Pension Baron am Schottentor':16.359819,

      'Austria Trend Hotel Schloss Wilhelminenberg Wien':16.2856,

      'Derag Livinghotel Kaiser Franz Joseph Vienna':16.341080,

      'NH Collection Barcelona Podium':2.1779,

      'City Hotel Deutschmeister':16.36663,

      'Hotel Park Villa':16.345682,

      'Cordial Theaterhotel Wien':16.351585,

      'Holiday Inn Paris Montmartre':2.333087,

      'Roomz Vienna':16.420643,

      'Mercure Paris Gare Montparnasse':2.323595,

      'Renaissance Barcelona Hotel':2.167494,

      'Hotel Advance':2.162828}
#filling the latitude information

df['lat'] = df['lat'].fillna(df['Hotel_Name'].apply(lambda x: loc_lat.get(x)))

#filling longitude information

df['lng'] = df['lng'].fillna(df['Hotel_Name'].apply(lambda x: loc_lng.get(x)))
#looking whether information is correctly filled or not.

msno.matrix(df)
#saving the data to pickle files

df.to_pickle('After_filling_Nans')
#loading the data from the pickle file

df = pd.read_pickle('After_filling_Nans')
df.Hotel_Name.describe()
# Let's look at the top 10 reviewed Hotels

Hotel_Name_count = df.Hotel_Name.value_counts()

Hotel_Name_count[:10].plot(kind='bar',figsize=(10,8))
import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 50, 18

rcParams["axes.labelsize"] = 16

from matplotlib import pyplot

import seaborn as sns
data_plot = df[["Hotel_Name","Average_Score"]].drop_duplicates()

sns.set(font_scale = 2.5)

a4_dims = (30, 12)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.countplot(ax = ax,x = "Average_Score",data=data_plot)
text = ""

for i in range(df.shape[0]):

    text = " ".join([text,df["Reviewer_Nationality"].values[i]])
from wordcloud import WordCloud

wordcloud = WordCloud(background_color='black', width = 600,\

                      height=200, max_font_size=50, max_words=40).generate(text)

wordcloud.recolor(random_state=312)

plt.imshow(wordcloud)

plt.title("Wordcloud for countries ")

plt.axis("off")

plt.show()
df.Reviewer_Nationality.describe()
# Let's look at the Top 10 Reviewer's Nationalities

Reviewer_Nat_Count = df.Reviewer_Nationality.value_counts()

print(Reviewer_Nat_Count[:10])
df.Review_Date.describe()
# Let's look at the top 10 Reviews given dates

Review_Date_count = df.Review_Date.value_counts()

Review_Date_count[:10].plot(kind='bar')
Reviewers_freq = df.Total_Number_of_Reviews_Reviewer_Has_Given.value_counts()

Reviewers_freq[:10].plot(kind='bar')
Reviewers_freq[:10]
#Loading the unique Hotel's information to plot them on the map

temp_df = df.drop_duplicates(['Hotel_Name'])

len(temp_df)
map_osm = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )



temp_df.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]])

                                             .add_to(map_osm), axis=1)



map_osm
pos_words = df.Review_Total_Positive_Word_Counts.value_counts()

pos_words[:10]
a = df.loc[df.Review_Total_Positive_Word_Counts == 0]

print('No of completely Negative reviews in the dataset:',len(a))

b = a[['Positive_Review','Negative_Review']]

b[:10]
neg_words = df.Review_Total_Negative_Word_Counts.value_counts()

neg_words[:10]
a = df.loc[df.Review_Total_Negative_Word_Counts == 0 ]

print('No of completely positive reviews in the dataset:',len(a))

b = a[['Positive_Review','Negative_Review']]

b[:10]
# For classifying positive and negative reviews

df['pos_count']=0

df['neg_count']=0
# since we found the words are in mixed case letters and with trailing whitespace 

#we remove those white spaces and converting the reviews to lowercases

df['Negative_Review']=[x.lower().strip() for x in df['Negative_Review']]

df['Positive_Review']=[x.lower().strip() for x in df['Positive_Review']]
#if the Positive_Review contains the words 'no positive' and 'nothing' are considered as a Negative_Review.

# if the Negative_Review contains the word 'everything' it is also considered as Negative_Review.

# we are maiking those reveiews as 1 in neg_count(attribute).

df["neg_count"] = df.apply(lambda x: 1 if x["Positive_Review"] == 'no positive' or \

                           x['Positive_Review']=='nothing' or \

                           x['Negative_Review']=='everything' \

                           else x['pos_count'],axis = 1)
#if the Negative_Review contains the words 'no negative' and 'nothing' are considered as a Positive_Review.

#if the Positive_Review contains the word 'Everything' it is also considered as positive_Review. 

#we are making those reviews as 1 in the pos_count(attribute). 

df["pos_count"] = df.apply(lambda x: 1 if x["Negative_Review"] == 'no negative' or \

                           x['Negative_Review']=='nothing' or \

                           x['Positive_Review']=='everything' \

                           else x['pos_count'],axis = 1)
#seeing how many reviews are classified as positive one's

df.pos_count.value_counts()
#seeing how many reviews are classified as negative one's

df.neg_count.value_counts()
# Calculating no of positive and negative reviews for each Hotel and storing them into reviews dataset. 

reviews = pd.DataFrame(df.groupby(["Hotel_Name"])["pos_count","neg_count"].sum())
reviews.head()
# Adding index to the reviews dataframe

reviews["HoteL_Name"] = reviews.index

reviews.index = range(reviews.shape[0])

reviews.head()
#calculating total number of reviews for each hotel

reviews["total"] = reviews["pos_count"] + reviews["neg_count"]

#calculating the positive ratio for each Hotel.

reviews["pos_ratio"] = reviews["pos_count"].astype("float")/reviews["total"].astype("float")
#looking at the famous 20 hotels location in the map. Famous Hotels are calculated based on the total

#no of reviews the Hotel has.

famous_hotels = reviews.sort_values(by = "total",ascending=False).head(100)

pd.set_option('display.max_colwidth', 2000)

popular = famous_hotels["HoteL_Name"].values[:20]

popular_hotels =df.loc[df['Hotel_Name'].isin(popular)][["Hotel_Name",\

                                "Hotel_Address",'Average_Score','lat','lng']].drop_duplicates()

maps_osm = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )

popular_hotels.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]])

                                             .add_to(maps_osm), axis=1)



maps_osm
#look at the Hotel_Name and Hotel_Address of those Hotels

popular_hotels
#Looking at top 20 famous hotels with positive reviews.

pos = famous_hotels.sort_values(by = "pos_ratio",ascending=False)["HoteL_Name"].head(20).values

famous_pos = df.loc[df['Hotel_Name'].isin(pos)][["Hotel_Name","Hotel_Address",'lat','lng','Average_Score']].drop_duplicates()

positive_map = folium.Map(location=[47, 6], zoom_start=5, tiles = 'Stamen Toner' )

famous_pos.apply(lambda row:folium.Marker(location=[row["lat"], row["lng"]])

                                             .add_to(positive_map), axis=1)



positive_map
#look at the Hotel_Name and Hotel_Address of those Hotels

famous_pos
#saving the dataframe to pickle file

reviews.to_pickle('reviews')
#loading the positive reviews and negative reviews to a single column as text

pos_reviews = df['Positive_Review'].values

pos_reviews = pos_reviews.tolist()

neg_reviews = df['Negative_Review'].values

neg_reviews = neg_reviews.tolist()

text = pos_reviews+neg_reviews
#providing score attribute to the review

score = ['positive' for i in range(len(pos_reviews))]

score += ['negative' for i in range(len(neg_reviews))]

#performing one-hot encoding to the score attrubute.(1- positive and 0- negative)

for i in range(0,len(score)):

    if score[i] == 'positive':

        score[i] = 1

    else:

        score[i] = 0
#loading required data to dataframe.

text_df = pd.DataFrame()

text_df['reviews'] = text

text_df['score'] = score

text_df.head()
# Perfoming preprocessing

start_time = time.time()

text = text_df['reviews'].values

print("Removing stop words...........................")

stop = set(stopwords.words('english'))

words = []

summary = []

all_pos_words = []

all_neg_words = []

for i in range(0,len(text)):

    if type(text[i]) == type('') :

        sentence = text[i]

        sentence = re.sub("[^a-zA-Z]"," ", sentence)

        buffer_sentence = [i for i in sentence.split() if i not in stop]

        word = ''

        for j in buffer_sentence:

            if len(j) >= 2:

                if i<=(len(text)/2): 

                    all_pos_words.append(j)

                else:

                    all_neg_words.append(j)

                word +=' '+j

        summary.append(word)    

print("performing stemming............................")

porter = PorterStemmer()

for i in range(0,len(summary)):

    summary[i] = porter.stem(summary[i])

print("--- %s seconds ---" % (time.time() - start_time))
# no of words in positive and negative reviews

len(all_pos_words),len(all_neg_words)
# displaying the frequency of words in positive and negative reviews 

freq_dist_pos = Counter(all_pos_words)

freq_dist_neg = Counter(all_neg_words)

print('Most common positive words : ',freq_dist_pos.most_common(20))

print('Most common negative words : ',freq_dist_neg.most_common(20))
# no of positive and negative words

len(freq_dist_neg),len(freq_dist_pos)
#converting the summary numpy array

score = text_df['score'].values
# loading the data to dataframe and saving it into pickle file

text_df = pd.DataFrame()

text_df['Summary'] = summary

text_df['score'] = score

text_df.to_pickle('text_df')