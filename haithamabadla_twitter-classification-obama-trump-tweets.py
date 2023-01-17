## Load libraries



import re

import json

import string

import numpy as np

import pandas as pd

from collections import Counter



import matplotlib.pyplot as plt

import seaborn as sns



import spacy

from spacy.lang.en.stop_words import STOP_WORDS



from textblob import TextBlob



from wordcloud import WordCloud

from PIL import Image # To mask the wordcloud, I have to import PIL - Pillow library
## Set Default Settings



sns.set_style('darkgrid')

nlp = spacy.load('en_core_web_sm')

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', None)
## Load CSV file

df_tweets = pd.read_csv('../input/Trump_Obama_Tweets.csv')



# Display the first few rows to understand your data

df_tweets.head()
# Features types

df_tweets.dtypes
# Check null values

df_tweets.isnull().sum()
# To avoid displaying scientific numbers like '9.500000e+01'

pd.options.display.float_format = '{:20,.2f}'.format
# Default method will display numeric features only

df_tweets.describe().round(2).T
# Using include=['O'] to display categorical features

df_tweets.describe(include=['O']).T
# Tweets shares

df_tweets.user.value_counts(normalize = True).round(3) * 100 
# Remove URLs from the tweets

def re_remove_url(x):

    return re.sub(r'http\S+', '', x)



#Extracts hashtags from tweets

def extract_hashtags(x):

    try:

        hashtags = re.findall(r"#(\w+)", x) # Extract hashtags

        if not hashtags:

            return np.nan

        elif isinstance(hashtags, list): # Check if it's a list object

            return ', '.join(hashtags) # Convert the list to str object

        else:

            return hashtags

    except:

        return np.nan

     

#Extracts mentions from tweets

def extract_mentions(x):

    try:

        mentions = re.findall(r"@(\w+)", x) # Extract mentions

        if not mentions:

            return np.nan

        if isinstance(mentions, list):

            return ', '.join(mentions)

        else:

            return mentions

    except:

        return np.nan



# Add exctracted data in new columns

df_tweets['tweets']   = df_tweets.text.apply(lambda x: re_remove_url(x))

df_tweets['hashtags'] = df_tweets.text.apply(lambda x: extract_hashtags(x))

df_tweets['mentions'] = df_tweets.text.apply(lambda x: extract_mentions(x))



# Drop unwanted columns

df_tweets.drop(['text'], axis = 1, inplace = True)
# Extract additional information about the tweets

tweets_length_list           = []

tweets_spaces_list           = []

tweets_uppercase_list        = []

tweets_punctuations_list     = []

tweets_questionmark_list     = []

tweets_exclamation_mark_list = []



def extract_text_details(x):

    tweets_length_list.append(len(x))                                                 # Length of the tweet

    tweets_spaces_list.append(sum([1 for l in x if l.isspace()]))                     # Total number of spaces exists in the tweet

    tweets_uppercase_list.append(sum([1 for l in x if l.isupper()]))                  # Total number of uppercases used in the tweet

    tweets_punctuations_list.append(sum([1 for l in x if l in string.punctuation]))   # Total number of punctuation exists in the tweet

    tweets_questionmark_list.append(x.count('?'))                                     # Total number of question marks in tweet 

    tweets_exclamation_mark_list.append(x.count('!'))                                 # Total number of exclamation marks in tweet



_ = df_tweets.tweets.apply(lambda x: extract_text_details(x)) # Since the function doesnt return values, it returns 'None' by default. Instead of displaying them, I stored them in temp object '_'

del _ # Delete _ object



df_tweets['tweets_length']           = tweets_length_list

df_tweets['tweets_spaces']           = tweets_spaces_list

df_tweets['tweets_uppercase']        = tweets_uppercase_list

df_tweets['tweets_punctuations']     = tweets_punctuations_list

df_tweets['tweets_questionmark']     = tweets_questionmark_list

df_tweets['tweets_exclamation_mark'] = tweets_exclamation_mark_list
# Extract polarity and subjectivity of the tweets 

polarity_list     = []

subjectivity_list = []



def polarity_subjectivity(x):

    analysis = TextBlob(x)

    polarity_list.append(round(analysis.polarity, 2))

    subjectivity_list.append(round(analysis.subjectivity, 2))

    

_ = df_tweets.tweets.apply(lambda x: polarity_subjectivity(x))

del _



df_tweets['polarity']     = polarity_list

df_tweets['subjectivity'] = subjectivity_list
# Very Positive / Positive / Very Negative / Negative / Neutral

def polarity_status(x):

    if x == 0:

        return 'Neutral'

    elif x > 0.00 and x < 0.50:

        return 'Positive'

    elif x >= 0.50:

        return 'Very Positive'

    elif x < 0.00 and x > -0.50:

        return 'Negative'

    elif x <= -0.50:

        return 'Very Negative'

    else:

        return 'Unknown'



# Very Positive / Positive / Very Negative / Negative / Neutral

def subjectivity_status(x):

    if x == 0:

        return 'Very Objective'

    elif x > 0.00 and x < 0.40:

        return 'Objective'

    elif x >= 0.40 and x < 0.70:

        return 'Subjective'

    elif x >= 0.70:

        return 'Very Subjective'



# Extract / Classify polarity and subjectivity

df_tweets['polarity_status'] = df_tweets.polarity.apply(lambda x: polarity_status(x))

df_tweets['subjectivity_status'] = df_tweets.subjectivity.apply(lambda x: subjectivity_status(x))
# Positive / Negative / Neutral numeric

# Very Positive and Positive are going to be ['is_positive']

neutral_list  = []

positive_list = []

negative_list = []



def polarity_status(x):

    if x == 0:

        neutral_list.append(1)

        positive_list.append(0)

        negative_list.append(0)

    elif x > 0.00:

        neutral_list.append(0)

        positive_list.append(1)

        negative_list.append(0)

    elif x < 0.00:

        neutral_list.append(0)

        positive_list.append(0)

        negative_list.append(1)

    

_ = df_tweets.polarity.apply(lambda x: polarity_status(x))

del _



df_tweets['is_neutral']  = neutral_list

df_tweets['is_positive'] = positive_list

df_tweets['is_negative'] = negative_list
# Convert [date] feature type to datetime type inorder to manipulate dates and times 

df_tweets.date = pd.to_datetime(df_tweets.date)
# Extract tweeting times [early, morning, noon, evening, midnight]

early_list    = []

morning_list  = []

noon_list     = []

evening_list  = []

midnight_list = []



def part_of_the_day(x):

    try:

        if x >= 5: 

            early_list.append(1)

            morning_list.append(0)

            noon_list.append(0)

            evening_list.append(0)

            midnight_list.append(0)

            return 'Early Morning'



        elif x >= 8: 

            early_list.append(0)

            morning_list.append(1)

            noon_list.append(0)

            evening_list.append(0)

            midnight_list.append(0)

            return 'Morning'



        elif x >= 12: 

            early_list.append(0)

            morning_list.append(0)

            noon_list.append(1)

            evening_list.append(0)

            midnight_list.append(0)

            return 'Afternoon'



        elif x >= 18: 

            early_list.append(0)

            morning_list.append(0)

            noon_list.append(0)

            evening_list.append(1)

            midnight_list.append(0)

            return 'Evening'



        elif x >= 0 and x < 5:

            early_list.append(0)

            morning_list.append(0)

            noon_list.append(0)

            evening_list.append(0)

            midnight_list.append(1)

            return 'Mid Night'

    except:

        early_list.append(np.nan)

        morning_list.append(np.nan)

        noon_list.append(np.nan)

        evening_list.append(np.nan)

        midnight_list.append(np.nan)

        return np.nan

    

df_tweets['part_of_day'] = df_tweets.date.dt.hour.apply(lambda x: part_of_the_day(x))



df_tweets['is_early']    = early_list

df_tweets['is_morning']  = morning_list

df_tweets['is_noon']     = noon_list

df_tweets['is_evening']  = evening_list

df_tweets['is_midnight'] = midnight_list 
is_norp_list    = []  # Nationalities or religious or political groups.

is_time_list    = []

is_org_list     = []  # Companies, agencies, institutions, etc.

is_gpe_list     = []  # Countries, cities, states.

is_loc_list     = []  # Non-GPE locations, mountain ranges, bodies of water.

is_product_list = []    

is_workart_list = []  # Titles of books, songs, etc.

is_fac_list     = []  # Buildings, airports, highways, bridges, etc.



is_noun_list    = []  # girl, cat, tree, air, beauty

is_pron_list    = []  # I, you, he, she, myself, themselves, somebody

is_adv_list     = []  # very, tomorrow, down, where, there

is_propn_list   = []  # Mary, John, London, NATO, HBO

is_verb_list    = []   

is_intj_list    = []  # psst, ouch, bravo, hello



def extract_tweet_style(x):

    

    doc = nlp(x)

    

    is_norp_list.append(sum([1 for i in doc.ents if i.label_ == 'NORP']))

    is_time_list.append(sum([1 for i in doc.ents if i.label_ == 'TIME']))

    is_org_list.append(sum([1 for i in doc.ents if i.label_ == 'ORG']))

    is_gpe_list.append(sum([1 for i in doc.ents if i.label_ == 'GPE']))

    is_loc_list.append(sum([1 for i in doc.ents if i.label_ == 'LOC']))

    is_product_list.append(sum([1 for i in doc.ents if i.label_ == 'PRODUCT']))

    is_workart_list.append(sum([1 for i in doc.ents if i.label_ == 'WORK_OF_ART']))

    is_fac_list.append(sum([1 for i in doc.ents if i.label_ == 'FAC']))



    is_noun_list.append((sum([1 for i in doc if i.pos_ == 'NOUN'])))

    is_pron_list.append((sum([1 for i in doc if i.pos_ == 'PRON'])))

    is_adv_list.append((sum([1 for i in doc if i.pos_ == 'ADV'])))

    is_propn_list.append((sum([1 for i in doc if i.pos_ == 'PROPN'])))

    is_verb_list.append((sum([1 for i in doc if i.pos_ == 'VERB'])))

    is_intj_list.append((sum([1 for i in doc if i.pos_ == 'INTJ'])))





_ = df_tweets.tweets.apply(lambda x: extract_tweet_style(x))

del _

                   

df_tweets['is_norp']     = is_norp_list

df_tweets['is_time']     = is_time_list

df_tweets['is_org']      = is_org_list

df_tweets['is_gpe']      = is_gpe_list

df_tweets['is_loc']      = is_loc_list

df_tweets['is_product']  = is_product_list

df_tweets['is_workart']  = is_workart_list

df_tweets['is_fac']      = is_fac_list



df_tweets['is_noun']     = is_noun_list

df_tweets['is_pron']     = is_pron_list

df_tweets['is_adv']      = is_adv_list

df_tweets['is_propn']    = is_propn_list

df_tweets['is_verb']     = is_verb_list

df_tweets['is_intj']     = is_intj_list
df_tweets.groupby(['user']).agg({'is_norp': 'sum',

                                 'is_time': 'sum',

                                 'is_org': 'sum',

                                 'is_gpe': 'sum',

                                 'is_loc': 'sum',

                                 'is_product': 'sum',

                                 'is_workart': 'sum',

                                 'is_fac': 'sum',

                                 'is_noun': 'sum',

                                 'is_pron': 'sum',

                                 'is_adv': 'sum',

                                 'is_propn': 'sum',

                                 'is_verb': 'sum',

                                 'is_intj': 'sum'})
# Extract months, week days, days 

df_tweets['month'] = df_tweets.date.dt.month

df_tweets['day'] = df_tweets.date.dt.day

df_tweets['week_day'] = df_tweets.date.dt.weekday # Weekday as number

df_tweets['week_day_name'] = df_tweets.date.dt.weekday_name # Weekday as text

df_tweets['hour'] = df_tweets.date.dt.hour
# Create Dictionaries

round1_cols      = ['user', 'week_day_name', 'part_of_day', 'polarity_status', 'subjectivity_status']

round1_titles    = ['Twitter Profile', 'Week Day', 'Part of The Day', 'Polarity', 'Subjectivity']

round1_cols_dict = dict(zip(range(0, len(round1_cols)), round1_cols))



# Create Pie Plot

fig, ax = plt.subplots(ncols = 5, figsize= (40,8))



for indx, col in round1_cols_dict.items():

    

    legend_list = df_tweets[col].value_counts(normalize = True).sort_index(ascending=True).keys().tolist()

    x = df_tweets[col].value_counts(normalize = True).sort_index(ascending=True)

    

    ax[indx].pie(data = df_tweets, x = x, autopct='%1.1f%%', textprops = {'fontsize': 11, 'color': 'w', 'weight': 'bold'})

    ax[indx].add_patch(plt.Circle((0,0), 0.35, fc = 'white'))

    

    ax[indx].legend(legend_list, loc = 2)

    ax[indx].set_title(round1_titles[indx], size = 15)



plt.show()
grouped = df_tweets.groupby([df_tweets.week_day_name, df_tweets.user]).size().reset_index().rename(columns = {0: 'counts'})

grouped
# Annotation function

def annotate_perct(ax_plot, total, add_height, rot):

    '''

    Definition - 

    

    Parameters - 

        1. ax_plot: is the graph object

        2. total: is the length of the dataframe or the sum of specific column, the use of this parameter depends on the objective of the graph

        3. add_height: the additional hight added to the actual hight in order to display the annotation on top of the bar. 

        4. rot: whether to display annotation with angles by passing [i.e. 75 / 85/ 90] or horizontal as it is (the default, which is 0) 

        

    Additional Explaination - 

        Once the hight of each bar is extracted, first I check if it's null (incase theres no values for specific cases) to assign 0 for the hight otherwise, I just add the extra hight provided (if any)        

    '''

    for p in ax_plot.patches:

        if np.isnan(p.get_height()): 

            height = 0

            ax_plot.text(p.get_x() + p.get_width()/2., height, '', ha="center", va='center', fontsize=10, rotation = rot)  

        else:

            height = p.get_height()

            ax_plot.text(p.get_x() + p.get_width()/2., height + add_height, '{}  ( {}% )'.format(int(height), round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10, rotation = rot)
temp = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



sns.set_style('whitegrid')



plt.figure(figsize = (20,9))



_ = sns.barplot(data = grouped, x = 'week_day_name', y = 'counts', hue = 'user', order = temp)

annotate_perct(ax_plot = _, add_height = 0.7, total = grouped.counts.sum(), rot= 0)



_.set_title('', pad = 40, weight= 'bold', size = 15)

_.set_xlabel('', weight= 'bold')

_.set_xticklabels(temp, rotation = 0,  weight= 'bold', fontsize = 15)

_.set_ylabel('Total Tweets', fontsize = 15, weight= 'bold')

_.margins(0)



plt.show()
# Define donut_plot function

def donut_plot(data_1, data_2, target, plot_title):



    fig, ax = plt.subplots(ncols = 2, figsize= (15,7))

    

    # Data preproccessing

    x = pd.DataFrame((data_1[target].value_counts(normalize=True).sort_index() * 100).round(1).reset_index().rename(columns = {'index': 'variable', 'gender': 'churn_yes'}))

    x['trump'] = (data_2[target].value_counts(normalize=True).sort_index() * 100).round(1).values



    cols_list = x.columns[1:3]

    hue_list = x.variable.unique()



    profile_list = ['Obama', 'Trump']



    for indx in range(0,2):

        # Technically its pie chart :) but I'm drawing a white circle to make them donuts!

        ax[indx].pie(data = x, x = cols_list[indx], autopct='%1.1f%%', textprops = {'fontsize': 11, 'color': 'w', 'weight': 'bold'}) 

        ax[indx].add_patch(plt.Circle((0,0), 0.35, fc = 'white'))

        label = ax[indx].annotate('{}'.format(profile_list[indx]), xy = (0, 0), fontsize = 13, ha = "center")  # weight = 'bold'

        ax[indx].legend(hue_list, loc = 2)

        ax[indx].set_title(plot_title, size = 16)

    plt.show()



# Seperate dataframe into obama and trump's dataframes

obama_df = df_tweets[df_tweets.user == 'Barak Obama']

trump_df = df_tweets[df_tweets.user == 'Donald Trump']



# Call donut plot function for each column

for indx, col in round1_cols_dict.items():

    if indx != 0: # Avoid Twitter profile comparison

        donut_plot( data_1 = obama_df, 

                    data_2 = trump_df, 

                    target = round1_cols_dict[indx], 

                    plot_title = '{}'.format(round1_titles[indx].title()) ) # .title() to capitilize the first character
## Detailed Language Observation



# Aggregate dataframe 

summarized_df = df_tweets.groupby(['user']).agg({  'favorite_counts': 'sum',       # Tweets overview

                                                   'retweet_counts': 'sum',

                                                   'is_positive': 'sum',

                                                   'is_negative': 'sum',



                                                   'tweets_length': 'sum',         # Tweets writing style

                                                   'tweets_uppercase': 'sum',

                                                   'tweets_punctuations': 'sum',

                                                   'tweets_questionmark': 'sum',



                                                   'is_norp': 'sum',               # Tweets detailed writing style

                                                   'is_time': 'sum',

                                                   'is_org': 'sum',

                                                   'is_gpe': 'sum',

                                                   'is_loc': 'sum',

                                                   'is_product': 'sum',

                                                   'is_workart': 'sum',

                                                   'is_fac': 'sum',

                                                   'is_noun': 'sum',

                                                   'is_pron': 'sum',

                                                   'is_adv': 'sum',

                                                   'is_propn': 'sum',

                                                   'is_verb': 'sum',

                                                   'is_intj': 'sum'  }).reset_index()



# Seperate dataframe 

obama_summarized_df = summarized_df[summarized_df.user == 'Barak Obama'].drop('user', axis = 1).copy()

trump_summarized_df = summarized_df[summarized_df.user == 'Donald Trump'].drop('user', axis = 1).copy()



# Get columns

summarized_cols      = obama_summarized_df.columns



# Create features for each round in list format

round_1_cols = summarized_cols[:5]

round_2_cols = summarized_cols[5:10]

round_3_cols = summarized_cols[10:15]

round_4_cols = summarized_cols[15:20]

round_5_cols = summarized_cols[20:22]



# Combine all lists into one list

temp_list = [round_1_cols, round_2_cols, round_3_cols, round_4_cols, round_5_cols]



# Dictionary to rename the titles

rep_title_dict = {  'favorite_counts': 'Tweets Likes',

                    'retweet_counts': 'Re-Tweets',

                    'is_positive': 'Positivity',

                    'is_negative': 'Negativity',

                    'tweets_length': 'Length of Tweets',

                    'tweets_uppercase': 'Uppercase Characters Used',

                    'tweets_punctuations': 'Punctuations Used',

                    'tweets_questionmark': 'Questionmark Used',

                    'is_norp': 'Nationalities | Religious | Political Groups',

                    'is_time': 'Mentioned Time Related',

                    'is_org': 'Corporate | Governmental',

                    'is_gpe': 'Countries | Cities | States',

                    'is_loc': 'Location Mentioned',

                    'is_product': 'Objects | Vehicles | Foods',

                    'is_workart': 'Books | Songs',

                    'is_fac': 'Buildings | Airports | Highways',

                    'is_noun': 'Noun Used',

                    'is_pron': 'Pronoun Used',

                    'is_adv': 'Adverb Used',

                    'is_propn': 'Propn (like Apple, UK, US)',

                    'is_verb': 'Verb Used',

                    'is_intj': 'Bravo | Hello | Ouch' }



# Create function to plot summarized details

def summarized_donut_plot(data_1, data_2, indx_list):#, plot_title):

    

    if indx_list == 4: 

        fig, ax = plt.subplots(ncols = 2, figsize= (13,6))

    else:

        fig, ax = plt.subplots(ncols = 5, figsize= (35,6))

        

    for indx, target in enumerate(temp_list[indx_list]):



        total = df_tweets[target].sum()

    

        # Data preproccessing

        x  = round(float(data_1[target] / total * 100), 1)

        y  = round(float(data_2[target] / total * 100), 1)



        #print(total, x, y)

        

        user_list = ['Obama', 'Trump']

        results_list = [x, y]



        x = pd.DataFrame(data = results_list, index = user_list)



        ax[indx].pie(x, autopct='%1.1f%%', textprops = {'fontsize': 11, 'color': 'w', 'weight': 'bold'})

        ax[indx].add_patch(plt.Circle((0,0), 0.35, fc = 'white'))



        ax[indx].legend(user_list, loc = 2)

        ax[indx].set_title(rep_title_dict[target], size = 13)



    plt.show()



# Call pie plot function for each column

for indx, sublist in enumerate(temp_list):

    #print(indx, sublist)

    summarized_donut_plot( data_1 = obama_summarized_df, data_2 = trump_summarized_df, indx_list = indx ) 
## Second round of cleaning before we use spaCy



# Stopwords external list

unwanted_text_list = ['“','”', 'lol', 'lmao', 'tell', 'twitter', 'list', 'whatever', 'yes', 'like', 

                      'im', 'know', 'just', 'dont', 'thats', 'right', 'youre', 'got', 'gonna','think',

                      'said', 'amp', 'omg', 'say', 'boy', 'lot', 'sir', 'office']
# Function to clean tweets using spacy from punctuations, stopwords and lemmatize them

def cleaning_tweets(x):

    # Spacy pipeline

    tweet = nlp(x)

    # Extract lemmatized words in lower case format if not digits, not punctuation, not stopword, and lenght not less than 2 

    tweet = ' '.join([token.lemma_.lower() for token in tweet if not token.is_stop and not token.is_punct and not token.text.isdigit() and len(token.text) > 2])

    tweet = ' '.join([token for token in tweet.split() if token not in unwanted_text_list])

    return tweet



# Store clean tweets

df_tweets['clean_tweets'] = df_tweets.tweets.apply(lambda x: cleaning_tweets(x))
# Check

df_tweets.loc[:, ['tweets', 'clean_tweets']][:5]
## Words count



# Initiate lists

obama_words_list = []

trump_words_list  = []



# Function to append word by word to their specific list

def collect_words(x, user_list):

    words = nlp(x.lower())

    [user_list.append(token.text) for token in words if not token.is_stop and not token.is_punct and not token.is_space]



# Send tweets to the function

_ = df_tweets.loc[df_tweets.user == 'Barak Obama', 'clean_tweets'].apply(lambda x: collect_words(x, obama_words_list))

_ = df_tweets.loc[df_tweets.user == 'Donald Trump', 'clean_tweets'].apply(lambda x: collect_words(x, trump_words_list))



# Apply counter function to count words

obama_freq_words = Counter(obama_words_list)

trump_freq_words = Counter(trump_words_list)



# Store the top 100 words

obama_top100_words = obama_freq_words.most_common(100)

trump_top100_words = trump_freq_words.most_common(100)



# Print top 5 words from each list

print(obama_top100_words[:5])

print(trump_top100_words[:5])
## WordCloud



# Combine all tweets into one for each user

obama_full_words = ' '.join([sent for sent in df_tweets.loc[df_tweets.user == 'Barak Obama', 'clean_tweets']])

trump_full_words = ' '.join([sent for sent in df_tweets.loc[df_tweets.user == 'Donald Trump', 'clean_tweets']])



# Initiate the WorldCloud

wc_obama = WordCloud(background_color="white", colormap="Blues", max_font_size=200, random_state=42) # blue color

wc_trump = WordCloud(background_color="white", colormap="Reds", max_font_size=200, random_state=42) # red color



# Function to display most common words

def plot_wordcloud(lists, titles, wc):

    

    fig, ax = plt.subplots(ncols = 2, figsize= (40,12))

    

    for indx, sents in enumerate(lists):

        wc[indx].generate(sents)

        ax[indx].imshow(wc[indx], interpolation = 'bilinear')

        ax[indx].axis('off')

        ax[indx].set_title(titles[indx], pad = 14, weight = 'bold')

    plt.show()



# Send user's tweets to the function

plot_wordcloud( lists  = [obama_full_words, trump_full_words],

                titles = ["Obama's Top 100 Words" , "Trump's Top 100 Words"],

                wc     = [wc_obama, wc_trump] )
# Drop unwanted columns and convert user profile into numeric reps

df_tweets['profile'] = df_tweets.user.replace({'Donald Trump': 0, 'Barak Obama': 1})

df_tweets.drop(['id', 'user', 'tweets'], axis = 1, inplace = True)
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split, GridSearchCV



import xgboost as xgb

import lightgbm as lgb
def process_data(data, feature, target):

    

    shuf_df = shuffle(data)                             # Shuffle DataFrame

    shuf_df.reset_index(drop = True, inplace = True)    # Reset DataFrame index

    

    X = data[feature]

    y = data[target]

    

    return X, y



X, y = process_data( data    = df_tweets, 

                     feature = ['clean_tweets'], 

                     target  = ['profile'] )
# Split train, test dataframe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 42)



# Check shapes

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train.values.shape, X_test.values.shape, y_train.values.ravel().shape, y_test.values.ravel().shape)
# Initiate TFIDF Vect

tfidf_vect = TfidfVectorizer(ngram_range = (1,3), stop_words = 'english', sublinear_tf = True)



# Fit & transform the X_train

tfidfVect_train    = tfidf_vect.fit_transform(X_train.clean_tweets)

tfidfVect_train_df = pd.DataFrame(tfidfVect_train.toarray(), columns = tfidf_vect.get_feature_names())

# Transform the X_test

tfidfVect_test     = tfidf_vect.transform(X_test.clean_tweets)

tfidfVect_test_df  = pd.DataFrame(tfidfVect_test.toarray(), columns = tfidf_vect.get_feature_names())
# Initiate lists 

model_name_list = []

accuracyScore_list = []

recallScore_list = []

precisionScore_list = []

rocAucScore_list = []

f1Score_list = []



def evaluate_classifier(model, Xtrain, Xtest, ytrain, ytest):

    # Initiate the Naive Bayes 'MultinomialNB' classifier

    clf = model[1]



    # Fit classifier with the X_train_tfidf, y_train

    clf.fit(Xtrain, ytrain)



    # Store predicted values ub y_pred

    y_pred = clf.predict(Xtest)

    

    accuracyScore  = accuracy_score(ytest, y_pred)

    recallScore    = recall_score(ytest, y_pred)

    precisionScore = precision_score(ytest, y_pred)

    rocAucScore    = roc_auc_score(ytest, y_pred)

    f1Score        = f1_score(ytest, y_pred)



    model_name_list.append(model[0])

    accuracyScore_list.append(accuracyScore)

    recallScore_list.append(recallScore)

    precisionScore_list.append(precisionScore)

    rocAucScore_list.append(rocAucScore)

    f1Score_list.append(f1Score)

    

    print('Accuracy Score: {}\n\n'.format(round(accuracyScore * 100),1))



    print('Confusion Matrix:')

    sns.heatmap(confusion_matrix(ytest, y_pred), annot = True, xticklabels = ["Trump","Obama"], yticklabels = ["Obama", "Trump"])

    plt.show()



    print('\n\nClassification Report: \n{}\n'.format(classification_report(ytest, y_pred)))
# Evaluate Naive Bayes 'MultinomialNB' classifier

evaluate_classifier( model  = ['MultinomialNB', MultinomialNB(alpha= 0.01)], 

                     Xtrain = tfidfVect_train_df.values, 

                     Xtest  = tfidfVect_test_df.values, 

                     ytrain = y_train.values.ravel(), 

                     ytest  = y_test.values.ravel())
# Evaluate XGBoost classifier

evaluate_classifier( model  = ['XGBoost', xgb.XGBClassifier(learning_rate = 0.5)], 

                     Xtrain = tfidfVect_train_df.values, 

                     Xtest  = tfidfVect_test_df.values, 

                     ytrain = y_train.values.ravel(), 

                     ytest  = y_test.values.ravel())
# Evaluate LogisticRegression classifier

evaluate_classifier( model  = ['LogisticRegression', LogisticRegression()], 

                     Xtrain = tfidfVect_train_df.values, 

                     Xtest  = tfidfVect_test_df.values, 

                     ytrain = y_train.values.ravel(), 

                     ytest  = y_test.values.ravel())
# Assemble scores from lists to dataframe

def assemble_scores():



    results_dict = { 'Model': model_name_list,

                     'Accuracy_Score': accuracyScore_list,

                     'Recall_Score': recallScore_list,

                     'Precision_Score': precisionScore_list,

                     'ROC_AUC_Score': rocAucScore_list,

                     'F1_Score': f1Score_list }



    results_df = pd.DataFrame(results_dict)

    

    return results_df



# Call assemble_scoes function to assemble scores into dataframe

results_df = assemble_scores()



# Display results

results_df
# Change the shape of the dataframe using pd.melt

results_melted_df = pd.melt(frame = results_df, id_vars = ['Model'], value_vars = ['Accuracy_Score', 'Recall_Score', 'Precision_Score', 'ROC_AUC_Score', 'F1_Score'], var_name = 'Score_Type', value_name = 'Score')



# Plot scores againt models

plt.figure(figsize= (20,8))



_ = sns.barplot( data = results_melted_df, x = 'Score', y = 'Model', hue = 'Score_Type', palette= "Paired")



_.set_title('Models Metrics / Scores', pad = 10, weight= 'bold')

_.set_xlabel('Performance Score', weight= 'bold')

_.set_ylabel('Model', weight= 'bold')



plt.show()
## Final text transformer model



# Initiate TFIDF Vect

tfidf_vectorize = TfidfVectorizer(ngram_range = (1,3), stop_words = 'english', sublinear_tf = True)



# Fit & transform the 

tfidfVect_X    = tfidf_vectorize.fit_transform(X.clean_tweets)

tfidfVect_X_df = pd.DataFrame(tfidfVect_X.toarray(), columns = tfidf_vectorize.get_feature_names())
## Final classifier



# GridSearchCV

cls_nb = MultinomialNB()



para_grid = { 'alpha': [0.001, 0.01, 0.1, 0.5, 1],

              'fit_prior' : [False, True] }



cls_nb_gscv = GridSearchCV( iid = False, estimator = cls_nb, param_grid = para_grid, cv = 10, return_train_score = True, n_jobs = -1 )

cls_nb_gscv_fit = cls_nb_gscv.fit(tfidfVect_X_df.values, y.values.ravel())



cls_nb_results_df = pd.DataFrame(cls_nb_gscv_fit.cv_results_)

cls_nb_results_df.sort_values('mean_test_score', ascending = False)[:5]
print('Best Score: {}\nBest Parameters: {}'.format(round(cls_nb_gscv_fit.best_score_ *100, 1) , cls_nb_gscv_fit.best_params_))
T1 = 'Nancy just said she “just doesn’t understand why?” Very simply, without a Wall it all doesn’t work. Our Country has a chance to greatly reduce Crime, Human Trafficking, Gangs and Drugs. Should have been done for decades. We will not Cave!'

T2 = 'Without a Wall there cannot be safety and security at the Border or for the U.S.A. BUILD THE WALL AND CRIME WILL FALL!'

T3 = 'The Fake News Media loves saying “so little happened at my first summit with Kim Jong Un.” Wrong! After 40 years of doing nothing with North Korea but being taken to the cleaners, & with a major war ready to start, in a short 15 months, relationships built, hostages & remains....'



O1 = 'I’ve always drawn inspiration from what Dr. King called life’s most persistent and urgent question: "What are you doing for others?" Let’s honor his legacy by standing up for what is right in our communities and taking steps to make a positive impact on the world.'

O2 = 'In 2018 people stepped up and showed up like never before. Keep it up in 2019. We’ve got a lot of work to do, and I’ll be right there with you. Happy New Year, everybody!'

O3 = 'I hope you find inspiration in the stories of Dejah, Moussa, Sandor, Hong and Jonny. Their journeys began with a decision to build the better future they wanted to see. The same is true for you. What matters isn’t the size of the step you take; what matters is that you take it.'



Q3 = 'Nadler just said that I “pressured Ukraine to interfere in our 2020 Election.” Ridiculous, and he knows that is not true. Both the President & Foreign Minister of Ukraine said, many times, that there “WAS NO PRESSURE.” Nadler and the Dems know this, but refuse to acknowledge!'



tweets_list = [T1, O1, T2, O2, T3, O3, Q3]



for indx, tweet in enumerate(tweets_list):

    result = cls_nb_gscv_fit.predict(tfidf_vectorize.transform([tweet])).item()

    print(indx, 'Donald Trump' if result == 0 else 'Barak Obama')