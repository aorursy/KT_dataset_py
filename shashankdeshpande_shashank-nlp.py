# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import json

from fuzzywuzzy import fuzz

import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import f1_score, accuracy_score



from sklearn.linear_model import LogisticRegression

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt

# from lxml import html

import html

from nltk.corpus import stopwords 

nltk_stopwords = set(stopwords.words('english'))

from geopy.geocoders import Nominatim

from nltk.stem import WordNetLemmatizer 

from tqdm import tqdm

tqdm.pandas()

import warnings

warnings.filterwarnings("ignore")

try:

    import preprocessor as p

    import reverse_geocode

    import splitter

except ModuleNotFoundError:

    !pip install tweet-preprocessor

    !pip install reverse-geocode==1.4

    !apt-get install -y libenchant1c2a

    !pip install pyenchant

    !pip install compound-word-splitter

    import preprocessor as p

    import reverse_geocode

    import splitter

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



SUBMISSION = True
train.head()
train.info()
train.isnull().sum()
emoticons_happy = set([

    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',

    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',

    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',

    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',

    '<3'

    ])

emoticons_sad = set([

    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',

    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',

    ':c', ':{', '>:\\', ';('

    ])



def get_smiley_type(tweet_text):

    e_type = "missing"

    happy = any(map(lambda x:x in emoticons_happy, tweet_text.split()))

    sad = any(map(lambda x:x in emoticons_sad, tweet_text.split()))

    if happy and sad:

        e_type = "mixed"

    elif happy:

        e_type = "happy"

    elif sad:

        e_type = "sad"

    return e_type



def get_emoji(text):

    emoji = ""

    parsed_obj = p.parse(text)

    emoji_list = parsed_obj.emojis

    if emoji_list:

        emoji = ", ".join(map(lambda x:x.match, emoji_list))

    return emoji
lemmatizer = WordNetLemmatizer()

hashtag_pattern = re.compile(r"#(\w+)")

mention_pattern = re.compile(r"@(\w+)")

url_pattern = re.compile(r"(?P<url>https?:\/\/[^\s]+)")



def split_compound_words(word):

    new_word = word.strip("#")

    compound = splitter.split(new_word)

    if compound:

        new_word = " ".join(compound)

    return new_word



def get_clean_text(text):

    text = text.strip().lower()

    text = text.replace("\n"," ")

    text = text.replace("\t"," ")

    

    # remove html & URL

    text = html.unescape(text)

    phrase = re.sub(url_pattern," ",text)

    

    phrase = re.sub(r"\bvia\b"," ", phrase)

    phrase = re.sub(r"\brt\b"," ", phrase)

    

    # process hashtags/mentions

    phrase = re.sub(hashtag_pattern, lambda x: split_compound_words(x.group()) ,phrase)

    phrase = re.sub(mention_pattern, " ", phrase)

    

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"[^a-zA-Z0-9 ]"," ", phrase)

    text = re.sub(r"\s+"," ", phrase)

#     text = " ".join(map(lambda x:lemmatizer.lemmatize(x), text.split()))

    text = text.strip().lower()

    return text
# location_dict = {}

# unique_locations = train.location.unique()

# unique_locations = set(unique_locations)

# print("len: ", len(unique_locations))



# geolocator = Nominatim(user_agent="twitter_location")

# from geopy.extra.rate_limiter import RateLimiter

# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)



# for i in tqdm(unique_locations):

#     loc_dict = {}

#     geo_obj = geocode(i)

#     if geo_obj:

#         lat, long = geo_obj[-1]

#         city_country = reverse_geocode.search([(lat, long)])

#         loc_dict = city_country[0]

#         loc_dict.update({"lat":lat, "long":long})

#     location_dict[i] = loc_dict



with open("/kaggle/input/lat-long-data/kaggle.json") as f:

    location_dict = json.load(f)



def add_city_country(loc):

    loc_data = location_dict.get(loc,{})

    city = loc_data.get("city","")

    country = loc_data.get("country","")

    return city, country
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



def process_df(df):

#     df["smiley"] = df["text"].progress_apply(get_smiley_type)

#     df["smiley_label"] = le.fit_transform(df["smiley"])

    

    df["clean_text"] = df["text"].progress_apply(get_clean_text)

    

#     df["emoji"] = df["text"].progress_apply(get_emoji)

    

#     new_cols = df["location"].progress_apply(add_city_country)

#     city_country_df = pd.DataFrame.from_records(new_cols, columns =['city', 'country']) 

#     df = df.join(city_country_df)

#     df["city_label"] = le.fit_transform(df["city"])

#     df["country_label"] = le.fit_transform(df["country"])

    return df
train = process_df(train)

test = process_df(test)

train.head()
test.head()
to_remove_indexes = []



# Remove duplicates using fuzzywuzzy

# for index, row in train.iterrows():

#     current_clean_text = row["clean_text"]

#     if index not in to_remove_indexes and current_clean_text and len(current_clean_text.split())>=4:

#         match_df = train[train.clean_text.str.contains("\b" + current_clean_text + "\b", regex=True)]

#         match_df = match_df[match_df["clean_text"].apply(lambda x: fuzz.ratio(current_clean_text,x)>90)]

#         if len(match_df.target.unique())>1:

#             to_remove_indexes.extend(match_df.index.tolist())



# Remove duplicates using exact match

duplicates = train[train.duplicated(["clean_text"], keep=False)].sort_values(by=['clean_text'])

duplicates = duplicates.groupby('clean_text')

for text, indexes in duplicates.groups.items():

    target_vals = []

    for idx in indexes:

        target_vals.append(train.iloc[idx].target)

    if len(set(target_vals))>1:

        to_remove_indexes.extend(indexes)
train.iloc[to_remove_indexes]
train.drop(train.index[to_remove_indexes], inplace=True)

train.shape
disaster = " ".join(train[train["target"]==1]["clean_text"].values)

normal = " ".join(train[train["target"]==0]["clean_text"].values)



disaster_wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = nltk_stopwords, 

                min_font_size = 10).generate(disaster)

normal_wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = nltk_stopwords, 

                min_font_size = 10).generate(normal) 

  

# plot the WordCloud image                        

fig = plt.figure(figsize = (8, 8), facecolor = None) 

ax1 = fig.add_subplot(2,1,1)

ax1.imshow(disaster_wordcloud) 

ax1.axis("off")

ax2 = fig.add_subplot(2,1,2)

ax2.imshow(normal_wordcloud) 

ax2.axis("off")

fig.show()
if SUBMISSION:

    train_df = train

    test_df = test

else:

    TEST_SIZE = 0.2

    train_df, test_df = train_test_split(train, test_size=TEST_SIZE, random_state=40)



print("Train: ", train_df.shape)

print("Test: ", test_df.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(

    stop_words=nltk_stopwords,

    min_df=2,

    ngram_range=(1,2),

#     max_features = 5000,

    sublinear_tf=True

)

tfidf_vec.fit(train_df["clean_text"])





def get_tfidf_vectors(df):

    vectors = tfidf_vec.transform(df["clean_text"])

    vectors = vectors.todense()

    vectors_df = pd.DataFrame(vectors)

    return vectors_df
train_vec_df = get_tfidf_vectors(train_df)

test_vec_df = get_tfidf_vectors(test_df)

print("Train vec: ", train_vec_df.shape)

print("Test vec: ", test_vec_df.shape)
features_to_add= [] #["country_label"]



def add_other_features(vectors_df, df):

    for each in features_to_add:

        vectors_df[each] = df[each].values

    return vectors_df



final_train_vec_df = add_other_features(train_vec_df, train_df)

final_test_vec_df = add_other_features(test_vec_df, test_df)

print("Train vec: ", final_train_vec_df.shape)

print("Valid vec: ", final_test_vec_df.shape)
log_reg = LogisticRegression()

train_x, train_y = final_train_vec_df, train_df["target"]



from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

foldScore=[]

# train_x.reset_index(drop=True)

# train_y.reset_index(drop=True)

for fold_idx, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):

    Xtr, X_test = train_x.iloc[train_index], train_x.iloc[test_index]

    ytr, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

    log_reg.fit(Xtr, ytr)

    val_acc=log_reg.score(X_test,y_test)

    tra_acc=log_reg.score(Xtr,ytr)

    print(f"Fold {fold_idx+1} ==> {val_acc} {tra_acc}")

    foldScore.append([val_acc,tra_acc])
test_x = final_test_vec_df

y_pred = log_reg.predict(test_x)



if not SUBMISSION:

    test_y = test_df["target"]

    print("F1 Score: ", f1_score(test_y, y_pred))

    print("\nAccuracy Train: ", log_reg.score(train_x, train_y))

    print("Accuracy Test: ", log_reg.score(test_x, test_y))

    test_df["predicted_target"] = y_pred

else:

    test_df["target"] = y_pred
test_df.head()
test_df.to_csv("submission.csv", index=False, columns=["id","target"])