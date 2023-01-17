# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("hello world")
print('test')

import pandas as pd 
import re 
import string 
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk.corpus  
from nltk.text import Text 
from PIL import Image
from textblob import TextBlob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
# Load the tweets
Tweets = pd.read_csv("../input/tweets-about-distance-learning/tweets_raw.csv", index_col = 0)

# Drop uneccessary 'Unnamed: 0.1 column'
Tweets.drop("Unnamed: 0.1", axis=1, inplace=True)

# Drop duplicated rows
Tweets.drop_duplicates(inplace=True)
Tweets.reset_index(drop=True, inplace=True)

# Change "Created at" to the datatype datetime
Tweets["Created at"] = pd.to_datetime(Tweets["Created at"])

# Fill in missing values for location 
Tweets["Location"].fillna("unknown", inplace=True)

Tweets.info()
# Pre-Processing Function 
def preprocess_text(text): 
    # get lowercase
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove urls
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    # remove punctuation
    text = text.translate(text.maketrans('', '', string.punctuation))
    # strip whitespace
    text = text.strip()
    
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    words = [w for w in tokens if not w in stop_words]
    text = " ".join(w for w in words)
    
    # Remove non-alphabetic characters and keep the words contains three or more letters
    tokens = word_tokenize(text)
    words = [w for w in
             tokens if w.isalpha() and len(w)>2]
    text = " ".join(w for w in words)
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    words = [lemmatizer.lemmatize(w, pos = 'a') for w in tokens]
    result = " ".join(w for w in words)
    return result
# Process tweets
Tweets['processed_post'] = [preprocess_text(post) for post in Tweets['Content'].values]
def get_sentiment(post): 
    blob = TextBlob(post) 
    return blob.sentiment 

from textblob import TextBlob
# Get sentiment and append into dataframe
polarity = []
subjectivity = []
for post in Tweets['processed_post']:
    polar = get_sentiment(post).polarity
    polarity.append(polar)
    subjec = get_sentiment(post).subjectivity
    subjectivity.append(subjec)
    
Tweets['polarity'] = polarity
Tweets['subjectivity'] = subjectivity
# Classify tweets as positive, negative, or neutral based on polarity
polarity = [None] * len(Tweets) 

for i in range(0, len(Tweets)):
    if Tweets["polarity"][i] > 0:
        polarity[i] = "Positive"
    elif Tweets["polarity"][i] == 0:
        polarity[i] = "Neutral"
    else:
        polarity[i] = "Negative"

Tweets["Label"] = polarity
# Examine distribution of tweet sentiments
import seaborn as sns 
import matplotlib.pyplot as plt 

ax = sns.countplot(x ='Label', data = Tweets, palette = "Set2")
ax.set(xlabel='Polarity')
plt.show()
# Remove outliers from favorites for a better visualization
# of the relationship between favorites and polarity
favorites = Tweets[Tweets["Favorites"] < 8000]

sns.relplot(x="polarity", y="Favorites", data=favorites)
sns.relplot(x="subjectivity", y="Favorites", data=favorites)
retweets = Tweets[Tweets["Retweet-Count"] < 1000]
sns.relplot(x="polarity", y="Retweet-Count", data=retweets)
sns.relplot(x="subjectivity", y="Retweet-Count", data=retweets)
# Examine the relationship between subjectivity and polarity
sns.catplot(x="Label", y="subjectivity", kind="box", data=Tweets, palette = "Set2")
# Visualize the Polarity scores
plt.figure(figsize = (10, 10)) 
sns.scatterplot(x="polarity", y="subjectivity", hue="Label", data=Tweets, palette = "Set2")
plt.title("Subjectivity vs Polarity")
plt.show()
# Look at popular words for positive and negative tweets
# Create DataFrames for each label
positive_df = Tweets.sort_values(by=["polarity","Favorites","Retweet-Count", ], axis=0,
    ascending=[False, False, False])[["Content","Retweet-Count","Favorites","polarity","processed_post"]].head(50)

negative_df = Tweets.sort_values(by=["polarity", "Favorites", "Retweet-Count"], axis=0, 
    ascending=[True, False, False])[["Content","Retweet-Count","Favorites","polarity","processed_post"]].head(50)

objective_df = Tweets.sort_values(by=["subjectivity","Favorites","Retweet-Count", ], axis=0,
    ascending=[True, False, False])[["Content","Retweet-Count","Favorites","subjectivity","processed_post"]].head(30)

subjective_df = Tweets.sort_values(by=["subjectivity","Favorites","Retweet-Count", ], axis=0,
    ascending=[False, False, False])[["Content","Retweet-Count","Favorites","subjectivity","processed_post"]].head(30)

posts = positive_df.processed_post.values
positive_text = " ".join(t for t in posts)

posts = negative_df.processed_post.values
negative_text = " ".join(t for t in posts)

posts = subjective_df.processed_post.values
subjective_text = " ".join(t for t in posts)

posts = objective_df.processed_post.values
objective_text = " ".join(t for t in posts)
# Identify the stopwords used to gather tweets
tfidf_stops = ["online", "class", "course", "learning", "learn","teach", "teaching", "distance", \
               "distancelearning", "education", "teacher", "student", "grade", "classes", "computer", "resource", \
               "onlineeducation", "onlinelearning", "school", "students", "class", "virtual", "eschool", "thing", \
               "virtuallearning", "educated", "educates", "teaches", "studies", "study", "semester", "elearning", \
               "teachers", "lecturer", "lecture", "amp", "academic", "admission", "academician", "account", "action",\
               "add", "app", "announcement", "application", "adult", "classroom", "system", "video", "essay", "training", \
               "homework","work","assignment", "paper", "get", "math", "project", "science", "physics", "lesson", "schools", \
               "courses", "assignments", "know", "instruction","email", "discussion","home", "college", "exam", "university", \
               "use", "fall", "term", "proposal", "one", "review", "proposal", "calculus", "search", "research", "algebra", \
               "internet", "remote", "remotelearning"]
# Positive tweets
mask = np.array(Image.open("../input/word-cloud-mask/Unknown.png"))

wordcloud = WordCloud(mask = mask, stopwords = tfidf_stops, max_font_size=100, max_words = 500, width = 400, height = 300, background_color = 'white',
                      scale = 6, collocations=False, colormap = 'summer').generate(positive_text)
plt.figure(figsize=(24, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# Negative Tweets
mask = np.array(Image.open("../input/word-cloud-mask/Unknown.png"))

wordcloud = WordCloud(mask = mask, stopwords = tfidf_stops, max_font_size=100, max_words = 500, width = 400, height = 300, background_color = 'white',
                      scale = 6, collocations=False, colormap = 'cool').generate(negative_text)
plt.figure(figsize=(24, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()