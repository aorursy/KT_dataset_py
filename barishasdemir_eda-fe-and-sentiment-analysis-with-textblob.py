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
import matplotlib.pyplot as plt

import seaborn as sns

import re



%matplotlib inline



# We want to see whole content (non-truncated)

pd.set_option('display.max_colwidth', None)


# Load the tweets

tweets_raw = pd.read_csv("/kaggle/input/tweets-about-distance-learning/tweets_raw.csv")



# Print the first five rows

display(tweets_raw.head())



# Print the summary statistics

print(tweets_raw.describe())



# Print the info

print(tweets_raw.info())


# We do not need first two columns. Let's drop them out.

tweets_raw.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)



# Drop duplicated rows

tweets_raw.drop_duplicates(inplace=True)



# Created at column's type should be datatime

tweets_raw["Created at"] = pd.to_datetime(tweets_raw["Created at"])



# Print the info again

print(tweets_raw.info())
# Print the minimum datetime

print("Since:",tweets_raw["Created at"].min())



# Print the maximum datetime

print("Until",tweets_raw["Created at"].max())
# Fill the missing values with unknown tag

tweets_raw["Location"].fillna("unknown", inplace=True)



# Print the unique locations and number of unique locations

print("Unique Values:",tweets_raw["Location"].unique())

print("Unique Value count:",len(tweets_raw["Location"].unique()))
# Set the seaborn style

sns.set()

# Plot the histogram of hours

sns.distplot(tweets_raw["Created at"].dt.hour, bins=24)

plt.title("Hourly Distribution of Tweets")

plt.show()
# Display the most popular tweets

display(tweets_raw.sort_values(by=["Favorites","Retweet-Count", ], axis=0, ascending=False)[["Content","Retweet-Count","Favorites"]].head(20))
import nltk

"""

nltk.download('punkt')

nltk.download('wordnet')

"""

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer 

"""

def process_tweets(tweet):

    

    # Remove links

    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)

    

    # Remove mentions and hashtag

    tweet = re.sub(r'\@\w+|\#','', tweet)

    

    # Tokenize the words

    tokenized = word_tokenize(tweet)



    # Remove the stop words

    tokenized = [token for token in tokenized if token not in stopwords.words("english")] 



    # Lemmatize the words

    lemmatizer = WordNetLemmatizer()

    tokenized = [lemmatizer.lemmatize(token, pos='a') for token in tokenized]



    # Remove non-alphabetic characters and keep the words contains three or more letters

    tokenized = [token for token in tokenized if token.isalpha() and len(token)>2]

    

    return tokenized

    

# Call the function and store the result into a new column

tweets_raw["Processed"] = tweets_raw["Content"].str.lower().apply(process_tweets)



"""

# After function call I have saved the file as tweets_processed.csv.

tweets_raw = pd.read_csv("/kaggle/input/tweets-processed/tweets_processed.csv", parse_dates=["Created at"])



# Print the first fifteen rows of Processed

display(tweets_raw[["Processed"]].head(15))
# Import TfidfVectorizer from sklearn

from sklearn.feature_extraction.text import TfidfVectorizer



# Create our contextual stop words

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



# Initialize a Tf-idf Vectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words= tfidf_stops)



# Fit and transform the vectorizer

tfidf_matrix = vectorizer.fit_transform(tweets_raw["Processed"])



# Let's see what we have

display(tfidf_matrix)



# Create a DataFrame for tf-idf vectors and display the first five rows

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns= vectorizer.get_feature_names())

display(tfidf_df.head())
# Import wordcloud

from wordcloud import WordCloud



# Create a new DataFrame called frequencies

frequencies = pd.DataFrame(tfidf_matrix.sum(axis=0).T,index=vectorizer.get_feature_names(),columns=['total frequency'])



# Sort the words by frequency

frequencies.sort_values(by='total frequency',ascending=False, inplace=True)

# Display the most 20 frequent words

display(frequencies.head(20))
# Join the indexes

frequent_words = " ".join(frequencies.index)+" "



# Initialize the word cloud

wc = WordCloud(width = 500, height = 500, min_font_size = 10, max_words=2000, background_color ='white', stopwords= tfidf_stops)



# Generate the world clouds for each type of label

tweets_wc = wc.generate(frequent_words)



# Plot the world cloud                     

plt.figure(figsize = (10, 10), facecolor = None) 

plt.imshow(tweets_wc, interpolation="bilinear") 

plt.axis("off") 

plt.title("Common words in the tweets")

plt.tight_layout(pad = 0) 

plt.show()
# Get the tweet lengths

tweets_raw["Length"] = tweets_raw["Content"].str.len()



# Get the number of words in tweets

tweets_raw["Words"] = tweets_raw["Content"].str.split().str.len()



# Display the new columns

display(tweets_raw[["Length", "Words"]])
# Since the processes takes a lot, I have used preprocessed column on Kaggle



# Import pycountry

import pycountry

"""

def get_countries(location):

    

    # If location is a country name return its alpha2 code

    if pycountry.countries.get(name= location):

        return pycountry.countries.get(name = location).alpha_2

    

    # If location is a subdivisions name return the countries alpha2 code

    try:

        pycountry.subdivisions.lookup(location)

        return pycountry.subdivisions.lookup(location).country_code

    except:

        # If the location is neither country nor subdivision return the "unknown" tag

        return "unknown"



# Call the function and store the country codes in the Country column

tweets_raw["Country"] = tweets_raw["Location"].apply(get_countries)



# Print the unique values

print(tweets_raw["Country"].unique())

"""



# Print the number of unique values

print("Number of unique values:",len(tweets_raw["Country"].unique()))
# We need to exclude unknowns

countries = tweets_raw[tweets_raw.Country!='unknown']



# Select the top 20 countries

top_countries = countries["Country"].value_counts(sort=True).head(20)



# Convert alpha2 country codes to country names and store in a list

country_fullnames = []

for alpha2 in top_countries.index:

    country_fullnames.append(pycountry.countries.get(alpha_2=alpha2).name)



# Visualize the top 20 countries

plt.figure(figsize=(12,10))

sns.barplot(y=country_fullnames,x=top_countries, orient="h", palette="RdYlGn")

plt.xlabel("Tweet count")

plt.ylabel("Countries")

plt.title("Top 20 Countries")

plt.show()
# Import the TextBlob

from textblob import TextBlob



"""

# Add polarities and subkectivities into the DataFrame by using TextBlob

tweets_raw["Polarity"] = tweets_raw["Processed"].apply(lambda word: TextBlob(word).sentiment.polarity)

tweets_raw["Subjectivity"] = tweets_raw["Processed"].apply(lambda word: TextBlob(word).sentiment.subjectivity)

"""



# Since the processes takes a lot, I have used preprocessed column on Kaggle

tweets_raw = pd.read_csv("/kaggle/input/tweets-sentiments/tweets_sentiments.csv", parse_dates=["Created at"])



# Display the Polarity and Subjectivity columns

display(tweets_raw[["Polarity","Subjectivity"]].head(10))
# Define a function to classify polarities

def analyse_polarity(polarity):

    if polarity > 0:

        return "Positive"

    if polarity == 0:

        return "Neutral"

    if polarity < 0:

        return "Negative"



# Apply the funtion on Polarity column and add the results into a new column

tweets_raw["Label"] = tweets_raw["Polarity"].apply(analyse_polarity)



# Display the Polarity and Subjectivity Analysis

display(tweets_raw[["Label"]].head(10))
# Print the value counts of the Label column

print(tweets_raw["Label"].value_counts())
# Change the datatype as "category"

tweets_raw["Label"] = tweets_raw["Label"].astype("category")



# Visualize the Label counts

sns.countplot(tweets_raw["Label"])

plt.title("Label Counts")

plt.show()



# Visualize the Polarity scores

plt.figure(figsize = (10, 10)) 

sns.scatterplot(x="Polarity", y="Subjectivity", hue="Label", data=tweets_raw)

plt.title("Subjectivity vs Polarity")

plt.show()
# Display the positive tweets

display(tweets_raw.sort_values(by=["Polarity","Favorites","Retweet-Count", ], axis=0, ascending=[False, False, False])[["Content","Retweet-Count","Favorites","Polarity"]].head(20))



# Display the negative tweets

display(tweets_raw.sort_values(by=["Polarity", "Favorites", "Retweet-Count"], axis=0, ascending=[True, False, False])[["Content","Retweet-Count","Favorites","Polarity"]].head(20))
def make_wordcloud(data, label):



    # Initialize a Tf-idf Vectorizer

    polarity_vectorizer = TfidfVectorizer(max_features=5000, stop_words= tfidf_stops)



    # Fit and transform the vectorizer

    tfidf_matrix_polarity = polarity_vectorizer.fit_transform(tweets_raw["Processed"])



    # Create a new DataFrame called frequencies

    frequencies_polarity = pd.DataFrame(tfidf_matrix_polarity.sum(axis=0).T,index=polarity_vectorizer.get_feature_names(),columns=['total frequency'])



    # Sort the words by frequency

    frequencies_polarity.sort_values(by='total frequency',ascending=False, inplace=True)



    # Join the indexes

    frequent_words_polarity = " ".join(frequencies_polarity.index)+" "



    # Initialize the word cloud

    wc = WordCloud(width = 500, height = 500, min_font_size = 10, max_words=2000, background_color ='white', stopwords= tfidf_stops)



    # Generate the world clouds for each type of label

    tweets_polarity = wc.generate(frequent_words_polarity)



    # Plot the world cloud                     

    plt.figure(figsize = (10, 10), facecolor = None) 

    plt.imshow(tweets_polarity, interpolation="bilinear") 

    plt.axis("off") 

    plt.title("Common words in the " + label +" tweets")

    plt.tight_layout(pad = 0) 

    plt.show() 



# Create DataFrames for each label

positive_popular_df = tweets_raw.sort_values(by=["Polarity","Favorites","Retweet-Count", ], axis=0, ascending=[False, False, False])[["Content","Retweet-Count","Favorites","Polarity","Processed"]].head(50)

negative_popular_df = tweets_raw.sort_values(by=["Polarity", "Favorites", "Retweet-Count"], axis=0, ascending=[True, False, False])[["Content","Retweet-Count","Favorites","Polarity","Processed"]].head(50)



# Call the function

make_wordcloud(positive_popular_df, "positive")

make_wordcloud(negative_popular_df, "negative")
# Get the positive/negative counts by country

positives_by_country = tweets_raw[tweets_raw.Country!='unknown'].groupby("Label")["Country"].value_counts().Negative.sort_values(ascending=False)

negatives_by_country =tweets_raw[tweets_raw.Country!='unknown'].groupby("Label")["Country"].value_counts().Positive.sort_values(ascending=False)



# Print them out

print("Positive \n")

print(positives_by_country)

print("\nNegative\n")

print(negatives_by_country)



# Create a mask for top 1 countries (by tweets count)

mask = tweets_raw["Country"].isin(top_countries.index[:10]).values



# Create a new DataFrame only includes top10 country

top_20df = tweets_raw.iloc[mask,:]



# Visualize the top 20 countries

plt.figure(figsize=(12,10))

sns.countplot(x="Country", hue="Label", data=top_20df, order=top_20df["Country"].value_counts().index)

plt.xlabel("Countries")

locs, labels = plt.xticks()

plt.xticks(locs, country_fullnames[:10])

plt.xticks(rotation=45)

plt.ylabel("Tweet count")

plt.title("Top 10 Countries")

plt.show()
positive = tweets_raw.loc[tweets_raw.Label=="Positive"]["Created at"].dt.hour

negative = tweets_raw.loc[tweets_raw.Label=="Negative"]["Created at"].dt.hour



plt.hist(positive, alpha=0.5, bins=24, label="Positive", density=True)

plt.hist(negative, alpha=0.5, bins=24, label="Negative", density=True)

plt.xlabel("Hour")

plt.ylabel("PDF")

plt.title("Hourly Distribution of Tweets")

plt.legend(loc='upper right')

plt.show()