import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load the data 

# Since volume of data is very large ~500k reviews, so we can take a subset of reviews to do the analysis faster

data = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv").head(10000)



# top 2 rows of the data

data.head(2)
# Number of rows and columns, ~568k

data.shape
# Columns of the data, their data type

data.info()
# Is there any null values for any of the columns ?

data.isna().sum()

# Profile name and summary has null values
# Let' remove the Id columnas it's of no use to us. It's just unique id of the row

data.drop(labels=["Id"], axis=1, inplace=True)
# Is there any duplicate review ?

data[data.duplicated(subset=["Text"], keep="first")].shape
# There are 174k duplicate reviews text.

# Let's see the results with all the columns

data[data.duplicated(subset=["UserId", "ProductId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", 

                             "Score", "Time", "Summary", "Text"])].shape
# Even if the duplicate review texts are 174k, but duplicate records are very very less, why is that?

# Let's remove the product id

data[data.duplicated(subset=["UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", 

                             "Score", "Time", "Summary", "Text"])].shape
# Now with product id removed, we have 172k records as duplicate

data[data.duplicated(subset=["UserId", "Time", "Summary", "Text"])].head(10)
# Lets check for a particular user id

data[data["UserId"] == "AQM74O8Z4FMS0"]
# Okay, it seems product with little variations are given different product ids. For example, 

# all the flavours of Junior horlicks (chocolate, vanilla, strawberry etc.) 500gm pack will have different product 

# ids, but when the review was given  

# for one flavour, they made it available for all other flavours, that's why we have this duplication 

# and the fuss about product id.

# We can check for other user id also
data[data["UserId"] == "A2EPNS38TTLZYN"]
# The same user has given same reviews to different products at the very same time and the no. people who found it 

# useful is also the same. How is it possible?

# Possible only when the same review has been generated using the review of similar product.

# Also, there are cases when helpfulness values are also different for the same text of review.

# Since we are only concerned for the unique reviews, we will ignore those also.
# If any two reviews have same userid, timestamp, summary and text, then we will take them as duplicates.

data = data.drop_duplicates(subset=["UserId", "Time", "Summary", "Text"])
# from 568k to 395k

data.shape
data[data.duplicated("Text")].shape
# Thereare still 1.5k review texts which are duplicates.

# Let' see what's different in them ?
text = data["Text"]

data[text.isin(text[text.duplicated()])].sort_values("Text")
# Few are different user id but same time and few are with different summaries. 

# You can delete them but I am not gonna tocuh them, I will keep them there.
# Still are there null values?

data[data.isnull().any(1)]

# Few of the profile names are missing and few summaries 
# Filling up the null user names by ""

data["ProfileName"] = data["ProfileName"].fillna("")
# Filling up the null summaries by ""

data["Summary"] = data["Summary"].fillna("")
# We have two records where helpfulness numerator is greater than helpfulness denominator

# We will remove those reviews

data = data[data["HelpfulnessNumerator"] <= data["HelpfulnessDenominator"]]
# HelpfulnessNumertaor - number of people found the review useful

# HelpfulnessDenominator - total number of people who told whether they found the review useful or not
# Let' see how the score is divided

(data["Score"].value_counts()/data.shape[0])*100

# We can make a pie chart for this
data["Time"] = pd.to_datetime(data["Time"], unit="s")
data["Year"], data["Month"] = data["Time"].dt.year, data["Time"].dt.month
data["Year"].value_counts().plot(kind="bar")
data["Month"].value_counts().plot(kind="bar")
# One hot encoded values of year and month 

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

month_and_year_ohe = ohe.fit_transform(data[["Month", "Year"]])
month_and_year_ohe.toarray().shape
ohe.categories_
column_names = []

for month_num in ohe.categories_[0]:

    column_names.append("Month_" + str(int(month_num)))



for year in ohe.categories_[1]:

    column_names.append("Year_" + str(int(year)))
month_and_year_processed = pd.DataFrame(month_and_year_ohe.toarray(), columns=column_names)
month_and_year_processed.head()
# COLLINEARITY ISSUE ?
# printing some random reviews

sent_0 = data['Text'].values[10]

print(sent_0)

print("="*50)



sent_1000 = data['Text'].values[1000]

print(sent_1000)

print("="*50)



sent_1500 = data['Text'].values[1500]

print(sent_1500)

print("="*50)



sent_4900 = data['Text'].values[4900]

print(sent_4900)

print("="*50)
# https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string

import re



def cleanhtml(raw_html):

  cleaner = re.compile('<.*?>')

  cleantext = re.sub(cleaner, ' ', raw_html)

  return cleantext
print(cleanhtml(sent_0))
# it's, i'm, shouldn't, can't, cake's, you'll,  pumpin', won't, i'll, doesn't, i've, that's, arn't, isn't, 'coz

#  face'll, don;t, let's, he'd, wasn't, that's, didn't, i'd, get'em, what's, don't, got'em, get'n, they're, comin', 

# ain't, there's, i'be, they'd, which'll, anddon't, had'nt, that's, deosn't, she's, won't, should'nt, he's, recv'd, 

# livin', keep'em, it'sa, can't ,  you're, dind't, they've
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

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

    phrase = re.sub(r"\'em", " them", phrase)

    return phrase
sent_1500 = decontracted(sent_1500)

print(sent_1500)

print("="*50)
#remove words with numbers python: https://stackoverflow.com/a/18082370/4084039

sent_0 = re.sub("\S*\d\S*", "", sent_0).strip()

print(sent_0)
#remove special character: https://stackoverflow.com/a/5843547/4084039

sent_1500 = re.sub('[^A-Za-z0-9]+', ' ', sent_1500)

print(sent_1500)
# From spacy 

import spacy 

nlp = spacy.load('en', disable=['parser', 'ner'])

print(" ".join([word.lemma_ for word in nlp(sent_1500.lower())]))
# We can also use Stemming than lemmatization 

from nltk.stem import SnowballStemmer

sno = SnowballStemmer('english')

print(" ".join([sno.stem(word) for word in sent_1500.lower().split(" ")]))
# lemmatization from nltk

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

print(" ".join([wnl.lemmatize(word.lower()) for word in sent_1500.split(" ")]))
def make_lemmatized(sent):

    " ".join([wnl.lemmatize(word.lower()) for word in sent_1500.split(" ")])
# we are removing the words from the stopwords list: 'no', 'nor', 'not'



stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
# Combining all the above stundents 

from tqdm import tqdm

import re

from bs4 import BeautifulSoup

preprocessed_reviews = []

# tqdm is for printing the status bar

for sentence in tqdm(data['Text'].values):

    sentence = re.sub(r"http\S+", "", sentence)

    sentence = cleanhtml(sentence)

    sentence = decontracted(sentence)

    sentence = re.sub("\S*\d\S*", "", sentence).strip()

    sentence = re.sub('[^A-Za-z]+', ' ', sentence)

    # https://gist.github.com/sebleier/554280

    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)

    sentence = " ".join([wnl.lemmatize(word) for word in sentence.split(" ")])

    preprocessed_reviews.append(sentence.strip())
data["Text_cleaned"] = preprocessed_reviews
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
def plot_word_cloud(review_score, column_name):

    temp_data = data[data["Score"] == review_score][column_name].values

    reviews_with_given_score = " ".join(temp_data)

    cloud = WordCloud(background_color = "white", width=800, height=400, max_words = 300, stopwords = set(STOPWORDS))

    wordcloud = cloud.generate(reviews_with_given_score)

    plt.figure(figsize=(12, 8))

    plt.imshow(wordcloud)
plot_word_cloud(review_score=1, column_name="Text_cleaned")
plot_word_cloud(review_score=2, column_name="Text_cleaned")
plot_word_cloud(review_score=3, column_name="Text_cleaned")
plot_word_cloud(review_score=4, column_name="Text_cleaned")
plot_word_cloud(review_score=5, column_name="Text_cleaned")
# Let's do it for summary and profile names also

# SUMMARY
preprocessed_summaries = []

for sentence in tqdm(data['Text'].values):

    sentence = re.sub(r"http\S+", "", sentence)

    sentence = cleanhtml(sentence)

    sentence = decontracted(sentence)

    sentence = re.sub("\S*\d\S*", "", sentence).strip()

    sentence = re.sub('[^A-Za-z]+', ' ', sentence)

    # https://gist.github.com/sebleier/554280

    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)

    sentence = " ".join([wnl.lemmatize(word) for word in sentence.split(" ")])

    preprocessed_summaries.append(sentence.strip())
data["Summary_cleaned"] = preprocessed_summaries
plot_word_cloud(review_score=5, column_name="Summary_cleaned")
plot_word_cloud(review_score=4, column_name="Summary_cleaned")
plot_word_cloud(review_score=3, column_name="Summary_cleaned")
plot_word_cloud(review_score=2, column_name="Summary_cleaned")
plot_word_cloud(review_score=1, column_name="Summary_cleaned")
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

count_vect.fit(preprocessed_reviews)

print("Some features names: ", count_vect.get_feature_names()[:10])

print("="*50)



final_counts = count_vect.transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_counts))

print("the shape of out text BOW vectorizer ",final_counts.get_shape())

print("the number of unique words ", final_counts.get_shape()[1])
#bi-gram, tri-gram and n-gram



# removing stop words like "not" should be avoided before building n-grams

# count_vect = CountVectorizer(ngram_range=(1,2))

# please do read the CountVectorizer documentation 

# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# you can choose these numebrs min_df=10, max_features=5000, of your choice

count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)

final_bigram_counts = count_vect.fit_transform(preprocessed_reviews)

print("some feature names ", count_vect.get_feature_names()[:10])

print("the type of count vectorizer ",type(final_bigram_counts))

print("the shape of out text BOW vectorizer ",final_bigram_counts.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])
column_names = ["BoW_" + str(feat_name) for feat_name in count_vect.get_feature_names()]

bow_processed_reviews = pd.DataFrame(final_bigram_counts.toarray(), columns=column_names)
from sklearn.feature_extraction.text import TfidfVectorizer



tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)

tf_idf_vect.fit(preprocessed_reviews)

print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])

print('='*50)



final_tf_idf = tf_idf_vect.transform(preprocessed_reviews)

print("the type of count vectorizer ",type(final_tf_idf))

print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())

print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])
column_names = ["TfIdf_" + str(feat_name) for feat_name in tf_idf_vect.get_feature_names()]

tfidf_processed_reviews = pd.DataFrame(final_tf_idf.toarray(), columns=column_names)
tfidf_processed_reviews.head()
# We can train our own Word2Vec model using the corpus of reviews we have

i=0

list_of_words=[]

for sentence in data["Text_cleaned"].values:

    list_of_words.append(sentence.split(" "))
# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.W17SRFAzZPY

# you can comment this whole cell

# or change these varible according to your need



from gensim.models import Word2Vec

min_count = 10  # min count of word to be selected for w2v algo

size=100  # size or dimension of the vector 



# min_count = 5 considers only words that occured atleast 5 times

w2v_model = Word2Vec(list_of_words, min_count=min_count, size=size, workers=10)

print(w2v_model.wv.most_similar('great'))

print('='*50)

print(w2v_model.wv.most_similar('worst'))

w2v_words = list(w2v_model.wv.vocab)

print("number of words that occured minimum {} times:  {}".format(min_count, len(w2v_words)))

print("sample words ", w2v_words[0:50])
# WE CAN ALWAYS USE THE W2V MODEL OF GOOGLE AND OTHER POPULAR ONES ALSO
# average Word2Vec

# compute average word2vec for each review.

def cal_avg_w2v(column_name):

    sent_vectors = [] 

    for sent in tqdm(data[column_name].values): 

        sent_vec = np.zeros(size) 

        cnt_words = 0 

        for word in sent.split(" "): 

            if word in w2v_words:

                vec = w2v_model.wv[word]

                sent_vec += vec

                cnt_words += 1

        if cnt_words != 0:

            sent_vec /= cnt_words

        sent_vectors.append(sent_vec)

    return sent_vectors
avg_w2v_processed_text = cal_avg_w2v(column_name="Text_cleaned")   
model = TfidfVectorizer()

model.fit(data["Text_cleaned"].values)

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
def cal_tfidf_avg_w2v(column_name):



    tfidf_feat = model.get_feature_names() 

    tfidf_sent_vectors = []

    row=0;

    for sent in tqdm(data[column_name].values): # for each review/sentence 

        sent_vec = np.zeros(size) 

        weight_sum =0

        for word in sent.split(" "): # for each word in a review/sentence

            if word in w2v_words and word in tfidf_feat:

                vec = w2v_model.wv[word]

                # tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]

                # to reduce the computation we are 

                # dictionary[word] = idf value of word in whole courpus

                # sent.count(word) = tf valeus of word in this review

                tf_idf = dictionary[word]*(sent.count(word)/len(sent))

                sent_vec += (vec * tf_idf)

                weight_sum += tf_idf

        if weight_sum != 0:

            sent_vec /= weight_sum

        tfidf_sent_vectors.append(sent_vec)

        row += 1

    return tfidf_sent_vectors
tfidf_wt_w2v_processed_text = cal_tfidf_avg_w2v("Text_cleaned")
# We can apply these processes to Summary and Profile name also.

# Then may be with these features we can do some PCA, get some features and apply model.



# HAPPY MODELING :)