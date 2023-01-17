# Importing libraries
import warnings
warnings.filterwarnings('ignore')
import sqlite3
import pandas as pd
import numpy as np
from time import time
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import gensim
# Connection to the dataset
con = sqlite3.connect('../input/database.sqlite')

# It is given that the table name is 'Reviews'
# Creating pandas dataframe and storing into variable 'dataset' by help of sql query
dataset = pd.read_sql_query("""
SELECT *
FROM Reviews
""", con)

# Getting the shape of actual data: row, column
display(dataset.shape)
# Displaying first 5 data points
display(dataset.head())
# Considering only those reviews which score is either 1,2 or 4,5
# Since, 3 is kind of neutral review, so, we are eliminating it
filtered_data = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3
""", con)
# Getting shape of new dataset
display(filtered_data.shape)
# Changing the scores into 'positive' or 'negative'
# Score greater that 3 is considered as 'positive' and less than 3 is 'negative'
def partition(x):
    if x>3:
        return 'positive'
    return 'negative'

actual_score = filtered_data['Score']
positiveNegative = actual_score.map(partition)
filtered_data['Score'] = positiveNegative
# Sorting data points according to the 'ProductId'
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

# Eliminating the duplicate data points based on: 'UserId', 'ProfileName', 'Time', 'Summary'
final = sorted_data.drop_duplicates(subset={'UserId', 'ProfileName', 'Time', 'Summary'}, keep='first', inplace=False)

# Eliminating the row where 'HelpfulnessDenominator' is greater than 'HelpfulnessNumerator' as these are the wrong entry
final = final[final['HelpfulnessDenominator'] >= final['HelpfulnessNumerator']]

# Getting shape of final data frame
display(final.shape)
%%time

# Creating the set of stopwords
stop = set(stopwords.words('english'))

# For stemming purpose
snow = nltk.stem.SnowballStemmer('english')

# Defining function to clean html tags
def cleanhtml(sentence):
    cleaner = re.compile('<.*>')
    cleantext = re.sub(cleaner, ' ', sentence)
    return cleantext

# Defining function to remove special symbols
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|.|!|*|@|#|\'|"|,|)|(|\|/]', r'', sentence)
    return cleaned


# Important steps to clean the text data. Please trace it out carefully
i = 0
str1 = ''
all_positive_words = []
all_negative_words = []
final_string = []
s=''
for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if ((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if (cleaned_words.lower() not in stop):
                    s = (snow.stem(cleaned_words.lower())).encode('utf-8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s)
                    if (final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    str1 = b" ".join(filtered_sentence)
    final_string.append(str1)
    i += 1
    
# Adding new column into dataframe to store cleaned text
final['CleanedText'] = final_string
final['CleanedText'] = final['CleanedText'].str.decode('utf-8')

# Creating new dataset with cleaned text for future use
conn = sqlite3.connect('final.sqlite')
c = conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)

# Getting shape of new datset
print(final.shape)
# Creating connection to read from database
conn = sqlite3.connect('./final.sqlite')

# Creating data frame for visualization using sql query
final = pd.read_sql_query("""
SELECT *
FROM Reviews
""", conn)
# Displaying first 3 data points of newly created datset
display(final.head(3))
# Getting the number of data points in each class: positive or negative
display(final['Score'].value_counts())
# Taking equal sample of negative and positive reviews to keep it balanced.
# If it is not balanced then there is chance that one class lebel can dominant other class label which might be sever probelm sometimes.
positive_points = final[final['Score'] == 'positive'].sample(n=3000)
negative_points = final[final['Score'] == 'negative'].sample(n=3000)

# Concatenating both of above
total_points = pd.concat([positive_points, negative_points])
%%time
# Initializing vectorizer for bigram
count_vect = CountVectorizer(ngram_range=(1,1))

# Initializing standard scaler
std_scaler = StandardScaler(with_mean=False)

# Creating count vectors and converting into dense representation
sample_points = total_points['CleanedText']
sample_points = count_vect.fit_transform(sample_points)
sample_points = std_scaler.fit_transform(sample_points)
sample_points = sample_points.todense()

# Storing class label in variable
labels = total_points['Score']

# Getting shape
print(sample_points.shape, labels.shape)
%%time
from sklearn.manifold import TSNE

tsne_data = sample_points
tsne_labels = labels

# Initializing with most explained variance
model = TSNE(n_components=2, random_state=15)

# Fitting model
tsne_data = model.fit_transform(tsne_data)

# Adding labels to the data point
tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with default parameters")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()

%%time
from sklearn.manifold import TSNE

tsne_data = sample_points
tsne_labels = labels

# Initializing with most explained variance
model = TSNE(n_components=2, random_state=15, perplexity=20, n_iter=2000)

# Fitting model
tsne_data = model.fit_transform(tsne_data)

# Adding labels to the data point
tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with perplexity: 20, n_iter:2000")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()

%%time

# Initializing tf-idf vectorizer for bigram
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))

tfidf_data = total_points['CleanedText']
tfidf_data = tfidf_vect.fit_transform(tfidf_data)
tfidf_data = tfidf_data.todense()

tfidf_labels = labels

print(tfidf_data.shape, tfidf_labels.shape)
%%time

model = TSNE(n_components=2, random_state=15)

# Fitting model
tsne_data = model.fit_transform(tfidf_data)


# Attaching feature and label
tsne_data = np.vstack((tsne_data.T, tfidf_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with default parameters")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()
%%time
from sklearn.manifold import TSNE

tsne_data = sample_points
tsne_labels = labels

# Initializing with most explained variance
model = TSNE(n_components=2, random_state=15, perplexity=20, n_iter=2000)

# Fitting model
tsne_data = model.fit_transform(tsne_data)

# Adding labels to the data point
tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with perplexity: 20, n_iter:2000")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()
# Getting text from Review
w2v_points = total_points['Text']
w2v_labels = labels.copy()
import re
def cleanhtml(sentence):
    cleantext = re.sub('<.*>', '', sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|#|@|.|,|)|(|\|/]', r'', sentence)
    return cleaned
# Creating list of sentences
sent_list = []
for sent in w2v_points:
    sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):
                sentence.append(cleaned_words.lower())
            else:
                continue
    sent_list.append(sentence)
print(sent_list[1])
# Initializing model for words occur atleast 5 times
w2v_model = gensim.models.Word2Vec(sent_list, min_count=5, size=50, workers=4)

# Applying model for word2vec
w2v_words = w2v_model[w2v_model.wv.vocab]
print("Number of words occur min 5 times: ", len(w2v_words))
print(w2v_words.shape)
# Getting 10 similar words
display(w2v_model.wv.most_similar("sweet"))
# Producing average word to vec vectors
import numpy as np
sent_vectors = []
for sent in sent_list:
    sent_vec = np.zeros(200)
    cnt_words = 0
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
sent_vectors = np.nan_to_num(sent_vectors)
print(sent_vectors.shape)
%%time

model = TSNE(n_components=2, random_state=15)

# Fitting model
w2v_points = model.fit_transform(sent_vectors)


# Attaching feature and label
tsne_data = np.vstack((w2v_points.T, w2v_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with default parameters")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()
%%time
from sklearn.manifold import TSNE

tsne_data = sample_points
tsne_labels = labels

# Initializing with most explained variance
model = TSNE(n_components=2, random_state=15, perplexity=20, n_iter=2000)

# Fitting model
tsne_data = model.fit_transform(tsne_data)

# Adding labels to the data point
tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with perplexity: 20, n_iter:2000")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()

%%time

tfidf_feat = tfidf_vect.get_feature_names()
tfidf_w2v_vectors = []
row = 0
for sent in sent_list:
    sent_vec = np.zeros(200)
    weight_sum = 0
    for word in sent:
        if word in w2v_words:
            vec = w2v_model.wv[word]
            tf_idf = final_tf_idf[row, tfidf_feat.index(word)]
            sent_vec += (vec*tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_w2v_vectors.append(sent_vec)
    row += 1

%%time

# Defining model for two features with most explained variance
model = TSNE(n_components=2, random_state=15)

# Fitting model
tfidf_w2v_points = model.fit_transform(tfidf_w2v_vectors)

# Attaching feature and label
tsne_data = np.vstack((tfidf_w2v_points.T, labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with default parameters")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()
%%time
from sklearn.manifold import TSNE

tsne_data = sample_points
tsne_labels = labels

# Initializing with most explained variance
model = TSNE(n_components=2, random_state=15, perplexity=20, n_iter=2000)

# Fitting model
tsne_data = model.fit_transform(tsne_data)

# Adding labels to the data point
tsne_data = np.vstack((tsne_data.T, tsne_labels)).T

# Creating data frame
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'label'))

# Plotting graph for class labels
sb.FacetGrid(tsne_df, hue='label', size=5).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title("TSNE with perplexity: 20, n_iter:2000")
plt.xlabel("Dim_1")
plt.ylabel("Dim_2")
plt.show()
