import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import textblob as tb
review_df = pd.read_csv('../input/yelp_review.csv')
business_df = pd.read_csv('../input/yelp_business.csv')
name_df = business_df[['business_id', 'name']]
review_df = pd.merge(review_df, name_df, how = 'left', left_on = 'business_id', right_on = 'business_id')
review_df.head()
snowball = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [snowball.stem(word) for word in tokenizer.tokenize(text.lower())]
def vectorize_reviews(reviews):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize, max_features = 1000)
    X = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    return X, words
def print_clusters(company_id, K = 8, num_words = 10):
    company_df = review_df[review_df['business_id'] == company_id]
    company_name = company_df['name'].unique()[0]
    reviews = company_df['text'].values
    X, words = vectorize_reviews(reviews)
    
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(X)
    
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-num_words-1:-1]
    print('Groups of ' + str(num_words) + ' words typically used together in reviews for ' + \
          company_name)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
#Tacos El Gordo in Downtown Las Vegas
bus_id = 'CiYLq33nAyghFkUR15pP-Q'
company_df = review_df[review_df['business_id'] == bus_id]
sns.countplot(x = company_df['stars'])
print_clusters(bus_id, K = 5, num_words = 12)
def vectorize_reviews2(reviews):
    vectorizer = TfidfVectorizer(stop_words = 'english', tokenizer = tokenize, \
                        min_df = 0.0025, max_df = 0.05, max_features = 1000, ngram_range = (1, 3))
    X = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names()
    return X, words
def print_clusters2(company_id, K = 8, num_words = 10):
    company_df = review_df[review_df['business_id'] == company_id]
    company_name = company_df['name'].unique()[0]
    reviews = company_df['text'].values
    X, words = vectorize_reviews2(reviews)
    
    kmeans = KMeans(n_clusters = K)
    kmeans.fit(X)
    
    common_words = kmeans.cluster_centers_.argsort()[:,-1:-num_words-1:-1]
    print('Groups of ' + str(num_words) + ' words typically used together in reviews for ' + \
          company_name)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
print_clusters2(bus_id, K = 3, num_words = 12)
def elbow_plot(X, k_start, k_end):
    
    distortions = []
    K = range(k_start, k_end + 1)
    for k in K:
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(K, distortions)
    plt.xticks(K)
    plt.title('Elbow curve')
    
reviews = company_df['text'].values
X, words = vectorize_reviews2(reviews)
elbow_plot(X, 1, 10)
agg = review_df.groupby('name').filter(lambda x: len(x) > 100)
agg = agg.groupby('name')['stars'].mean()
agg[agg == 3.0]
review_df[review_df['name'] == '"Ginseng Korean BBQ II"']
bus_id2 = 'EkuSy_kM8dpGrlb2pTxCBw'
company_df2 = review_df[review_df['business_id'] == bus_id2]
sns.countplot(x = company_df2['stars'])
print_clusters2(bus_id2, K = 3, num_words = 20)
reviews2 = company_df2['text'].values
X2, words2 = vectorize_reviews2(reviews2)
elbow_plot(X, 1, 10)
def calc_polarity(text):
    blob = tb.TextBlob(text)
    return blob.sentiment.polarity

def calc_subjectivity(text):
    blob = tb.TextBlob(text)
    return blob.sentiment.subjectivity
def get_pol_sub(company_id):
    company_df = review_df[review_df['business_id'] == company_id]
    company_name = company_df['name'].unique()[0]
    company_df['polarity'] = company_df['text'].apply(calc_polarity)
    company_df['subjectivity'] = company_df['text'].apply(calc_subjectivity)
    
    print('Company:' + company_name + '\nMean Polarity: ' + str(company_df['polarity'].mean())\
          + '\nMean Subjectivity: ' + str(company_df['subjectivity'].mean()))
get_pol_sub(bus_id)
get_pol_sub(bus_id2)
