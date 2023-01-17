!pip install chart_studio
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import re 
import string
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, plot, iplot

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
from os import path
import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')
data_dict = {}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_dict[filename] = os.path.join(dirname,filename)
        print(os.path.join(dirname, filename))
print(data_dict)
# Load in training data
train = pd.read_csv(data_dict['Corona_NLP_train.csv'], encoding = 'latin1')
# Copy training data
df = train.copy()
df.head()
# Load in test data
test_df = pd.read_csv(data_dict['Corona_NLP_test.csv'], encoding = 'latin1')
test_df.head()
# Check for nulls
df.info()
# Replace na with 'None'
df['Location'].fillna('None', inplace = True)
df.head()
# Join stopwords together and set them for use in cleaning function.
", ".join(stopwords.words('english'))
stops = set(stopwords.words('english'))

# Function that cleans tweets for classification. 
def clean_tweet(tweet):
    # Remove hyperlinks.
    tweet= re.sub(r'https?://\S+|www\.\S+','',tweet)
    # Remove html
    tweet = re.sub(r'<.*?>','',tweet)
    # Remove numbers (Do we want to remove numbers? Death toll?)
    tweet = re.sub(r'\d+','',tweet)
    # Remove mentions
    tweet = re.sub(r'@\w+','',tweet)
    # Remove punctuation
    tweet = re.sub(r'[^\w\s\d]','',tweet)
    # Remove whitespace
    tweet = re.sub(r'\s+',' ',tweet).strip()
    # Remove stopwords
    tweet = " ".join([word for word in str(tweet).split() if word not in stops])
    
    return tweet.lower()
# Check function
example2 = df['OriginalTweet'][1]
clean_tweet(example2)
# Apply text cleaning function to training and test dataframes.
df['newTweet'] = df['OriginalTweet'].apply(lambda x: clean_tweet(x))
test_df['newTweet'] = test_df['OriginalTweet'].apply(lambda x: clean_tweet(x))
df.head()
def token_stem(tweet):
    tk = TweetTokenizer()
    stemmer = PorterStemmer()
    tweet = tk.tokenize(tweet)
    tweet = [stemmer.stem(word) for word in tweet]
    tweet =  tweet = " ".join([word for word in tweet])
    return tweet
def token_lemma(tweet):
    tk = TweetTokenizer()
    lemma = WordNetLemmatizer()
    tweet = tk.tokenize(tweet)
    tweet = [lemma.lemmatize(word) for word in tweet]
    tweet = " ".join([word for word in tweet])
    return tweet
tweet = df['newTweet'][1]
tweet
print(token_stem(tweet))
print('\n')
print(token_lemma(tweet))
df['stemTweet'] = df['newTweet'].apply(lambda x: token_stem(x))
df['lemmaTweet'] = df['newTweet'].apply(lambda x: token_lemma(x))
df.head()
# Create more useful labels for classification.
# We will take the original 5 possibilites and
# reduce them to 3, removing the "extremelys".
def make_label(sentiment):
    
    label = ''
    if 'Positive' in sentiment: 
        label = 1
    if 'Negative' in sentiment:
        label = -1
    if 'Neutral' in sentiment:
        label = 0
    return label
# Apply make_label funtion to training and test dataframes.
df['label'] = df['Sentiment'].apply(lambda x: make_label(x))
test_df['label'] = test_df['Sentiment'].apply(lambda x: make_label(x))
df.head()
# Some frequent US locations
us_filters = ('New York', 'New York, NY', 'NYC', 'NY', 'Washington, DC', 'Los Angeles, CA',
             'Seattle, Washington', 'Chicago', 'Chicago, IL', 'California, USA', 'Atlanta, GA',
             'San Francisco, CA', 'Boston, MA', 'New York, USA', 'Texas, USA', 'Austin, TX',
              'Houston, TX', 'New York City', 'Philadelphia, PA', 'Florida, USA', 'Seattle, WA',
             'Washington, D.C.', 'San Diego, CA', 'Las Vegas, NV', 'Dallas, TX', 'Denver, CO',
             'New Jersey, USA', 'Brooklyn, NY', 'California', 'Michigan, USA', 'Minneapolis, MN',
             'Virginia, USA', 'Miami, FL', 'Texas', 'Los Angeles', 'United States', 'San Francisco',
             'Indianapolis, IN', 'Pennsylvania, USA', 'Phoenix, AZ', 'New Jersey', 'Baltimore, MD',
             'CA', 'FL', 'DC', 'TX', 'IL', 'MA', 'PA', 'GA', 'NC', 'NJ', 'WA', 'VA', 'PAK', 'MI', 'OH',
             'CO', 'AZ', 'D.C.', 'WI', 'MD', 'MO', 'TN', 'Florida', 'IN', 'NV', 'MN', 'OR','LA', 'Michigan',
             'CT', 'SC', 'OK', 'Illinois', 'Ohio', 'UT', 'KY', 'Arizona', 'Colorado')

# Various nation's frequent locations
uk_filters = ('England', 'London', 'london', 'United Kingdom', 'united kingdom',
              'England, United Kingdom', 'London, UK', 'London, England',
              'Manchester, England', 'Scotland, UK', 'Scotland', 'Scotland, United Kingdom',
              'Birmingham, England', 'UK', 'Wales')
india_filters = ('New Delhi, India', 'Mumbai', 'Mumbai, India', 'New Delhi', 'India', 
                 'Bengaluru, India')
australia_filters = ('Sydney, Australia', 'New South Wales', 'Melbourne, Australia', 'Sydney',
                     'Sydney, New South Wales', 'Melbourne, Victoria', 'Melbourne', 'Australia')
canada_filters = ('Toronto, Ontario', 'Toronto', 'Ontario, Canada', 'Toronto, Canada', 'Canada',
                  'Vancouver, British Columbia', 'Ontario', 'Victoria', 'British Columbia', 'Alberta',)
south_africa_filters = ('Johannesburg, South Africa', 'Cape Town, South Africa', 'South Africa')
nigeria_filters = ('Lagos, Nigeria')
kenya_filters = ('Nairobi, Kenya')
france_filters = ('Paris, France')
ireland_filters = ('Ireland')
new_zealand_filters = ('New Zealand')
pakistan_filters = ('Pakistan')
malaysia_filters = ('Malaysia')
uganda_filters = ('Kampala, Uganda', 'Uganda')
singapore_filters = ('Singapore')
germany_filters = ('Germany', 'Deutschland')
switz_filters = ('Switzerland')
uae_filters = ('United Arab Emirates', 'Dubai')
spain_filters = ('Spain')
belg_filters = ('Belgium')
phil_filters = ('Philippines')
hk_filters = ('Hong Kong')
ghana_filters = ('Ghana')
# These all have large counts. Need to be removed from rest of data
other_filters = ('None', 'Worldwide', 'Global', 'Earth', '??')
df['country'] = df['Location'].apply(lambda x: x.split(",")[-1].strip() if ("," in x) else x)
df.head()
# Changing strings found with filters into 3 digit codes
df['country'] = df['country'].apply(lambda x: 'USA' if x in us_filters else x)
df['country'] = df['country'].apply(lambda x: 'GBR' if x in uk_filters else x)
df['country'] = df['country'].apply(lambda x: 'IND' if x in india_filters else x)
df['country'] = df['country'].apply(lambda x: 'AUS' if x in australia_filters else x)
df['country'] = df['country'].apply(lambda x: 'CAN' if x in canada_filters else x)
df['country'] = df['country'].apply(lambda x: 'ZAF' if x in south_africa_filters else x)
df['country'] = df['country'].apply(lambda x: 'KEN' if x in kenya_filters else x)
df['country'] = df['country'].apply(lambda x: 'NGA' if x in nigeria_filters else x)
df['country'] = df['country'].apply(lambda x: 'SGP' if x in singapore_filters else x)
df['country'] = df['country'].apply(lambda x: 'FRA' if x in france_filters else x)
df['country'] = df['country'].apply(lambda x: 'NZL' if x in new_zealand_filters else x)
df['country'] = df['country'].apply(lambda x: 'PAK' if x in pakistan_filters else x)
df['country'] = df['country'].apply(lambda x: 'MYS' if x in malaysia_filters else x)
df['country'] = df['country'].apply(lambda x: 'IRL' if x in ireland_filters else x)
df['country'] = df['country'].apply(lambda x: 'UGA' if x in uganda_filters else x)
df['country'] = df['country'].apply(lambda x: 'DEU' if x in germany_filters else x)
df['country'] = df['country'].apply(lambda x: 'CHE' if x in switz_filters else x)
df['country'] = df['country'].apply(lambda x: 'ARE' if x in uae_filters else x)
df['country'] = df['country'].apply(lambda x: 'ESP' if x in spain_filters else x)
df['country'] = df['country'].apply(lambda x: 'BEL' if x in belg_filters else x)
df['country'] = df['country'].apply(lambda x: 'PHL' if x in phil_filters else x)
df['country'] = df['country'].apply(lambda x: 'GHA' if x in ghana_filters else x)
df['country'] = df['country'].apply(lambda x: 'HKG' if x in hk_filters else x)
df['country'] = df['country'].apply(lambda x: 'None' if x in other_filters else x)
df['country'].value_counts()
# 0:30 because that's where the labeled countries end
places_df = pd.DataFrame(df['country'].value_counts()[0:30])
places_df.reset_index(inplace = True)
places_df.rename(columns = {'index':'Country', 'country':'Tweets'}, inplace = True)
# Remove 'None' location
places_df = places_df[places_df['Country'] != 'None']
data = dict(type='choropleth',
            colorscale = 'inferno',
            locations = places_df['Country'],
            z = places_df['Tweets'],
            #locationmode = 'USA-states',
            text = places_df['Tweets'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Number of Tweets"}
            ) 

layout = dict(title = 'Number of Tweets By Country',
              geo = dict(#scope='usa',
                         showlakes = False,
                         lakecolor = 'rgb(85,173,240)',
                         projection_type='equirectangular')
             )

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
# image courtesy of https://tse2.mm.bing.net/th?id=OIP.VLv_PpEOc8TDwuTNvj5hWQHaHa&pid=Api
#img = Image.open(data_dict['rona4.jpeg'])
mask = np.array(Image.open(data_dict['rona4.jpeg']))

# Positive WordCloud
pos_df = df[df['label'] == 1]
pos_text = pos_df['newTweet'].to_string(index = False)
pos_text = re.sub(r'\n','',pos_text)
pos_cloud = WordCloud(colormap = 'Greens', mask = mask).generate(pos_text)

# Neutral WordCloud
neut_df = df[df['label'] == 0]
neut_text = neut_df['newTweet'].to_string(index = False)
neut_text = re.sub(r'\n','', neut_text)
neut_cloud = WordCloud(colormap = 'Blues', mask = mask).generate(neut_text)

# Negative wordcloud
neg_df = df[df['label'] == -1]
neg_text = neg_df['newTweet'].to_string(index = False)
neg_text = re.sub(r'\n','', neg_text)
neg_cloud = WordCloud(colormap = 'Reds', mask = mask).generate(neg_text)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = [30,20])
ax1.imshow(pos_cloud)
ax1.set_title('Positive Cloud', fontsize = 30)
ax1.axis('off')
ax2.imshow(neut_cloud)
ax2.set_title('Neutral Cloud', fontsize = 30)
ax2.axis('off')
ax3.imshow(neg_cloud)
ax3.set_title('Negative Cloud', fontsize = 30)
ax3.axis('off')
def ngram_df(corpus,nrange,n=None):
    vec = CountVectorizer(stop_words = 'english',ngram_range=nrange).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df
unigram_df = ngram_df(df['newTweet'],(1,1),20)
bigram_df = ngram_df(df['newTweet'],(2,2),20)
trigram_df = ngram_df(df['newTweet'],(3,3),20)
unigram_df['text'][::-1]
sns.set(font_scale = 1.3)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(data = unigram_df, y = 'text', x = 'count')
sns.set(font_scale = 1.3)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(data = bigram_df, y = 'text', x = 'count')
sns.set(font_scale = 1.3)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(data = trigram_df, y = 'text', x = 'count')
# Set X and y.
X = df['newTweet']
y = df['label']

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = dict({'SVC': LinearSVC(max_iter = 5000),
            'Logisitc': LogisticRegression(max_iter = 5000),
            'NaiveBayes': MultinomialNB(),
            'RandomForest': RandomForestClassifier(),
           })
def make_models(clf, vectorizer, X_train, y_train, cv = 5):
    
    acc_df = pd.DataFrame(index=range(cv * len(clf)))
    results = []
    for classifier in clf.keys():
        model = Pipeline([('vectorizer',vectorizer),
                   ('clf', clf[classifier])])
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train , y_train, cv = cv)
        model_name = classifier
        for fold, score in enumerate(scores):
            results.append((model_name, fold, score))
    
    acc_df = pd.DataFrame(results, columns=['model_name', 'fold', 'accuracy'])
    
    return acc_df
# Number of folds for K-fold cross validation
cv = 10
tfidf_df = make_models(clf, TfidfVectorizer(), X_train, y_train, cv)
count_df = make_models(clf, CountVectorizer(), X_train, y_train, cv)
tfidf_df['vectorizer'] = 'tfidf'
count_df['vectorizer'] = 'count'
combined_df = tfidf_df.append(count_df)
combined_df['method'] = 'none'
combined_df.head(10)
# Set X and y.
X = df['stemTweet']
y = df['label']

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

stem_tfidf_df = make_models(clf, TfidfVectorizer(), X_train, y_train, cv)
stem_count_df = make_models(clf, CountVectorizer(), X_train, y_train, cv)

stem_tfidf_df['method'] = 'stem'
stem_tfidf_df['vectorizer'] = 'tfidf'
stem_count_df['method'] = 'stem'
stem_count_df['vectorizer'] = 'count'
stem_df = stem_tfidf_df.append(stem_count_df)
# Set X and y.
X = df['lemmaTweet']
y = df['label']

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lemma_tfidf_df = make_models(clf, TfidfVectorizer(), X_train, y_train, cv)
lemma_count_df = make_models(clf, CountVectorizer(), X_train, y_train, cv)

lemma_tfidf_df['vectorizer'] = 'tfidf'
lemma_tfidf_df['method'] = 'lemma'
lemma_count_df['vectorizer'] = 'count'
lemma_count_df['method'] = 'lemma'
lemma_df = lemma_tfidf_df.append(lemma_count_df)
all_df = lemma_df.append(stem_df)
all_df = all_df.append(combined_df)
# Skip to here to avoid running the models
all_df = pd.read_csv(data_dict['pipe_results.csv'])
sns.set(font_scale = 1.4)
sns.catplot(x = 'model_name', y = 'accuracy', hue = 'method', height = 7,
            data = all_df, kind = 'box', col = 'vectorizer', palette = 'rainbow')
no_nb = all_df[all_df['model_name'] != 'NaiveBayes']
no_nb_rf = no_nb[no_nb['model_name'] != 'RandomForest']
sns.set(font_scale = 1.4)
sns.catplot(x = 'model_name', y = 'accuracy', hue = 'method', height = 7,
            data = no_nb_rf, kind = 'box', col = 'vectorizer', palette = 'rainbow')
#all_df = pd.read_csv('pipe_results.csv')
accuracies = all_df.groupby(['model_name', 'method', 'vectorizer']).accuracy.mean()
stdDev = all_df.groupby(['model_name', 'method', 'vectorizer']).accuracy.std()
metrics_df = pd.concat([accuracies, stdDev], axis = 1, ignore_index = True)
metrics_df.columns = ['mean_acc', 'mean_std']
metrics_df.sort_values(by = ['mean_acc','method'], ascending = False).head(10)
# Set X and y.
X = df['newTweet']
y = df['label']

# Set vectorizer for feature extraction.
vectorizer = TfidfVectorizer()

# Split data into training and test sets to fit the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define model for predictions
model = Pipeline([('vectorizer',vectorizer),
                  ('clf', LinearSVC(max_iter = 5000))])

model.fit(X_train, y_train)
train_preds = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, train_preds))
print('\n')
print(classification_report(y_test, train_preds))
# Set X and y.
X2 = test_df['newTweet']
y2 = test_df['label']


test_preds = model.predict(X2)
print('Accuracy:', accuracy_score(y2, test_preds))
print('\n')
print(classification_report(y2, test_preds))
# Dictionary of parameters that can be tuned
model.get_params()
# GridSearchCV goes through specified parameter values and finds the best ones. 
from sklearn.model_selection import GridSearchCV

# We'll try a few different options here.
hyperparameters = { 'vectorizer__max_df': [1, 0.9, 0.95, .85],
                    'vectorizer__ngram_range': [(1,1), (1,2), (2,2),(2,3)],
                  }
model_tune = GridSearchCV(model, hyperparameters, cv=5)

# Fit and tune model
model_tune.fit(X_train, y_train)
# These are the best parameters according to the GridSearch
model_tune.best_params_
# Gridsearch will refit the model on the best settings
model_tune.refit
preds = model_tune.predict(X2)
print('Accuracy:', accuracy_score(y2, preds))
print('\n')
print(classification_report(y2, preds))
test_df['pred_label'] = test_preds
test_df.head()
mislabel_df = test_df[test_df['label'] != test_preds]
mislabel_df.head()
mislabel_df['Location'].fillna('None', inplace = True)
mislabel_df['country'] = mislabel_df['Location'].apply(lambda x: x.split(",")[-1].strip() if ("," in x) else x)

# Changing strings found with filters into 3 digit codes
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'USA' if x in us_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'GBR' if x in uk_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'IND' if x in india_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'AUS' if x in australia_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'CAN' if x in canada_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'ZAF' if x in south_africa_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'KEN' if x in kenya_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'NGA' if x in nigeria_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'SGP' if x in singapore_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'FRA' if x in france_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'NZL' if x in new_zealand_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'PAK' if x in pakistan_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'MYS' if x in malaysia_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'IRL' if x in ireland_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'UGA' if x in uganda_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'DEU' if x in germany_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'CHE' if x in switz_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'ARE' if x in uae_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'ESP' if x in spain_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'BEL' if x in belg_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'PHL' if x in phil_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'GHA' if x in ghana_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'HKG' if x in hk_filters else x)
mislabel_df['country'] = mislabel_df['country'].apply(lambda x: 'None' if x in other_filters else x)
mislabel_df['country'].value_counts()
mislabel_df.head()
mislabel_df.tail()
# Neutral tweet
print(mislabel_df['OriginalTweet'][7])
print('\n')
print(mislabel_df['newTweet'][7])
# Negative tweet
print(mislabel_df['OriginalTweet'][15])
print('\n')
print(mislabel_df['newTweet'][15])
# Positive tweet
print(mislabel_df['OriginalTweet'][3779])
print('\n')
print(mislabel_df['newTweet'][3779])
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer

from torch.utils.data import TensorDataset

import transformers
from transformers import BertForSequenceClassification

#import numpy as np
#import pandas as pd
#import re
from sklearn.preprocessing import LabelEncoder

# Encode the classes for BERT. We'll keep using the 3 labels we made earlier.  
encoder = LabelEncoder()
df['encoded_sentiment'] = encoder.fit_transform(df['label'])
# Set X and y.
X = df['newTweet']
y = df['encoded_sentiment']

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
# Encoding the words in the training data into vectors.
encoded_data_train = tokenizer.batch_encode_plus(
    X_train, 
    truncation = True,
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=50, 
    return_tensors='pt'
)

# Encoding the words in the test data into vectors.
encoded_data_test = tokenizer.batch_encode_plus(
    X_test, 
    truncation = True,
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=50, 
    return_tensors='pt'
)
# Get inputs and attention masks from previously encoded data. 
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(y_test.values)

# Instantiate TensorDataset
dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
# Initialize the model. 
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
# DataLoaders for running the model
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=128)

dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=128)
# Setting hyperparameters
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
                  
epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda')
model.to(device)

for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels':         batch[2].to(device),
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')

def evaluate(dataloader_test):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_test:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_test) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
val_loss, predictions, true_vals = evaluate(dataloader_test)
val_f1 = f1_score_func(predictions, true_vals)
print('Val Loss = ', val_loss)
print('Val F1 = ', val_f1)
encoded_classes = encoder.classes_
predicted_category = [encoded_classes[np.argmax(x)] for x in predictions]
true_category = [encoded_classes[x] for x in true_vals]
x = 0
for i in range(len(true_category)):
    if true_category[i] == predicted_category[i]:
        x += 1
        
print('Accuracy Score = ', x / len(true_category))
print(classification_report(true_category, predicted_category))
test_df['encoded_sentiment'] = encoder.fit_transform(test_df['label'])

# Set X and y.
X = test_df['newTweet']
y = test_df['encoded_sentiment']

encoded_data_test = tokenizer.batch_encode_plus(
    X, 
    truncation = True,
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=50, 
    return_tensors='pt'
)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(y.values)

# Pytorch TensorDataset Instance
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=128)
val_loss, predictions, true_vals = evaluate(dataloader_test)
val_f1 = f1_score_func(predictions, true_vals)
encoded_classes = encoder.classes_
predicted_category = [encoded_classes[np.argmax(x)] for x in predictions]
true_category = [encoded_classes[x] for x in true_vals]

x = 0
for i in range(len(true_category)):
    if true_category[i] == predicted_category[i]:
        x += 1
        
print('Accuracy Score = ', x / len(true_category))
print('\n')
print(classification_report(true_category, predicted_category))
