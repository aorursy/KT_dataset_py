'''
from apiclient.discovery import build
import pandas as pd

# Data to be stored
category = []
no_of_samples = 1700

# Gathering Data using the Youtube API
api_key = "AIzaSyAS9eTgOEnOJ2GlJbbqm_0bR1onuRQjTHE"
youtube_api = build('youtube','v3', developerKey = api_key)

# Travel Data
tvl_titles = []
tvl_descriptions = []
tvl_ids = []

req = youtube_api.search().list(q='travel vlogs', part='snippet', type='video', maxResults = 50)
res = req.execute()
while(len(tvl_titles)<no_of_samples):
    for i in range(len(res['items'])):
        tvl_titles.append(res['items'][i]['snippet']['title'])
        tvl_descriptions.append(res['items'][i]['snippet']['description'])
        tvl_ids.append(res['items'][i]['id']['videoId'])
        category.append('travel')
            
    if('nextPageToken' in res):
        next_page_token = res['nextPageToken']
        req = youtube_api.search().list(q='travelling', part='snippet', type='video', maxResults = 50, pageToken=next_page_token)
        res = req.execute()
    else:
        break


# Science Data
science_titles = []
science_descriptions = []
science_ids = []

next_page_token = None
req = youtube_api.search().list(q='robotics', part='snippet', type='video', maxResults = 50)
res = req.execute()
while(len(science_titles)<no_of_samples):
    if(next_page_token is not None):
        req = youtube_api.search().list(q='robotics', part='snippet', type='video', maxResults = 50, pageToken=next_page_token)
        res = req.execute()
    for i in range(len(res['items'])):
        science_titles.append(res['items'][i]['snippet']['title'])
        science_descriptions.append(res['items'][i]['snippet']['description'])
        science_ids.append(res['items'][i]['id']['videoId'])
        category.append('science and technology')
            
    if('nextPageToken' in res):
        next_page_token = res['nextPageToken']
    else:
        break
    
# Food Data
food_titles = []
food_descriptions = []
food_ids = []

next_page_token = None
req = youtube_api.search().list(q='delicious food', part='snippet', type='video', maxResults = 50)
res = req.execute()
while(len(food_titles)<no_of_samples):
    if(next_page_token is not None):
        req = youtube_api.search().list(q='delicious food', part='snippet', type='video', maxResults = 50, pageToken=next_page_token)
        res = req.execute()
    for i in range(len(res['items'])):
        food_titles.append(res['items'][i]['snippet']['title'])
        food_descriptions.append(res['items'][i]['snippet']['description'])
        food_ids.append(res['items'][i]['id']['videoId'])
        category.append('food')
            
    if('nextPageToken' in res):
        next_page_token = res['nextPageToken']
    else:
        break

# Food Data
manufacturing_titles = []
manufacturing_descriptions = []
manufacturing_ids = []

next_page_token = None
req = youtube_api.search().list(q='3d printing', part='snippet', type='video', maxResults = 50)
res = req.execute()
while(len(manufacturing_titles)<no_of_samples):
    if(next_page_token is not None):
        req = youtube_api.search().list(q='3d printing', part='snippet', type='video', maxResults = 50, pageToken=next_page_token)
        res = req.execute()
    for i in range(len(res['items'])):
        manufacturing_titles.append(res['items'][i]['snippet']['title'])
        manufacturing_descriptions.append(res['items'][i]['snippet']['description'])
        manufacturing_ids.append(res['items'][i]['id']['videoId'])
        category.append('manufacturing')
            
    if('nextPageToken' in res):
        next_page_token = res['nextPageToken']
    else:
        break
    
# History Data
history_titles = []
history_descriptions = []
history_ids = []

next_page_token = None
req = youtube_api.search().list(q='archaeology', part='snippet', type='video', maxResults = 50)
res = req.execute()
while(len(history_titles)<no_of_samples):
    if(next_page_token is not None):
        req = youtube_api.search().list(q='archaeology', part='snippet', type='video', maxResults = 50, pageToken=next_page_token)
        res = req.execute()
    for i in range(len(res['items'])):
        history_titles.append(res['items'][i]['snippet']['title'])
        history_descriptions.append(res['items'][i]['snippet']['description'])
        history_ids.append(res['items'][i]['id']['videoId'])
        category.append('history')
            
    if('nextPageToken' in res):
        next_page_token = res['nextPageToken']
    else:
        break
    
# Art and Music Data
art_titles = []
art_descriptions = []
art_ids = []

next_page_token = None
req = youtube_api.search().list(q='painting', part='snippet', type='video', maxResults = 50)
res = req.execute()
while(len(art_titles)<no_of_samples):
    if(next_page_token is not None):
        req = youtube_api.search().list(q='painting', part='snippet', type='video', maxResults = 50, pageToken=next_page_token)
        res = req.execute()
    for i in range(len(res['items'])):
        art_titles.append(res['items'][i]['snippet']['title'])
        art_descriptions.append(res['items'][i]['snippet']['description'])
        art_ids.append(res['items'][i]['id']['videoId'])
        category.append('art and music')
            
    if('nextPageToken' in res):
        next_page_token = res['nextPageToken']
    else:
        break
 
    
# Construct Dataset
final_titles = tvl_titles + science_titles + food_titles + manufacturing_titles + history_titles + art_titles
final_descriptions = tvl_descriptions + science_descriptions + food_descriptions + manufacturing_descriptions + history_descriptions + art_descriptions
final_ids = tvl_ids + science_ids + food_ids + manufacturing_ids + history_ids + art_ids
data = pd.DataFrame({'Video Id': final_ids, 'Title': final_titles, 'Description': final_descriptions, 'Category': category}) 
data.to_csv('Videos_data.csv')
'''
import pandas as pd
import nltk
#nltk.download()
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# Import Data
vdata = pd.read_csv('Videos_data.csv')
vdata = data.iloc[:, 1:]     # Remove extra un-named column
vdata.head(10)
# Missing Values
num_missing_desc = data.isnull().sum()[2]    # No. of values with msising descriptions
print('Number of missing values: ' + str(num_missing_desc))
vdata = data.dropna()

# Change to lowercase
vdata['Title'] = vdata['Title'].map(lambda x: x.lower())
vdata['Description'] = vdata['Description'].map(lambda x: x.lower())

# Remove numbers
vdata['Title'] = vdata['Title'].map(lambda x: re.sub(r'\d+', '', x))
vdata['Description'] = vdata['Description'].map(lambda x: re.sub(r'\d+', '', x))

# Remove Punctuation
vdata['Title']  = vdata['Title'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
vdata['Description']  = vdata['Description'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))

# Remove white spaces
vdata['Title'] = vdata['Title'].map(lambda x: x.strip())
vdata['Description'] = vdata['Description'].map(lambda x: x.strip())

# Tokenize into words
vdata['Title'] = vdata['Title'].map(lambda x: word_tokenize(x))
vdata['Description'] = vdata['Description'].map(lambda x: word_tokenize(x))
 
# Remove non alphabetic tokens
vdata['Title'] = vdata['Title'].map(lambda x: [word for word in x if word.isalpha()])
vdata['Description'] = vdata['Description'].map(lambda x: [word for word in x if word.isalpha()])

# filter out stop words
stop_words = set(stopwords.words('english'))
vdata['Title'] = vdata['Title'].map(lambda x: [w for w in x if not w in stop_words])
vdata['Description'] = vdata['Description'].map(lambda x: [w for w in x if not w in stop_words])

# Word Lemmatization
lem = WordNetLemmatizer()
vdata['Title'] = vdata['Title'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
vdata['Description'] = vdata['Description'].map(lambda x: [lem.lemmatize(word,"v") for word in x])

# Turn lists back to string
vdata['Title'] = vdata['Title'].map(lambda x: ' '.join(x))
vdata['Description'] = vdata['Description'].map(lambda x: ' '.join(x))
vdata.head(10)
# Encode classes
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(vdata.Category)
vdata.Category = le.transform(vdata.Category)
vdata.head(10)
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_title = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
tfidf_desc = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
labels = vdata.Category
features_title = tfidf_title.fit_transform(vdata.Title).toarray()
features_description = tfidf_desc.fit_transform(vdata.Description).toarray()
print('Title Features Shape: ' + str(features_title.shape))
print('Description Features Shape: ' + str(features_description.shape))
# Plotting class distribution
vdata['Category'].value_counts().sort_values(ascending=False).plot(kind='bar', y='Number of Samples', 
                                                                title='Number of samples for each class')
# Best 5 keywords for each class using Title Feaures
from sklearn.feature_selection import chi2
import numpy as np
N = 10
for current_class in list(le.classes_):
    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_title, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_title.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")
# Best 5 keywords for each class using Description Features
from sklearn.feature_selection import chi2
import numpy as np
N = 10
for current_class in list(le.classes_):
    current_class_id = le.transform([current_class])[0]
    features_chi2 = chi2(features_description, labels == current_class_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf_desc.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(current_class))
    print("Most correlated unigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(unigrams[-N:])))
    print("Most correlated bigrams:")
    print('-' *30)
    print('. {}'.format('\n. '.join(bigrams[-N:])))
    print("\n")
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(vdata.iloc[:, 1:3], vdata['Category'], random_state = 0)
X_train_title_features = tfidf_title.transform(X_train['Title']).toarray()
X_train_desc_features = tfidf_desc.transform(X_train['Description']).toarray()
features = np.concatenate([X_train_title_features, X_train_desc_features], axis=1)
X_train.head()
y_train.head()
# Naive Bayes
nb = MultinomialNB().fit(features, y_train)
# SVM
svm = linear_model.SGDClassifier(loss='modified_huber',max_iter=1000, tol=1e-3).fit(features,y_train)
# AdaBoost
adaboost = AdaBoostClassifier(n_estimators=40,algorithm="SAMME").fit(features,y_train)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 20000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50
# This is fixed.
EMBEDDING_DIM = 100

# Combining titles and descriptions into a single sentence
titles = vdata['Title'].values
descriptions = vdata['Description'].values
data_for_lstms = []
for i in range(len(titles)):
    temp_list = [titles[i], descriptions[i]]
    data_for_lstms.append(' '.join(temp_list))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data_for_lstms)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Convert the data to padded sequences
X = tokenizer.texts_to_sequences(data_for_lstms)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

# One-hot Encode labels
Y = pd.get_dummies(vdata['Category']).values
print('Shape of label tensor:', Y.shape)

# Splitting into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 42)
# Define LSTM Model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Training LSTM Model
epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)
import matplotlib.pyplot as plt
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();
X_train, X_test, y_train, y_test = train_test_split(vdata.iloc[:, 1:3], vdata['Category'], random_state = 0)
X_test_title_features = tfidf_title.transform(X_test['Title']).toarray()
X_test_desc_features = tfidf_desc.transform(X_test['Description']).toarray()
test_features = np.concatenate([X_test_title_features, X_test_desc_features], axis=1)
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import scikitplot as skplt

X_test_title_features = tfidf_title.transform(X_test['Title']).toarray()
X_test_desc_features = tfidf_desc.transform(X_test['Description']).toarray()
test_features = np.concatenate([X_test_title_features, X_test_desc_features], axis=1)

# Naive Bayes
y_pred = nb.predict(test_features)
y_probas = nb.predict_proba(test_features)

print(metrics.classification_report(y_test, y_pred, 
                                    target_names=list(le.classes_)))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
plt.title('Precision-Recall Curve - Naive Bayes')
plt.show()
# SVM
y_pred = svm.predict(test_features)
y_probas = svm.predict_proba(test_features)

print(metrics.classification_report(y_test, y_pred, 
                                    target_names=list(le.classes_)))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - SVM')
plt.show()

skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
plt.title('Precision-Recall Curve - SVM')
plt.show()
# Adaboost Classifier
y_pred = adaboost.predict(test_features)
y_probas = adaboost.predict_proba(test_features)

print(metrics.classification_report(y_test, y_pred, 
                                    target_names=list(le.classes_)))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Adaboost Classifier')
plt.show()

skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
plt.title('Precision-Recall Curve - Adaboost Classifier')
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 42)
y_probas = model.predict(X_test)
y_pred = np.argmax(y_probas, axis=1)
y_test = np.argmax(Y_test, axis=1)

print(metrics.classification_report(y_test, y_pred, 
                                    target_names=list(le.classes_)))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=list(le.classes_), yticklabels=list(le.classes_))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - LSTM')
plt.show()

skplt.metrics.plot_precision_recall_curve(y_test, y_probas)
plt.title('Precision-Recall Curve - LSTM')
plt.show()
#import data
adata = pd.read_csv('collected_sports_data.csv' )
adata.head(10)
#delete columns which are not required
adata.drop(adata.iloc[:, 4:42], inplace = True, axis = 1) 
adata.head(10)
# Change to lowercase
adata['Title 1'] = adata['Title 1'].map(lambda x: x.lower())
adata['Meta Description 1'] = adata['Meta Description 1'].map(lambda x: x.lower())
adata['H1-1'] = adata['H1-1'].map(lambda x: x.lower())

# Remove Punctuation
adata['Title 1'] = adata['Title 1'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
adata['Meta Description 1'] = adata['Meta Description 1'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
adata['H1-1'] = adata['H1-1'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))

# Remove white spaces
adata['Title 1'] = adata['Title 1'].map(lambda x: x.strip())
adata['Meta Description 1'] = adata['Meta Description 1'].map(lambda x: x.strip())
adata['H1-1'] = adata['H1-1'].map(lambda x: x.strip())

adata.head(10)
import pandas as pd
def find(dec,k):
    r=[]
    for i in dec.index:
        if k in dec['Meta Description 1'][i]:
            r.append(dec['Original Url'][i])
    return r


# Import Data
#adata = pd.read_csv('collected_sports_data.csv' )
adata=adata[['Original Url', 'Meta Description 1']]

#Search unigram keyword which is extracted from videos data.

result=find(adata, "travel") 
for i in result:
    print(" Url Link ",i)
