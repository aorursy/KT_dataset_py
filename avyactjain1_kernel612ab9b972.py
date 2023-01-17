import numpy as np

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import *

stemmer = PorterStemmer()



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('../input/nlp-getting-started/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.model_selection import train_test_split as tts

from sklearn import decomposition, ensemble

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.utils import to_categorical

import nltk
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')
print('Training data shape (rows, cols): ', df_train.shape)

df_train.head()
# keyword and location columns have some nulls

df_train.info()
print('Test data shape (rows, cols): ', df_test.shape)

df_test.head()
# keyword and location columns have some nulls

df_test.info()
# Null check

df_train['keyword'].isnull().value_counts() / df_train.shape[0]
df_train['location'].isnull().value_counts() / df_train.shape[0]
df_train['text'].isnull().value_counts() / df_train.shape[0]
df_test['keyword'].isnull().value_counts() / df_test.shape[0]
df_test['location'].isnull().value_counts() / df_test.shape[0]
df_test['text'].isnull().value_counts() / df_test.shape[0]
# Target Distribution (0 or 1)

dist_class = df_train['target'].value_counts()

labels = ['Non-disaster tweet', 'Disaster tweet']



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



sns.barplot(x=dist_class.index, y=dist_class, ax=ax1).set_title("Target Count")



ax2.pie(dist_class,

        labels=labels,

        counterclock=False,

        startangle=90,

        autopct='%1.1f%%',

        pctdistance=0.7)

plt.title("Target Frequency Proportion")

plt.show
disaster_tweet_length = df_train[df_train['target']==1]['text'].str.len()

nondisaster_tweet_length = df_train[df_train['target']==0]['text'].str.len()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



ax1.hist(disaster_tweet_length, color='red')

ax1.set_title("Disaster Tweets")



ax2.hist(nondisaster_tweet_length, color='green')

ax2.set_title("Non-Disaster Tweets")



fig.suptitle("Characters in tweets")

plt.show()
disaster_tweet_words = df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))

nondisaster_tweet_words = df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



ax1.hist(disaster_tweet_words, color='red')

ax1.set_title("Disaster Tweets")



ax2.hist(nondisaster_tweet_words, color='green')

ax2.set_title("Non-Disaster Tweets")



fig.suptitle("Words in tweets")

plt.show()
df_train_keyword = pd.DataFrame({

    'keyword': df_train['keyword'].value_counts().index,

    'count': df_train['keyword'].value_counts().values

})



df_train_location = pd.DataFrame({

    'location': df_train['location'].value_counts().index,

    'count': df_train['location'].value_counts().values

})





print('Number fo unique keywords in training data: ', df_train_keyword.shape[0])



px.bar(

    df_train_keyword,

    x='keyword',

    y='count',

    title="Each unique keyword count in training data"

).show()



px.bar(

    df_train_location,

    x=df_train_location['location'][:20],

    y=df_train_location['count'][:20],

    title="Top 20 location countin training data"

).show()
df_train[df_train['target'] == 1]['keyword'].value_counts()
df_train[df_train['target'] == 0]['keyword'].value_counts()
df_train[df_train['target'] == 1]['location'].value_counts()
df_train[df_train['target'] == 0]['location'].value_counts()
df_test_keyword = pd.DataFrame({

    'keyword': df_test['keyword'].value_counts().index,

    'count': df_test['keyword'].value_counts().values

})



df_test_location = pd.DataFrame({

    'location': df_test['location'].value_counts().index,

    'count': df_test['location'].value_counts().values

})



print('Number fo unique keywords in test data: ', df_test_keyword.shape[0])



px.bar(

    df_test_keyword,

    x='keyword',

    y='count',

    title="Each unique keyword count in test data"

).show()



px.bar(

    df_test_location,

    x=df_test_location['location'][:20],

    y=df_test_location['count'][:20],

    title="Top 20 location count in test data"

).show()
disaster_tweet = dict(df_train[df_train['target']==1]['keyword'].value_counts())



stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white").generate_from_frequencies(disaster_tweet)



plt.figure(figsize=[10,6])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
nondisaster_tweet = dict(df_train[df_train['target']==0]['keyword'].value_counts())



wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white").generate_from_frequencies(nondisaster_tweet)



plt.figure(figsize=[10,6])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
test_tweet = dict(df_test['keyword'].value_counts())



wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color="white").generate_from_frequencies(test_tweet)



plt.figure(figsize=[10,6])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



import string

def remove_punc(text):

    table = str.maketrans('','',string.punctuation)

    return text.translate(table)
# https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt

slang_abbrev_dict = {

    'AFAIK': 'As Far As I Know',

    'AFK': 'Away From Keyboard',

    'ASAP': 'As Soon As Possible',

    'ATK': 'At The Keyboard',

    'ATM': 'At The Moment',

    'A3': 'Anytime, Anywhere, Anyplace',

    'BAK': 'Back At Keyboard',

    'BBL': 'Be Back Later',

    'BBS': 'Be Back Soon',

    'BFN': 'Bye For Now',

    'B4N': 'Bye For Now',

    'BRB': 'Be Right Back',

    'BRT': 'Be Right There',

    'BTW': 'By The Way',

    'B4': 'Before',

    'B4N': 'Bye For Now',

    'CU': 'See You',

    'CUL8R': 'See You Later',

    'CYA': 'See You',

    'FAQ': 'Frequently Asked Questions',

    'FC': 'Fingers Crossed',

    'FWIW': 'For What It\'s Worth',

    'FYI': 'For Your Information',

    'GAL': 'Get A Life',

    'GG': 'Good Game',

    'GN': 'Good Night',

    'GMTA': 'Great Minds Think Alike',

    'GR8': 'Great!',

    'G9': 'Genius',

    'IC': 'I See',

    'ICQ': 'I Seek you',

    'ILU': 'I Love You',

    'IMHO': 'In My Humble Opinion',

    'IMO': 'In My Opinion',

    'IOW': 'In Other Words',

    'IRL': 'In Real Life',

    'KISS': 'Keep It Simple, Stupid',

    'LDR': 'Long Distance Relationship',

    'LMAO': 'Laugh My Ass Off',

    'LOL': 'Laughing Out Loud',

    'LTNS': 'Long Time No See',

    'L8R': 'Later',

    'MTE': 'My Thoughts Exactly',

    'M8': 'Mate',

    'NRN': 'No Reply Necessary',

    'OIC': 'Oh I See',

    'OMG': 'Oh My God',

    'PITA': 'Pain In The Ass',

    'PRT': 'Party',

    'PRW': 'Parents Are Watching',

    'QPSA?': 'Que Pasa?',

    'ROFL': 'Rolling On The Floor Laughing',

    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',

    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',

    'SK8': 'Skate',

    'STATS': 'Your sex and age',

    'ASL': 'Age, Sex, Location',

    'THX': 'Thank You',

    'TTFN': 'Ta-Ta For Now!',

    'TTYL': 'Talk To You Later',

    'U': 'You',

    'U2': 'You Too',

    'U4E': 'Yours For Ever',

    'WB': 'Welcome Back',

    'WTF': 'What The Fuck',

    'WTG': 'Way To Go!',

    'WUF': 'Where Are You From?',

    'W8': 'Wait',

    '7K': 'Sick:-D Laugher'

}



def unslang(text):

    if text.upper() in slang_abbrev_dict.keys():

        return slang_abbrev_dict[text.upper()]

    else:

        return text
def tokenization(text):

    text = re.split('\W+', text)

    return text



stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):

    text = [word for word in text if word not in stopword]

    return text



def stemming(text):

    text = [stemmer.stem(word) for word in text]

    return text

for datas in [df_train,df_test]:

    cleaned_text_list = []

    cleaned_keyword_list = []

    cleaned_location_list = []

    

    for index,row in datas.iterrows():

        if(row["text"]):

            cleaned_text = remove_url(row["text"])

            cleaned_text = remove_html(cleaned_text)

            cleaned_text = remove_emoji(cleaned_text)

            cleaned_text = unslang(cleaned_text)

            cleaned_text = remove_punc(cleaned_text)

            cleaned_text = tokenization(cleaned_text.lower())

            cleaned_text = remove_stopwords(cleaned_text)

            cleaned_text = stemming(cleaned_text)

            cleaned_text = " ".join(cleaned_text)

            

            cleaned_text_list.append(cleaned_text)



            

        

    datas["cleaned_text"] = cleaned_text_list
df_train.head(10)
df_train['text'][100]
df_train['cleaned_text'][100]
df_test.head(10)
df_test['text'][100]
df_test['cleaned_text'][100]
df_train = pd.get_dummies(df_train, columns=["keyword"], drop_first=True)
df_test  = pd.get_dummies(df_test, columns=["keyword"], drop_first=True)
df_train.drop(columns=["location", "text"], axis = 1, inplace = True)
df_train.head(10)
df_test.head()
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=64)

vectorizer.fit(df_train['cleaned_text'])
X = df_train.drop(columns=["target"])

y = df_train["target"].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)
text_data_train = vectorizer.transform(X_train['cleaned_text']).todense()

text_data_valid = vectorizer.transform(X_valid['cleaned_text']).todense()



keyword_data_train = (X_train.drop(["cleaned_text", "id"],axis = 1).values)

keyword_data_valid = (X_valid.drop(["cleaned_text", "id"],axis = 1).values)



text_data_train = np.asarray(text_data_train)

text_data_valid = np.asanyarray(text_data_valid)

keyword_data_train = np.asarray(keyword_data_train)

keyword_data_valid = np.asarray(keyword_data_valid)
num_keywords = 220  # Number of unique issue tags

num_words = 64  # Size of vocabulary specified when preprocessing text data

num_classes = 1 



body_input = keras.Input(shape=(None,), name='text')  # Variable-length sequence of ints

keywords_input = keras.Input(shape=(num_keywords,), name='keywords')  # Binary vectors of size `num_keywords`



# Embed each word in the text into a 64-dimensional vector

body_features = layers.Embedding(num_words + 1, 64)(body_input)



# Reduce sequence of embedded words in the body into a single 32-dimensional vector

body_features = layers.LSTM(32)(body_features)



# Merge all available features into a single large vector via concatenation

x = layers.concatenate([body_features, keywords_input])

x = layers.Dropout(0.2)(x)

x = layers.Dense(16, activation="relu", name = "hiddenlayer")(x)



# Stick a logistic regression for priority prediction on top of the features

# Stick a department classifier on top of the features

class_pred = layers.Dense(num_classes, activation='sigmoid', name='classes')(x)



# Instantiate an end-to-end model predicting both priority and department

model = keras.Model(inputs=[body_input, keywords_input],

                    outputs=[class_pred])
keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
model.compile(optimizer=keras.optimizers.Adam(),

              loss={

                    'classes': 'binary_crossentropy'},)
model.fit({'text': text_data_train, 'keywords':keyword_data_train },

          {'classes':(y_train)},

          epochs=20,

          batch_size=128)
predictions  = model.predict({'text': text_data_valid, 'keywords': keyword_data_valid.astype(np.float32)})
prediction_list = []

for prediction in predictions:

    if prediction >= 0.5:

        prediction_list.append(1)

    else:

        prediction_list.append(0)
print(metrics.accuracy_score(prediction_list, y_valid))
text_data_test = vectorizer.transform(df_test['cleaned_text']).todense()

keyword_data_test = (df_test.drop(["cleaned_text", "id", "location","text"],axis = 1).values)
y_test_pred  = model.predict({'text': text_data_test, 'keywords': keyword_data_test.astype(np.float32)})
y_test_pred_list = []

for prediction in y_test_pred:

    if prediction >= 0.5:

        y_test_pred_list.append(1)

    else:

        y_test_pred_list.append(0)
submission_file = pd.DataFrame({'id': df_test['id'], 'target': y_test_pred_list})
submission_file.to_csv("submission_avyactJain.csv")