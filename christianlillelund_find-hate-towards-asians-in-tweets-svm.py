import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split

import re



nrows = 1000000

df = pd.read_csv('/kaggle/input/Xenophobia.csv', nrows=nrows, encoding='latin1')



# Drop columns not used for modelling

cols_to_drop = ['status_id', 'created_at', 'location']

df.drop(cols_to_drop, axis=1, inplace=True)

            

# Convert text to string type

df['text'] = df['text'].astype(str)



print("Total number of samples:", len(df))



df.head()
# Print a random tweet as a sample

sample_index = 25

print(df.iloc[sample_index])
# Helper function to remove unwanted patterns

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt



# Remove Twitter handles from the data 

df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")



# Remove punctuations, numbers, and special characters

df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")



# Remove all words below 3 characters

df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# Tokenize the tweets

tokenized_tweet = df['text'].apply(lambda x: x.split())
# Stem the tweets

from nltk.stem.porter import *

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
# Put tokenized tweets back in dataframe

for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['text'] = tokenized_tweet
from textblob import TextBlob



# Helper function to match negative tweets

def is_tweet_negative(tweet, threshold):

    testimonial = TextBlob(tweet)

    polarity = testimonial.sentiment.polarity

    if polarity < threshold:

        return True

    return False



# Helper function to match xenophoic tweets

def is_tweet_xenophobic(tweet):

    for s in search_terms:

        if s in tweet:

            return True

    return False



# Define search terms that appear in xenophic tweets

search_terms = ['alien', 'asian', 'china', 'criminal', 'floater', 'foreigner', 'greenhorn',

                'illegal', 'intruder', 'invader', 'migrant', 'newcomer', 'odd one out', 'outsider',

                'refugee', 'send her back', 'send him back', 'send them back', 'settler', 'stranger']



# Find all negative tweets using polarity (< 0)

df['Negative'] = df.text.apply(lambda x: is_tweet_negative(x, threshold=0))



# Find all xenophobic tweets looking only at the negative tweets

df['Xenophobic'] = df.loc[df['Negative'] == True].apply(lambda x: is_tweet_xenophobic(x.text), axis=1)

df[['Xenophobic']] = df[['Xenophobic']].fillna(value=False)
print(f"Number of tweets: {len(df)}")

print(f"Number of positive tweets: {len(df.loc[df['Negative'] == False])}")

print(f"Number of negative tweets: {len(df.loc[df['Negative'] == True])}")

print(f"Number of negative tweets, but benign: {len(df.loc[(df['Negative'] == True) & (df['Xenophobic'] == False)])}")

print(f"Number of xenophobic tweets: {len(df.loc[(df['Negative'] == True) & (df['Xenophobic'] == True)])}")
# Print a xenophobic tweet as a sample

tweet = df.loc[(df['Negative'] == True) & (df['Xenophobic'] == True)].iloc[20]

print(tweet.text)
# Print ratio of target variable

target_ratio = df['Xenophobic'].value_counts()[1]/df['Xenophobic'].value_counts()[0]

print(f"Ratio of non-xenophobic/xenophobic: {np.round(target_ratio, 3)}\n")



print('Split before random under-sampling:')

print(df.Xenophobic.value_counts())



# Plot the two classes

df['Xenophobic'].value_counts().plot(kind='bar', title='Count (Xenophobic) before')
# Apply random undersampling to fix the imbalancness in the data

# Thanks https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets



count_class_0, count_class_1 = df.Xenophobic.value_counts()

df_class_0 = df[df['Xenophobic'] == 0]

df_class_1 = df[df['Xenophobic'] == 1]



df_class_0_under = df_class_0.sample(2*count_class_1)

df_undersampled = pd.concat([df_class_0_under, df_class_1], axis=0)



target_ratio = df_undersampled['Xenophobic'].value_counts()[1]/df_undersampled['Xenophobic'].value_counts()[0]

print(f"Ratio of non-xenophobic/xenophobic: {np.round(target_ratio, 3)}\n")



print('Split after random under-sampling:')

print(df_undersampled.Xenophobic.value_counts())



df_undersampled = df_undersampled.reset_index(drop=True)

df_undersampled.Xenophobic.value_counts().plot(kind='bar', title='Count (Xenophobic) after');
# Plot a funnel chart



plt.style.use('seaborn')

from plotly import graph_objs as go

import plotly.express as px



temp = df_undersampled.groupby('Xenophobic').count()['text'].reset_index()

temp['label'] = temp['Xenophobic'].apply(lambda x : 'Xenophobic Tweet' if x==1 else 'Non Xenophobic Tweet')



fig = go.Figure(go.Funnelarea(

    text = temp.label,

    values = temp.text,

    title = {"position" : "top center", "text" : "Funnel Chart for target distribution"}

    ))

fig.show()
# Plot number of words in tweets



fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

tweet_len=df_undersampled[df_undersampled['Xenophobic']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='red')

ax1.set_title('Xenophobic Tweets')

tweet_len=df_undersampled[df_undersampled['Xenophobic']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='blue')

ax2.set_title('Non Xenophobic Tweets')

fig.suptitle('Number of words in tweets')

plt.show()
from collections import Counter



df_undersampled['temp_list'] = df_undersampled['text'].apply(lambda x:str(x).split())



top = Counter([item for sublist in df_undersampled['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(25))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x='count',y='Common_words',title='Common words in tweet',orientation='h',width=700,height=700,color='Common_words')

fig.show()
from wordcloud import WordCloud, STOPWORDS



text = df_undersampled['text'].values

cloud = WordCloud(stopwords=STOPWORDS,

                  background_color='white',

                  max_words=150,

                  width=2000,

                  height=1500).generate(" ".join(text))



plt.imshow(cloud)

plt.axis('off')

plt.show()
# Set up variables

df_undersampled = df_undersampled.drop(['Negative'], axis=1)

df_undersampled = df_undersampled.reset_index(drop=True)



X = df_undersampled

y = X['Xenophobic']

X.drop(['Xenophobic'], axis=1, inplace=True)



# Split the data using stratify 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)



# Reset the index

X_train = X_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)

X_test_stats = X_test.copy()
# Convert the text feature into a vectors of tokens

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',

                             lowercase=True, stop_words='english')

X_train_cv = cv.fit_transform(X_train.text)

X_test_cv = cv.transform(X_test.text)



# Scale numerical features (followers, retweets etc.)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = ['favorite_count', 'retweet_count', 'followers_count', 'friends_count', 'statuses_count']

X_train_sc = scaler.fit_transform(X_train[cols])

X_test_sc = scaler.transform(X_test[cols])



# Merge the numerical features with our count vectors

import scipy.sparse as sp

train_count = sp.csr_matrix(X_train_cv)

train_num = sp.csr_matrix(X_train_sc)

X_train = sp.hstack([train_count, train_num])



test_count = sp.csr_matrix(X_test_cv)

test_num = sp.csr_matrix(X_test_sc)

X_test = sp.hstack([test_count, test_num])
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix

n_classes = 2



clf = SGDClassifier(alpha=1e-3, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# Plot scores and make a confusion matrix for non-xenophobic/xenophobic predictions



from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix

n_classes = 2



cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))



print(f'Number of samples to classify: {len(X_test.toarray())}\n')

print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

print(f'Precision score: {precision_score(y_test, y_pred)}')

print(f'Recall score: {recall_score(y_test, y_pred)}\n')

print(f'Confusion matrix: \n{cm}')
# Normalize the confusion matrix and plot it



labels = ['non-xenophobic', 'xenophobic']

plt.figure(figsize=(6,6))

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, square=True, annot=True, cbar=False,

            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')

plt.ylabel('True label')
# Plot the ROC curve for the SVM classifier

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(8,8))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='SVM')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.show()
# Show how the first 50 test tweets were classified and their true label

testing_predictions = []

for i in range(len(X_test.toarray())):

    if y_pred[i] == 1:

        testing_predictions.append('Xenophobic')

    else:

        testing_predictions.append('Non-xenophobic')

check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'text':list(X_test_stats.text)})

check_df.replace(to_replace=0, value='No-Xenophobic', inplace=True)

check_df.replace(to_replace=1, value='Xenophobic', inplace=True)

check_df.iloc[:50]
# Show first 10 tweets that were classified as being xenophobic

pd.DataFrame(X_test_stats.text.iloc[y_pred].head(10)).reset_index(drop=True)