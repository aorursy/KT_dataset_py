#Import Libraries

import numpy as np 

import random  

import pandas as pd 

import os

import glob

import seaborn as sns

import matplotlib.pyplot as plt



import string

import itertools

from nltk import word_tokenize, FreqDist

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE

from sklearn.preprocessing import scale



import gensim

from gensim.models.word2vec import Word2Vec 

from tqdm import tqdm



import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook



import keras

from keras.preprocessing import text, sequence

from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import Input, Dense, LSTM, Embedding, Dropout
#List of csv files with twitter data

PATH = "../input/russian-troll-tweets"

print(os.listdir(PATH))
#Create filenames

filenames = glob.glob(os.path.join(PATH, "*.csv"))

print(filenames)
#Combine csv files

df_raw = pd.concat((pd.read_csv(f) for f in filenames))

print(df_raw.shape)

df_raw.tail()
#number of unique authors

len(df_raw.author.unique())
#Look at types of data included in each field.  

df_raw.info()
#Are there any missing values?  

df_raw.isna().sum().sort_values(ascending = False)
#Remove 1 content N/A row

df_raw.dropna(subset = ['content'], inplace = True)

df_raw.isna().sum().sort_values(ascending = False)
#Looking at data in more detail

df_raw.describe(include="all")
#dropping columns that won't be used

df = df_raw.drop(['external_author_id', 'harvested_date'], axis=1)

df.head()
#review languages of tweets

df.language.value_counts(normalize=True).head()
#update data to only contain english tweets

df = df.loc[df.language == 'English']

print(df.shape)

df.drop(['language'], axis = 1, inplace = True)

df.head()
#review missing data again

df.isna().sum().sort_values(ascending = False)
#Look at region in more detail

df.region.unique()
#value counts of region

df.region.value_counts(normalize=True)
#rename region nan values to unknown

df['region'].fillna(value='Unknown', inplace = True)

df.region.unique()
#recheck regions

df.region.value_counts(normalize=True)
#update data to only contain United States tweets

df = df.loc[df.region == 'United States']

print(df.shape)

df.drop(['region'], axis = 1, inplace = True)

df.head()
#review nan values again

df.isna().sum().sort_values(ascending = False)
#look at account_type in more detail

df.account_type.unique()
#update account_type

df['account_type'].fillna(value='Unknown', inplace = True)

df['account_type'].replace({'?': 'Unknown', 'right': 'Right', 

                            'left': 'Left', 'news': 'News', 

                           'local': 'Local', 'ZAPOROSHIA': 'Zaporoshia'}, 

                           inplace = True)

df.account_type.unique()
#revisit missing data

df.isna().sum().sort_values(ascending = False)
#look further at post_type

df.post_type.unique()
#Check retweet column vs post_type

df_post_type_nan = df.loc[df.post_type.isna()]

df_post_type_nan.retweet.unique()
#Check retweet column vs post_type

df_post_type_notnan = df.loc[df.post_type.notnull()]

df_post_type_notnan.retweet.unique()
#update post_type

df['post_type'].fillna(value='NOT_RETWEET', inplace = True)

df.post_type.unique()
#confirm there is no more missing data

df.isna().sum().sort_values(ascending = False)
#look at info

df.info()
# convert publish_date to datetime format

df['publish_date'] = pd.to_datetime(df['publish_date']).dt.date

df.head()
#check publish_date

df.info()
#review updated data

df.describe(include="all")
#review data first 5 rows

df.head()
#resample data for eda purposes

eda_sample = df.take(np.random.permutation(len(df))[:50000])

print(len(eda_sample))

eda_sample.head()
#Check original distribution

df.hist(figsize = (16,8))
#Check eda distribution

eda_sample.hist(figsize = (16,8))
#look at original categorical features

fig, ax =plt.subplots(3,1)

sns.countplot(df['post_type'], ax=ax[0], color='#95a5a6', order=df['post_type'].value_counts().index)

sns.countplot(df['account_type'], ax=ax[1], color='#2ecc71', order=df['account_type'].value_counts().index)

sns.countplot(df['account_category'], ax=ax[2], color='#3498db', order=df['account_category'].value_counts().index)

ax[0].set_xlabel('Post Type')

ax[1].set_xlabel('Account Type')

ax[2].set_xlabel('Account Category')

ax[0].set_ylabel('Tweet Count')

ax[1].set_ylabel('Tweet Count')

ax[2].set_ylabel('Tweet Count')

fig.set_size_inches(15, 20)

fig.show()
#look at eda sample categorical features

fig, ax =plt.subplots(3,1)

sns.countplot(eda_sample['post_type'], ax=ax[0], color='#95a5a6', order=eda_sample['post_type'].value_counts().index)

sns.countplot(eda_sample['account_type'], ax=ax[1], color='#2ecc71', order=eda_sample['account_type'].value_counts().index)

sns.countplot(eda_sample['account_category'], ax=ax[2], color='#3498db',order=eda_sample['account_category'].value_counts().index)

ax[0].set_xlabel('Post Type')

ax[1].set_xlabel('Account Type')

ax[2].set_xlabel('Account Category')

ax[0].set_ylabel('Tweet Count')

ax[1].set_ylabel('Tweet Count')

ax[2].set_ylabel('Tweet Count')

fig.set_size_inches(15, 20)

fig.show()
#account_category

plt.figure(figsize=(16,8))

ax = sns.scatterplot(x="followers", y="following", hue="account_category",data=eda_sample)

plt.title('Follower and Following Counts by Account Category', fontsize = 15)

plt.xlabel('Number of Followers', fontsize = 13)

plt.ylabel('Number of Following', fontsize = 13)

plt.legend(loc = 'best')
#account_type

plt.figure(figsize=(16,8))

ax = sns.scatterplot(x="followers", y="following", hue="account_type",data=eda_sample)

plt.title('Follower and Following Counts by Account Type', fontsize = 15)

plt.xlabel('Number of Followers', fontsize = 13)

plt.ylabel('Number of Following', fontsize = 13)
#Look at the NonEnglish account category

non_engl_acc_cat = eda_sample.loc[eda_sample.account_category == 'NonEnglish']

non_engl_acc_cat.account_type.value_counts(normalize=True)
#post_type

plt.figure(figsize=(16,8))

ax = sns.scatterplot(x="followers", y="following", hue="post_type",data=eda_sample)

plt.title('Follower and Following Counts by Post Type', fontsize = 15)

plt.xlabel('Number of Followers', fontsize = 13)

plt.ylabel('Number of Following', fontsize = 13)
# Count the number of times a date appears and convert to dataframe

tweet_trend = pd.DataFrame(df['publish_date'].value_counts())



# index is date, columns indicate tweet count on that day

tweet_trend.columns = ['tweet_count']



# sort the dataframe by the dates to have them in order

tweet_trend.sort_index(ascending = True, inplace = True)



# plot

tweet_trend['tweet_count'].plot(linestyle = "-", figsize = (16,8), color = 'blue')

plt.title('Tweet Count by Date', fontsize = 15)

plt.xlabel('Date', fontsize = 13)

plt.ylabel('Tweet Count', fontsize = 13)
#Important dates from Trumps campaign

#https://www.reuters.com/article/us-usa-election-timeline-factbox/timeline-pivotal-moments-in-trumps-presidential-campaign-idUSKBN1341FJ

dates_list = ['2015-06-16', '2015-12-07', '2016-02-01',

              '2016-03-01', '2016-03-03', '2016-03-11',

              '2016-05-03', '2016-05-26', '2016-06-20', 

              '2016-07-15', '2016-07-21', '2016-08-17',

              '2016-09-01', '2016-10-07', '2016-11-08']



# create a series of these dates.

important_dates = pd.Series(pd.to_datetime(dates_list))



#add columns to identify important events, and mark a 0 or 1.

tweet_trend['Important Events'] = False

tweet_trend.loc[important_dates, 'Important Events'] = True

tweet_trend.head()

tweet_trend['values'] = 0

tweet_trend.loc[important_dates, 'values'] = 1
# mark important events

tweet_trend['tweet_count'].plot(linestyle = "-", figsize = (12,8), rot = 45, color = 'blue', label = 'Tweet Counts')



# plot dots for where values in the tweet_trend df are 1

plt.plot(tweet_trend[tweet_trend['Important Events'] == True].index.values,

         tweet_trend.loc[tweet_trend['Important Events'] == True, 'values'],

         marker = 'o', color = 'red', linestyle = 'none', label = "Important Dates in Trump's Campaign")



# Adding a 30 day moving average on top to view the trend

plt.plot(tweet_trend['tweet_count'].rolling(window = 30, min_periods = 10).mean(), 

         color = 'red', label = '30 Day Moving Avg # of Tweets')



plt.title('Tweet Count by Date', fontsize = 15)

plt.xlabel('Date', fontsize = 13)

plt.ylabel('Tweet Count', fontsize = 13)

plt.legend(loc = 'best')
#Look further at the huge spike in 2016 that had 17,500+ tweets

df.publish_date.value_counts()[:20]
#look at account categorys of those who tweeted on the most tweeted day 10/6/2016

df_2016_10_06 = df.loc[df['publish_date'] == pd.to_datetime('2016-10-06')]

print(len(df_2016_10_06))



sns.countplot(df_2016_10_06['account_category'], color='#3498db', order=df_2016_10_06['account_category'].value_counts().index)

plt.xlabel("Account Category", fontsize = 13)

plt.ylabel("Tweet Count", fontsize = 13)

plt.title("Tweet Count on October 6, 2016", fontsize = 15)

fig.set_size_inches(15, 20)

fig.show()
#look at account categorys of those who tweeted on the most tweeted day 10/7/2016

df_2016_10_07 = df.loc[df['publish_date'] == pd.to_datetime('2016-10-07')]

print(len(df_2016_10_07))



sns.countplot(df_2016_10_07['account_category'], color='#2ecc71', order=df_2016_10_07['account_category'].value_counts().index)

plt.xlabel("Account Category", fontsize = 13)

plt.ylabel("Tweet Count", fontsize = 13)

plt.title("Tweet Count on October 7, 2016", fontsize = 15)

fig.set_size_inches(15, 20)

fig.show()
# Look at data again

print(len(df))

df.head()
# Cleaning up the tweets column in our dataframe

def standardize_text(df, content_field):

    df[content_field] = df[content_field].str.replace(r"http\S+", "")

    df[content_field] = df[content_field].str.replace(r"http", "")

    df[content_field] = df[content_field].str.replace(r"@\S+", "")

    df[content_field] = df[content_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")

    df[content_field] = df[content_field].str.replace(r"@", "at")

    df[content_field] = df[content_field].str.lower()

    return df



df = standardize_text(df, "content")
#review dataframe

df.head()
#Additional cleaning with stopwords

stopwords_list = stopwords.words('english') + list(string.punctuation)

stopwords_list += ["''", '""', '...', '``']
#Tokenizing and removing stopwords

def process_content(data):

    tokens = word_tokenize(data)

    stopwords_removed = [token for token in tokens if token not in stopwords_list]

    return stopwords_removed   



df['tokens'] = df['content'].apply(process_content)

df['text'] = df['tokens'].apply(' '.join)
#Further clean/check

df = df.reset_index()

df.drop(['content'], axis=1, inplace=True)

df.head()
## review tokens in more detail

word_tot = [word for tokens in df['tokens'] for word in tokens]

word_unique = set(word_tot)

tweet_len = [len(tokens) for tokens in df['tokens']]



print('{} total words with a vocabulary size of {}'.format(len(word_tot), len(word_unique)))

print('Maximum sentence length is {}'.format(max(tweet_len)))
#Look at histogram of tweet lengths

fig = plt.figure(figsize=(10, 10)) 

plt.xlabel('Tweet Length')

plt.ylabel('Number of Tweets')

plt.hist(tweet_len)

plt.show()
# join tweets to a single string

all_words = ' '.join(df['text'])



wordcloud = WordCloud(stopwords=STOPWORDS,width=800,height=500,

                      max_font_size=110, collocations=False).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# join tweets to a single string - Right Trolls

df_right = df.loc[df.account_category == "RightTroll"]

right_words = ' '.join(df_right['text'])



wordcloud = WordCloud(stopwords=STOPWORDS,width=800,height=500,

                      max_font_size=110, collocations=False).generate(right_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# join tweets to a single string - Left Trolls

df_left = df.loc[df.account_category == "LeftTroll"]

left_words = ' '.join(df_left['text'])



wordcloud = WordCloud(stopwords=STOPWORDS,width=800,height=500,

                      max_font_size=110, collocations=False).generate(left_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# join tweets to a single string - News Feed

df_news = df.loc[df.account_category == "NewsFeed"]

news_words = ' '.join(df_news['text'])



wordcloud = WordCloud(stopwords=STOPWORDS,width=800,height=500,

                      max_font_size=110, collocations=False).generate(news_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# join tweets to a single string - 10/6/2016 large spike in tweets

df_2016_10_06 = df.loc[df['publish_date'] == pd.to_datetime('2016-10-06')]

df_100616_words = ' '.join(df_2016_10_06['text'])



wordcloud = WordCloud(stopwords=STOPWORDS,width=800,height=500,

                      max_font_size=110, collocations=False).generate(df_100616_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
#Update dataframe for modeling

clean_data = df.drop(['index','author', 'publish_date', 

                      'following', 'followers', 'updates', 'account_type', 

                      'new_june_2018', 'retweet', 'post_type'], axis=1)

clean_data.head()
print(len(clean_data))
clean_data.account_category.value_counts(normalize=True)
#renaming account_categories

clean_data['account_category'].replace({'HashtagGamer': 'Other','NonEnglish': 'Other', 'Unknown': 'Other', 

                            'Fearmonger': 'Other', 'Commercial': 'Other'}, inplace = True)

print(clean_data.shape)

clean_data.account_category.value_counts(normalize=True)
#Define X and y

X = np.array(clean_data.tokens)

y = np.array(clean_data.account_category)



#Create Train/Validate/Test data splits

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=40)
#labelize tweets

LabeledSentence = gensim.models.doc2vec.LabeledSentence

tqdm.pandas(desc="progress-bar") #estimate time to completion



#return labelized tweets

def labelizeTweets(tweets, label_type):

    labelized = []

    for i,v in tqdm(enumerate(tweets)):

        label = '%s_%s'%(label_type,i)

        labelized.append(LabeledSentence(v, [label]))

    return labelized



#split labelized tweets by train/test/split data

X_train = labelizeTweets(X_train, 'TRAIN')

X_val = labelizeTweets(X_val, 'VALIDATE')

X_test = labelizeTweets(X_test, 'TEST')
#Check the first element of the labelized trained data to make sure it worked

X_train[0]
#Build the Word2Vec Model

tweet_w2v = Word2Vec(size=200, window = 5, min_count=10, workers=4) #initialize model

tweet_w2v.build_vocab([x.words for x in tqdm(X_train)]) #create vocabulary

tweet_w2v.train([x.words for x in tqdm(X_train)], total_examples=tweet_w2v.corpus_count, epochs=2) #train model
#Check that the Word2Vec code worked correctly 

tweet_w2v['happy']
#example producing the most similar words

tweet_w2v.most_similar('happy')
# defining the chart

output_notebook()

plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)



# getting a list of word vectors. limit to 10,000. each is of 200 dimensions

word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]



# dimensionality reduction. converting the vectors to 2d vectors

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)

tsne_w2v = tsne_model.fit_transform(word_vectors)



# putting everything in a dataframe

tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])

tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]



# plotting the corresponding word appears when you hover on the data point.

plot_tfidf.scatter(x='x', y='y', source=tsne_df)

hover = plot_tfidf.select(dict(type=HoverTool))

hover.tooltips={"word": "@words"}

show(plot_tfidf)

# Create tf-idf matrix

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)

matrix = vectorizer.fit_transform([x.words for x in X_train])

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

print('vocab size :', len(tfidf))
#Build the vector producing averaged tweets

def buildWordVector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0.

    for word in tokens:

        try:

            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]

            count += 1.

        except KeyError: # handling the case where the token is not

                         # in the corpus. useful for testing.

            continue

    if count != 0:

        vec /= count

    return vec
#Convert into vector and scale

train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_train))])

train_vecs_w2v = scale(train_vecs_w2v)



val_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_val))])

val_vecs_w2v = scale(val_vecs_w2v)



test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, X_test))])

test_vecs_w2v = scale(test_vecs_w2v)
#Using a Random Forest Classifier model

rfc =  RandomForestClassifier(n_estimators=100, verbose=True)
#Fitting a Random Forest Classifier

rfc.fit(train_vecs_w2v, y_train)

y_pred_rf = rfc.predict(val_vecs_w2v)
#Define a function to get metrics

def get_metrics(y_val, y_pred):  

    # true positives / (true positives+false positives)

    precision = precision_score(y_val, y_pred, pos_label=None, average='weighted')             

    # true positives / (true positives + false negatives)

    recall = recall_score(y_val, y_pred, pos_label=None,average='weighted')

    # harmonic mean of precision and recall

    f1 = f1_score(y_val, y_pred, pos_label=None, average='weighted')

    # true positives + true negatives/ total

    accuracy = accuracy_score(y_val, y_pred)

    return accuracy, precision, recall, f1
#View metrics for random forest classifier

accuracy, precision, recall, f1 = get_metrics(y_val, y_pred_rf)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, 

                                                                       recall, f1))
#Plot Confusion Matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.winter):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=30)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, fontsize=20)

    plt.yticks(tick_marks, classes, fontsize=20)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.



    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 

                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    

    plt.tight_layout()

    plt.ylabel('True label', fontsize=30)

    plt.xlabel('Predicted label', fontsize=30)



    return plt
#Plot confusion matrix for random forest model

class_names = list(set(y))

cm = confusion_matrix(y_val, y_pred_rf)

fig = plt.figure(figsize=(15, 15))

plot = plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Confusion matrix')

plt.show()

print(cm)
# Convert labels to categorical one-hot encoding

ohe_y_train = pd.get_dummies(y_train)

ohe_y_val = pd.get_dummies(y_val)

ohe_y_test = pd.get_dummies(y_test)
#For a single-input model using RMSprop optimizer

random.seed(123)

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(200,)))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 32 samples

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=32,

                    validation_data=(val_vecs_w2v, ohe_y_val))
#view loss and accuracy dictionary

history.history
#LSTM

#model = Sequential()

#model.add(Embedding(len(tfidf), 200))

#model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))

#model.add(Dense(4, activation='softmax'))

#model.compile(optimizer='rmsprop',

#              loss='binary_crossentropy',

#              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 32 samples

#history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, 

#                    batch_size=32,validation_data=(val_vecs_w2v, ohe_y_val))
#using adam optimization

random.seed(123)

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(200,)))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 32 samples

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=32, 

                   validation_data=(val_vecs_w2v, ohe_y_val))
#Increase layers in this model using RMSprop optimizer and an increased batch size

random.seed(123)

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(200,)))

model.add(Dense(25, activation='relu'))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 256 samples

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=256,

                   validation_data=(val_vecs_w2v, ohe_y_val))
#L2 Regularizer

random.seed(123)

model = Sequential()

model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.005), input_shape=(200,)))

model.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 256 samples

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=256, 

         validation_data=(val_vecs_w2v, ohe_y_val))
#L1 Regularizer

random.seed(123)

model = Sequential()

model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l1(0.005), input_shape=(200,)))

model.add(Dense(25, activation='relu', kernel_regularizer=regularizers.l1(0.005)))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 256 samples

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=256,

                   validation_data=(val_vecs_w2v, ohe_y_val))
#Dropout Regularizer 

random.seed(123)

model = Sequential()

model.add(Dropout(0.3, input_shape=(200,)))

model.add(Dense(50, activation='relu'))

model.add(layers.Dropout(0.3))

model.add(Dense(25, activation='relu'))

model.add(layers.Dropout(0.3))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 256 samples

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=2, batch_size=256,

                   validation_data=(val_vecs_w2v, ohe_y_val))
#Increase epochs 

random.seed(123)

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(200,)))

model.add(Dense(25, activation='relu'))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001, rho=0.9),

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Train the model, iterating on the data in batches of 256 samples with 100 epochs

history = model.fit(train_vecs_w2v, ohe_y_train, epochs=400, batch_size=256,

                   validation_data=(val_vecs_w2v, ohe_y_val))
#plot to see how many epochs makes sense

history_dict = history.history

train_loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(train_loss_values) + 1)

plt.plot(epochs, train_loss_values, 'g', label='Training loss')

plt.plot(epochs, val_loss_values, 'r', label='Validation loss')



plt.title('Training loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
#determine predictions

y_pred_nn = model.predict_classes(val_vecs_w2v)

y_pred_nn
#reformat y_val for confusion matrix and metrics

ohe_y_val_arr = np.array(ohe_y_val)

y_val_nn = np.argmax(ohe_y_val_arr, axis=1)

y_val_nn
#View metrics for neural network

accuracy, precision, recall, f1 = get_metrics(y_val_nn, y_pred_nn)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, 

                                                                       recall, f1))
#Plot confusion matrix for neural network

class_names =list(set(y))

cm = confusion_matrix(y_val_nn, y_pred_nn)

fig = plt.figure(figsize=(15, 15))

plot = plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Confusion matrix')

plt.show()

print(cm)
#View metrics for random forest classifier on test data

y_pred_test = rfc.predict(test_vecs_w2v)

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred_test)

print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, 

                                                                       recall, f1))
#Plot confusion matrix for test data using random forest model

class_names = list(set(y))

cm = confusion_matrix(y_test, y_pred_test)

fig = plt.figure(figsize=(15, 15))

plot = plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Confusion matrix')

plt.show()

print(cm)