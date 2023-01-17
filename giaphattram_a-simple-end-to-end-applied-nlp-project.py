import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re

import nltk

import spacy

nlp = spacy.load('en_core_web_sm')

from collections import Counter

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head(2)
def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase) #This confuses contraction with possession

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
train['cleanedText'] = train.text.map(lambda x: decontracted(x))

test['cleanedText'] = test.text.map(lambda x: decontracted(x))
def removeURL(text):

    for eachToken in nlp(text):

        match = re.search("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", eachToken.text)

        if match:

            text = text.replace(eachToken.text, "")

    return text

train['cleanedText'] = train.cleanedText.apply(removeURL)

test['cleanedText'] = test.cleanedText.apply(removeURL)
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

# for each in "~`!@#$%^&?*()_+?|/<>.,[]-:;'":

#     stopWords.add(each)

stopWords.remove('no')

stopWords.remove('not')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from nltk.tokenize import word_tokenize

train['cleanedText'] = train.cleanedText.map(lambda x: ' '.join([lemmatizer.lemmatize(each.text.lower()) for each in nlp(x) if each.text.lower() not in stopWords]))

test['cleanedText'] = test.cleanedText.map(lambda x: ' '.join([lemmatizer.lemmatize(each.text.lower()) for each in nlp(x) if each.text.lower() not in stopWords]))

train[['text','cleanedText']].head()
# Concatenate keyword

train['keyword'] = train.keyword.astype(str)

test['keyword'] = test.keyword.astype(str)

train['cleanedText'] = train.apply(lambda x: x.cleanedText if x.keyword == 'nan' else x.keyword.lower() + ' ' + x.cleanedText, axis = 1)

test['cleanedText'] = test.apply(lambda x: x.cleanedText if x.keyword == 'nan' else x.keyword.lower() + ' ' + x.cleanedText, axis = 1)



# Concatenate location

train['location'] = train.location.astype(str)

test['location'] = train.location.astype(str)

train['cleanedText'] = train.apply(lambda x: x.cleanedText if x.location == 'nan' else x.location.lower() + ' ' + x.cleanedText, axis = 1)

test['cleanedText'] = test.apply(lambda x: x.cleanedText if x.location == 'nan' else x.location.lower() + ' ' + x.cleanedText, axis = 1)
# Build a manual counter

counter = Counter()

for eachWord in word_tokenize(' '.join(train.cleanedText.tolist())):

    if (eachWord.lower() not in stopWords)&(eachWord.lower()!=''):

        counter[eachWord.lower()] += 1
print("Columns in train: {}".format(', '.join(train.columns.values)))

print(f"Count of tweets in train: {train.shape[0]}")

print('Count of true tweets: {}'.format(train.target.value_counts()[1]))

print('Count of fake tweets: {}'.format(train.target.value_counts()[0]))

print("Top 10 keywords: {}".format(', '.join(train.keyword.value_counts()[:10].index.values)))

print('Top 10 locations: {}'.format(', '.join(train.location.value_counts()[:10].index.values)))

print("Total number of words in all tweets: {}".format(len(counter)))

print("Most common words or symbols used in tweets: {}".format(', '.join([each[0] for each in counter.most_common()[:20]])))
from wordcloud import WordCloud

allTokens = [each.text for each in nlp(' '.join(train.cleanedText.tolist()))]

allTokensTrueTweets = [each.text for each in nlp(' '.join(train[train.target == 1].cleanedText.tolist()))]

allTokensFalseTweets = [each.text for each in nlp(' '.join(train[train.target == 0].cleanedText.tolist()))]

frequency_dist = nltk.FreqDist(allTokens)

frequency_dist_trueTweets = nltk.FreqDist(allTokensTrueTweets)

frequency_dist_falseTweets = nltk.FreqDist(allTokensFalseTweets)

wordcloud = WordCloud().generate_from_frequencies(frequency_dist)

wordcloud_trueTweets = WordCloud().generate_from_frequencies(frequency_dist_trueTweets)

wordcloud_falseTweets = WordCloud().generate_from_frequencies(frequency_dist_falseTweets)



plt.imshow(wordcloud)

plt.title("All Tweets")

plt.show()

plt.imshow(wordcloud_trueTweets)

plt.title("True Tweets")

plt.show()

plt.imshow(wordcloud_falseTweets)

plt.title("False Tweets")

plt.show()
fig, axs = plt.subplots(3,2, sharey = 'row', figsize = (10,10))

train.apply(lambda x: len(x.cleanedText) if x.target == 1 else None, axis = 1).hist(bins = 20, ax = axs[0,0])

axs[0,0].set_title("Character Count Distribution for True Tweets")

train.apply(lambda x: len(x.cleanedText) if x.target == 0 else None, axis = 1).hist(bins = 20, ax = axs[0,1])

axs[0,1].set_title("Character Count Distribution for False Tweets")

train.apply(lambda x: len(x.cleanedText.split(' ')) if x.target == 1 else None, axis = 1).hist(bins = 20, ax = axs[1,0])

axs[1,0].set_title("Token Count Distribution for True Tweets")

train.apply(lambda x: len(x.cleanedText.split(' ')) if x.target == 0 else None, axis = 1).hist(bins = 20, ax = axs[1,1])

axs[1,1].set_title("Token Count Distribution for False Tweets")

train.apply(lambda x: np.mean(list(map(len, x.cleanedText.split(' ')))) if x.target == 1 else None, axis = 1).hist(bins = 20, ax = axs[2,0])

axs[2,0].set_title("Average Token Length for True Tweets")

train.apply(lambda x: np.mean(list(map(len, x.cleanedText.split(' ')))) if x.target == 0 else None, axis = 1).hist(bins = 20, ax = axs[2,1])

axs[2,1].set_title("Average Token Length for False Tweets")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()

vect.fit(train.cleanedText)

tfidf_dtm = pd.DataFrame(vect.transform(train.cleanedText.tolist()).toarray(), index = train.id, columns = vect.get_feature_names())

print("Vocabulary size from all the tweets: {}".format(len(vect.vocabulary_)))
import random

train_index = random.sample(list(np.arange(0, tfidf_dtm.shape[0])), int(tfidf_dtm.shape[0]/2))

train_index.sort()

train_features = tfidf_dtm.iloc[train_index]

test_features = tfidf_dtm.iloc[[index for index in np.arange(0, tfidf_dtm.shape[0]) if index not in train_index]]

train_target = train.iloc[train_index].target

test_target = train.iloc[[index for index in np.arange(0, train.shape[0]) if index not in train_index]].target

print("Train set has {} true tweets and {} false tweets.".format(train_target.value_counts().loc[1], train_target.value_counts().loc[0]))

print("Test set has {} true tweets and {} false tweets.".format(test_target.value_counts().loc[1], test_target.value_counts().loc[0]))
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB().fit(train_features, train_target)

test_predictions = classifier.predict(test_features)

from sklearn.metrics import classification_report

print(classification_report(test_target, test_predictions))
test_dtm = pd.DataFrame(vect.transform(test.text).toarray(), index = test.id, columns = vect.get_feature_names())

classifier.predict(test_dtm)

submission_prediction = pd.DataFrame(classifier.predict(test_dtm), columns = ['target'])

submission_prediction['id'] = test.id

submission_prediction = submission_prediction[['id', 'target']]

os.chdir("/kaggle/working/")

submission_prediction.to_csv('submission.csv', index = False)
os.chdir("/kaggle/input/gloveicg")

gloveFile = open("/kaggle/input/gloveicg/Glove/glove.6B.300d.txt")

print("First line in gloveFile: {}".format(gloveFile.readline()))
gloveModel = {}

gloveFile.seek(0) # Return the cursor back to the beginning of the file

for line in gloveFile.readlines():

    lineElements = line.split(" ")

    token = lineElements[0]

    tokenEmbedding = [float(each) for each in lineElements[1:]]

    gloveModel[token] = tokenEmbedding

gloveMatrix = pd.DataFrame(gloveModel).T

gloveMatrix.head()
def findClosestWordToWord(word):

    diffMatrix = gloveMatrix - gloveModel[word]

    diffMatrix = diffMatrix * diffMatrix

    diffMatrix.drop(word, axis = 0, inplace = True)

    return diffMatrix.sum(axis = 1).idxmin()

def findClosestWordToEmbedding(embedding, excludedWords):

    diffMatrix = gloveMatrix - embedding

    diffMatrix = diffMatrix * diffMatrix

    diffMatrix.drop(excludedWords, axis = 0, inplace = True)

    return diffMatrix.sum(axis = 1).idxmin()
temp = gloveMatrix.loc['father'] - gloveMatrix.loc['mother'] + gloveMatrix.loc['daughter']

print("Father to Mother is like {} to Daughter".format(findClosestWordToEmbedding(temp, ['father', 'mother', 'daughter']).title()))

print("The closest word to Data is {}".format(findClosestWordToWord('data').title()))
print("Size of vocabulary in GloVe: ",gloveMatrix.shape[0])

print("Size of vocabulary from all tweets: ",len(vect.get_feature_names()))

print("Size of overlapped vocabulary:" , len(set(gloveMatrix.index).intersection(set(vect.get_feature_names()))))



print("Since we want to use word embedding to train model, we need to remove tokens that do not appear in GloVe")

train['cleanedText2'] = train.cleanedText.map(lambda x: ' '.join([each for each in word_tokenize(x) if each in gloveModel.keys()]))

test['cleanedText2'] = test.cleanedText.map(lambda x: ' '.join([each for each in word_tokenize(x) if each in gloveModel.keys()]))



train.dropna(subset=['text', 'cleanedText', 'cleanedText2'], inplace = True)

# test.dropna(subset=['text', 'cleanedText', 'cleanedText2'], inplace = True)



# Drop tweets whose cleanedText2 is empty

temp = (train.cleanedText2.apply(len) == 0)

train.drop(temp[temp == True].index, axis = 0, inplace = True)

del temp



# temp = (test.cleanedText2.apply(len) == 0)

# test.drop(temp[temp == True].index, axis = 0, inplace = True)

# del temp



train[['text', 'cleanedText', 'cleanedText2']].head(2)
# Interestingly, i see '13,000' is reserved in cleanedText2 for a row

# Finding out whether 13,000 is in GloVe vocabulary

print("'13,000' is in GloVe vocabulary: {}".format('13,000' in gloveModel.keys()))

print("The word closest to '13,000' is '{}'".format(findClosestWordToWord('13,000')))
# Create a GloVe dictionary where keys are tokens and values are respective numeric indices 

gloveDict = {}

for index, eachWord in enumerate(gloveMatrix.index):

    gloveDict[eachWord] = index

    

# Using GloVe dictionary, convert train.cleanedText2 to sequences of numeric indices of respective tokens

def texts_to_sequences(textSeries):

    return textSeries.map(lambda x: [gloveDict['.']] if len(x) == 0 else [gloveDict[each] for each in x.split(" ")])

    # if cleanedText2 is empty then fill the string with '.' (to meet submission criterion of full test set)

    

train['sequence'] = texts_to_sequences(train.cleanedText2)

test['sequence'] = texts_to_sequences(test.cleanedText2)  



# Padding these sequences with periods '.'. I use the length of the longest sequence in train as the max for both train and submission test sets 

maxlen = np.max(train.sequence.map(lambda x: len(x)))

train['padded_sequence'] = train.sequence.map(lambda x: x[:maxlen] if len(x) >= maxlen else x + [gloveDict['.']]*(maxlen-len(x)))

test['padded_sequence'] = test.sequence.map(lambda x: x[:maxlen] if len(x) >= maxlen else x + [gloveDict['.']]*(maxlen-len(x)))
train_index = random.sample(list(np.arange(0, train.shape[0])), int(train.shape[0]/2))

train_index.sort()

train_features = train.iloc[train_index]['padded_sequence']

train_features = pd.DataFrame([each for each in train_features])

train_target = train.iloc[train_index]['target']

test_features = train.iloc[[i for i in np.arange(0, train.shape[0]) if i not in train_index]]['padded_sequence']

test_features = pd.DataFrame([each for each in test_features])

test_target = train.iloc[[i for i in np.arange(0, train.shape[0]) if i not in train_index]]['target']

print("Train set has {} true tweets and {} false tweets.".format(train_target.value_counts().loc[1], train_target.value_counts().loc[0]))

print("Test set has {} true tweets and {} false tweets.".format(test_target.value_counts().loc[1], test_target.value_counts().loc[0]))
from tensorflow.keras import metrics

from tensorflow.keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

METRICS = [

    metrics.BinaryAccuracy(name = 'accuracy'),

    metrics.Precision(name = 'precision'),

    metrics.Recall(name = 'recall'),

    metrics.AUC(name = 'auc')

]

early_stopping = EarlyStopping(

    monitor = 'val_auc',

    verbose = 1,

    patience = 10,

    mode = 'max',

    restore_best_weights = True

)

checkpoint = ModelCheckpoint('model-{epoch}-{loss}-{val_loss}.h5', verbose = 1, \

                             monitor = 'val_loss', save_best_only = True, mode = 'auto')
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, BatchNormalization, Dropout

from keras.initializers import Constant

from keras.optimizers import SGD, Adam, RMSprop

model = Sequential()

embedding_layer = Embedding(input_dim = gloveMatrix.shape[0], output_dim = gloveMatrix.shape[1],\

                            embeddings_initializer = Constant(gloveMatrix.to_numpy()),\

                           input_length = maxlen, trainable = False)

model.add(embedding_layer)

model.add(Bidirectional(LSTM(units=64, dropout = 0.5, recurrent_dropout = 0.5)))

model.add(Dense(32))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr = 0.0002), metrics = METRICS)

model.summary()
history = model.fit(train_features, train_target, epochs = 35, batch_size = 32, \

                    validation_data = (test_features, test_target))
historyDF = pd.DataFrame(history.history)

historyDF.loss.plot()

plt.title("Train Loss")

plt.show()

historyDF.val_loss.plot()

plt.title("Test Loss")

plt.show()

# plt.title('Visualizing Loss')
test_predictions = pd.Series(model.predict(test_features).reshape(-1,),name = 'test_prediction')

threshold = 0.45

test_classPredictions = pd.Series(np.where(test_predictions > threshold, 1, 0), name = 'test_classPredictions')

print(classification_report(test_target, test_classPredictions))
submissionTest_features = test['padded_sequence']

submissionTest_features = pd.DataFrame([each for each in submissionTest_features])



submissionTest_Predictions = model.predict(submissionTest_features).reshape(-1,)

submissionTest_classPredictions = pd.Series(np.where(submissionTest_Predictions > 0.35, 1, 0), name = 'target')



submission_prediction = pd.DataFrame({'id': test.id.values, 'target': submissionTest_classPredictions.values})



os.chdir("/kaggle/working")

submission_prediction = submission_prediction.to_csv("submission.csv", index = False)