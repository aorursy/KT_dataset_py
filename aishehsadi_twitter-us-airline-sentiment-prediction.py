#import needed libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from nltk.corpus import stopwords

import string

from collections import Counter 

from wordcloud import WordCloud,STOPWORDS





from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from mlxtend.plotting import plot_confusion_matrix



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



from tensorflow.keras.layers import Embedding

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import LSTM
#define work directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read data 

data= pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")

#let's see samples of the data 

data.head()
#data size

data.shape
data.info()
data['airline_sentiment'].value_counts()
negativereason_values_count= data['negativereason'].value_counts()

print(negativereason_values_count)

print("all negative resons :" ,(negativereason_values_count).sum())

#plot negative reasons distribution

x =(data['negativereason']).value_counts().keys()

x_pos = np.arange(len(x))

y = (data['negativereason'].value_counts()).values





barlist = plt.bar(x_pos, y, align='center')

plt.xticks(x_pos, x,rotation=90)

plt.xlabel('Negative Reason')

plt.ylabel('Count')

plt.xlim(-1,len(x) )

plt.ylim(0,3000)



plt.show()
airline_values= data['airline'].value_counts()

print(airline_values)

#plot airlines with tweets distribution

x =(data['airline']).value_counts().keys()

x_pos = np.arange(len(x))

y = (data['airline'].value_counts()).values





barlist = plt.bar(x_pos, y, align='center')

plt.xticks(x_pos, x,rotation=90)

plt.xlabel('Airline')

plt.ylabel('Count')

plt.xlim(-1,len(x) )

plt.ylim(0,4000)



plt.show()
#plot airlines with tweets distribution

data_neg = data[data['airline_sentiment']=='negative']



x =(data_neg['airline']).value_counts().keys()

x_pos = np.arange(len(x))

y = (data_neg['airline'].value_counts()).values





barlist = plt.bar(x_pos, y, align='center')

plt.xticks(x_pos, x,rotation=45)

plt.xlabel('Airline')

plt.ylabel('Negative Count')

plt.xlim(-1,len(x) )

plt.ylim(0,3000)



plt.show()
#plot airlines with tweets distribution

data_pos = data[data['airline_sentiment']=='positive']



x =(data_pos['airline']).value_counts().keys()

x_pos = np.arange(len(x))

y = (data_pos['airline'].value_counts()).values







barlist = plt.bar(x_pos, y, align='center')

plt.xticks(x_pos, x,rotation=45)

plt.xlabel('Airline')

plt.ylabel('Positive Count')

plt.xlim(-1,len(x) )

plt.ylim(0,700)



plt.show()
#plot airlines with tweets distribution

data_neut= data[data['airline_sentiment']=='neutral']



x =(data_neut['airline']).value_counts().keys()

x_pos = np.arange(len(x))

y = (data_neut['airline'].value_counts()).values







barlist = plt.bar(x_pos, y, align='center')

plt.xticks(x_pos, x,rotation=45)

plt.xlabel('Airline')

plt.ylabel('Neutral Count')

plt.xlim(-1,len(x) )

plt.ylim(0,800)



plt.show()
def plotAirlineSentiment(airline):

    data_air= data[data['airline']==airline]



    x =(data_air['airline_sentiment']).value_counts().keys()

    x_pos = np.arange(len(x))

    y = (data_air['airline_sentiment'].value_counts()).values







    barlist = plt.bar(x_pos, y, align='center')

    plt.xticks(x_pos, x)

    plt.xlabel('Sentiment')

    plt.ylabel('Sentiment Count')

    plt.xlim(-1,len(x) )

    plt.ylim(0, 3000)

    plt.title(airline)
plt.figure(1,figsize=(15, 15))

plt.subplot(231)

plotAirlineSentiment('United')

plt.subplot(232)

plotAirlineSentiment('US Airways')

plt.subplot(233)

plotAirlineSentiment('American')

plt.subplot(234)

plotAirlineSentiment('Southwest')

plt.subplot(235)

plotAirlineSentiment('Delta')

plt.subplot(236)

plotAirlineSentiment('Virgin America')
def plotAirlineNegativeReason(airline):

    data_air= data[data['airline']==airline]



    x =(data_air['negativereason']).value_counts().keys()

    x_pos = np.arange(len(x))

    y = (data_air['negativereason'].value_counts()).values







    barlist = plt.bar(x_pos, y, align='center')

    plt.xticks(x_pos, x,rotation=90)

    plt.xlabel('Negative Reason')

    plt.ylabel('Tweets Count')

    plt.xlim(-1,len(x) )

    plt.ylim(0, 1000)

    plt.title(airline)
plotAirlineNegativeReason('United')

#seprate each sentiment tweets 

tweets = data["text"]

neg_tweets = data[data["airline_sentiment"] =="negative"]["text"]

neut_tweets = data[data["airline_sentiment"] =="neutral"]["text"]

pos_tweets = data[data["airline_sentiment"] =="positive"]["text"]
stopwordslist = set(stopwords.words('english'))

airline_names =["united", "usairways", "americanair" ,"southwestair", "deltaair", "virginamericair" ,"flight"]

allunWantedwords= list(stopwordslist) + airline_names

# function to remove stop words, remove airline names, words starts with @ 

def pre_processData(data):

  

    data =list(data)

    new_data =[]

    

    

    for l in range(len(data)):

        line = data[l]

        line = line.lower()

        line=' '.join(word for word in line.split() if not word.startswith('@'))

        querywords = line.split()  

        resultwords  = [word for word in querywords if word not in stopwordslist]

        resultwords2  = [word for word in resultwords if word not in airline_names]



        newline = ' '.join(resultwords2)       

        newline = newline.translate(str.maketrans('', '', string.punctuation))

        new_data.append(newline)

        

    return new_data
#pre_process tweets from the three categories

neg_tweets= pre_processData(neg_tweets)

neut_tweets= pre_processData(neut_tweets)

pos_tweets= pre_processData(pos_tweets)
#function to find some statistics about tweets texts

def tweetStat(tweetsList):

    min_len= min(len(x) for x in tweetsList) 

    max_len= max(len(x) for x in tweetsList) 

    avg_len= sum(len(x) for x in tweetsList) / len(tweetsList)



    return min_len, max_len, avg_len
print("Negative tweets stats (min, max, avg): ", tweetStat(neg_tweets))

print("Neutral tweets stats (min, max, avg): ", tweetStat(neut_tweets))

print("Positive tweets stats (min, max, avg): ", tweetStat(pos_tweets))
#function to get most k frquent words in a set of tweets

def getFrequentWords(textList, k):  #k frequent words

    texts = ' '.join(textList)

    split_it = texts.split() 

    count = Counter(split_it) 

    most_occur = count.most_common(k) 



    print (most_occur)

    return most_occur
print("Most frquent 10 words in negative tweets are:" )

negwords = getFrequentWords(neg_tweets,10)
print("Most frquent 10 words in neutral tweets are:" )

neut_words =getFrequentWords(neut_tweets,10)
print("Most frquent 10 words in positive tweets are:" )

pos_words= getFrequentWords(pos_tweets,10)
#function to get wordcloud 

def getWordCloud(texts):

    texts =" ".join(texts)

    wordcloud = WordCloud(stopwords=allunWantedwords,

                          background_color='white',

                          width=2000,

                          height=2000

                         ).generate(texts)

    

    return wordcloud
#negtaive tweets word cloud 

neg_wordcloud = getWordCloud(neg_tweets)

plt.figure(1,figsize=(7, 7))

plt.imshow(neg_wordcloud)

plt.axis('off')

plt.show()

#neutral tweets word cloud 

neut_wordcloud = getWordCloud(neut_tweets)

plt.figure(2,figsize=(7, 7))

plt.imshow(neut_wordcloud)

plt.axis('off')

plt.show()
#positive tweets word cloud 

pos_wordcloud = getWordCloud(pos_tweets)

plt.figure(2,figsize=(7, 7))

plt.imshow(pos_wordcloud)

plt.axis('off')

plt.show()
#store the required two columns in a new data frame 

cleanedTweets= pre_processData(data["text"])  # apply preprocessing step 

finalData= pd.DataFrame()

finalData['text']= cleanedTweets

finalData['airline_sentiment']= data['airline_sentiment']

finalData.head()
#convert airline_sentiment values (negative, neutral, and positive) to numerical values (0,1,and 2) and map the data to it

sentiments = data['airline_sentiment'].astype('category').cat.categories.tolist()

replace_map_comp = {'airline_sentiment' : {k: v for k,v in zip(sentiments,list(range(0,len(sentiments))))}}

print(replace_map_comp)



finalData.replace(replace_map_comp, inplace=True)

finalData.head()
# store tweets text in x & the target label in y

x=  finalData['text'].values

y= finalData['airline_sentiment'].values
#split data into training & test with 15%

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.15, random_state=42)



print("x_train:" ,x_train.shape, ", y_train: ", len(y_train))

print("x_test: ",x_test.shape, ", y_test: ", len(y_test))

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(x_train)

test_features=v.transform(x_test)
#convert from sparse (contain a lot of zeros) to dense

final_train_features=train_features.toarray()

final_test_features= test_features.toarray()

print(final_train_features.shape)

print(final_test_features.shape)
print('training model (this could take sometime)...')

clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)

clf.fit(final_train_features, y_train)



print('calculating results...')

predictions_train = clf.predict(final_train_features)

predictions_test = clf.predict(final_test_features)



accuracy = accuracy_score(predictions_train,y_train)

print(" Logistic Regression Train accuracy is: {:.4f}".format(accuracy))



accuracy = accuracy_score(predictions_test,y_test)

print(" Logistic Regression Test accuracy is: {:.4f}".format(accuracy))
#print other performance measures, espically the data is unbalanced

print(classification_report(predictions_test , y_test))

#calculate the confusion matrix and plot it



cm=confusion_matrix(predictions_test , y_test)

class_names =  ['Negative', 'Neutral', 'Positive']

fig, ax = plot_confusion_matrix(conf_mat=cm,

                                colorbar=True,

                                show_absolute=True,

                                show_normed=True,

                                class_names=class_names)

plt.show()
print('training model (this could take sometime)...')

clf =     RandomForestClassifier(n_estimators=10)

clf.fit(final_train_features, y_train)



print('calculating results...')

predictions_train = clf.predict(final_train_features)

predictions_test = clf.predict(final_test_features)



accuracy = accuracy_score(predictions_train,y_train)

print("Random Forest Train accuracy is: {:.4f}".format(accuracy))



accuracy = accuracy_score(predictions_test,y_test)

print("Random Forest Test accuracy is: {:.4f}".format(accuracy))

#print other performance measures, espically the data is unbalanced

print(classification_report(predictions_test , y_test))
#calculate the confusion matrix and plot it



cm=confusion_matrix(predictions_test , y_test)

class_names =  ['Negative', 'Neutral', 'Positive']

fig, ax = plot_confusion_matrix(conf_mat=cm,

                                colorbar=True,

                                show_absolute=True,

                                show_normed=True,

                                class_names=class_names)

plt.show()
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(x_train)

train_features =  tfidf_vect.transform(x_train)

test_features =  tfidf_vect.transform(x_test)

#convert from sparse (contain a lot of zeros) to dense

final_train_features=train_features.toarray()

final_test_features= test_features.toarray()
print('training model (this could take sometime)...')

clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)

clf.fit(final_train_features, y_train)



print('calculating results...')



predictions_train = clf.predict(final_train_features)

predictions_test = clf.predict(final_test_features)



accuracy = accuracy_score(predictions_train,y_train)

print("Logisitc regression Train accuracy is: {:.4f}".format(accuracy))



accuracy = accuracy_score(predictions_test,y_test)

print("Logisitc regression Test accuracy is: {:.4f}".format(accuracy))



from sklearn.ensemble import RandomForestClassifier



print('training model (this could take sometime)...')

clf = RandomForestClassifier(n_estimators=10)

clf.fit(final_train_features, y_train)



print('calculating results...')

predictions_train = clf.predict(final_train_features)

predictions_test = clf.predict(final_test_features)



accuracy = accuracy_score(predictions_train,y_train)

print("Random forest Train accuracy is: {:.4f}".format(accuracy))



accuracy = accuracy_score(predictions_test,y_test)

print("Random forest Test accuracy is: {:.4f}".format(accuracy))
embeddings_index = {}

f = open('../input/glove840b300dtxt/glove.840B.300d.txt')



for line in f:

    values = line.split(' ')

    word = values[0] ## The first entry is the word

    coefs = np.asarray(values[1:], dtype='float32') 

    embeddings_index[word] = coefs

f.close()



print('GloVe data loaded')

print('Loaded %s word vectors.' % len(embeddings_index))
#encode train texts and test texts using the a tokenizer

MAX_NUM_WORDS = 1000

MAX_SEQUENCE_LENGTH = 135 #from the stats we found previously

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(x)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



sequences_train = tokenizer.texts_to_sequences(x_train)

x_train_seq = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)



sequences_test = tokenizer.texts_to_sequences(x_test)

x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)



#convert labels to one hot vectors

labels_train = to_categorical(np.asarray(y_train))

labels_test = to_categorical(np.asarray(y_test))



print("train data :")

print(x_train_seq.shape)

print(labels_train.shape)



print("test data :")

print(x_test_seq.shape)

print(labels_test.shape)
# Find number of unique words in our tweets

vocab_size = len(word_index) + 1 # +1 is for UNKNOWN words
# Define size of embedding matrix: number of unique words x embedding dim (300)

embedding_matrix = np.zeros((vocab_size, 300))



# fill in matrix

for word, i in word_index.items():  # dictionary

    embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe

    if embedding_vector is not None:

        # add to matrix

        embedding_matrix[i] = embedding_vector # each row of matrix
#DL model: pass the encoded data to an embedding layer and use the Glove pre_trained weights, then pass the 

# output to an LSTM layer follwed by 2 dense layers.

# the optimizer used is Adam, since it achivied higher accurcies usually.



cell_size= 256

deepLModel1 = Sequential()

embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],

                           input_length = MAX_SEQUENCE_LENGTH, trainable=False)

deepLModel1.add(embedding_layer)

deepLModel1.add(LSTM(cell_size, dropout = 0.2))

deepLModel1.add(Dense(64,activation='relu'))

deepLModel1.add(Flatten())

deepLModel1.add(Dense(3, activation='softmax'))

deepLModel1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

deepLModel1.summary()
#train the model

deepLModel1_history = deepLModel1.fit(x_train_seq, labels_train, validation_split = 0.15,

                    epochs=100, batch_size=256)
# Find train and test accuracy

loss, accuracy = deepLModel1.evaluate(x_train_seq, labels_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))



loss, accuracy = deepLModel1.evaluate(x_test_seq, labels_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
predictions_test = deepLModel1.predict_classes(x_test_seq)

#print other performance measures, espically the data is unbalanced

print(classification_report(predictions_test , y_test))
#calculate the confusion matrix and plot it



cm=confusion_matrix(predictions_test , y_test)

class_names =  ['Negative', 'Neutral', 'Positive']

fig, ax = plot_confusion_matrix(conf_mat=cm,

                                colorbar=True,

                                show_absolute=True,

                                show_normed=True,

                                class_names=class_names)

plt.show()