# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

from keras import Sequential

from keras import layers

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import sqlite3

import re

import string

from scipy import signal

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
cn = sqlite3.connect('../input/database.sqlite')

query = "SELECT * FROM Tweets"

df = pd.read_sql_query(query, cn)

df.head()
output_for_ml_task = df[['airline_sentiment','text']]

# Remane header

header = ['tweet_id', 'sentiment', 'sentiment_confidence','negative_reason', 'negative_reason_confidence', 'airline_name','sentiment_gold', 'user_name', 'negative_reason_gold','retweet_count', 'text', 'tweet_coordinate', 'date','tweet_location', 'user_timezone']

df.columns = header

# Map every string to a weighted number

mapping = {'negative': -1,

           'neutral': 0,

           'positive': 1,

           '' : np.nan}



# Apply mapping

df.replace({'sentiment': mapping}, inplace=True)



df.replace({'sentiment_gold': mapping}, inplace=True)



# Drop data ids and confidence

df.drop(['tweet_id','sentiment_confidence', 'negative_reason_confidence'], axis=1, inplace=True)



# Convert to datetime

df['date'] = pd.to_datetime(df['date'])

# Set the index

df.set_index('date', inplace=True)



df['negative_reason'].replace('', 'None',  inplace=True)

df['negative_reason_gold'].replace('', 'None',  inplace=True)

df['tweet_location'].replace('', 'None',  inplace=True)

df['tweet_coordinate'].replace('', np.nan,  inplace=True)



df.head()
df.sentiment.value_counts()
df.negative_reason.value_counts()[1::]

pd.Series(df["negative_reason"]).value_counts()[1::].plot(kind = "bar",

                                                           figsize=(24,12),

                                                           title = "Negative Reasons", 

                                                           fontsize=20)

plt.xlabel('Negative Reasons', fontsize=20)

plt.ylabel('# of negative reasons', fontsize=20)
def plot_sub_negative_reason_for_airline(Airline):

    airline = df[df['airline_name']==Airline]

    count = airline['negative_reason'].value_counts()

    Index = [i+1 for i in range(len(count)-1)]

    label = list(count.keys())

    x = label.index('None')

    label.pop(x)

    count = list(count)

    count.pop(x)

    plt.bar(Index,count,width=0.5)

    plt.xticks(Index, label, rotation='vertical')

    plt.title('Negative reasons for %s'%Airline)

airline_name = df['airline_name'].unique()

plt.figure(1,figsize=(16,20))



plt.subplots_adjust(hspace=0.4)



for i in range(len(airline_name)):

    plt.subplot(len(airline_name)/3,3,i+1)

    plot_sub_negative_reason_for_airline(airline_name[i])
print("Day   # of retweet \n%s"%df.retweet_count.value_counts())
def plot_sub_retweet_per_day_for_airline(Airline):

    airline = df[df['airline_name']==Airline]

    count = airline['retweet_count'].index.day.value_counts()

    Index = airline.index.day.drop_duplicates()

    plt.plot(Index,count)

    plt.xticks(Index, list(Index))

    plt.title('Retweet per day for %s'%Airline)

airline_name = df['airline_name'].unique()

plt.figure(1,figsize=(18,18))

for i in range(len(airline_name)):

    plt.subplot(len(airline_name)/3,3,i+1)

    plot_sub_retweet_per_day_for_airline(airline_name[i])
def plot_sub_sentiment_for_airline(Airline):

    airline = df[df['airline_name']==Airline]

    count = airline['sentiment'].value_counts()

    Index = [i for i in range(len(count))]

    plt.bar(Index,count,width=0.5)

    plt.xticks(Index, list(mapping.keys()))

    plt.title('Sentiment for %s'%Airline)

airline_name = df['airline_name'].unique()

plt.figure(1,figsize=(18,18))

for i in range(len(airline_name)):

    plt.subplot(len(airline_name)/3,3,i+1)

    plot_sub_sentiment_for_airline(airline_name[i])
x = [x for x in df['tweet_coordinate'].tolist() if x]

x = [incom for incom in x if str(incom) != 'nan']

coor = []

for i in x:

    la, lo = i.replace('[','').replace(']','').split(',')

    la, lo = float(la), float(lo)

    if la == 0 and lo == 0:

        continue

    coor.append([float(la), float(lo)])
latitude, longitude = list(map(list, zip(*coor)))





from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



# How much to zoom from coordinates (in degrees)

zoom_scale = 1



bbox = [np.min(latitude)-zoom_scale,np.max(latitude)+zoom_scale, 

        np.min(longitude)-zoom_scale,np.max(longitude)+zoom_scale]



plt.figure(figsize=(24,12))

# Define the projection, scale, the corners of the map, and the resolution.

m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],\

            llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='i')



# Draw coastlines and fill continents

m.drawcoastlines()

m.fillcontinents(color='white')



# build and plot coordinates onto map

x,y = m(longitude,latitude)

m.plot(x,y,'r*',markersize=10)

plt.title("Tweet Location")

plt.show()
distance = []

central_mass_latitude, central_mass_longitude = sum(latitude)/len(latitude), sum(longitude)/len(longitude)

number_of_positions = len(latitude)

_=[distance.append(np.sqrt((central_mass_latitude - latitude[i])**2 + (central_mass_longitude - longitude[0])**2)) for i in range(number_of_positions)]
def central_limit_theorem(rv):

    return (rv - np.average(rv))/(np.std(rv)/len(rv)**.5)
def handle_gaussian(l):

    # Make list to be as gaussian dist.

    l = l.tolist()

    l.sort()

    n = len(l)

    i = int(n/2)

    halve1 = l[0:i]

    halve2 = l[i+1:n-1]

    halve2.reverse()

    return halve1 + halve2
data = np.array(distance,dtype='float64')
from scipy.stats import shapiro

stat, p = shapiro(data)

print(p)
gaussian = signal.gaussian(len(data), std=np.std(central_limit_theorem(data)))



fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)



ax.plot(central_limit_theorem(handle_gaussian(data)), label='Data Curve', linestyle='--')

ax.plot(gaussian,label="Gaussian's Curve", linewidth=4)

ax.legend()



plt.show()
data = np.array(df.negative_reason.value_counts().tolist(),dtype='float64')

from scipy.stats import shapiro

stat, p = shapiro(data)

print(p)
gaussian = signal.gaussian(len(data), std=np.std(central_limit_theorem(data)))



fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)



ax.plot(central_limit_theorem(handle_gaussian(data)), label='Data Curve', linestyle='--')

ax.plot(gaussian,label="Gaussian's Curve", linewidth=4)

ax.legend()



plt.show()
data = np.array(df['retweet_count'].value_counts().tolist(),dtype='float64')

from scipy.stats import shapiro

stat, p = shapiro(data)

print(p)
gaussian = signal.gaussian(len(data), std=np.std(central_limit_theorem(data)))



fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)



ax.plot(central_limit_theorem(handle_gaussian(data)), label='Data Curve', linestyle='--')

ax.plot(gaussian,label="Gaussian's Curve", linewidth=4)

ax.legend()



plt.show()
from scipy.stats import spearmanr

data1, data2 = np.array(df.retweet_count.tolist()), np.array(df.sentiment.tolist())

corr, p = spearmanr(data1, data2)

print(p)
gaussian = signal.gaussian(len(data), std=np.std(central_limit_theorem(data)))



fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)



ax.plot(handle_gaussian(data1), label='Retweet Count Curve', linestyle='--')

ax.plot(handle_gaussian(data2),label="Sentiment Curve", linestyle=':')

ax.legend()



plt.show()
percentage = 80/100

length_of_data = len(df.text.tolist())

Threshold = int(length_of_data*percentage)

input_data = df.text.tolist()

X_train, X_test  = np.array(input_data[0:Threshold]), np.array(input_data[Threshold + 1:length_of_data])

Y_train, Y_test = np.array(output_for_ml_task[0:Threshold]['airline_sentiment'].tolist()), np.array(output_for_ml_task[Threshold + 1:length_of_data]['airline_sentiment'].tolist())

print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(X_train.shape), 

      "\nTest set: \t\t{}".format(X_test.shape))
# Preprocessing and tokenizing

def preprocessing(line):

    line = line.lower()

    line = re.sub(r"[{}]".format(string.punctuation), " ", line)

    return line

tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing)

tfidf = tfidf_vectorizer.fit_transform(X_train)

kmeans = KMeans(n_clusters=3).fit(tfidf)
transformation = ['negative','positive','neutral']

num_transformation = [-1,1,0]

# Test the model

transformation[kmeans.predict(tfidf_vectorizer.transform(['bad travel'])).tolist()[0]]
prediction = kmeans.predict(tfidf_vectorizer.transform(X_train))

Y_model = [transformation[i] for i in prediction]

accuracy = sum(np.array(Y_train) == np.array(Y_model)) / len(np.array(Y_train))

print("Training Accuracy: {:.4f}".format(accuracy))



prediction = kmeans.predict(tfidf_vectorizer.transform(X_test))

Y_model = [transformation[i] for i in prediction]

accuracy = sum(np.array(Y_test) == np.array(Y_model)) / len(np.array(Y_test))

print("Testing Accuracy: {:.4f}".format(accuracy))
plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

# Encode Y to be numerical value

y_train_ = [mapping[i] for i in Y_train]

y_test_ = [mapping[i] for i in Y_test]

print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)



X_train_ = tokenizer.texts_to_sequences(X_train)

X_test_ = tokenizer.texts_to_sequences(X_test)



vocab_size = len(tokenizer.word_index) + 1
from keras.preprocessing.sequence import pad_sequences



embedding_dim = 75

maxlen = 100

X_train_ = pad_sequences(X_train_, padding='post', maxlen=maxlen)

X_test_ = pad_sequences(X_test_, padding='post', maxlen=maxlen)
from keras.layers import Activation

from keras.utils.generic_utils import get_custom_objects



# My activation function

def custom_activation(x):

    return x/(1 + abs(x))



get_custom_objects().update({'custom_activation': Activation(custom_activation)})
model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, trainable=False))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation=Activation(custom_activation)))

model.add(layers.Dense(1, activation='tanh'))



model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])



model.summary()
history = model.fit(X_train_, y_train_,

                    epochs=50,

                    validation_data=(X_test_, y_test_),

                    batch_size=10)
loss, accuracy = model.evaluate(X_train_, y_train_, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))



loss, accuracy = model.evaluate(X_test_, y_test_, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
model.predict(pad_sequences(tokenizer.texts_to_sequences(['bad travel']), padding='post', maxlen=maxlen))[0][0]
error_analysis = []

y_kMean = []

y_deep = []

for i,w in enumerate(X_test):

    

    desired = y_test_[i]

    kMean = num_transformation[kmeans.predict(tfidf_vectorizer.transform([w])).tolist()[0]]

    deep = model.predict(pad_sequences(tokenizer.texts_to_sequences([w]), padding='post', maxlen=maxlen))[0][0]

    

    y_kMean.append(kMean)

    y_deep.append(deep)

    error_analysis.append([w,desired,kMean,deep])

error_analysis_frame = pd.DataFrame(error_analysis, columns=['Text','Desired', 'K_Mean', 'Deep_model'])

error_analysis_frame.head()
def make_roc(y, score):

    roc_x = []

    roc_y = []

    min_score = min(score)

    max_score = max(score)

    thr = np.linspace(min_score, max_score, 30)

    FP=0

    TP=0

    N = sum(y)

    P = len(y) - N



    for (i, T) in enumerate(thr):

        for i in range(0, len(score)):

            if (score[i] > T):

                if (y[i]==1):

                    TP = TP + 1

                if (y[i]==0):

                    FP = FP + 1

        roc_x.append(FP/float(N))

        roc_y.append(TP/float(P))

        FP=0

        TP=0



    return roc_x, roc_y
roc_x, roc_y = make_roc(error_analysis_frame.Desired, error_analysis_frame.K_Mean)

plt.plot(roc_x,roc_y, label='ROC for K-Mean')

plt.title('Receiver Operating Characteristic')

plt.legend()

plt.show()
roc_x, roc_y = make_roc(error_analysis_frame.Desired, error_analysis_frame.Deep_model)

plt.plot(roc_x,roc_y, label='ROC for deep model')

plt.title('Receiver Operating Characteristic')

plt.legend()

plt.show()