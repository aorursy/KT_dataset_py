import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading required python packages and libraries

import nltk

import pandas as pd

import re

import string

import random

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from wordcloud import WordCloud,STOPWORDS
# Load the csv file using Pandas and print first 5 lines

data = pd.read_csv("../input/urduromansentiment/Roman Urdu DataSet.csv",header=None)

data.head()
# Adding column names

data.columns =['body_text','sentiment','unknown']
# Print unique values in column 1 and 2

print ('Unique values of the sentiments are', data.sentiment.unique())

print ('Unique values of the unknonwn column are', data.unknown.unique())
# 'Neative' sentiment will be most likely Negative, so it is replaced accordingly. 

data[data['sentiment']=='Neative']='Negative'
# Verify we replaced all the 'Neative'

print ('Unique values of the sentiments are', data.sentiment.unique())
# Checking Null values in the data

data.isna().sum()
# Dropping the text body row which has a null value

data = data.dropna(subset=['body_text'])
# Last column can be dropped as it does not contain any useful information. Here axis=1, means column. 

data = data.drop('unknown', axis=1)
data.head()
data.describe()
print ('Number of sentiments in each category are as below')

print (data.sentiment.value_counts())



print ('\nPerecentage sentiments in each category are as below')

print (data.sentiment.value_counts()/data.shape[0]*100)



data.sentiment.value_counts().plot(kind='bar')
# Dropping neutral sentiment sentences. 

data = data[data.sentiment != 'Neutral']
data = data.reset_index(drop=True)
data.head()
data.sentiment.value_counts().plot(kind='bar')
data.describe()
text_wordcloud = " ".join(word.lower() for word in data.body_text)

print ('There are total {} words in text provided.'.format(len(text_wordcloud)))
str2 = [] 

def freq(str): 

  

    # Break the string into list of words  

    str = str.split()          

    #str2 = [] 

  

    # Loop till string values present in list str 

    for i in str:              

  

        # Checking for the duplicacy 

        if i not in str2: 

  

            # Append value in str2 

            str2.append(i)  

              

    for i in range(0, len(str2)): 

        if(str.count(str2[i])>100): 

            print('Frequency of word,', str2[i],':', str.count(str2[i]))

            

freq(text_wordcloud)
print ('Number of unique words in vocabulary are',len(str2))
UrduStopWordList = [line.rstrip('\n') for line in open('../input/urdustopwords/stopwords.txt')]



print (UrduStopWordList)
stopwords_with_urdu = set(STOPWORDS)

stopwords_with_urdu.update(UrduStopWordList)





wordcloud = WordCloud(stopwords=stopwords_with_urdu,

                      background_color='white',

                      width=3000,

                      height=2500

                     ).generate(text_wordcloud)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
neg_text_wordcloud = " ".join(word.lower() for word in data[data['sentiment']=='Negative']['body_text'])

print ('There are total {} words in sentences with negative sentiments.'.format(len(neg_text_wordcloud)))
# Plotting Plotting words in setences with negative sentiment

wordcloud = WordCloud(stopwords=stopwords_with_urdu,

                      background_color='white',

                      width=3000,

                      height=2500

                     ).generate(neg_text_wordcloud)

plt.figure(1,figsize=(12, 12))

plt.title('Negative Sentiment Words',fontsize = 20)

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
pos_text_wordcloud = " ".join(word.lower() for word in data[data['sentiment']=='Positive']['body_text'])

print ('There are total {} words in text with positive sentements.'.format(len(pos_text_wordcloud)))
# Plotting words in positive sentiment sentences



wordcloud = WordCloud(stopwords=stopwords_with_urdu,

                      background_color='white',

                      width=3000,

                      height=2500

                     ).generate(pos_text_wordcloud)

plt.figure(1,figsize=(12, 12))

plt.title('Positive Sentiment Words',fontsize = 20)

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
def clean_text(text):

    #Change each character to lowercase and avoid any punctuation. Finally join word back. 

    text = "".join([char.lower() for char in text if char not in string.punctuation])

    

    # Use non word characters to split the sentence

    tokens = re.split('\W+', text)



    # Remove the stop words - commonly used words such as I, we, you, am, is etc in Urdu language 

    # that do not contribute to sentiment. 

    text = [word for word in tokens if word not in stopwords_with_urdu]

    return text
data['body_text'] = data['body_text'].apply(lambda x: clean_text(x))
data.head()
y = data['sentiment']

y = np.array(list(map(lambda x: 1 if x=="Positive" else 0, y)))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data['body_text'],y, test_size=0.2,random_state=42,stratify=data['sentiment'])
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)

X_test = tokenizer.texts_to_matrix(X_test)
X_train.shape,X_test.shape
n_words = X_train.shape[1]

print (n_words)
vocab_size = len(tokenizer.word_index) + 1

print (vocab_size)
from keras.preprocessing.text import Tokenizer

from keras.utils.vis_utils import plot_model

from keras.models import Sequential

from keras.layers import Dense



def define_model(n_words):

    # define network

    model = Sequential()

    model.add(Dense(100, input_shape=(n_words,), activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(25, activation='relu'))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # compile network

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model

    model.summary()

    plot_model(model, to_file='model.png', show_shapes=True)

    return model



model = define_model(n_words)

# fit network

model.fit(X_train, y_train, epochs=50, verbose=2)

# evaluate

loss, acc = model.evaluate(X_test, y_test, verbose=0)

print('Test Accuracy: %f' % (acc*100))
y_pred = model.predict(X_test)
from sklearn.metrics import precision_recall_fscore_support as score, roc_auc_score

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from mlxtend.plotting import plot_confusion_matrix



print ('Classification Report for Classifier:\n',classification_report(y_test, y_pred.round(),digits=3))

print ('\nConfussion matrix for Classifier:\n'),confusion_matrix(y_test, y_pred.round())
cm = confusion_matrix(y_test,y_pred.round())

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)

plt.xticks(range(2), ['Negative','Positive'], fontsize=20)

plt.yticks(range(2),['Negative','Positive'] , fontsize=20)

plt.show()
k = random.randint(0,data.shape[0])

message = data['body_text'][k]

message
data['sentiment'][k]
X_predict_vect = tokenizer.texts_to_matrix(message)
y_message =max(model.predict(X_predict_vect))

y_message_rnd = np.round(y_message)



if y_message_rnd==1:

    print ("Positive")

else:

    print ("Negative")