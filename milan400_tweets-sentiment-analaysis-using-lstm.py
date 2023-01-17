import numpy as np 

import pandas as pd 



from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

import tensorflow as tf

from tensorflow.keras.layers import Dropout



import nltk

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train_df.head()
# Drop Nan Values

X = train_df.dropna()



# Get training data

X = train_df.drop('sentiment', axis = 1)



#  Get target label

y = train_df['sentiment']
# copying data for further processing

messages = X.copy()



# The reset_index() function is used to generate a new DataFrame or Series with the index reset

messages.reset_index(inplace = True)



#downloading stop words

nltk.download('stopwords')
# Dataset Preprocessing

ps = PorterStemmer()



corpus = []



for i in range(0, len(messages)):

    # replace with space words other than a-1, A-Z

    

    review = re.sub('[^a-zA-Z]', ' ', str(messages['text'][i]))

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
# vocabulray size

voc_size = 5000
# One Hot Encoding

onehot_repr = [one_hot(words, voc_size) for words in corpus]
# making all sentences of same length

sent_length = 30

embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)
# Finding the numberof labels

num_labels = len(set(train_df['sentiment']))
# initializing the number of features

embedding_vector_features = 40



## Creating model

model=Sequential()

model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))

model.add(LSTM(100))

model.add(Dense(num_labels,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print(model.summary())
from sklearn import preprocessing



# encode label to int

le = preprocessing.LabelEncoder()

y = le.fit_transform(y)



X_final = np.array(embedded_docs)

y_final = np.array(y)



from keras.utils import to_categorical

y_final = to_categorical(y_final)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size = 64)
def return_x_y(X):

    

    # Drop Nan Values

    X = X.fillna(0)

    

    messages = X.copy()



    messages.reset_index(inplace = True)



    # Dataset Preprocessing

    ps = PorterStemmer()



    corpus = []



    for i in range(0, len(messages)):

        # replace with space words other than a-1, A-Z



        review = re.sub('[^a-zA-Z]', ' ', str(messages['text'][i]))

        review = review.lower()

        review = review.split()



        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

        review = ' '.join(review)

        corpus.append(review)



    # vocabulray size

    voc_size = 5000



    onehot_repr = [one_hot(words, voc_size) for words in corpus]



    # Embedding Representation

    # making all sentences of same length

    sent_length = 30

    embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)



    X_final = np.array(embedded_docs)

    

    

    return X_final, X
# reading test data and pre-processing

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

X_test,X_test_drop = return_x_y(test_df)
# making prediction

y_pred_test = model.predict_classes(X_test)
submission_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
len(X_test_drop['textID']), len(y_pred_test)
df_sub = pd.DataFrame()

df_sub['id'] = X_test_drop['textID']

df_sub['text'] = X_test_drop['text']

df_sub['sentiment_predicted'] = le.inverse_transform(y_pred_test)

df_sub['sentiment_actual'] = X_test_drop['sentiment']
df_sub.to_csv('gender_submission.csv', index=False)
# Visualizing the data

df_sub.head()
#Creating a confusion matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve,auc

import seaborn as sns

import matplotlib.pyplot as plt



cm = confusion_matrix(df_sub['sentiment_actual'].values , df_sub['sentiment_predicted'].values)

cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
#Transform to df for easier plotting

final_cm = pd.DataFrame(cm, index = le.classes_,

                     columns = le.classes_

                    )
plt.figure(figsize = (5,5))

sns.heatmap(final_cm, annot = True,cmap='Greys',cbar=False)

plt.title('Emotion Classify')

plt.ylabel('True class')

plt.xlabel('Prediction class')

plt.show()