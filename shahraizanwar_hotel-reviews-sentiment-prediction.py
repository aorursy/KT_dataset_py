# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.layers as L

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.optimizers import Adam



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt

from wordcloud import WordCloud 

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

import seaborn as sns





import numpy as np 

import pandas as pd



import nltk

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import TweetTokenizer

from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords



import random as rn



import re
seed_value = 1337

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

rn.seed(seed_value)
data = pd.read_csv('../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')

data.head()
print('Examples in data: {}'.format(len(data)))
data.isna().sum()
class_dist = data['Rating'].value_counts()



def ditribution_plot(x,y,name):

    fig = go.Figure([

        go.Bar(x=x, y=y)

    ])



    fig.update_layout(title_text=name)

    fig.show()
ditribution_plot(x= class_dist.index, y= class_dist.values, name= 'Class Distribution')
def wordCloud_generator(data, title=None):

    wordcloud = WordCloud(width = 800, height = 800,

                          background_color ='black',

                          min_font_size = 10

                         ).generate(" ".join(data.values))

    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud, interpolation='bilinear') 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 

    plt.title(title,fontsize=30)

    plt.show() 
wordCloud_generator(data['Review'], title="Most used words in reviews")
X = data['Review'].copy()

y = data['Rating'].copy()
def data_cleaner(review):

    

    # remove digits

    review = re.sub(r'\d+',' ', review)

    

    #removing stop words

    review = review.split()

    review = " ".join([word for word in review if not word in stop_words])

    

    #Stemming

    #review = " ".join([ps.stem(w) for w in review])

    

    return review



ps = PorterStemmer() 

stop_words = stopwords.words('english')



X_cleaned = X.apply(data_cleaner)

X_cleaned.head()
length_dist = [len(x.split(" ")) for x in X_cleaned]

plt.hist(length_dist)

plt.show()
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_cleaned)



X = tokenizer.texts_to_sequences(X_cleaned)



max_length = max([len(x) for x in X])

vocab_size = len(tokenizer.word_index)+1

exp_sen = 1



print("Vocabulary size: {}".format(vocab_size))

print("max length of sentence: {}".format(max_length))

print("\nExample:\n")

print("Sentence:\n{}".format(X_cleaned[exp_sen]))

print("\nAfter tokenizing :\n{}".format(X[exp_sen]))



X = pad_sequences(X, padding='post', maxlen=350)

print("\nAfter padding :\n{}".format(X[exp_sen]))
encoding = {1: 0,

            2: 1,

            3: 2,

            4: 3,

            5: 4

           }



labels = ['1', '2', '3', '4', '5']

           

y = data['Rating'].copy()

y.replace(encoding, inplace=True)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=67, stratify=y

)
# hyper parameters

EPOCHS = 3

BATCH_SIZE = 100

embedding_dim = 16

units = 76



model = tf.keras.Sequential([

    L.Embedding(vocab_size, embedding_dim, input_length=X.shape[1]),

    L.Bidirectional(L.LSTM(units,return_sequences=True)),

    #L.LSTM(units,return_sequences=True),

    L.Conv1D(64,3),

    L.MaxPool1D(),

    L.Flatten(),

    L.Dropout(0.5),

    L.Dense(128, activation="relu"),

    L.Dropout(0.5),

    L.Dense(64, activation="relu"),

    L.Dropout(0.5),

    L.Dense(5, activation="softmax")

])





model.compile(loss=SparseCategoricalCrossentropy(),

              optimizer='adam',metrics=['accuracy']

             )



model.summary()
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.12, batch_size=BATCH_SIZE, verbose=2)
fig = px.line(

    history.history, y=['accuracy', 'val_accuracy'],

    labels={'index': 'epoch', 'value': 'accuracy'}

)



fig.show()
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}

)



fig.show()
pred = model.predict_classes(X_test)
print('Accuracy: {}'.format(accuracy_score(pred, y_test)))
print("Mean absolute error: {}".format(mean_absolute_error(pred,y_test)))
print("Root mean square error: {}".format(np.sqrt(mean_squared_error(pred,y_test))))
conf = confusion_matrix(y_test, pred)



cm = pd.DataFrame(

    conf, index = [i for i in labels],

    columns = [i for i in labels]

)



plt.figure(figsize = (12,7))

sns.heatmap(cm, annot=True, fmt="d")

plt.show()
print(classification_report(y_test, pred, target_names=labels))