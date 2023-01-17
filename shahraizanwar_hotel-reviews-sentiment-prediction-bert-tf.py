# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

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



import random as rn
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

train_texts, test_texts, train_labels, test_labels = train_test_split(

    X.values.tolist(), y, test_size=.33, random_state=67, stratify=y)



print("Examples in train data: {}".format(len(train_texts)))

print("Examples in test data: {}".format(len(test_texts)))
from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')



seq_len = 350



train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=seq_len)

test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=seq_len)



input_ids_train = np.array(train_encodings['input_ids']) 

attention_mask_train = np.array(train_encodings['attention_mask'])



input_ids_test = np.array(test_encodings['input_ids']) 

attention_mask_test = np.array(test_encodings['attention_mask'])
# Example

exp_sen = 1



print("\nExample:\n")

print("Sentence:\n{}".format(train_texts[exp_sen]))

print("\nAfter tokenizing :\n{}".format(tokenizer.encode(train_texts[exp_sen])))

print("\nAfter padding :\n{}".format(input_ids_train[exp_sen]))
from transformers import TFDistilBertModel



bert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
# Creating bert model

inp_ids = L.Input(shape=(seq_len,), dtype=tf.int32) # Shape:(batch_size, seq_len)

attention_mask = L.Input(shape=(seq_len,), dtype=tf.int32) # Shape:(batch_size, seq_len)

last_hidden_state = bert(inp_ids,attention_mask=attention_mask)[0] # Shape:(batch_size, seq_len, 768)

out = last_hidden_state[:,0,:] # Shape:(Batch_size, 768)



bert_model = tf.keras.Model(inputs=[inp_ids, attention_mask], outputs=out)



# Passing data from bert pretrained model and extracting the final state.



print("Passing train data")

bert_output_train = bert_model.predict(

    [input_ids_train,attention_mask_train], batch_size=16, verbose=1)



print("Passing test data")

bert_output_test = bert_model.predict(

    [input_ids_test,attention_mask_test], batch_size=16, verbose=1)
print("Bert output train: {}".format(bert_output_train.shape))

print("Bert output test: {}".format(bert_output_test.shape))
seed_value = 1337

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

rn.seed(seed_value)



model = tf.keras.Sequential([

    L.Input(shape=(768)),

    L.Dense(128,activation='relu'),

    L.Dropout(0.5),

    L.Dense(5, activation="softmax")

])





model.compile(loss=SparseCategoricalCrossentropy(),

              optimizer='adam',metrics=['accuracy']

             )



model.summary()
# Passing bert output for training

history = model.fit(

    bert_output_train, train_labels, epochs=22, validation_split=0.12, batch_size=32, verbose=2)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}

)



fig.show()
fig = px.line(

    history.history, y=['accuracy', 'val_accuracy'],

    labels={'index': 'epoch', 'value': 'accuracy'}

)



fig.show()
pred = model.predict_classes(bert_output_test)
print('Accuracy: {}'.format(accuracy_score(pred, test_labels)))
print("Mean absolute error: {}".format(mean_absolute_error(pred,test_labels)))
print("Root mean square error: {}".format(np.sqrt(mean_squared_error(pred,test_labels))))
conf = confusion_matrix(test_labels, pred)



cm = pd.DataFrame(

    conf, index = [i for i in labels],

    columns = [i for i in labels]

)



plt.figure(figsize = (12,7))

sns.heatmap(cm, annot=True, fmt="d")

plt.show()
print(classification_report(test_labels, pred, target_names=labels))