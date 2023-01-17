#Install Text Processing Library

!pip install texthero -q
# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# TensorFlow

import tensorflow as tf

import tensorflow_hub as hub



#Library for Text Processing

import texthero as hero



#Sk Learn Library

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



#Garbage

import gc



#Warnings

import warnings

warnings.filterwarnings("ignore")



#Tabulate

from tabulate import tabulate
# Raw Data

url = '../input/fakenewsdetection/news.csv'

raw_data = pd.read_csv(url, header='infer')



#For Simplicity we'll focus on Title & Label columns

data = raw_data[['title','label']]
print("Total Records: ",data.shape[0])
#Creating a new column with processed text

data['title_clean'] = data['title'].pipe(hero.clean)
#Encode the Label to convert it into numerical values [Fake = 0; Real = 1]

lab_enc = LabelEncoder()



#Applying to the dataset

data['label'] = lab_enc.fit_transform(data['label'])
#Inspect

data.head()
# Data Split with Original Title Split into training[90%] & test[10%]

x_train,x_test,y_train,y_test = train_test_split(data['title'], data.label, test_size=0.1, random_state=0)



# Data Split with Cleaned Title Split into training[90%] & test[10%]

xc_train,xc_test,yc_train,yc_test = train_test_split(data['title_clean'], data.label, test_size=0.1, random_state=0)
# Pre-Trained Text Embedding Model & Layer Definition

Embed = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'

Trainable_Module = True

hub_layer = hub.KerasLayer(Embed, input_shape=[], dtype=tf.string, trainable=Trainable_Module)



# Build Model (Original Title Text)

model = tf.keras.Sequential()

model.add(hub_layer)           #pre-trained text embedding layer

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))



# Build Model (Cleaned Title Text)

model_c = tf.keras.Sequential()

model_c.add(hub_layer)           #pre-trained text embedding layer

model_c.add(tf.keras.layers.Dense(16, activation='relu'))

model_c.add(tf.keras.layers.Dense(1))





print(" -- Original Title Text Model Summary --")

model.summary()

print('\n')

print(" -- Cleaned Title Text Model Summary --")

model_c.summary()
# Model Compile (Original Title Text)

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy']

              )



# Model Compile (Cleaned Title Text)

model_c.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy']

              )
EPOCHS = 20           #feel free to change this

BATCH_SIZE = 256      #feel free to change this



history = model.fit(x_train,y_train, batch_size = BATCH_SIZE,

                    epochs = EPOCHS, validation_split= 0.1,

                    verbose=1)
EPOCHS = 20           #feel free to change this

BATCH_SIZE = 256      #feel free to change this



history_c = model_c.fit(xc_train,yc_train, batch_size = BATCH_SIZE,

                    epochs = EPOCHS, validation_split= 0.1,

                    verbose=1)
#Original Title Text

accr = model.evaluate(x_test,y_test, verbose=0)

#print("Model Accuracy on Original Title Text: ",'{:.2%}'.format(accr[1]))



#Cleaned Title Text

accr_c = model_c.evaluate(xc_test,yc_test, verbose=0)

#print("Model Accuracy on Cleaned Title Text: ",'{:.2%}'.format(accr_c[1]))



tab_data = [ [ "Model Trained on Original Title Text", '{:.2%}'.format(accr[0]), '{:.2%}'.format(accr[1]) ],

             [ "Model Trained on Cleaned Title Text", '{:.2%}'.format(accr_c[0]), '{:.2%}'.format(accr_c[1]) ]

           ]



print(tabulate(tab_data, headers=['','LOSS','ACCURACY'], tablefmt='pretty'))
# 5 Randomly Sampled Records from Original Title Text

pred_df = pd.DataFrame({ 'Random_News_Titles' : x_test.sample(n=5, random_state=1) })

pred_df.reset_index(inplace=True,drop=True)



# Function to convert numerical label to string

def label2str(x):

    if x == 0:

        return "Fake"

    else:

        return "Real"



# Function to highlight mismatching column values

def highlight(x):

    y = 'yellow'

    g = 'green'



    mismtch = x['ModelPred_OgTxt'] != x['ModelPred_CleandTxt']

    mtch = x['ModelPred_OgTxt'] == x['ModelPred_CleandTxt']

    



    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)

    

    df1['ModelPred_OgTxt'] = np.where(mismtch, 'background-color: {}'.format(y), df1['ModelPred_OgTxt'])

    df1['ModelPred_OgTxt'] = np.where(mtch, 'background-color: {}'.format(g), df1['ModelPred_OgTxt'])

    

    df1['ModelPred_CleandTxt'] = np.where(mismtch, 'background-color: {}'.format(y), df1['ModelPred_CleandTxt'])

    df1['ModelPred_CleandTxt'] = np.where(mtch, 'background-color: {}'.format(g), df1['ModelPred_CleandTxt'])

    

    return df1    

    



# Add a new column with predictions

pred_df['ModelPred_OgTxt'] = [label2str(x) for x in model.predict_classes(pred_df.Random_News_Titles)]

pred_df['ModelPred_CleandTxt'] = [label2str(x) for x in model_c.predict_classes(pred_df.Random_News_Titles)]





# Applying the highlight style to the prediction dataframe

pred_df.style.apply(highlight, axis=None)