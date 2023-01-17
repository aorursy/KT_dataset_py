#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
!pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
#importing database
asgmt1df = pd.read_excel('../input/merklesokrati/Data Analyst Assignment (1).xlsx')
#printing out the first five rows of the data
asgmt1df.head()
asgmt1df.pop('Date')
asgmt1df.pop('product')
asgmt1df.pop('phase')
asgmt1df['age'].unique()
#one hot encoding the data
campaign_platform = {'Google Ads': 0,'Facebook Ads': 1} 
age = {'18-24': 0,'25-34': 1,'35-44': 2,'45-54': 3,'55-64': 4,'65 or more': 5}
campaign_type = {'Search': 0,'Conversions': 1} 
communication_medium = {'Search Keywords': 0,'Creative': 1}
subchannel = {'Brand': 0,'Competitor': 1,'Generic': 2,'Facebook Ads' : 3} 
device = {'Desktop': 0,'Mobile': 1,'Tablet': 2,'device' : 3} 
audience_type = {"'-": 0,"Audience 1": 1,"Audience 2": 2,"Audience 3" : 3} 
creative_type = {"'-": 0,"Carousal": 1,"Image": 2} 
creative_name = {"'-": 0,"Carousal": 1,"Click": 2,"Girl" : 3} 
asgmt1df['age'] = asgmt1df['age'].replace(to_replace= 'Undetermined', value= '35-44')
asgmt1df.age = [age[item] for item in asgmt1df.age]
asgmt1df.campaign_platform = [campaign_platform[item] for item in asgmt1df.campaign_platform] 
asgmt1df.campaign_type = [campaign_type[item] for item in asgmt1df.campaign_type] 
asgmt1df.communication_medium = [communication_medium[item] for item in asgmt1df.communication_medium] 
asgmt1df.subchannel = [subchannel[item] for item in asgmt1df.subchannel] 
asgmt1df.device = [device[item] for item in asgmt1df.device] 
asgmt1df.audience_type = [audience_type[item] for item in asgmt1df.audience_type] 
asgmt1df.creative_type = [creative_type[item] for item in asgmt1df.creative_type] 
asgmt1df.creative_name = [creative_name[item] for item in asgmt1df.creative_name] 
#spliting the data
X_train, X_test, y_train, y_test = train_test_split(asgmt1df.loc[:, asgmt1df.columns != 'link_clicks'], asgmt1df['link_clicks'], test_size=0.2)
#model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
model.summary()
EPOCHS = 1000

history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])