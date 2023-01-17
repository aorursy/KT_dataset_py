import numpy as np 
import pandas as pd 

import os

!pip install ktrain
import tensorflow as tf
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
#loading the train dataset
ticket_data = pd.read_excel('../input/amsticketdata/TicketData.xlsx', dtype = str)
train, test = train_test_split(ticket_data, test_size=0.2)
train.head()
(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=train,
                                                                   text_column = 'Short description',
                                                                   label_columns = 'Label',
                                                                   val_df = test,
                                                                   maxlen = 320,
                                                                   preprocess_mode = 'bert')
model = text.text_classifier(name = 'bert',
                             train_data = (X_train, y_train),
                             preproc = preproc)
learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 14)
# # find out best learning rate?
learner.lr_find(max_epochs=2)
learner.lr_plot()
learner.lr_plot(n_skip_beginning=625, n_skip_end=825)
#Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback

learner.fit_onecycle(lr = 2.8e-4, epochs = 1)
predictor = ktrain.get_predictor(learner.model, preproc)

#sample dataset to test on

data = ['Job control ',
        'Job Production control  abended with a rc 8',
        'P&D Messaging/ user unable to sign in']
predictor.predict(data)
predictor.save('/kaggle/working/bert')
from IPython.display import FileLink
FileLink(r'bert/tf_model.h5')
# FileLink(r'bert/tf_model.preproc')
