!pip install ktrain
import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf
tf.__version__
predictor1 = ktrain.load_predictor('../input/imdbbert/')

#sample dataset to test on

data = ['Cinematography was amazing, however the story was boring. One time watch',
        'The protogonist acted very well. Director has done amazing job. Always nice to wacth Nolan movies',
        'They are both immature, selfish, and self-centered people. They hurt EVERYBODY around them playing their silly game']

predictor1.predict(data)
import os
os.chdir(r'/kaggle/working')
!git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
#loading the train dataset
data_train = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx', dtype = str)

#loading the test dataset
data_test = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx', dtype = str)

#dimension of the dataset

print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)
#printing last rows of train dataset

data_train.tail()
# text.texts_from_df return two tuples
# maxlen means it is considering that much words and rest are getting trucated
# preprocess_mode means tokenizing, embedding and transformation of text corpus(here it is considering BERT model)


(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,
                                                                   text_column = 'Reviews',
                                                                   label_columns = 'Sentiment',
                                                                   val_df = data_test,
                                                                   maxlen = 500,
                                                                   preprocess_mode = 'bert')
# name = "bert" means, here we are using BERT model.

model = text.text_classifier(name = 'bert',
                             train_data = (X_train, y_train),
                             preproc = preproc)
#here we have taken batch size as 6 as from the documentation it is recommend to use this with maxlen as 500

learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 6)
# find out best learning rate?
# learner.lr_find(max_epochs=2)
# learner.lr_plot(n_skip_beginning=2200, n_skip_end=4100)

#Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback
## After running the lr_find learnin rate with 2e-5 was having minimum loss. So using it for this dataset

learner.fit_onecycle(lr = 2e-5, epochs = 1)

predictor = ktrain.get_predictor(learner.model, preproc)


## Save the model for future use. Also use FileLink to download the files, so that you can upload it later
predictor.save('/kaggle/working/bert')
!ls bert/
FileLink(r'bert/tf_model.preproc')
#sample dataset to test on

data = ['Cinematography was amazing, however the story was boring. One time watch',
        'The protogonist acted very well. Director has done amazing job. Always nice to wacth Nolan movies',
        'They are both immature, selfish, and self-centered people. They hurt EVERYBODY around them playing their silly game']
predictor.predict(data)