!pip install ktrain
# getting larget Dataset
!git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git
import numpy as np
import pandas as pd
import ktrain
import tensorflow as tf
from ktrain import text
data_train = pd.read_excel('./IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx',dtype=str)
data_train.head()
data_test = pd.read_excel('./IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx',dtype=str)
data_test.head()
(xtrain,ytrain),(xtest,ytest),preprocess = text.texts_from_df(train_df = data_train,
                                                              text_column='Reviews',
                                                              label_columns='Sentiment',
                                                              val_df=data_test,
                                                              maxlen=400,
                                                              preprocess_mode='bert' )

xtrain[0].shape
model = text.text_classifier(name='bert',
                             train_data = (xtrain,ytrain),
                            #  val_data=(xtest,ytest),
                             preproc=preprocess
                             )
# Get the Learning Rate
learner = ktrain.get_learner(model=model,
                             train_data = (xtrain,ytrain),
                             val_data=(xtest,ytest),
                             batch_size =6)
# this will take days to learn

# learner.lr_find()
# learner.lr_plot()

# Optimal learning rate is for this model 2e5

learner.fit_onecycle(lr=2e-5,epochs=1)
learner
predictor = ktrain.get_predictor(learner.model,preprocess)
data = ['The movie was really bad I do not like it ',
        'the film was really sucked, I want my money back',
        'what a beautiful movie'        
        ]
predictor.predict(data)
predictor.save('/contents/bert')
