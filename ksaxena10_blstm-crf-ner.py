# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install -q tensorflow_gpu>=2.0

!pip install ktrain

%reload_ext autoreload

%autoreload 2

%matplotlib inline

import os

os.environ['DISABLE_V2_BEHAVIOR'] = '1'
import tensorflow as tf; print(tf.__version__)
import ktrain

from ktrain import text
DATAFILE = './../input/entity-annotated-corpus/ner_dataset.csv'

(trn, val, preproc) = text.entities_from_txt(DATAFILE,

                                             embeddings='word2vec',

                                             sentence_column='Sentence #',

                                             word_column='Word',

                                             tag_column='Tag', 

                                             data_format='gmb')
text.print_sequence_taggers()
model = text.sequence_tagger('bilstm-crf', preproc)
learner = ktrain.get_learner(model, train_data=trn, val_data=val)
# find good learning rate

#learner.lr_find()             # briefly simulate training to find good learning rate

#learner.lr_plot()             # visually identify best learning rate
learner.fit(1e-3, 1)
learner.validate(class_names=preproc.get_classes())
learner.view_top_losses(n=1)
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.predict('As of 2019,Narendra modi has been prime minister of india.')
