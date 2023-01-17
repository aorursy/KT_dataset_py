import os

import numpy as np

import pandas as pd

from fastai.text import * 

from fastai.callbacks import CSVLogger

from shutil import copyfile
#Setting path for learner

path = Path(os.path.abspath(os.curdir))
# Create directory

dirName = 'models'

 

try:

    # Create target Directory

    os.mkdir(dirName)

    print("Directory " , dirName ,  " Created ") 

except FileExistsError:

    print("Directory " , dirName ,  " already exists")
#copying files into working path

modelpath = Path('../input/awd-lstm-1')



copyfile(modelpath/"models/final.pth", path/"models/final.pth")

copyfile(modelpath/"models/ft_enc1.pth", path/"models/ft_enc1.pth")

copyfile(modelpath/"data_clas_export.pkl", path/"data_clas_export.pkl")

copyfile(modelpath/"data_lm_export.pkl", path/"data_lm_export.pkl")
"""

#reading into pandas and renaming columns for easier api access

filepath = Path('../input/quora-insincere-questions-classification')

trn = pd.read_csv(filepath/'train.csv')

tst = pd.read_csv(filepath/'test.csv')



#For training language model, using both train and test data for more data to learn from

df = pd.concat([trn,tst], sort=False)

df.rename(columns={'target':'label', 'question_text':'text'},inplace=True)

df = df[['label','text']]

df.head(2)



#Simple 90-10 split into train/validation set

train = df[:int(len(df)*.9)]

valid = df[int(len(df)*.9):]



"""
# Language model data

#data_lm = TextLMDataBunch.from_df(path, train, valid)

#data_lm.save('data_lm_export.pkl')
data_lm = load_data(path, 'data_lm_export.pkl')
#Training a language model, i.e. to predict the next few words

#learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, callback_fns=[partial(CSVLogger, append=True)])
#learn.lr_find()

#learn.recorder.plot()
#learn.fit_one_cycle(4, 1e-2)

#learn.save('fit_head'); learn.load('fit_head')
#learn.unfreeze()

#learn.lr_find(); learn.recorder.plot()
#learn.fit_one_cycle(4, 1e-3)

#learn.save_encoder('ft_enc1')
#learn.predict("Why are people", n_words=10)
"""

trn.rename(columns={'target':'label', 'question_text':'text'},inplace=True)

df = trn[['label','text']]



train = df[:int(len(df)*.80)]

valid = df[int(len(df)*.80):]

"""
data_clas = load_data(path, 'data_clas_export.pkl', bs=16)

# Classifier model data

#data_clas = TextClasDataBunch.from_df(path, train, valid, vocab=data_lm.train_ds.vocab, bs=16)

#data_clas.save('data_clas_export.pkl') ; data_clas = load_data(path, 'data_clas_export.pkl', bs=16)
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=.3, metrics=[accuracy, FBeta(beta=1, average='binary')],

                               callback_fns=[partial(CSVLogger, append=True)])

learn.load_encoder('ft_enc1') #encoder from first training has 42% accuracy in predicting next word
learn.fit_one_cycle(4, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(4, slice(1e-3/(2.6**4), 1e-3))
learn.freeze_to(-3)

learn.fit_one_cycle(4, slice(1e-4/(2.6**4), 1e-4))
learn.unfreeze()

learn.fit_one_cycle(4, slice(1e-5/(2.6**4),1e-5))
learn.predict("Why are foreigners so lazy?")
learn.predict("When was SMU founded and why?")
learn.save('clas-1')