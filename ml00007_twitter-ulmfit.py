#importing the following libraries-

import numpy as np

import pandas as pd

import seaborn as sns

import re

from pathlib import Path

from sklearn.model_selection import train_test_split

from fastai.text import *

# we will be using the fast.ai library to build the ULMFiT model as there are specific methods in the fast.ai library for the different techniques involved in ULMFiT
#loading data from Kaggle



data_path=Path('../input/twitter-airline-sentiment/')

input_name='Tweets.csv'

input_path=data_path/input_name

df_input=pd.read_csv(input_path)

print("The shape of the input csv file containing all tweets is",df_input.shape)

# some example of entries in the input data table

df_input.sample(10,random_state=0)
# we only require rge airline_sentiment and text row for our input and targetted output, thus we select just those columns

df_input_crop=df_input[['airline_sentiment','text']]

df_input_crop.sample(10)
#before, we start dividing the dataset into train/test/validation sets, we need to know whether the data is class imbalanced

#class imbalance can occur based on the sentiment class or airline class-



print('The distribution of data according to the different sentiment classes is:\n',df_input_crop['airline_sentiment'].value_counts())

print('The distribution of data according to the different sentiment classes is:\n',df_input['airline'].value_counts())

# to visualise the class imbalance due to both sentiment and airline class, we will use bar chart-

sns.countplot(y='airline', hue='airline_sentiment', data=df_input)








regex=r"@(VirginAmerica|united|SouthwestAir|Delta|USAirways|AmericanAir|JetBlue)"

def replace(text):

    return re.sub(regex, '@airline',text, flags=re.IGNORECASE)

df_input_crop['text']=df_input_crop['text'].apply(replace)

df_input_crop['text'].sample(10)



# now all the airlines names have been replaced with the word 'airline'







train_data,valid_data=train_test_split(df_input_crop, test_size=0.2)
moms=(0.8,0.7)

wd=0.1
#fast.ai needs a path to work on

fastai_path=Path('./').resolve()

fastai_path
# Here TextLMDataBunch formats the input data such that data can be called immediately for use in model training

input_data=TextLMDataBunch.from_df(fastai_path, train_data,valid_data)

# input_data.save('input_data.pkl')
#learn_process is our model 

learn_process=language_model_learner(input_data, AWD_LSTM, drop_mult=0.3)

#learning process occurs using wikipedia data

learn_process.freeze()
#lr_find helps to fine a good learning rate

learn_process.lr_find()

#plots the losses over a range of learning rates

learn_process.recorder.plot()
#from the plot above we can see that 5e-01 is the learning rate with minimum loss

# we will take our max learning rate in 1cyle to be tenth of that: 5e-02

learn_process.fit_one_cycle(1,5e-02,moms=moms,wd=wd)
learn_process.unfreeze()
learn_process.save_encoder('encoder')
trainNvalid, test=train_test_split(df_input_crop, test_size=0.2)
train, valid=train_test_split(trainNvalid, test_size=0.2)
# formatting data such that it can be easily fed into model

data_classified= TextClasDataBunch.from_df(fastai_path, train, valid, test_df=test, vocab= input_data.train_ds.vocab, text_cols='text', label_cols='airline_sentiment', bs=32)
data_classified.show_batch()
# updating our model learn_process with classifier tuning

learn_process=text_classifier_learner(data_classified, AWD_LSTM, drop_mult=0.5)
learn_process.load_encoder('encoder')
# freeze all layers before unfreezing gradually in the following step

learn_process.freeze()

learn_process.lr_find()

learn_process.recorder.plot()
# 3e-01/10

lr=3.0E-02

learn_process.fit_one_cycle(1,lr,moms=moms, wd=wd)
# here we begin the gradual unfreezing strategy where we unfreeze layer by layer in a cumulative fashion

learn_process.freeze_to(-2)

lr/=2

# 2.6 was found be an optimal factor from experimentation (Howard and Ruder)

learn_process.fit_one_cycle(1, slice(lr/(2.6**4),lr),moms=moms, wd=wd)
learn_process.freeze_to(-3)

lr/=2

learn_process.fit_one_cycle(1, slice(lr/(2.6**4),lr),moms=moms, wd=wd)
#unfreeze all layers 

learn_process.unfreeze()
lr/=5

learn_process.fit_one_cycle(3, slice(lr/(2.6**4),lr), moms=moms, wd=wd)
#an example of the model predicting a review 

learn_process.predict('quite a good experience, not perfect tho')



# end of model development
# test acc



vals=TextClassificationInterpretation.from_learner(learn_process)

test_acc=accuracy(vals.preds,vals.y_true)

print('Test acc is-',test_acc)



#confusion matrix, normalised to show accuracies for better understanding

vals.plot_confusion_matrix(normalize=True)
# A function to flip the order of words in tweets



input_data_backward=df_input_crop

for i in range(len(input_data_backward['text'])):

    

#     break

    inputWords = input_data_backward['text'][i].split(" ")  

    inputWords=inputWords[-1::-1] 

    output = ' '.join(inputWords) 

    input_data_backward.at[i,'text']=output

#     print(i)

#     break

input_data_backward.sample(30)

# df_input_crop.sample(10)

train_data_backward,valid_data_backward=train_test_split(input_data_backward, test_size=0.2)
#stage 1: learning process

learn_process_backward=language_model_learner(input_data, AWD_LSTM, drop_mult=0.3)

#learning process occurs using wikipedia data

learn_process_backward.freeze()
learn_process_backward.lr_find()

learn_process_backward.recorder.plot()
learn_process_backward.fit_one_cycle(1,5.0E-02,moms=moms,wd=wd)
learn_process.unfreeze()
# learn_process_backward.save_encoder('encoder_backward')
# trainNvalid_backward, test_backward=train_test_split(input_data_backward, test_size=0.2)
learn_process.fit_one_cycle(3,5.0E-03,moms=moms,wd=wd)
learn_process.save_encoder('backward_encoder')
trainNvalid_backward, test_backward=train_test_split(input_data_backward, test_size=0.2)
train_backward, valid_backward=train_test_split(trainNvalid_backward, test_size=0.2)
data_classified_backward= TextClasDataBunch.from_df(fastai_path, train_backward, valid_backward, test_df=test_backward, vocab= input_data.train_ds.vocab, text_cols='text', label_cols='airline_sentiment', bs=32)
data_classified.show_batch()
learn_process_backward=text_classifier_learner(data_classified_backward, AWD_LSTM, drop_mult=0.5)
learn_process_backward.load_encoder('backward_encoder')
learn_process_backward.freeze()

learn_process_backward.lr_find()

learn_process_backward.recorder.plot()
lr=3.0E-02

learn_process_backward.fit_one_cycle(1,lr,moms=moms, wd=wd)
learn_process_backward.freeze_to(-2)

lr/=2

learn_process_backward.fit_one_cycle(1, slice(lr/(2.6**4),lr),moms=moms, wd=wd)
learn_process_backward.freeze_to(-3)

lr/=2

learn_process_backward.fit_one_cycle(1, slice(lr/(2.6**4),lr),moms=moms, wd=wd)
learn_process_backward.unfreeze()
lr/=5

learn_process_backward.fit_one_cycle(15, slice(lr/(2.6**4),lr), moms=moms, wd=wd)
learn_process_backward.predict('quite a good experience, not perfect tho')
forward_preds, forward_targets=learn_process.get_preds(ordered=True)



print('Forward classifier results (validation set): \nValidation accuracy: {:.2f}, Validation error rate: {:.2f}'.format(accuracy(forward_preds, forward_targets), error_rate(forward_preds, forward_targets)))
backward_preds, backward_targets=learn_process_backward.get_preds(ordered=True)



print('Forward classifier results (validation set): \nValidation accuracy: {:.2f}, Validation error rate: {:.2f}'.format(accuracy(backward_preds, backward_targets), error_rate(backward_preds, backward_targets)))
ensemble_preds =(forward_preds + backward_preds)/2

# get combined(mean) accuracy on validation set

print('Ensemble classifier results (validation set): \nValidation accuracy: {:.2f}, Validation error rate: {:.2f}'.format(accuracy(ensemble_preds, forward_targets), error_rate(ensemble_preds, forward_targets)))