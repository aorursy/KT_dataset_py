!pip install simpletransformers
import sys

import pandas as pd
from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy as np
import logging
import sys
import sklearn
import os, re, string
import random
import torch

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
seed = 1337

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
delimiter = "\t"
xlnettrain = pd.DataFrame(pd.read_csv("/content/drive/My Drive/Colab Files/train_DrugExp_Text (4).tsv", delimiter, header=None, names = ['labels','text']))
xlnettest = pd.DataFrame(pd.read_csv("/content/drive/My Drive/Colab Files/test_DrugExp_Text (1).tsv", delimiter, header=None, names = ['labels','text']))
inputDataValidation = pd.DataFrame(pd.read_csv("/content/drive/My Drive/Colab Files/validation_DrugExp_Text (1).tsv", delimiter, header=None, names =  ['target','text']))
xlnettrain.loc[xlnettrain.labels != 1, 'labels'] = 0
xlnettest.loc[xlnettest.labels != 1, 'labels'] = 0
# %%time

# def clean(tweet): 
            
#     # Special characters
#     tweet = re.sub(r"\x89Û_", "", tweet)
#     tweet = re.sub(r"\x89ÛÒ", "", tweet)
#     tweet = re.sub(r"\x89ÛÓ", "", tweet)
#     tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)
#     tweet = re.sub(r"\x89ÛÏ", "", tweet)
#     tweet = re.sub(r"China\x89Ûªs", "China's", tweet)
#     tweet = re.sub(r"let\x89Ûªs", "let's", tweet)
#     tweet = re.sub(r"\x89Û÷", "", tweet)
#     tweet = re.sub(r"\x89Ûª", "", tweet)
#     tweet = re.sub(r"\x89Û\x9d", "", tweet)
#     tweet = re.sub(r"å_", "", tweet)
#     tweet = re.sub(r"\x89Û¢", "", tweet)
#     tweet = re.sub(r"\x89Û¢åÊ", "", tweet)
#     tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)
#     tweet = re.sub(r"åÊ", "", tweet)
#     tweet = re.sub(r"åÈ", "", tweet)
#     tweet = re.sub(r"JapÌ_n", "Japan", tweet)    
#     tweet = re.sub(r"Ì©", "e", tweet)
#     tweet = re.sub(r"å¨", "", tweet)
#     tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)
#     tweet = re.sub(r"åÇ", "", tweet)
#     tweet = re.sub(r"å£3million", "3 million", tweet)
#     tweet = re.sub(r"åÀ", "", tweet)
    
#     # Contractions
#     tweet = re.sub(r"crohn s", "Chrons", tweet)
#     tweet = re.sub(r"Crohn s", "Chrons", tweet)
#     tweet = re.sub(r"crohn's", "Chrons", tweet)
#     tweet = re.sub(r"Crohn's", "Chrons", tweet)
#     tweet = re.sub(r"he's", "he is", tweet)
#     tweet = re.sub(r"there's", "there is", tweet)
#     tweet = re.sub(r"We're", "We are", tweet)
#     tweet = re.sub(r"That's", "That is", tweet)
#     tweet = re.sub(r"won't", "will not", tweet)
#     tweet = re.sub(r"they're", "they are", tweet)
#     tweet = re.sub(r"Can't", "Cannot", tweet)
#     tweet = re.sub(r"wasn't", "was not", tweet)
#     tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
#     tweet = re.sub(r"aren't", "are not", tweet)
#     tweet = re.sub(r"isn't", "is not", tweet)
#     tweet = re.sub(r"What's", "What is", tweet)
#     tweet = re.sub(r"haven't", "have not", tweet)
#     tweet = re.sub(r"hasn't", "has not", tweet)
#     tweet = re.sub(r"There's", "There is", tweet)
#     tweet = re.sub(r"He's", "He is", tweet)
#     tweet = re.sub(r"It's", "It is", tweet)
#     tweet = re.sub(r"You're", "You are", tweet)
#     tweet = re.sub(r"I'M", "I am", tweet)
#     tweet = re.sub(r"shouldn't", "should not", tweet)
#     tweet = re.sub(r"wouldn't", "would not", tweet)
#     tweet = re.sub(r"i'm", "I am", tweet)
#     tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
#     tweet = re.sub(r"I'm", "I am", tweet)
#     tweet = re.sub(r"Isn't", "is not", tweet)
#     tweet = re.sub(r"Here's", "Here is", tweet)
#     tweet = re.sub(r"you've", "you have", tweet)
#     tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
#     tweet = re.sub(r"we're", "we are", tweet)
#     tweet = re.sub(r"what's", "what is", tweet)
#     tweet = re.sub(r"couldn't", "could not", tweet)
#     tweet = re.sub(r"we've", "we have", tweet)
#     tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
#     tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
#     tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
#     tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
#     tweet = re.sub(r"who's", "who is", tweet)
#     tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
#     tweet = re.sub(r"y'all", "you all", tweet)
#     tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
#     tweet = re.sub(r"would've", "would have", tweet)
#     tweet = re.sub(r"it'll", "it will", tweet)
#     tweet = re.sub(r"we'll", "we will", tweet)
#     tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
#     tweet = re.sub(r"We've", "We have", tweet)
#     tweet = re.sub(r"he'll", "he will", tweet)
#     tweet = re.sub(r"Y'all", "You all", tweet)
#     tweet = re.sub(r"Weren't", "Were not", tweet)
#     tweet = re.sub(r"Didn't", "Did not", tweet)
#     tweet = re.sub(r"they'll", "they will", tweet)
#     tweet = re.sub(r"they'd", "they would", tweet)
#     tweet = re.sub(r"don't", "do not", tweet)
#     tweet = re.sub(r"don t", "do not", tweet)
#     tweet = re.sub(r"can't", "can not", tweet)
#     tweet = re.sub(r"won't", "will not", tweet)
#     tweet = re.sub(r"won t", "will not", tweet)
#     tweet = re.sub(r"can t", "can not", tweet)
#     tweet = re.sub(r"DON'T", "DO NOT", tweet)
#     tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
#     tweet = re.sub(r"they've", "they have", tweet)
#     tweet = re.sub(r"i'd", "I would", tweet)
#     tweet = re.sub(r"should've", "should have", tweet)
#     tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
#     tweet = re.sub(r"where's", "where is", tweet)
#     tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
#     tweet = re.sub(r"we'd", "we would", tweet)
#     tweet = re.sub(r"i'll", "I will", tweet)
#     tweet = re.sub(r"weren't", "were not", tweet)
#     tweet = re.sub(r"They're", "They are", tweet)
#     tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
#     tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
#     tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
#     tweet = re.sub(r"let's", "let us", tweet)
#     tweet = re.sub(r"it's", "it is", tweet)
#     tweet = re.sub(r"can't", "cannot", tweet)
#     tweet = re.sub(r"don't", "do not", tweet)
#     tweet = re.sub(r"you're", "you are", tweet)
#     tweet = re.sub(r"i've", "I have", tweet)
#     tweet = re.sub(r"that's", "that is", tweet)
#     tweet = re.sub(r"i'll", "I will", tweet)
#     tweet = re.sub(r"doesn't", "does not", tweet)
#     tweet = re.sub(r"i'd", "I would", tweet)
#     tweet = re.sub(r"didn't", "did not", tweet)
#     tweet = re.sub(r"ain't", "am not", tweet)
#     tweet = re.sub(r"you'll", "you will", tweet)
#     tweet = re.sub(r"I've", "I have", tweet)
#     tweet = re.sub(r"Don't", "do not", tweet)
#     tweet = re.sub(r"didn't", "did not", tweet)
#     tweet = re.sub(r"didn t", "did not", tweet)
#     tweet = re.sub(r"I'll", "I will", tweet)
#     tweet = re.sub(r"I'd", "I would", tweet)
#     tweet = re.sub(r"Let's", "Let us", tweet)
#     tweet = re.sub(r"you'd", "You would", tweet)
#     tweet = re.sub(r"It's", "It is", tweet)
#     tweet = re.sub(r"Ain't", "am not", tweet)
#     tweet = re.sub(r"Haven't", "Have not", tweet)
#     tweet = re.sub(r"haven't", "Have not", tweet)
#     tweet = re.sub(r"haven t", "Have not", tweet)
#     tweet = re.sub(r"havent", "Have not", tweet)
#     tweet = re.sub(r"wasnt", "was not", tweet)
#     tweet = re.sub(r"couldn t", "could not", tweet)
#     tweet = re.sub(r"I ve", "I have", tweet)
#     tweet = re.sub(r"I've", "I have", tweet)
#     tweet = re.sub(r"i've", "I have", tweet)
#     tweet = re.sub(r"i ve", "I have", tweet)
#     tweet = re.sub(r"i'll", "I will", tweet)
#     tweet = re.sub(r"i ll", "I will", tweet)
#     tweet = re.sub(r"it'll", "It will", tweet)
#     tweet = re.sub(r"it ll", "It will", tweet)
#     tweet = re.sub(r"could've", "could have", tweet)
#     tweet = re.sub(r"could ve", "could have", tweet)
#     tweet = re.sub(r"should've", "should have", tweet)
#     tweet = re.sub(r"should ve", "should have", tweet)
#     tweet = re.sub(r"would've", "would have", tweet)
#     tweet = re.sub(r"would ve", "would have", tweet)
#     tweet = re.sub(r"couldn't", "could not", tweet)
#     tweet = re.sub(r"Couldn t", "Could not", tweet)
#     tweet = re.sub(r"Couldn't", "Could not", tweet)
#     tweet = re.sub(r"wouldn't", "would not", tweet)
#     tweet = re.sub(r"Wouldn't", "Would not", tweet)
#     tweet = re.sub(r"wouldn t", "would not", tweet)
#     tweet = re.sub(r"Wouldn t", "Would not", tweet)
#     tweet = re.sub(r"hasn't", "has not", tweet)
#     tweet = re.sub(r"hasn t", "has not", tweet)
#     tweet = re.sub(r"doesn't", "does not", tweet)
#     tweet = re.sub(r"doesn t", "does not", tweet)
#     tweet = re.sub(r"does'nt", "does not", tweet)
#     tweet = re.sub(r"Doesn't", "Does not", tweet)
#     tweet = re.sub(r"Doesn t", "Does not", tweet)
#     tweet = re.sub(r"Does'nt", "Does not", tweet)
#     tweet = re.sub(r"wasn't", "was not", tweet)
#     tweet = re.sub(r"wasn t", "was not", tweet)
#     tweet = re.sub(r"had nt", "was not", tweet)
#     tweet = re.sub(r"had'nt", "had not", tweet)
#     tweet = re.sub(r"was nt", "was not", tweet)
#     tweet = re.sub(r"was'nt", "was not", tweet)
#     tweet = re.sub(r"Was nt", "Was not", tweet)
#     tweet = re.sub(r"Was'nt", "Was not", tweet)
#     tweet = re.sub(r"is nt", "is not", tweet)
#     tweet = re.sub(r"is'nt", "is not", tweet)
#     tweet = re.sub(r"isnt", "is not", tweet)
#     tweet = re.sub(r"could nt", "could not", tweet)
#     tweet = re.sub(r"could'nt", "could not", tweet)
#     tweet = re.sub(r"Could nt", "Could not", tweet)
#     tweet = re.sub(r"Could'nt", "Could not", tweet)
#     tweet = re.sub(r"sideeffects", "side effects", tweet)
#     tweet = re.sub(r"Could've", "Could have", tweet)
#     tweet = re.sub(r"youve", "you have", tweet)  
#     tweet = re.sub(r"mg", "milligrams", tweet) 
#     tweet = re.sub(r"donå«t", "do not", tweet)   
            
#     # Character entity references
#     tweet = re.sub(r"&gt;", ">", tweet)
#     tweet = re.sub(r"&lt;", "<", tweet)
#     tweet = re.sub(r"&amp;", "&", tweet)
    
#     # Typos, slang and informal abbreviations
#     tweet = re.sub(r"w/e", "whatever", tweet)
#     tweet = re.sub(r"w/", "with", tweet)
#     tweet = re.sub(r"USAgov", "USA government", tweet)
#     tweet = re.sub(r"c diff", "Clostridioides difficile", tweet)
#     tweet = re.sub(r"Ph0tos", "Photos", tweet)
#     tweet = re.sub(r"amirite", "am I right", tweet)
#     tweet = re.sub(r"exp0sed", "exposed", tweet)
#     tweet = re.sub(r"<3", "love", tweet)
#     tweet = re.sub(r"amageddon", "armageddon", tweet)
#     tweet = re.sub(r"Trfc", "Traffic", tweet)
#     tweet = re.sub(r"8/5/2015", "2015-08-05", tweet)
#     tweet = re.sub(r"WindStorm", "Wind Storm", tweet)
#     tweet = re.sub(r"8/6/2015", "2015-08-06", tweet)
#     tweet = re.sub(r"10:38PM", "10:38 PM", tweet)
#     tweet = re.sub(r"10:30pm", "10:30 PM", tweet)
#     tweet = re.sub(r"16yr", "16 year", tweet)
#     tweet = re.sub(r"lmao", "laughing my ass off", tweet)   
#     tweet = re.sub(r"TRAUMATISED", "traumatized", tweet)
#     # Words with punctuations and special characters
#     punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
#     for p in punctuations:
#         tweet = tweet.replace(p, f' {p} ')
        
#     # ... and ..
#     tweet = tweet.replace('...', ' ... ')
#     if '...' not in tweet:
#         tweet = tweet.replace('..', ' ... ')      
        
       
#     # Grouping same words without embeddings
#     tweet = re.sub(r"Bestnaijamade", "bestnaijamade", tweet)
#     tweet = re.sub(r"SOUDELOR", "Soudelor", tweet)
    
#     return tweet


# xlnettrain['text']=xlnettrain['text'].apply(lambda s :  clean(s))
# xlnettrain['text']=xlnettrain['text'].apply(lambda s :  clean(s))
# def remove_punct(text):
#     table=str.maketrans('','',string.punctuation)
#     return text.translate(table)
# xlnettrain['text']=xlnettrain['text'].apply(lambda s :  remove_punct(s))
# xlnettest['text']=xlnettest['text'].apply(lambda s :  remove_punct(s))
xlnettrain = xlnettrain[['text', 'labels']]
n=2
kf = KFold(n_splits=n, random_state=seed, shuffle=True)
results = []

for train_index, val_index in kf.split(xlnettrain):
    train_df = xlnettrain.iloc[train_index]
    eval_df = xlnettrain.iloc[val_index]
model_type = "xlnet"

if model_type == "bert":
    model_name = "bert-base-cased"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "distilbert":
    model_name = "distilbert-base-cased"

elif model_type == "distilroberta":
    model_type = "roberta"
    model_name = "distilroberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

elif model_type == "electra-small":
    model_type = "electra"
    model_name = "google/electra-small-discriminator"
    
elif model_type == "xlnet":
    model_name = "xlnet-base-cased"

train_args = {
"output_dir": "/content/drive/My Drive/Colab Files/outputs/",
"cache_dir": "/content/drive/My Drive/Colab Files/cache/",
"best_model_dir": "/content/drive/My Drive/Colab Files/outputs/best_model/",

"fp16": False,
"max_seq_length": 128,
"train_batch_size": 8,
"eval_batch_size": 8,
"gradient_accumulation_steps": 1,
"num_train_epochs": 5,
"weight_decay": 0,
"learning_rate": 4e-5,
"adam_epsilon": 1e-8,
"warmup_ratio": 0.06,
"warmup_steps": 0,
"max_grad_norm": 1.0,
"do_lower_case": True,
'sliding_window': True,                                                         
                                                             
"logging_steps": 50,
"evaluate_during_training": True,
"evaluate_during_training_steps": 2000,
"evaluate_during_training_verbose": True,
"use_cached_eval_features": False,
"save_eval_checkpoints": True,
"save_steps": 2000,
"no_cache": False,
"save_model_every_epoch": True,
"tensorboard_dir": None,
"overwrite_output_dir": True,  
}

if model_type == "xlnet":
    train_args["train_batch_size"] = 64
    train_args["gradient_accumulation_steps"] = 2


# Create a ClassificationModel
model = ClassificationModel(model_type, model_name, args=train_args)

# Train the model
model.train_model(train_df, eval_df=eval_df)

# # # Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
print(result['acc'])
results.append(result['acc'])
XLNpredictions, XLNraw_outputs = model.predict(xlnettest['text'])
dataset = pd.DataFrame(XLNpredictions)
dataset.to_csv('/content/drive/My Drive/7212 HW/XLNpredictions.csv', index=False)
print(dataset)

predictions, raw_outputs = model.predict(["I'd like to puts some CD-ROMS on my iPad, is that possible?' — Yes, but wouldn't that block the screen?" * 25])
print(predictions)
print(raw_outputs)
model_type = "electra-base"

if model_type == "bert":
    model_name = "bert-base-cased"

elif model_type == "roberta":
    model_name = "roberta-base"

elif model_type == "distilbert":
    model_name = "distilbert-base-cased"

elif model_type == "distilroberta":
    model_type = "roberta"
    model_name = "distilroberta-base"

elif model_type == "electra-base":
    model_type = "electra"
    model_name = "google/electra-base-discriminator"

elif model_type == "electra-small":
    model_type = "electra"
    model_name = "google/electra-small-discriminator"
    
elif model_type == "xlnet":
    model_name = "xlnet-base-cased"

train_args = {
"output_dir": "/content/drive/My Drive/Colab Files/output/",
"cache_dir": "/content/drive/My Drive/Colab Files/ch/",
"best_model_dir": "/content/drive/My Drive/Colab Files/outputs/bm/",

"fp16": False,
"max_seq_length": 128,
"train_batch_size": 8,
"eval_batch_size": 8,
"gradient_accumulation_steps": 1,
"num_train_epochs": 4,
"weight_decay": 0,
"learning_rate": 4e-5,
"adam_epsilon": 1e-8,
"warmup_ratio": 0.06,
"warmup_steps": 0,
"max_grad_norm": 1.0,
"do_lower_case": True,
'sliding_window': True,                                                         
                                                             
"logging_steps": 50,
"evaluate_during_training": True,
"evaluate_during_training_steps": 2000,
"evaluate_during_training_verbose": True,
"use_cached_eval_features": False,
"save_eval_checkpoints": True,
"save_steps": 2000,
"no_cache": False,
"save_model_every_epoch": True,
"tensorboard_dir": None,
"overwrite_output_dir": True,  
}

if model_type == "xlnet":
    train_args["train_batch_size"] = 64
    train_args["gradient_accumulation_steps"] = 2


# Create a ClassificationModel
model = ClassificationModel(model_type, model_name, args=train_args)

# Train the model
model.train_model(train_df, eval_df=eval_df)

# # # Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)
ELECTRApredictions, ELECTRAraw_outputs = model.predict(xlnettest['text'])
ELECTRAraw_outputs

print(result['acc'])
results.append(result['acc'])
predictions4, raw_outputs4 = model.predict(["This works y'all","I can't see good"])
print(predictions4)
print(raw_outputs4)
dataset = pd.DataFrame(ELECTRApredictions)
dataset.to_csv('/content/drive/My Drive/7212 HW/ELECTRApredictions2.csv', index=False)
print(dataset)
