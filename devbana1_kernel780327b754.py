import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import spacy
import random
from spacy.util import minibatch, compounding
import timeit

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
file_dict = {}
dir_na = ''
for dirname, _, filenames in os.walk('/kaggle/input/tweet-sentiment-extraction/'):
    dir_na = dirname
    for filename in filenames:
        path_ = os.path.join(dirname, filename)
        if str('train') in path_:
            file_dict['train'] = path_
        elif str('test') in path_:
            file_dict['test'] = path_
        else:
            file_dict['sub'] = path_
print(file_dict)

raw_data = pd.read_csv(file_dict['train'])
raw_data['sentiment'].replace({'neutral':0,'negative':-1,'positive':1}, inplace=True)
raw_data.head(5)
raw_data.dropna(inplace=True, axis=0)
raw_data.reset_index(inplace=True)
raw_data.columns
def pre_processing(x):
    x = str(x).lower()
    result = []
    for i in x.split(' '):
        if i.find('http') > -1:
            pass
        else:
            if len(i) == 1:
                pass
            else:
                result.append(i)
    x = ' '.join(result)
    REPLACE_BY_SPACE_RE = re.compile('[^0-9a-z# `]')
    x = re.sub('[*.]{2,}',' ',x)
    x = re.sub('\d{1,}',' ',x)
    x = REPLACE_BY_SPACE_RE.sub('', x)
    x = re.sub('\s{2,}',' ',x)
    x = re.sub('\A ','',x)
    return x 
    
raw_data['cleaned_text']= raw_data['selected_text'].apply(lambda x:pre_processing(x))
raw_data['cleaned_text']
def get_token_ration(x1, x2):
    return fuzz.token_set_ratio(str(x1), str(x2))
ratio_set = []
for i,j in raw_data.loc[:,['cleaned_text','text']].iterrows():
    ratio_set.append(get_token_ration(j[0],j[1]))
raw_data['token_ration'] = ratio_set
raw_data['text'] = raw_data['text'].apply(lambda x: str(x).lower())
raw_data.drop(raw_data.loc[raw_data['token_ration']<80].index, inplace=True, axis=0)
raw_data['cleaned_text'] = raw_data['cleaned_text'].apply(lambda x : re.sub('[+]\d{2,}',' ',x))
raw_data.isna().sum()
raw_data.drop_duplicates(subset='text',keep=False,inplace=True)
raw_data.reset_index(inplace=True)
raw_data
train_dict = {}
def prepare_datset(full_sen, token_sen):
    unique_token = []
    entities = []
    for i in token_sen.split(' '):
        count =0
        for j in unique_token:
            if (i in j) or (j in i):
                count +=1
        if len(i)>1:
            pre = re.search(rf'\b{i}\b',full_sen)
            if pre is None:
                pass
            else:
                if count == 0:
                    if i in unique_token:
                        pass
                    else:
                        if full_sen in train_dict.keys():
                            train_dict[full_sen].append((pre.span()[0],pre.span()[1],'twitter'))
                        else:
                            train_dict[full_sen] = [(pre.span()[0],pre.span()[1],'twitter')]
            unique_token.append(i)
                
for i,j in raw_data.loc[:,['text','cleaned_text']].iterrows():
    prepare_datset(j[0],j[1])
dataset_ = []
for key, value in train_dict.items():
    med_t = (key,{'entities':value})
    dataset_.append(med_t)
def main(model=None, output_dir=None, n_iter=20,TRAIN_DATA=None):
    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")
    print('Checking NER pipe in model')
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    print('Adding Entries')
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        print('Traning of NER Started')
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            optimi = nlp.begin_training()
        for itn in range(n_iter):
            print('Starting with Iteration {}'.format(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.3,  # dropout - make it harder to memorise data
                    sgd = optimi,
                    losses=losses,
                )
            print("Losses", losses)
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
main(None,'/kaggle/output/',100,dataset_)
nlp_new = spacy.load('/kaggle/output/')
test_data = pd.read_csv(file_dict['test'])
test_data.head(5)
test_data.isnull().sum()
sent_data = test_data['text'].values
sent_data
res = []
for i in sent_data:
    doc = nlp_new(str(i).lower())
    resul = []
    for e in doc.ents:
        resul.append(str(e))
    strr = ' '.join(resul)
    res.append(strr)
submis = pd.read_csv(file_dict['sub'])
submis['selected_text']=res
submis.to_csv('sample_submission.csv')