import pandas as pd
import numpy as np
import random
import os
import nltk
import re
from tqdm import tqdm
import spacy
from spacy.util import compounding
from spacy.util import minibatch

import warnings
warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train.dropna(inplace = True)
train_data = train.copy()
def get_training_data(sentiment):
    training_data = []
    for data in train_data.values:  
        if data[3] == sentiment:
            text = data[1]
            selected_text = data[2]
            start = text.find(selected_text)
            end = start + len(selected_text)
            training_data.append((text, {'entities' : [[start, end, 'selected_text']]}))
    return training_data
def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    else:
        model_out_path = 'models/model_neu'
    return model_out_path
def save_model(output_dir, nlp, new_model_name):
    output_dir = f'/kaggle/input/tse-spacy-model/{output_dir}'
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta['name'] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to ", output_dir)
def train(training_data, output_dir, n_iter = 20, model = None):
    if model is not None:
        nlp = spacy.load(output_dir)
        print("Loaded model '%s'", model)
    else:
        nlp = spacy.blank("en")
        print("Created Blank en model")
        
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    else:
        ner = nlp.get_pipe("ner")
        
    for _,annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    # get names of other pipe to disable them during training
    other_pipes = [x for x in nlp.pipe_names if x != 'ner']
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()
            
        for itn in tqdm(range(n_iter)):
            random.shuffle(training_data)
            batches = minibatch(training_data, size = compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                text, annotations = zip(*batch)
                nlp.update(
                    text,
                    annotations,
                    drop = 0.5,
                    losses = losses
                )
            print("losses : ", losses)
#         save_model(output_dir, nlp, 'st_ner')
sentiment = 'positive'

training_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(training_data, model_path, n_iter = 2, model = None)
sentiment = 'negative'

training_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(training_data, model_path, n_iter = 2, model = None)
sentiment = 'neutral'

training_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)

train(training_data, model_path, n_iter = 2, model = None)
TRAINED_MODELS_BASE_PATH = '../input/tse-spacy-model/models/'
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0] : ent_array[0][1]] if len(ent_array) > 1 else text
    return selected_text
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c))/(len(a) + len(b) - len(c))
if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading models from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
    
    jaccard_score = 0
    for index, row in tqdm(train_data.iterrows(), total = train_data.shape[0]):
        text = row.text
        if row.sentiment == 'positive':
            jaccard_score += jaccard(predict_entities(text, model_pos), row.selected_text)
        elif row.sentiment == 'negative':
            jaccard_score += jaccard(predict_entities(text, model_neg), row.selected_text)
        else:
            jaccard_score += jaccard(predict_entities(text, model_neu), row.selected_text)    
    print(f'Average Jaccard Score is {jaccard_score / train_data.shape[0]}')
if TRAINED_MODELS_BASE_PATH is not None:
    print("Loading models from ", TRAINED_MODELS_BASE_PATH)
    model_pos = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neg')
    model_neu = spacy.load(TRAINED_MODELS_BASE_PATH + 'model_neu')
    
    final_data = []
    for index, row in tqdm(test.iterrows(), total = test.shape[0]):
        text = row.text
        if row.sentiment == 'positive':
            final_data.append(predict_entities(text, model_pos))
        elif row.sentiment == 'negative':
            final_data.append(predict_entities(text, model_neg))
        else:
            final_data.append(predict_entities(text, model_neu))
testID = test['textID']
df = pd.DataFrame(list(zip(testID, final_data)), columns = ['textID', 'selected_text'])
df.head()
df.to_csv("submission.csv", index=False)
print("successfully saved")
