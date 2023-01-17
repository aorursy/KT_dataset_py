import numpy as np
import pandas as pd
import os
import nltk
import spacy
import random
from tqdm import tqdm
from spacy.util import minibatch, compounding

df_train  = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
df_train['Num_words_text'] = df_train['text'].apply(lambda x: len(str(x).split()))
df_train = df_train[df_train['Num_words_text'] >= 3]
df_train
def save_model(output_dir, nlp, model_name):
    """
    Saving the prediction model in some desired location.
    """
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta['name'] = model_name
        nlp.to_disk(output_dir)  
        print('Saved the model to: ', output_dir)
def output_model_path(sentiment):
    """
    On the basis of sentiment type, return the model path.
    """
    model_path = None
    if sentiment == 'positive':
        model_path = 'model/pos_model'
    if sentiment == 'negative':
        model_path = 'model/neg_model'
    return model_path
def get_training_data(sentiment):
    """
    On the basis of sentiment type, convert the input train data in your desired format
    """
    train_data = []
    for i, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected = row.selected_text
            text = row.text
            start = text.find(selected)
            end = start + len(selected)
            
            train_data.append((text,{'entities': [[start, end, 'selected_text']]}))
    return train_data
def train(train_data, output_dir, n_iter = 10, model = None):
    """
    Load the saved model or build a new blank model for training and prediction
    """
    if model is not None:
        nlp = spacy.load(output_dir)
        print('Loaded model %s' %model)
    else:
        nlp = spacy.blank('en')
        print('Created a blank model')
        
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last = True)
    else:
        nlp.get_pipe('ner')
        
    for _, annote in train_data:
        for ent in annote.get('entities'):
            ner.add_label(ent[2])
            
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()
            
    for itera in tqdm(range(n_iter)):
        random.shuffle(train_data)
        batches = minibatch(train_data, size = compounding(4.0, 50.0, 1.001))
        loss = {}
        for batch in batches:
            text, annote = zip(*batch)
            nlp.update(text, annote, drop = 0.5, losses = loss)
        print('Losses : ', loss)
    save_model(output_dir, nlp, 'senti_ner')
    
# training for positive sentiments

sentiment = 'positive'

train_data = get_training_data(sentiment)
model_path = output_model_path(sentiment)

train(train_data, model_path, n_iter = 15, model = None)
# training for negative sentiments

sentiment = 'negative'

train_data = get_training_data(sentiment)
model_path = output_model_path(sentiment)

train(train_data, model_path, n_iter = 18, model = None)
def predict_entities(text, model):
    data = model(text)
    ent = []
    for it in data.ents:
        start = text.find(it.text)
        end = start + len(it.text)
        new_ent = [start, end, it.label_]
        if new_ent not in ent:
            ent.append(new_ent)
    selected = text[ent[0][0]:ent[0][1]] if len(ent)>0 else text
    return selected
selected = []
base_model_path = './model/'

if base_model_path is not None:
    print('Loading models from ', base_model_path)
    
    pos_model = spacy.load(base_model_path + 'pos_model')
    neg_model = spacy.load(base_model_path + 'neg_model')
    
    for it, row in df_test.iterrows():
        text = row.text
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected.append(text)
        elif row.sentiment == 'positive':
            selected.append(predict_entities(text, pos_model))
        else:
            selected.append(predict_entities(text, neg_model))
df_test['selected_text'] = selected
df_sub['selected_text'] = df_test['selected_text']
df_sub.to_csv('submission.csv', index = False)
display(df_sub.head(10))
