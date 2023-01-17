import os

import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
display(len(df_train))
df_train.head()
df_test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
df_test.head()
# drop na
print(len(df_train))
df_train.dropna(axis = 0, how ='any',inplace=True)
print(len(df_train))
# Add tokes and counts.
df_train['text_tokes']   = df_train.text.str.split()
df_train['select_tokes'] = df_train.selected_text.str.split()
df_train['text_tokes_cnt'] = df_train.text_tokes.str.len()
df_train['select_tokes_cnt'] = df_train.select_tokes.str.len()
df_train.head(5)
# remove text=2 and neutrals as = self.
df_train = df_train[~(df_train.text_tokes_cnt<=2)]
df_train = df_train[(df_train.sentiment!='neutral')]
display(len(df_train))
display(df_train.sentiment.value_counts())
df_train.sample(5)
# Hold out set. Ten percent of total train split evenly between sentiment.
# num_rows_to_use = int((len(df_train) * .10)/2) 
# df_valid_pos = df_train[df_train.sentiment=='positive'].sample(num_rows_to_use)
# df_valid_neg = df_train[df_train.sentiment=='negative'].sample(num_rows_to_use)
# df_valid = pd.concat([df_valid_pos, df_valid_neg])
# display(len(df_valid))
# display(df_valid.sentiment.value_counts())
# df_valid.sample(5)
# Drop validation set from training set.
# display(f"Train before={len(df_train)}")
# df_train.drop(df_valid.index, inplace=True, axis=0)
# display(f"Train after={len(df_train)}") 
# Input & output the same.
# Seem problematic to me.
df_sames = df_train[df_train.text_tokes_cnt==df_train.select_tokes_cnt]
display(len(df_sames))
display(df_sames.sentiment.value_counts())
pd.options.display.max_colwidth = 1000
df_sames.sample(5)
# Spacy model building related.
import spacy
from tqdm import tqdm
import random
from spacy.util import minibatch, compounding

import warnings
warnings.filterwarnings("ignore")
def save_model(output_dir, nlp, new_model_name):
    ''' This Function Saves model to 
    given output directory'''
    
    output_dir = f'../working/{output_dir}'
    if output_dir is not None:        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
def get_model_out_path(sentiment):
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = 'models/model_pos'
    elif sentiment == 'negative':
        model_out_path = 'models/model_neg'
    return model_out_path
def get_training_data(sentiment, df_input):
    '''
    Returns Training data in the format needed to train spacy NER
    ID start and end point of the 'selected' text in the text 
    and used as your string entity info for spacy.
    '''
    SENTIMENT = ['negative', 'positive']
    if sentiment not in SENTIMENT:
        raise ValueError(f"{sentiment} not in {SENTIMENT})")
    train_data = []
    for index, row in df_input.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data
# pass model = nlp if you want to train on top of existing model 

def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    # Uses given model or instantiates a blank model.
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')
def train(train_data, output_dir, n_iter=20, model=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    ""
    if model is not None:
        nlp = spacy.load(output_dir)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")
    
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()


        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                            annotations,  # batch of annotations
                            drop=0.5,   # dropout - make it harder to memorise data
                            losses=losses, 
                            )
            print("Losses", losses)
    save_model(output_dir, nlp, 'st_ner')
def run_train(n_iter=20):
    """ Convenience so can comment out if not don't need to regenerat models. """
    for sentiment in ['positive', 'negative']:
        model_path = get_model_out_path(sentiment)
        train_data = get_training_data(sentiment, df_train)
        train(train_data, model_path, n_iter=n_iter)
%%time
run_train(n_iter=5)
MODELS_BASE_PATH = 'models/'
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end, ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text

def pred_set(df_set):
    """ Run NER models on data. """
    df_pred = df_set.copy()
    
    selected_texts = []

    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')

    for index, row in df_pred.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))

    df_pred['predicted_text'] = selected_texts

    return df_pred
model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
predict_entities("I love this model, it's great. I can't wait to lean more about training spacy models", model_pos)
# Metric.
def jaccard(compare_strings): 
    str1, str2 = compare_strings
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
# Predict on validation set - add to df.
df_pred = pred_set(df_train)
# Reminder that neutral are missing from this.
df_pred.sentiment.value_counts()
# Calculate jaccard.
# Do it here then move to above func - probably.
df_pred['jaccard'] = df_pred[['selected_text','predicted_text']].values.tolist()
df_pred['jaccard'] = df_pred.jaccard.apply(jaccard)
# Not much difference in accuracy per sentiment type
print(df_pred.jaccard.mean())
print(df_pred[df_pred.sentiment=='positive'].jaccard.mean())
print(df_pred[df_pred.sentiment=='negative'].jaccard.mean())
display(df_pred[['text','selected_text', 'sentiment', 'predicted_text', 'jaccard']].sample(20))
# Run NER on full test set.
selected_texts = []
MODELS_BASE_PATH = 'models/'

if MODELS_BASE_PATH is not None:
    print("Loading Models  from ", MODELS_BASE_PATH)
    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
        
    for index, row in df_test.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
        
df_test['selected_text'] = selected_texts
print(len(df_test))
print(df_submission.describe())
df_submission.info()
df_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

df_submission = df_test[['textID', 'selected_text']]
print(len(df_submission))
display(df_submission.head(10))
os.chdir("/kaggle/working/")
df_submission.to_csv("submission.csv", index=False)
print(len(df_submission))
print(df_submission.describe())
print(df_submission.info())
df_submission.head(10)
df_submission
!ls