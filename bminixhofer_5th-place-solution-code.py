import os
os.environ['USER'] = 'root'
os.system('pip install ../input/xlearn/xlearn/xlearn-0.40a1/')
import json
import random
import os
import gc
import html
import time
import re
from collections import defaultdict
import math

import cv2
import numpy as np
import pandas as pd
import lightgbm as lgb
import xlearn as xl
import scipy
from scipy import ndimage
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score, mean_squared_error, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from nltk.stem.snowball import SnowballStemmer, PorterStemmer

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
import keras

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchvision import models, transforms

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', UserWarning)
%matplotlib inline

# treat np.inf as nan value too, because machine learning algorithms should not get inf as input
pd.set_option("use_inf_as_na", True)
tqdm.pandas()
TRAIN_PATH = '../input/petfinder-adoption-prediction/train/train.csv'
TEST_PATH = '../input/petfinder-adoption-prediction/test/test.csv'
TRAIN_SENTIMENT_PATH = '../input/petfinder-adoption-prediction/train_sentiment/'
TEST_SENTIMENT_PATH = '../input/petfinder-adoption-prediction/test_sentiment/'
TRAIN_METADATA_PATH = '../input/petfinder-adoption-prediction/train_metadata/'
TEST_METADATA_PATH = '../input/petfinder-adoption-prediction/test_metadata/'
TRAIN_IMAGES_PATH = '../input/petfinder-adoption-prediction/train_images/'
TEST_IMAGES_PATH = '../input/petfinder-adoption-prediction/test_images/'
EMBEDDING_FILE_SMALL = '../input/petfinder-external/glove.6B.200d.txt'
EMBEDDING_FILE_CRAWL = '../input/petfinder-external/crawl-300d-2M.vec'
PRETRAINED_WEIGHT_PATH = '../input/petfinder-external/DenseNet-BC-121-32-no-top.h5'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df.head()
# state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP
state_gdp = {
    41336: 116.679,
    41325: 40.596,
    41367: 23.02,
    41401: 190.075,
    41415: 5.984,
    41324: 37.274,
    41332: 42.389,
    41335: 52.452,
    41330: 67.629,
    41380: 5.642,
    41327: 81.284,
    41345: 80.167,
    41342: 121.414,
    41326: 280.698,
    41361: 32.270
}

# state population: https://en.wikipedia.org/wiki/Malaysia
state_population = {
    41336: 33.48283,
    41325: 19.47651,
    41367: 15.39601,
    41401: 16.74621,
    41415: 0.86908,
    41324: 8.21110,
    41332: 10.21064,
    41335: 15.00817,
    41330: 23.52743,
    41380: 2.31541,
    41327: 15.61383,
    41345: 32.06742,
    41342: 24.71140,
    41326: 54.62141,
    41361: 10.35977
}

train_df["state_gdp"] = train_df['State'].map(state_gdp)
train_df["state_population"] = train_df['State'].map(state_population)
test_df["state_gdp"] = test_df['State'].map(state_gdp)
test_df["state_population"] = test_df['State'].map(state_population)
train_df["gdp_vs_population"] = train_df["state_gdp"] / train_df["state_population"]
test_df["gdp_vs_population"] = test_df["state_gdp"] / test_df["state_population"]
puncts = ['。', ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', 
          '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '_', '{', '}', 
          '^', '`', '<', '°', '™', '♥', '½', '…', '“', '”', '–', '●', '²', '¬', '↑',
          '—', '：', '’', '☆', 'é', '¯', '♦', '‘', '）', '↓', '、', '（', '，', '♪', 
          '³', '❤', 'ï', '√']

mispell_dict = {
    "I'd": 'I would',
    "I'll": 'I will',
    "I'm": 'I am',
    "I've": 'I have',
    "ain't": 'is not',
    "aren't": 'are not',
    "can't": 'cannot',
    'cancelled': 'canceled',
    'centre': 'center',
    'colour': 'color',
    "could've": 'could have',
    "couldn't": 'could not',
    "didn't": 'did not',
    "doesn't": 'does not',
    "don't": 'do not',
    'enxiety': 'anxiety',
    'favourite': 'favorite',
    "hadn't": 'had not',
    "hasn't": 'has not',
    "haven't": 'have not',
    "he'd": 'he would',
    "he'll": 'he will',
    "he's": 'he is',
    "here's": 'here is',
    "how's": 'how is',
    "i'd": 'i would',
    "i'll": 'i will',
    "i'm": 'i am',
    "i've": 'i have',
    "isn't": 'is not',
    "it'll": 'it will',
    "it's": 'it is',
    'labour': 'labor',
    "let's": 'let us',
    "might've": 'might have',
    "must've": 'must have',
    'organisation': 'organization',
    "she'd": 'she would',
    "she'll": 'she will',
    "she's": 'she is',
    "shouldn't": 'should not',
    "that's": 'that is',
    'theatre': 'theater',
    "there's": 'there is',
    "they'd": 'they would',
    "they'll": 'they will',
    "they're": 'they are',
    "they've": 'they have',
    'travelling': 'traveling',
    "wasn't": 'was not',
    'watsapp': 'whatsapp',
    "we'd": 'we would',
    "we'll": 'we will',
    "we're": 'we are',
    "we've": 'we have',
    "weren't": 'were not',
    "what's": 'what is',
    "where's": 'where is',
    "who'll": 'who will',
    "who's": 'who is',
    "who've": 'who have',
    "won't": 'will not',
    "would've": 'would have',
    "wouldn't": 'would not',
    "you'd": 'you would',
    "you'll": 'you will',
    "you're": 'you are',
    "you've": 'you have',
    '，': ',',
    '／': '/',
    '？': '?'
}
def safe_get(data, *keys):
    try:
        current = data
        for key in keys:
            current = current[key]
    except:
        current = None
    return current        
for df, path in ([train_df, TRAIN_SENTIMENT_PATH], [test_df, TEST_SENTIMENT_PATH]):
    doc_sent_mags = np.full(len(df), np.nan)
    doc_sent_scores = np.full(len(df), np.nan)
    sentence_texts = np.full(len(df), '', dtype=object)
    nf_count = 0
    
    for i, pet in enumerate(df['PetID']):
        try:
            with open(path + pet + '.json', 'r') as f:
                sentiment = json.load(f)
            
            doc_sent_mag = safe_get(sentiment, 'documentSentiment', 'magnitude')
            if doc_sent_mag is not None:
                doc_sent_mags[i] = doc_sent_mag

            doc_sent_score = safe_get(sentiment, 'documentSentiment', 'score')
            if doc_sent_score is not None:
                doc_sent_scores[i] = doc_sent_score

            text = ' '.join([x['text']['content'] for x in sentiment['sentences']])
            text = html.unescape(text)
            sentence_texts[i] = text
        except:
            nf_count += 1
            sentence_texts[i] = df.iloc[i]['Description']
            
    print('Not found: ', nf_count)
    df.loc[:, 'doc_sent_mag'] = doc_sent_mags
    df.loc[:, 'doc_sent_score'] = doc_sent_scores
    df.loc[:, 'sentence_text'] = sentence_texts
for df, path in ([train_df, TRAIN_METADATA_PATH], [test_df, TEST_METADATA_PATH]):
    for image in range(1, 31):
        vertex_xs = np.full(len(df), np.nan)
        vertex_ys = np.full(len(df), np.nan)
        bounding_confidences = np.full(len(df), np.nan)
        bounding_importance_fracs = np.full(len(df), np.nan)
        dominant_blues = np.full(len(df), np.nan)
        dominant_greens = np.full(len(df), np.nan)
        dominant_reds = np.full(len(df), np.nan)
        dominant_pixel_fracs = np.full(len(df), np.nan)
        dominant_scores =np.full(len(df), np.nan)
        top_label_descriptions = np.full(len(df), '', dtype=object)
        label_descriptions = np.full(len(df), '', dtype=object)
        label_scores = np.full(len(df), np.nan)
        nf_count = 0
        nl_count = 0

        for i, pet in enumerate(df['PetID']):
            try:
                with open(path + pet + f'-{image}.json', 'r') as f:
                    data = json.load(f)

                for feature_array, keys in [
                    (vertex_xs, ['cropHintsAnnotation', 'cropHints', 0, 
                                 'boundingPoly', 'vertices', 2, 'x']),
                    (vertex_ys, ['cropHintsAnnotation', 'cropHints', 0, 
                                 'boundingPoly', 'vertices', 2, 'y']),
                    (bounding_confidences, ['cropHintsAnnotation', 'cropHints', 0, 
                                            'confidence']),
                    (bounding_importance_fracs, ['cropHintsAnnotation', 'cropHints', 0, 
                                                 'importanceFraction']),
                    (dominant_blues, ['imagePropertiesAnnotation', 'dominantColors', 
                                      'colors', 0, 'color', 'blue']),
                    (dominant_greens, ['imagePropertiesAnnotation', 'dominantColors', 
                                       'colors', 0, 'color', 'green']),
                    (dominant_reds, ['imagePropertiesAnnotation', 'dominantColors', 
                                     'colors', 0, 'color', 'red']),
                    (dominant_pixel_fracs, ['imagePropertiesAnnotation', 'dominantColors', 
                                            'colors', 0, 'pixelFraction']),
                    (dominant_scores, ['imagePropertiesAnnotation', 'dominantColors', 
                                       'colors', 0, 'score'])
                ]:
                    x = safe_get(data, *keys)
                    if x is not None:
                        feature_array[i] = x
                    
                if data.get('labelAnnotations'):
                    file_annots = data['labelAnnotations'][:int(len(data['labelAnnotations']) * 0.3)]
                    
                    if len(file_annots) > 0:
                        label_score = np.asarray([x['score'] for x in file_annots]).mean()
                        label_description = ' '.join([x['description'] for x in file_annots])

                        top_label_descriptions[i] = file_annots[0]['description']
                        label_descriptions[i] = label_description
                        label_scores[i] = label_score
                else:
                    nl_count += 1
            except:
                nf_count += 1

        df.loc[:, f'vertex_x_{image}'] = vertex_xs
        df.loc[:, f'vertex_y_{image}'] = vertex_ys
        df.loc[:, f'bounding_confidence_{image}'] = bounding_confidences
        df.loc[:, f'bounding_importance_{image}'] = bounding_importance_fracs
        df.loc[:, f'dominant_blue_{image}'] = dominant_blues
        df.loc[:, f'dominant_green_{image}'] = dominant_greens
        df.loc[:, f'dominant_red_{image}'] = dominant_reds
        df.loc[:, f'dominant_pixel_frac_{image}'] = dominant_pixel_fracs
        df.loc[:, f'dominant_score_{image}'] = dominant_scores
        df.loc[:, f'top_label_description_{image}'] = top_label_descriptions
        df.loc[:, f'label_description_{image}'] = label_descriptions
        df.loc[:, f'label_score_{image}'] = label_scores
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(['\s*'.join(key) 
                                               for key in mispell_dict.keys()]))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[re.sub('\s', '', match.group(0))]
    return mispellings_re.sub(replace, text)

def fix_distribution(y_train, pred):
    """
    Convert predictions to labels such that the labels have 
    the same distribution as in the y_train array.
    """
    base = pd.Series([0, 0, 0, 0, 0], index=np.arange(0, 5))
    thresholds = (base + pd.Series(y_train).value_counts()).fillna(0).cumsum()
    thresholds = thresholds / len(y_train) * len(pred)
    
    pred_ranks = pd.Series(pred).rank()
    ranked_scores = np.zeros(len(pred))

    for j, threshold in list(enumerate(thresholds))[::-1]:
        ranked_scores[pred_ranks <= threshold] = j
    return ranked_scores

def process_text_rnn(text):
    """Process text for RNNs."""
    if text is None:
            return ''
    text = clean_text(text)
    text = replace_typical_misspell(text)
    for char in '()*,./:;\\\t\n':
        text = text.replace(char, '')
    text = re.sub('\s+', ' ', text)
    return text

stemmer = PorterStemmer()

def process_text(text):
    """
    Process text for other models (LGB, FFM). 
    Additionally replaces numbers by # and stems the text.
    """
    if text is None:
        return ''
    text = clean_text(text)
    text = replace_typical_misspell(text)
    text = clean_numbers(text)
    for char in '()*,./:;\\\t\n':
        text = text.replace(char, '')
    text = re.sub('\s+', ' ', text)
    text = ' '.join(stemmer.stem(word) for word in text.split())
    return text
aggregate_features = []

for feature in ['vertex_x', 'vertex_y', 'bounding_confidence', 'bounding_importance', 
                'dominant_blue', 'dominant_green', 'dominant_red', 
                'dominant_pixel_frac', 'dominant_score', 'label_score']:
    feature_names = [feature + f'_{i}' for i in range(1, 31)]
    
    for df in [train_df, test_df]:
        df[feature + '_mean'] = df[feature_names].mean(axis=1)
        df[feature + '_sum'] = df[feature_names].sum(axis=1)
        df[feature + '_var'] = df[feature_names].var(axis=1)
    
    aggregate_features += [feature + '_mean', feature + '_sum', feature + '_var']
for df in [train_df, test_df]:
    df['all_label_descriptions'] = df['label_description_1'].str.cat(
        df[[f'label_description_{i}' for i in range(2, 31)]], sep=' '
    ).replace('\s+', ' ', regex=True)
    df['no_name'] = (df['Name'].fillna('None').str.match(r'\b(No|Not|None)\b')) | \
                    (df['Name'].fillna('').apply(len) <= 2)
    df['sentence_text_rnn'] = df['sentence_text'].progress_apply(lambda x: process_text_rnn(x))
    df['sentence_text_stemmed'] = df['sentence_text'].progress_apply(lambda x: process_text(x))   
for features in [['Dewormed', 'Vaccinated']]:
    feature = '_'.join(features)

    train_df[feature] = train_df[features].astype(str).apply(lambda x: '_'.join(x), axis=1)
    test_df[feature] = test_df[features].astype(str).apply(lambda x: '_'.join(x), axis=1)
all_num_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'doc_sent_mag', 
                    'doc_sent_score', 'MaturitySize', 'FurLength', 'Health', 'state_gdp', 
                    'state_population', 'gdp_vs_population'] + aggregate_features

all_cat_features = ['Dewormed', 'Vaccinated', 'Sterilized', 'Breed1', 'Type', 
                    'Dewormed_Vaccinated', 'Breed2', 'State', 'Gender', 'Color1', 
                    'Color2', 'Color3', 'no_name', 'RescuerID']

for image in [1]:
    all_num_features += [f'dominant_score_{image}', f'dominant_pixel_frac_{image}', 
                         f'dominant_red_{image}', f'dominant_green_{image}', 
                         f'dominant_blue_{image}', f'bounding_importance_{image}', 
                         f'bounding_confidence_{image}', f'vertex_x_{image}', 
                         f'vertex_y_{image}', f'label_score_{image}']
    all_cat_features += [f'top_label_description_{image}']
for feature in all_num_features:
    train_df[feature].fillna(-1, inplace=True)
    test_df[feature].fillna(-1, inplace=True)
for feature in all_cat_features:
    if feature + '_count' in train_df.columns:
        train_df.drop(feature + '_count', axis=1, inplace=True)

    if feature + '_count' in test_df.columns:
        test_df.drop(feature + '_count', axis=1, inplace=True)    
    
    valcounts = train_df[feature].append(test_df[feature]).value_counts()
    
    train_df = train_df.join(valcounts.rename(feature + '_count'), on=feature, how='left')
    test_df = test_df.join(valcounts.rename(feature + '_count'), on=feature, how='left')
    
    if feature + '_count' not in all_num_features:
        all_num_features.append(feature + '_count')
bin_counts = defaultdict(lambda: 20)
bin_counts['RescuerID_count'] = 10

for feature in all_num_features:
    if '_binned' in feature:
        continue
    
    n_bins = bin_counts[feature]
    
    if n_bins:
        bins = np.unique(train_df[feature].quantile(np.linspace(0, 1, n_bins)).values)
        
        for df in [train_df, test_df]:
            n_bins = bin_counts[feature]
            df[feature + '_binned'] = pd.cut(
                df[feature], bins=bins, duplicates='drop'
            ).cat.codes

        if feature + '_binned' not in all_num_features:
            all_num_features.append(feature + '_binned')
for feature in all_cat_features:
    encoder = LabelEncoder()
    encoder.fit(train_df[feature].append(test_df[feature]))
    train_df[feature + '_label'] = encoder.transform(train_df[feature])
    test_df[feature + '_label'] = encoder.transform(test_df[feature])
for feature in all_num_features:
    if feature.endswith('_log'):
        continue
    
    train_df[feature + '_log'] = np.log(train_df[feature] + 2)
    test_df[feature + '_log'] = np.log(test_df[feature] + 2)
    if feature + '_log' not in all_num_features:
        all_num_features.append(feature + '_log')
for feature in all_num_features:
    scaler = StandardScaler()
    scaler.fit(train_df[feature].append(test_df[feature]).astype(np.float64).values[:, np.newaxis])
    
    train_df[feature + '_scaled'] = scaler.transform(train_df[feature].astype(np.float64).values[:, np.newaxis])
    test_df[feature + '_scaled'] = scaler.transform(test_df[feature].astype(np.float64).values[:, np.newaxis])
y_train = train_df['AdoptionSpeed'].values
n_splits = 5
n_repeats = 5

kfold = RepeatedStratifiedKFold(n_splits=n_splits, 
                                n_repeats=n_repeats,
                                random_state=42)
splits = list(kfold.split(np.empty_like(y_train), y_train))
import multiprocessing
from keras.applications.densenet import preprocess_input, DenseNet121
def resize_to_square(im, size):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

def load_image(path, pet_id, size, flip=False):
    image = cv2.imread(f'{path}{pet_id}-1.jpg')
        
    if flip:
        image = cv2.flip(image, 1)
    
    new_image = resize_to_square(image, size)
    new_image = preprocess_input(new_image)
    return new_image

def extract_img_features():
    img_size = 512
    batch_size = 16
    
    inp = Input((img_size, img_size, 3))

    backbone = DenseNet121(input_tensor=inp, weights=PRETRAINED_WEIGHT_PATH, include_top=False)
    x = backbone.output
    out = GlobalAveragePooling2D()(x)
    m = Model(inp, out)
    
    img_feature_dim = int(m.output.shape[1])
    
    img_features_train = np.zeros((len(train_df), img_feature_dim))
    img_features_test = np.zeros((len(test_df), img_feature_dim))
    img_features_train_flipped = np.zeros((len(train_df), img_feature_dim))

    for df, path, features, kwargs in [
        (train_df, TRAIN_IMAGES_PATH, img_features_train, {}),
        (test_df, TEST_IMAGES_PATH, img_features_test, {}),
        (train_df, TRAIN_IMAGES_PATH, img_features_train_flipped, {'flip': True})
    ]:
        pet_ids = df['PetID'].values
        n_batches = int(np.ceil(len(pet_ids) / batch_size))

        for b in tqdm(range(n_batches)):
            start = b*batch_size
            end = (b+1)*batch_size
            batch_pets = pet_ids[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i,pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image(path, pet_id, img_size, **kwargs)
                except:
                    pass
            batch_preds = m.predict(batch_images)
            for i,pet_id in enumerate(batch_pets):
                features[b * batch_size + i] = batch_preds[i]
                
    np.save('img_features_train.npy', img_features_train)
    np.save('img_features_test.npy', img_features_test)    
    np.save('img_features_train_flipped.npy', img_features_train_flipped)
p = multiprocessing.Process(target=extract_img_features)
p.start()
p.join()
img_features_train = np.load('img_features_train.npy')
img_features_test = np.load('img_features_test.npy')
img_features_train_flipped = np.load('img_features_train_flipped.npy')

img_feature_dim = img_features_train.shape[1]
img_feature_dim
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
    
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, factor=0.6, min_lr=1e-4, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration
        
        self.last_loss = np.inf
        self.min_lr = min_lr
        self.factor = factor
        
    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step(self, loss):
        if loss > self.last_loss:
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.max_lrs = [max(lr * self.factor, self.min_lr) for lr in self.max_lrs]
            
    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
    
def train_model(model, train, valid, test, loss_fn, output_dim, lr=0.001, validate=True, enable_clr=True, verbose=False):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    if enable_clr:
        step_size = 300
        scheduler = CyclicLR(optimizer, base_lr=0.002, max_lr=0.006,
                             step_size=step_size, mode='exp_range',
                             gamma=0.99994)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    best_loss = np.inf
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        
        model.epoch_started(epoch)
        
        for data in tqdm(train_loader, disable=(not verbose)):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)
            if enable_clr:
                scheduler.batch_step()
            
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        valid_preds = np.zeros((len(valid), output_dim))
        
        if validate:
            avg_val_loss = 0.
            for i, data in enumerate(valid_loader):
                x_batch = data[:-1]
                y_batch = data[-1]

                y_pred = model(*x_batch).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size, :] = y_pred.cpu().numpy()

            if avg_val_loss < best_loss:
                if verbose:
                    print('Saving model to best_model.torch')
                torch.save(model.state_dict(), 'best_model.torch')
                best_loss = avg_val_loss

            elapsed_time = time.time() - start_time
            if verbose:
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                      epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        else:
            elapsed_time = time.time() - start_time
            if verbose:
                print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                      epoch + 1, n_epochs, avg_loss, elapsed_time))
   
    valid_preds = np.zeros((len(valid), output_dim))
    
    avg_val_loss = 0.
    for i, data in enumerate(valid_loader):
        x_batch = data[:-1]
        y_batch = data[-1]
        
        y_pred = model(*x_batch).detach()

        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        
        valid_preds[i * batch_size:(i+1) * batch_size, :] = y_pred.cpu().numpy()
    
    if verbose:
        print('Validation loss: ', avg_val_loss)
    
    test_preds = np.zeros((len(test), output_dim))
    
    for i, x_batch in enumerate(test_loader):
        y_pred = model(*x_batch).detach()

        test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred.cpu().numpy()

    if validate:
        model.load_state_dict(torch.load('best_model.torch'))

        valid_preds_earlystop = np.zeros((len(valid), output_dim))

        avg_val_loss = 0.
        for i, data in enumerate(valid_loader):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch).detach()
            
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)            
            valid_preds_earlystop[i * batch_size:(i+1) * batch_size, :] = y_pred.cpu().numpy()
    
        if verbose:
            print('Validation loss: ', avg_val_loss)

        test_preds_earlystop = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(test_loader):
            y_pred = model(*x_batch).detach()

            test_preds_earlystop[i * batch_size:(i+1) * batch_size, :] = y_pred.cpu().numpy()

        return valid_preds, test_preds, valid_preds_earlystop, test_preds_earlystop
    else:
        return valid_preds, test_preds
max_features = None
maxlen = 200
train_text = list(train_df['sentence_text_rnn'].fillna(''))
test_text = list(test_df['sentence_text_rnn'].fillna(''))
all_text = train_text + test_text

tokenizer = Tokenizer(num_words=max_features, filters='')
tokenizer.fit_on_texts(all_text)
train_sequences = tokenizer.texts_to_sequences(train_text)
test_sequences = tokenizer.texts_to_sequences(test_text)

train_text_tokens = pad_sequences(train_sequences, maxlen=maxlen)
test_text_tokens = pad_sequences(test_sequences, maxlen=maxlen)
max_features = max_features or len(tokenizer.word_index) + 1
max_features
def load_glove(word_index, max_features, path):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    
    embeddings_index = []
    for o in tqdm(open(path)):
        try:
            embeddings_index.append(get_coefs(*o.split(" ")))
        except Exception as e:
            print(e)
    
    embeddings_index = dict(embeddings_index)
            
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    unknown_words = []
    
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word.lower())
        
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.upper())
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.capitalize())
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(stemmer.stem(word))
            
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            unknown_words.append(word)

    return embeddings_index, embedding_matrix, unknown_words

def load_crawl(word_index, max_features, path):
    def get_coefs(word,*arr): return word, np.asarray(list(arr) + [0] * max(0, (300 - len(arr))), dtype='float32')
    
    embeddings_index = []
    for o in tqdm(open(path)):
        if len(o) <= 100:
            continue

        try:
            embeddings_index.append(get_coefs(*o.strip().split(" ")))
        except Exception as e:
            print(e)

    embeddings_index = dict(embeddings_index)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, embed_size))
    unknown_words = []
    
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word.lower())
        
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.upper())
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.capitalize())
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(stemmer.stem(word))
        
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
        else:
            unknown_words.append(word)

    return embeddings_index, embedding_matrix, unknown_words
num_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'doc_sent_mag',
                'doc_sent_score', 'MaturitySize', 'FurLength', 'Health', 
                'RescuerID_count_binned', 'Breed1_count_log', 'Breed2_count_log', 
                'gdp_vs_population']

cat_features = ['Dewormed', 'Vaccinated', 'Sterilized', 'Breed1', 'Type',
                'Breed2', 'State', 'Gender', 'Color1', 'Color2', 'Color3', 'no_name']

for image in [1]:
    num_features += [f'dominant_score_{image}', f'dominant_pixel_frac_{image}', 
                     f'dominant_red_{image}', f'dominant_green_{image}', 
                     f'dominant_blue_{image}', f'bounding_importance_{image}', 
                     f'bounding_confidence_{image}', f'vertex_x_{image}', 
                     f'vertex_y_{image}', f'label_score_{image}']
    cat_features += [f'top_label_description_{image}']
batch_size = 128
n_epochs = 10
train_text = train_df['sentence_text_stemmed'].fillna('')
test_text = test_df['sentence_text_stemmed'].fillna('')
all_text = list(train_text) + list(test_text)

word_vectorizer = Pipeline([
    ('cv', TfidfVectorizer(min_df=2,
                           strip_accents='unicode', 
                           analyzer='word',
                           stop_words='english',
                           token_pattern=r'[\w@]{1,}',
                           ngram_range=(1, 3), 
                           use_idf=1, 
                           smooth_idf=1, 
                           sublinear_tf=1)),
])

word_vectorizer.fit(all_text)

word_features_train_desc = word_vectorizer.transform(train_text)
word_features_test_desc = word_vectorizer.transform(test_text)
train_text = train_df['all_label_descriptions'].fillna('')
test_text = test_df['all_label_descriptions'].fillna('')
all_text = list(train_text) + list(test_text)

word_vectorizer = Pipeline([
    ('cv', TfidfVectorizer(ngram_range=(1, 2),
                           use_idf=1,
                           smooth_idf=1,
                           sublinear_tf=1)), 
    ('dim_reduce', TruncatedSVD(n_components=5, random_state=10, algorithm='arpack'))
])

word_vectorizer.fit(all_text)

word_features_train_label = scipy.sparse.csr_matrix(word_vectorizer.transform(train_text))
word_features_test_label = scipy.sparse.csr_matrix(word_vectorizer.transform(test_text))
word_features_train = scipy.sparse.hstack([word_features_train_desc, word_features_train_label], format='coo')
word_features_test = scipy.sparse.hstack([word_features_test_desc, word_features_test_label], format='coo')

del word_features_train_desc
del word_features_test_desc
del word_features_train_label
del word_features_test_label
print(word_features_train.shape)
def scipy_sparse_to_pytorch_dense(matrix):
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda().to_dense()

word_tensor_train = scipy_sparse_to_pytorch_dense(word_features_train)
word_tensor_test = scipy_sparse_to_pytorch_dense(word_features_test)
features = [
    word_tensor_train,
    torch.tensor(img_features_train, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([train_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(train_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]
test_features = [
    word_tensor_test,
    torch.tensor(img_features_test, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([test_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(test_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]

y_train = train_df['AdoptionSpeed'].values
y_train_torch = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32).cuda()

test_dataset = torch.utils.data.TensorDataset(*test_features)
class RNNModel(nn.Module):
    def __init__(self, seed):
        super(RNNModel, self).__init__()
        
        seed_everything(seed)
        
        fc_dropout = 0.1
        embed_size = 300
        emb_dropout = 0.1
        cat_emb_dropout = 0.0
        emb_conc_dropout = 0.1
        img_act_dropout = 0.1
        img_dropout = 0.1
        num_dropout = 0.0
        word_dropout = 0.8
        batch_word_dropout = 0.4
        
        embedding_feature_sizes = [train_df[x + '_label'].append(test_df[x + '_label']).max() + 1 \
                                   for x in cat_features]
        embedding_sizes = [20, 20, 20, 40, 20, 40, 40, 20, 20, 20, 20, 20, 20]
        self.embeddings = nn.ModuleList([
            nn.Embedding(embedding_feature_sizes[i], embedding_sizes[i]) for i in range(len(cat_features))
        ])
        self.cat_embedding_dropout = nn.Dropout(cat_emb_dropout)
        
        self.embedding_linear = nn.Linear(np.sum(embedding_sizes), 32)
        self.embed_dropout = nn.Dropout(emb_conc_dropout)
        
        self.num_linear = nn.Linear(len(num_features), 64)
        self.num_dropout = nn.Dropout(num_dropout)
        
        self.img_act_linear = nn.Linear(img_feature_dim, 64)
        self.img_act_dropout = nn.Dropout(img_act_dropout)
        
        self.word_linear = nn.Linear(word_features_train.shape[1], 8)
        self.word_dropout = nn.Dropout(word_dropout)
        self.batch_word_dropout = nn.Dropout2d(batch_word_dropout)
        
        self.dropout = nn.Dropout(fc_dropout)
        self.linear = nn.Linear(self.img_act_linear.out_features + \
                                self.num_linear.out_features + \
                                self.embedding_linear.out_features + \
                                self.word_linear.out_features, 16)
        self.out = nn.Linear(self.linear.out_features, 1)
        
    def epoch_started(self, epoch):
        pass
    
    def forward(self, x_words, x_img_act, x_num, x_cat):
        activation = F.elu
            
        img_act_out = activation(
            self.img_act_dropout(
                self.img_act_linear(x_img_act)
            )
        )
        
        word_out = activation(
            self.word_dropout(self.batch_word_dropout(
                self.word_linear(
                    x_words
                ).unsqueeze(1)
            ).squeeze(1))
        )
        
        embedding_out = []
        for i in range(len(cat_features)):
            x = self.embeddings[i](x_cat[:, i])
            x = self.cat_embedding_dropout(x)
            embedding_out.append(x)
        
        embedding_conc = torch.cat(embedding_out, 1)
        embedding_conc = activation(
            self.embed_dropout(
                self.embedding_linear(embedding_conc)
            )
        )
        
        num_out = activation(
            self.num_dropout(
                self.num_linear(x_num)
            )
        )
        
        conc = torch.cat([img_act_out, num_out, embedding_conc, word_out], 1)

        conc = self.dropout(conc)
        conc = activation(
            self.linear(conc)
        )
        out = self.out(conc)
        
        return out
def run_nn(index, train_index, val_index):
    full_dataset = torch.utils.data.TensorDataset(*features, y_train_torch)
    train_dataset = torch.utils.data.Subset(full_dataset, train_index)
    valid_dataset = torch.utils.data.Subset(full_dataset, val_index)
    
    model = RNNModel(seed=index + 1)
    model.cuda()
    
    valid_preds, test_preds, _, _ = train_model(model, train_dataset, 
                                                valid_dataset, test_dataset, 
                                                loss_fn=nn.MSELoss(reduction='mean'),
                                                output_dim=1,
                                                enable_clr=True)

    return valid_preds[:, 0], test_preds[:, 0], model
%%time
train_preds_nn_1 = np.zeros((n_repeats, len(train_df)))
test_preds_nn_1 = np.zeros((n_repeats, len(test_df), n_splits))

for i, (train_index, val_index) in enumerate(splits):
    train_preds_nn_1[i // n_splits, val_index], test_preds_nn_1[i // n_splits, :, i % n_splits], _ = run_nn(i, train_index, val_index)
del word_features_train
del word_features_test
del features
del test_features
gc.collect()
torch.cuda.empty_cache()
batch_size = 128
n_epochs = 10
seed_everything()
_, crawl_matrix, unknown_words_crawl = load_crawl(tokenizer.word_index, max_features, EMBEDDING_FILE_CRAWL)
print('words unknown: ', len(unknown_words_crawl))
features = [
    torch.tensor(train_text_tokens, dtype=torch.long).cuda(),
    torch.tensor(img_features_train, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([train_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(train_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]
test_features = [
    torch.tensor(test_text_tokens, dtype=torch.long).cuda(),
    torch.tensor(img_features_test, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([test_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(test_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]

y_train = train_df['AdoptionSpeed'].values
y_train_torch = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32).cuda()

test_dataset = torch.utils.data.TensorDataset(*test_features)
class RNNModel(nn.Module):
    def __init__(self, seed):
        super(RNNModel, self).__init__()
        
        seed_everything(seed)
        
        fc_dropout = 0.1
        embed_size = 300
        emb_dropout = 0.15
        cat_emb_dropout = 0.0
        emb_conc_dropout = 0.1
        img_act_dropout = 0.1
        img_dropout = 0.1
        num_dropout = 0.0
        hidden_size = 20
        
        embedding_feature_sizes = [train_df[x + '_label'].append(test_df[x + '_label']).max() + 1 \
                                   for x in cat_features]
        embedding_sizes = [20, 20, 20, 40, 20, 40, 40, 20, 20, 20, 20, 20, 20]
        self.embeddings = nn.ModuleList([
            nn.Embedding(embedding_feature_sizes[i], embedding_sizes[i]) for i in range(len(cat_features))
        ])
        self.cat_embedding_dropout = nn.Dropout(cat_emb_dropout)
        
        self.embedding_linear = nn.Linear(np.sum(embedding_sizes), 32)
        self.embed_dropout = nn.Dropout(emb_conc_dropout)
        
        self.num_linear = nn.Linear(len(num_features), 64)
        self.num_dropout = nn.Dropout(num_dropout)
        
        self.img_act_linear = nn.Linear(img_feature_dim, 64)
        self.img_act_dropout = nn.Dropout(img_act_dropout)
        
        self.embedding = nn.Embedding(max_features, embed_size,
                                      _weight=nn.Parameter(torch.tensor(crawl_matrix, dtype=torch.float32)))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(emb_dropout)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.1)
        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        
        self.dropout = nn.Dropout(fc_dropout)
        self.linear = nn.Linear(self.img_act_linear.out_features + \
                                self.num_linear.out_features + \
                                self.embedding_linear.out_features + \
                                hidden_size * 4, 16)
        self.out = nn.Linear(self.linear.out_features, 1)
        
    def epoch_started(self, epoch):
        pass
 
    def forward(self, x_words, x_img_act, x_num, x_cat):
        activation = F.elu

        img_act_out = activation(
            self.img_act_dropout(
                self.img_act_linear(x_img_act)
            )
        )

        embedding_out = []
        for i in range(len(cat_features)):
            x = self.embeddings[i](x_cat[:, i])
            x = torch.squeeze(self.cat_embedding_dropout(torch.unsqueeze(x, 0)), 0)
            embedding_out.append(x)
        
        embedding_out = torch.cat(embedding_out, 1)
        embedding_out = activation(
            self.embed_dropout(
                self.embedding_linear(embedding_out)
            )
        )
        
        num_out = activation(
            self.num_dropout(
                self.num_linear(x_num)
            )
        )
        
        h_embedding = self.embedding(x_words)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)), 0)
        
        h_lstm, _ = self.lstm(h_embedding)
        h_lstm_atten = self.lstm_attention(h_lstm)

        max_pool, _ = torch.max(h_lstm, 1)
        
        conc = torch.cat([img_act_out, num_out, embedding_out, max_pool, h_lstm_atten], 1)

        conc = self.dropout(conc)
        conc = activation(
            self.linear(conc)
        )
        out = self.out(conc)
        
        return out

def run_nn(index, train_index, val_index):
    full_dataset = torch.utils.data.TensorDataset(*features, y_train_torch)
    train_dataset = torch.utils.data.Subset(full_dataset, train_index)
    valid_dataset = torch.utils.data.Subset(full_dataset, val_index)
    
    model = RNNModel(seed=index + 1)
    model.cuda()
    
    valid_preds, test_preds, _, _ = train_model(model, train_dataset, 
                                                valid_dataset, test_dataset, 
                                                loss_fn=nn.MSELoss(reduction='mean'),
                                                output_dim=1,
                                                enable_clr=True)
    
    return valid_preds[:, 0], test_preds[:, 0], model
%%time
train_preds_nn_2 = np.zeros((n_repeats, len(train_df)))
test_preds_nn_2 = np.zeros((n_repeats, len(test_df), n_splits))

for i, (train_index, val_index) in enumerate(splits):
    train_preds_nn_2[i // n_splits, val_index], test_preds_nn_2[i // n_splits, :, i % n_splits], _ = run_nn(i, train_index, val_index)
seed_everything()
_, glove_matrix_small, unknown_words_small = load_glove(tokenizer.word_index, max_features, EMBEDDING_FILE_SMALL)
print('words unknown: ', len(unknown_words_small))
features = [
    torch.tensor(train_text_tokens, dtype=torch.long).cuda(),
    torch.tensor(img_features_train_flipped, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([train_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(train_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]
test_features = [
    torch.tensor(test_text_tokens, dtype=torch.long).cuda(),
    torch.tensor(img_features_test, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([test_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(test_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]

y_train = train_df['AdoptionSpeed'].values
y_train_torch = torch.tensor(y_train[:, np.newaxis], dtype=torch.float32).cuda()
    
test_dataset = torch.utils.data.TensorDataset(*test_features)
class RNNModel(nn.Module):
    def __init__(self, seed):
        super(RNNModel, self).__init__()
        
        seed_everything(seed)
        
        fc_dropout = 0.1
        embed_size = 200
        emb_dropout = 0.2
        cat_emb_dropout = 0.0
        emb_conc_dropout = 0.1
        img_act_dropout = 0.1
        img_dropout = 0.1
        num_dropout = 0.0
        hidden_size = 20
        
        embedding_feature_sizes = [train_df[x + '_label'].append(test_df[x + '_label']).max() + 1 \
                                   for x in cat_features]
        embedding_sizes = [20, 20, 20, 40, 20, 40, 40, 20, 20, 20, 20, 20, 20]
        self.embeddings = nn.ModuleList([
            nn.Embedding(embedding_feature_sizes[i], embedding_sizes[i]) for i in range(len(cat_features))
        ])
        self.cat_embedding_dropout = nn.Dropout(cat_emb_dropout)
        
        self.embedding_linear = nn.Linear(np.sum(embedding_sizes), 32)
        self.embed_dropout = nn.Dropout(emb_conc_dropout)
        
        self.num_linear = nn.Linear(len(num_features), 64)
        self.num_dropout = nn.Dropout(num_dropout)
        
        self.img_act_linear = nn.Linear(img_feature_dim, 64)
        self.img_act_dropout = nn.Dropout(img_act_dropout)
        
        self.embedding = nn.Embedding(max_features, embed_size,
                                      _weight=nn.Parameter(torch.tensor(glove_matrix_small, dtype=torch.float32)))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(emb_dropout)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.1)
        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        
        self.dropout = nn.Dropout(fc_dropout)
        self.linear = nn.Linear(self.img_act_linear.out_features + \
                                self.num_linear.out_features + \
                                self.embedding_linear.out_features + \
                                hidden_size * 4, 16)
        self.out = nn.Linear(self.linear.out_features, 1)
        
    def epoch_started(self, epoch):
        if epoch >= 6:
            self.embedding.weight.requires_grad = True
 
    def forward(self, x_words, x_img_act, x_num, x_cat):
        activation = F.elu

        img_act_out = activation(
            self.img_act_dropout(
                self.img_act_linear(x_img_act)
            )
        )

        embedding_out = []
        for i in range(len(cat_features)):
            x = self.embeddings[i](x_cat[:, i])
            x = torch.squeeze(self.cat_embedding_dropout(torch.unsqueeze(x, 0)), 0)
            embedding_out.append(x)
        
        embedding_out = torch.cat(embedding_out, 1)
        embedding_out = activation(
            self.embed_dropout(
                self.embedding_linear(embedding_out)
            )
        )
        
        num_out = activation(
            self.num_dropout(
                self.num_linear(x_num)
            )
        )
        
        h_embedding = self.embedding(x_words)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)), 0)
        
        h_lstm, _ = self.lstm(h_embedding)
        h_lstm_atten = self.lstm_attention(h_lstm)

        max_pool, _ = torch.max(h_lstm, 1)
        
        conc = torch.cat([img_act_out, num_out, embedding_out, max_pool, h_lstm_atten], 1)

        conc = self.dropout(conc)
        conc = activation(
            self.linear(conc)
        )
        out = self.out(conc)
        
        return out

def run_nn(index, train_index, val_index):
    full_dataset = torch.utils.data.TensorDataset(*features, y_train_torch)
    train_dataset = torch.utils.data.Subset(full_dataset, train_index)
    valid_dataset = torch.utils.data.Subset(full_dataset, val_index)
    
    model = RNNModel(seed=index + 1)
    model.cuda()
    
    valid_preds, test_preds, _, _ = train_model(model, train_dataset, 
                                                valid_dataset, test_dataset, 
                                                loss_fn=nn.SmoothL1Loss(reduction='mean'),
                                                output_dim=1,
                                                enable_clr=True)
    return valid_preds[:, 0], test_preds[:, 0], model
%%time
train_preds_nn_3 = np.zeros((n_repeats, len(train_df)))
test_preds_nn_3 = np.zeros((n_repeats, len(test_df), n_splits))

for i, (train_index, val_index) in enumerate(splits):
    train_preds_nn_3[i // n_splits, val_index], test_preds_nn_3[i // n_splits, :, i % n_splits], _ = run_nn(i, train_index, val_index)
features = [
    torch.tensor(train_text_tokens, dtype=torch.long).cuda(),
    torch.tensor(img_features_train, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([train_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(train_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]
test_features = [
    torch.tensor(test_text_tokens, dtype=torch.long).cuda(),
    torch.tensor(img_features_test, dtype=torch.float32).cuda(),
    torch.tensor(np.hstack([test_df[[x + '_scaled' for x in num_features]].values]), dtype=torch.float32).cuda(),
    torch.tensor(test_df[[x + '_label' for x in cat_features]].values, dtype=torch.long).cuda(),
]
all_features = [
    torch.tensor(np.vstack([train_text_tokens, test_text_tokens]), dtype=torch.long).cuda(),
    torch.tensor(np.vstack([img_features_train, img_features_test]), dtype=torch.float32).cuda(),
    torch.tensor(train_df[[x + '_scaled' for x in num_features]].append(test_df[[x + '_scaled' for x in num_features]]).values, dtype=torch.float32).cuda(),
    torch.tensor(train_df[[x + '_label' for x in cat_features]].append(test_df[[x + '_label' for x in cat_features]]).values, dtype=torch.long).cuda(),
]

y_train_torch = torch.tensor(
    train_df[['Type', 'Age_scaled', 'Breed1_label']].append(test_df[['Type', 'Age_scaled', 'Breed1_label']]).values, 
    dtype=torch.float32
).cuda()
# rescale Type feature from 1 to 2 to 0 to 1 for binary crossentropy
y_train_torch[:, 0] -= 1

full_dataset = torch.utils.data.TensorDataset(*all_features, y_train_torch)
out_shape = 3 + train_df['Breed1_label'].append(test_df['Breed1_label']).max()

class ExtractorLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ExtractorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, y_pred, y_true):
        return self.bce_loss(y_pred[:, [0]], y_true[:, [0]]) + \
               self.mse_loss(y_pred[:, [1]], y_true[:, [1]]) + \
               self.ce_loss(y_pred[:, 2:], y_true[:, 2].long())
        
class ExtractorModel(nn.Module):
    def __init__(self, seed):
        super(ExtractorModel, self).__init__()
        
        seed_everything(seed)
        
        img_dropout = 0.1
        emb_dropout = 0.0
        hidden_size = 20
        embed_size = 300
        
        self.img_linear = nn.Linear(img_feature_dim, 64)
        self.img_dropout = nn.Dropout(img_dropout)
        
        self.linear = nn.Linear(self.img_linear.out_features, 16)
        self.out = nn.Linear(self.linear.out_features, out_shape)
        
    def epoch_started(self, epoch):
        pass
        
    def forward(self, x_words, x_img_act, x_num, x_cat, extract_features=False):
        activation = F.elu
        
        h1 = activation(
            self.img_dropout(
                self.img_linear(x_img_act)
            )
        )
        
        h2 = activation(
            self.linear(h1)
        )
        
        if extract_features:
            return torch.cat([h1, h2], 1)
        
        out = self.out(h2)
        return out

full_dataset = torch.utils.data.TensorDataset(*all_features, y_train_torch)
# set valid and test dataset to the same dataset to be able to use train_model without any modifications
valid_dataset = full_dataset
test_dataset = torch.utils.data.TensorDataset(*all_features)

model = ExtractorModel(seed=1234)
model.cuda()

(valid_preds, test_preds, 
 valid_preds_earlystop, test_preds_earlystop) = train_model(model, full_dataset, valid_dataset, test_dataset, 
                                                            loss_fn=ExtractorLoss(),
                                                            output_dim=out_shape,
                                                            enable_clr=True)
def predict_batches(model, x, shape=64, batch_size=64, **kwargs):
    loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
    
    out = np.zeros((len(x), shape))
    
    for i, batch in enumerate(loader):
        start, end = i * batch_size, (i + 1) * batch_size
        end = min(end, len(x))
        
        out[start:end] = model(*batch , **kwargs).detach().cpu().numpy()
    return out
train_dataset = torch.utils.data.TensorDataset(*features)
test_dataset = torch.utils.data.TensorDataset(*test_features)

train_activations = predict_batches(model, train_dataset, batch_size=512, shape=64 + 16, extract_features=True)
test_activations = predict_batches(model, test_dataset, batch_size=512, shape=64 + 16, extract_features=True)
train_text = train_df['sentence_text_stemmed'].fillna('')
test_text = test_df['sentence_text_stemmed'].fillna('')
all_text = list(train_text) + list(test_text)

word_vectorizer = Pipeline([
    ('cv', TfidfVectorizer(min_df=3,  
                           max_features=10000,
                           strip_accents='unicode', 
                           analyzer='word',
                           token_pattern=r'[\w@]{1,}',
                           ngram_range=(1, 3), 
                           use_idf=1, 
                           smooth_idf=1, 
                           sublinear_tf=1)),
    ('svd', TruncatedSVD(n_components=120, algorithm='arpack', random_state=10))
])

word_vectorizer.fit(all_text)

word_features_train_desc = word_vectorizer.transform(train_text)
word_features_test_desc = word_vectorizer.transform(test_text)
from sklearn.decomposition import NMF

train_text = train_df['all_label_descriptions'].fillna('')
test_text = test_df['all_label_descriptions'].fillna('')
all_text = list(train_text) + list(test_text)

word_vectorizer = Pipeline([
    ('cv', TfidfVectorizer(ngram_range=(1, 2),
                           use_idf=1,
                           smooth_idf=1,
                           sublinear_tf=1)), 
    ('dim_reduce', TruncatedSVD(n_components=5, random_state=10, algorithm='arpack'))
])

word_vectorizer.fit(all_text)

word_features_train_label = word_vectorizer.transform(train_text)
word_features_test_label = word_vectorizer.transform(test_text)
word_features_train = np.hstack([word_features_train_desc, word_features_train_label])
word_features_test = np.hstack([word_features_test_desc, word_features_test_label])
word_features_train.shape
num_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'doc_sent_mag',
                'doc_sent_score', 'MaturitySize', 'FurLength', 'Health', 
                'RescuerID_count', 'Breed1_count', 'Breed2_count', 'State_count', 
                'gdp_vs_population'] + aggregate_features

cat_features = ['Dewormed', 'Vaccinated', 'Sterilized', 'Breed1', 'Type', 
                'Dewormed_Vaccinated', 'Breed2', 'State', 'Gender', 'Color1',
                'Color2', 'Color3', 'no_name']

for image in [1]:
    cat_features += [f'top_label_description_{image}']
feature_names = [f'activation_{i}' for i in range(64)] + \
                [f'word_{i}' for i in range(word_features_train.shape[1])] + \
                num_features + cat_features
x_train = np.hstack([
    train_activations[:, :64],
    word_features_train,
    train_df[num_features],
    train_df[[x + '_label' for x in cat_features]]
])
x_test = np.hstack([
    test_activations[:, :64],
    word_features_test,
    test_df[num_features],
    test_df[[x + '_label' for x in cat_features]]    
])

print(len(feature_names))
y_train = train_df['AdoptionSpeed'].values
x_train.shape, x_test.shape
params = {
    'application': 'regression',
    'boosting': 'gbdt',
    'metric': 'mse',
    'num_leaves': 102,
    'learning_rate': 0.02,
    'bagging_fraction': 0.9951448659512921,
    'bagging_freq': 3,
    'feature_fraction': 0.6867901263802068,
    'verbosity': -1,
    'early_stop': 100,
    'verbose_eval': 1000,
    'num_rounds': 10000,
    'raw_seed': 1234,
    'max_bin': 127,
    'min_child_samples': 38,
    'lambda_l1': 0.42651295024341174,
    'lambda_l2': 0.15395842517107572,
    'max_depth': 14,
    'min_split_gain': 0.023658591149106636
}
def run_lightgbm(x_train, y_train, x_valid, y_valid, x_test, index):
    params['seed'] = params['raw_seed'] + index
    num_rounds = params['num_rounds']
    verbose_eval = params['verbose_eval']
    early_stop = params['early_stop']

    x_train_proc, x_valid_proc, x_test_proc = x_train, x_valid, x_test

    dtrain = lgb.Dataset(x_train_proc, y_train, feature_name=feature_names, 
                         categorical_feature=cat_features)
    dvalid = lgb.Dataset(x_valid_proc, y_valid, feature_name=feature_names,
                         categorical_feature=cat_features)

    model = lgb.train(params,
                      train_set=dtrain,
                      valid_sets=(dtrain, dvalid),
                      num_boost_round=num_rounds,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    return  model.predict(x_valid_proc), model.predict(x_test_proc)
%%time
train_preds_lgb = np.zeros((n_repeats, len(train_df)))
test_preds_lgb = np.zeros((n_repeats, len(test_df), n_splits))

for i, (train_index, val_index) in enumerate(splits):
    train_preds_lgb[i // n_splits, val_index], test_preds_lgb[i // n_splits, :, i % n_splits] = run_lightgbm(x_train[train_index], y_train[train_index],
                                                                                                             x_train[val_index], y_train[val_index], x_test, i)
train_text = train_df['sentence_text_stemmed'].fillna('')
test_text = test_df['sentence_text_stemmed'].fillna('')
all_text = list(train_text) + list(test_text)

word_vectorizer = Pipeline([
    ('cv', TfidfVectorizer(min_df=3,  
                           max_features=10000,
                           strip_accents='unicode', 
                           analyzer='word', 
                           token_pattern=r'[\w@]{1,}',
                           ngram_range=(1, 3), 
                           use_idf=1, 
                           smooth_idf=1, 
                           sublinear_tf=1)),
    ('svd', TruncatedSVD(n_components=10, algorithm='arpack', random_state=10))
])

word_vectorizer.fit(all_text)

word_features_train = word_vectorizer.transform(train_text)
word_features_test = word_vectorizer.transform(test_text)
num_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'doc_sent_mag',
                'doc_sent_score', 'MaturitySize', 'FurLength', 'Health', 'RescuerID_count',
                'Breed1_count', 'Breed2_count', 'State_count', 'gdp_vs_population']

cat_features = ['Dewormed', 'Vaccinated', 'Sterilized', 'Breed1', 'Type', 
                'Dewormed_Vaccinated', 'Breed2', 'State', 'Gender', 'Color1', 
                'Color2', 'Color3', 'no_name'] 

for image in [1]:
    num_features += [f'dominant_score_{image}', f'dominant_pixel_frac_{image}', 
                     f'dominant_red_{image}', f'dominant_green_{image}', 
                     f'dominant_blue_{image}', f'bounding_importance_{image}', 
                     f'bounding_confidence_{image}', f'vertex_x_{image}', 
                     f'vertex_y_{image}', f'label_score_{image}']
    cat_features += [f'top_label_description_{image}']
n_word_bins = 10
word_bins = np.unique(pd.Series(word_features_train.flatten()).quantile(np.linspace(0, 1, n_word_bins)))

n_img_bins = 10
img_bins = np.unique(pd.Series(train_activations[:, 64:].flatten()).quantile(np.linspace(0, 1, n_img_bins)))
feature_names = [f'activation_{i}' for i in range(16)] + \
                [f'word_{i}' for i in range(10)] + num_features + cat_features
x_train = np.hstack([
    np.digitize(word_features_train, word_bins),
    np.digitize(train_activations[:, 64:], img_bins),
    train_df[[x + '_binned' for x in num_features]],
    train_df[[x + '_label' for x in cat_features]]    
])
x_test = np.hstack([
    np.digitize(word_features_test, word_bins),
    np.digitize(test_activations[:, 64:], img_bins),
    test_df[[x + '_binned' for x in num_features]],
    test_df[[x + '_label' for x in cat_features]]    
])

print(len(feature_names))
y_train = train_df['AdoptionSpeed'].values
x_train.shape, x_test.shape
categories = num_features + [x + '_label' for x in cat_features]
field_features = defaultdict()

max_val = 1
w = np.round(math.sqrt(2) / math.sqrt(x_train.shape[1]), 20)

for feature_arr, filename in [
    (x_train, 'train.libffm'),
    (x_test, 'test.libffm')
]:
    with open(filename, 'w') as the_file:
        for i in tqdm(range(len(feature_arr))):
            ffeatures = []

            for j in range(feature_arr.shape[1]):
                feature = feature_arr[i, j]
                field = j

                ff = str(field) + '_____' + str(feature)

                if ff not in field_features:
                    if len(field_features) == 0:
                        field_features[ff] = 1
                        max_val += 1
                    else:
                        field_features[ff] = max_val + 1
                        max_val += 1

                fnum = field_features[ff]
                ffeatures.append('{}:{}:{}'.format(field, fnum, w))

            line = [str(y_train[i])] + ffeatures
            the_file.write('{}\n'.format(' '.join(line)))
max_val
ffm_train_data = pd.read_csv('train.libffm', squeeze=True, header=None)
def run_xlearn(index, train_index, val_index):
    x_train_proc, x_valid_proc = ffm_train_data[train_index], ffm_train_data[val_index]
    for path, data in [
        ['ffm_train.txt', x_train_proc],
        ['ffm_valid.txt', x_valid_proc]
    ]:
        with open(path, 'w') as f:
            f.write(data.str.cat(sep='\n'))

    model = xl.create_ffm()
    model.setTrain('./ffm_train.txt')
    model.setValidate('./ffm_valid.txt')
    
    param = {
        'task': 'reg', 
        'lr': 0.2,
        'lambda': 0.0,
        'opt': 'adagrad',
        'k': 4,
        'stop_window': 1
    }
    model.fit(param, './model.out')
    
    model.setTest("./ffm_train.txt")
    model.predict("./model.out", "./output.txt")
    
    with open('output.txt', 'r') as f:
        train_preds = np.array([float(x.strip()) for x in f.readlines()])
    
    model.setTest("./ffm_valid.txt")
    model.predict("./model.out", "./output.txt")
    
    with open('output.txt', 'r') as f:
        valid_preds = np.array([float(x.strip()) for x in f.readlines()])
    
    model.setTest("./test.libffm")
    model.predict("./model.out", "./output.txt")
    
    with open('output.txt', 'r') as f:
        test_preds = np.array([float(x.strip()) for x in f.readlines()])
    
    print(cohen_kappa_score(y_train[val_index], fix_distribution(y_train[train_index], valid_preds), weights='quadratic'))
    return valid_preds, test_preds
%%time
train_preds_ffm = np.zeros((n_repeats, len(train_df)))
test_preds_ffm = np.zeros((n_repeats, len(test_df), n_splits))

for i, (train_index, val_index) in enumerate(splits):
    train_preds_ffm[i // n_splits, val_index], test_preds_ffm[i // n_splits, :, i % n_splits] = run_xlearn(i, train_index, val_index)
params = {
    'application': 'binary',
    'boosting': 'gbdt',
    'metric': 'auc',
    'num_leaves': 80,
    'max_depth': 9,
    'learning_rate': 0.04,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'verbosity': -1,
    'early_stop': 50,
    'verbose_eval': 100,
    'num_rounds': 10000
}
num_features = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'doc_sent_mag',
                'doc_sent_score', 'MaturitySize', 'FurLength', 'Health', 'RescuerID_count',
                'Breed1_count', 'Breed2_count', 'State_count']

cat_features = ['Dewormed', 'Vaccinated', 'Sterilized', 'Breed1', 'Type',
                'Breed2', 'State', 'Gender', 'Color1', 'Color2', 'Color3', 'no_name']

for image in [1]:
    num_features += [f'dominant_score_{image}', f'dominant_pixel_frac_{image}', f'dominant_red_{image}', 
                     f'dominant_green_{image}', f'dominant_blue_{image}', f'bounding_importance_{image}',
                     f'bounding_confidence_{image}',  f'vertex_x_{image}', f'vertex_y_{image}', 
                     f'label_score_{image}']
    cat_features += [f'top_label_description_{image}']
feature_names = num_features + cat_features

x_train = np.hstack([
    train_df[num_features].append(test_df[num_features]),
    train_df[[x + '_label' for x in cat_features]].append(test_df[[x + '_label' for x in cat_features]])
])
y_train_adv = np.concatenate([np.zeros((len(train_df))), np.ones(len(test_df))])

print(f'n pos: {(y_train_adv == 1).sum()}')
print(f'n neg: {(y_train_adv == 0).sum()}')
x_train.shape, y_train_adv.shape
def run_adversarial_validation(x_train, y_train):
    splits = list(StratifiedKFold(n_splits=5, random_state=42, shuffle=True).split(x_train, y_train))
    feature_importance = np.zeros((len(feature_names)))
    models = []

    train_preds = np.zeros((x_train.shape[0]))
    num_rounds = params['num_rounds']
    verbose_eval = params['verbose_eval']
    early_stop = params['early_stop']

    for i, (train_index, val_index) in enumerate(splits):
        dtrain = lgb.Dataset(x_train[train_index], y_train[train_index], 
                             feature_name=feature_names)
        dvalid = lgb.Dataset(x_train[val_index], y_train[val_index], 
                             feature_name=feature_names)

        model = lgb.train(params,
                          train_set=dtrain,
                          valid_sets=(dtrain, dvalid),
                          num_boost_round=num_rounds,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)
        feature_importance += model.feature_importance() / model.feature_importance().sum()
        train_preds[val_index] = model.predict(x_train[val_index])
        models.append(model)

    print('Overall AUC: ', roc_auc_score(y_train, train_preds))
    return train_preds
adv_scores = run_adversarial_validation(x_train, y_train_adv)
sns.jointplot(np.arange(len(train_df)), adv_scores[:len(train_df)], size=10, stat_func=None,
              marginal_kws=dict(bins=15), joint_kws=dict(s=3))
print('NN1: ', cohen_kappa_score(y_train, fix_distribution(y_train, train_preds_nn_1.mean(axis=0)), weights='quadratic'))
print('NN2: ', cohen_kappa_score(y_train, fix_distribution(y_train, train_preds_nn_2.mean(axis=0)), weights='quadratic'))
print('NN3: ', cohen_kappa_score(y_train, fix_distribution(y_train, train_preds_nn_3.mean(axis=0)), weights='quadratic'))
print('LGB: ', cohen_kappa_score(y_train, fix_distribution(y_train, train_preds_lgb.mean(axis=0)), weights='quadratic'))
print('FFM: ', cohen_kappa_score(y_train, fix_distribution(y_train, train_preds_ffm.mean(axis=0)), weights='quadratic'))
pd.DataFrame(np.hstack([
    test_preds_nn_1.mean(axis=0).mean(axis=1)[:, np.newaxis],
    test_preds_nn_2.mean(axis=0).mean(axis=1)[:, np.newaxis],
    test_preds_nn_3.mean(axis=0).mean(axis=1)[:, np.newaxis],
    test_preds_lgb.mean(axis=0).mean(axis=1)[:, np.newaxis],
    test_preds_ffm.mean(axis=0).mean(axis=1)[:, np.newaxis]
]), columns=['NN1', 'NN2', 'NN3', 'LGB', 'FFM']).corr()
scores = []
sample_weights = adv_scores[:len(train_df)]
all_preds = []
to_stack_train = [
    train_preds_nn_1,
    train_preds_nn_2,
    train_preds_nn_3,
    train_preds_lgb,
    train_preds_ffm
]
to_stack_test = [
    test_preds_nn_1,
    test_preds_nn_2,
    test_preds_nn_3,
    test_preds_lgb,
    test_preds_ffm
]

for i in range(n_repeats):
    print(f'Repeat {i}')
    
    clf = linear_model.LinearRegression()
    x_train = np.hstack([x[i][:, np.newaxis] for x in to_stack_train])
    x_test = np.hstack([x[i].mean(axis=1)[:, np.newaxis] for x in to_stack_test])

    corrs = np.array([pd.DataFrame(x[i]).corr().values.mean() for x in to_stack_test])
    
    clf.fit(x_train, y_train, sample_weight=sample_weights)
    coef = clf.coef_
    print('coefs: ', coef)
    print('corrs: ', corrs)
    coef = coef / (corrs ** 2)
    coef = coef / coef.sum()
    print('Adjusted coefs: ', coef)
    print()
    
    all_preds.append((x_test * coef).sum(axis=1))
    
fixed_scores = fix_distribution(y_train, np.array(all_preds).mean(axis=0))
pd.Series(fixed_scores).plot.hist(bins=9)
submission = test_df[['PetID']].copy()
submission['AdoptionSpeed'] = fixed_scores.astype(np.int32)
submission.to_csv('submission.csv', index=False)