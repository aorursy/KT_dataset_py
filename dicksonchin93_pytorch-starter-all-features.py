# standard imports
import time
import random
import os
from IPython.display import display
import numpy as np
import pandas as pd
import gc
from sklearn import metrics
from sklearn import preprocessing
import glob

# pytorch imports
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.checkpoint as checkpoint

# imports for preprocessing the questions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# progress bars
from tqdm import tqdm
tqdm.pandas()

from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    

FILE_DIR = '../input/petfinder-adoption-prediction'
    
TRAIN_PATH = f'{FILE_DIR}/train/train.csv'
TEST_PATH = f'{FILE_DIR}/test/test.csv'
SAMPLE_SUBMISSION_PATH = f'{FILE_DIR}/test/sample_submission.csv'

maxlen = 50
max_namelen = 5

max_features = 95000
batch_size = 512
seed = 1018

def seed_torch(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch()

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
y = train_df['AdoptionSpeed'].values
breed_labels = pd.read_csv(f'{FILE_DIR}/breed_labels.csv')
color_labels = pd.read_csv(f'{FILE_DIR}/color_labels.csv')
state_labels = pd.read_csv(f'{FILE_DIR}/state_labels.csv')


breed_labels['Breed1'] = breed_labels['BreedID']
breed_labels['Breed2'] = breed_labels['BreedID']
color_labels['Color1'] = color_labels['ColorID']
color_labels['Color2'] = color_labels['ColorID']
color_labels['Color3'] = color_labels['ColorID']
state_labels['State'] = state_labels['StateID']
def get_data(df):

    df = df.merge(breed_labels[['Breed1','Type']], on='Breed1', how='left').rename(columns={"Type": "breed_one_catdog"})
    df = df.merge(breed_labels[['Breed2','Type']], on='Breed2',how='left').rename(columns={"Type": "breed_two_catdog"})
    df = df.merge(color_labels[['Color1','ColorName']], on='Color1', how='left').rename(columns={"ColorName": "color1_name"})
    df = df.merge(color_labels[['Color2','ColorName']], on='Color2', how='left').rename(columns={"ColorName": "color2_name"})
    df = df.merge(color_labels[['Color3','ColorName']], on='Color3', how='left').rename(columns={"ColorName": "color3_name"})    
    df = df.merge(state_labels[['State','StateName']], on='State', how='left')
    return df
train_df = get_data(train_df)
test_df = get_data(test_df)
def preprocess_text(text):
    punct = [ '"', ')', '(', '-', '|', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
        '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾']
    for p in punct:
        text = text.replace(p, f' {p} ')
    return text
train_df['Description'] = train_df['Description'].astype(str)
test_df['Description'] = test_df['Description'].astype(str)
train_df['Description'] = train_df['Description'].progress_apply(preprocess_text)
test_df['Description'] = test_df['Description'].progress_apply(preprocess_text)
train_df['Name'] = train_df['Name'].astype(str)
test_df['Name'] = test_df['Name'].astype(str)
train_df['Name'] = train_df['Name'].progress_apply(preprocess_text)
test_df['Name'] = test_df['Name'].progress_apply(preprocess_text)
tknzr = Tokenizer(num_words=max_features, lower=False)
tknzr.fit_on_texts(pd.concat([
    train_df['Description'], 
    train_df['Name'], 
    test_df['Description'], 
    test_df['Name'], 
]).values)
tr_desc_seq = tknzr.texts_to_sequences(train_df['Description'].values)
te_desc_seq = tknzr.texts_to_sequences(test_df['Description'].values)
tr_name_seq = tknzr.texts_to_sequences(train_df['Name'].values)
te_name_seq = tknzr.texts_to_sequences(test_df['Name'].values)
tr_desc_pad = pad_sequences(tr_desc_seq, maxlen=maxlen)
te_desc_pad = pad_sequences(te_desc_seq, maxlen=maxlen)
tr_name_pad = pad_sequences(tr_name_seq, maxlen=max_namelen)
te_name_pad = pad_sequences(te_name_seq, maxlen=max_namelen)
categoricals = ['Breed1','Breed2','Gender','Color1','Color2','Color3','State','Vaccinated','MaturitySize','Dewormed','Health','RescuerID',
               'Sterilized','Type_y','Type_x','FurLength']
for cat in tqdm(categoricals):
    dtype = train_df[cat].dtype
    print(train_df[cat].dtype, cat)
    if dtype == 'int64':
        train_df[cat] = train_df[cat].fillna(-1)
        test_df[cat] = test_df[cat].fillna(-1)
    elif dtype == 'float64':
        train_df[cat] = train_df[cat].fillna(-1.0)
        test_df[cat] = test_df[cat].fillna(-1.0)
    else:
        train_df[cat] = train_df[cat].fillna('unknown')
        test_df[cat] = test_df[cat].fillna('unknown')
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate([train_df[cat].values,test_df[cat].values]))
    train_df[cat] = le.transform(train_df[cat])
    test_df[cat] = le.transform(test_df[cat])
def to_categorical_idx(col, df_trn, df_test):
    merged = pd.concat([df_trn[col], df_test[col]])
    train_size = df_trn[col].shape[0]
    idxs, uniques = pd.factorize(merged)
    return idxs[:train_size], idxs[train_size:], uniques

tr_breed1, te_breed1, tknzr_breed1    = to_categorical_idx('Breed1', train_df, test_df)
tr_breed2, te_breed2, tknzr_breed2    = to_categorical_idx('Breed2', train_df, test_df)
tr_gen, te_gen, tknzr_gen   = to_categorical_idx('Gender', train_df, test_df)
tr_col1, te_col1, tknzr_col1   = to_categorical_idx('Color1', train_df, test_df)
tr_col2, te_col2, tknzr_col2   = to_categorical_idx('Color2', train_df, test_df)
tr_col3, te_col3, tknzr_col3   = to_categorical_idx('Color3', train_df, test_df)
tr_state, te_state, tknzr_state  = to_categorical_idx('State', train_df, test_df)
tr_vac, te_vac, tknzr_vac  = to_categorical_idx('Vaccinated', train_df, test_df)
tr_msize, te_msize, tknzr_msize  = to_categorical_idx('MaturitySize', train_df, test_df)
tr_dworm, te_dworm, tknzr_dworm  = to_categorical_idx('Dewormed', train_df, test_df)
tr_health, te_health, tknzr_health  = to_categorical_idx('Health', train_df, test_df)
tr_rid, te_rid, tknzr_rid = to_categorical_idx('RescuerID', train_df, test_df)
tr_ster, te_ster, tknzr_ster = to_categorical_idx('Sterilized', train_df, test_df)
tr_ty, te_ty, tknzr_ty= to_categorical_idx('Type_y', train_df, test_df)
tr_tx, te_tx, tknzr_tx= to_categorical_idx('Type_x', train_df, test_df)
tr_fl, te_fl, tknzr_fl= to_categorical_idx('FurLength', train_df, test_df)

from nltk.stem import WordNetLemmatizer
            
def load_fasttext(word_index):
    EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))
    return embeddings_index 

def load_glove(word_index):
    EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)
    return embeddings_index 
embeddings_index_1 = load_fasttext(tknzr.word_index)
embeddings_index_2 = load_glove(tknzr.word_index)

all_embs = np.stack(embeddings_index_2.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tknzr.word_index
nb_words = min(max_features, len(word_index)+1)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 300))
for word, i in word_index.items():
    if i >= nb_words: continue
    embedding_vector = embeddings_index_2.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
max_emb = 64

emb_breed1   = min(max_emb,(len(tknzr_breed1) + 1)//2)
emb_breed2   = min(max_emb,(len(tknzr_breed2) + 1)//2)
emb_gen   = min(max_emb,(len(tknzr_gen) + 1)//2)
emb_col1   = min(max_emb,(len(tknzr_col1) + 1)//2)
emb_col2   = min(max_emb,(len(tknzr_col2) + 1)//2)
emb_col3   = min(max_emb,(len(tknzr_col3) + 1)//2)
emb_state   = min(max_emb,(len(tknzr_state) + 1)//2)
emb_vac   = min(max_emb,(len(tknzr_vac) + 1)//2)
emb_msize   = min(max_emb,(len(tknzr_msize) + 1)//2)
emb_dworm   = min(max_emb,(len(tknzr_dworm) + 1)//2)
emb_health   = min(max_emb,(len(tknzr_health) + 1)//2)
emb_rid = min(max_emb,(len(tknzr_rid) + 1)//2)
emb_ster   = min(max_emb,(len(tknzr_ster) + 1)//2)
emb_ty   = min(max_emb,len(tknzr_ty))
emb_tx   = min(max_emb,len(tknzr_tx))
emb_fl   = min(max_emb,(len(tknzr_fl) + 1)//2)

train_df.head()
train_id = train_df['PetID']
test_id = test_df['PetID']

doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train_id:
    try:
        with open('../input/train_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

train_df.loc[:, 'doc_sent_mag'] = doc_sent_mag
train_df.loc[:, 'doc_sent_score'] = doc_sent_score

doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test_id:
    try:
        with open('../input/test_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

test_df.loc[:, 'doc_sent_mag'] = doc_sent_mag
test_df.loc[:, 'doc_sent_score'] = doc_sent_score
vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in train_id:
    try:
        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
print(nl_count)
train_df.loc[:, 'vertex_x'] = vertex_xs
train_df.loc[:, 'vertex_y'] = vertex_ys
train_df.loc[:, 'bounding_confidence'] = bounding_confidences
train_df.loc[:, 'bounding_importance'] = bounding_importance_fracs
train_df.loc[:, 'dominant_blue'] = dominant_blues
train_df.loc[:, 'dominant_green'] = dominant_greens
train_df.loc[:, 'dominant_red'] = dominant_reds
train_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train_df.loc[:, 'dominant_score'] = dominant_scores
train_df.loc[:, 'label_description'] = label_descriptions
train_df.loc[:, 'label_score'] = label_scores


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in test_id:
    try:
        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
test_df.loc[:, 'vertex_x'] = vertex_xs
test_df.loc[:, 'vertex_y'] = vertex_ys
test_df.loc[:, 'bounding_confidence'] = bounding_confidences
test_df.loc[:, 'bounding_importance'] = bounding_importance_fracs
test_df.loc[:, 'dominant_blue'] = dominant_blues
test_df.loc[:, 'dominant_green'] = dominant_greens
test_df.loc[:, 'dominant_red'] = dominant_reds
test_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
test_df.loc[:, 'dominant_score'] = dominant_scores
test_df.loc[:, 'label_description'] = label_descriptions
test_df.loc[:, 'label_score'] = label_scores
train_image_path = '../input/petfinder-adoption-prediction/train_images'
test_image_path = '../input/petfinder-adoption-prediction/test_images'
from PIL import Image
from torch.utils import data as D

class PureImageDataset(D.Dataset):
    """
    A customized data loader.
    """
    def __init__(self, path, pet_id):
        """ Intialize the dataset
        """
        self.filenames = []
       
        self.transform = transforms.ToTensor()
    
        filenames = glob.glob(os.path.join(path, '*.jpg'))
        tmp_filenames = []
        for image_path in filenames:
            tmp_filenames.append(image_path.split('/')[-1].split('-')[0])
        for pid in pet_id:
            try:
                self.filenames.append(filenames[tmp_filenames.index(pid)])
            except:
                self.filenames.append(np.nan)
        self.len = len(self.filenames)
        
    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        try:
            image = Image.open(self.filenames[index])
            image = image.resize((224,224))
            images = np.transpose(image, (2, 0, 1))            
            images = image_to_tensor_transform(images)
        except:
            image = Image.new('RGB', (224,224))
            image = np.asarray(image)
            images = np.transpose(image, (2, 0, 1))     
            images = image_to_tensor_transform(images)
       
        return images
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
numerical_feats = ['Quantity','FurLength','Age','Quantity','Fee','VideoAmt','PhotoAmt','doc_sent_mag', 'doc_sent_score', 'dominant_score', 
                   'dominant_pixel_frac', 'dominant_red', 'dominant_green', 'dominant_blue', 'bounding_importance', 'bounding_confidence',
                   'vertex_x', 'vertex_y', 'label_score']
train_features = train_df[numerical_feats].fillna(0).values
test_features = test_df[numerical_feats].fillna(0).values
ss = StandardScaler()
ss.fit(np.vstack((train_features, test_features)))
train_features = ss.transform(train_features)
test_features = ss.transform(test_features)
from torchvision import transforms, datasets, models
from tqdm import tqdm_notebook as tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
cnn_model =  models.resnet50(pretrained=False)
cnn_model.load_state_dict(torch.load('../input/pretrained-pytorch-models/resnet50-19c8e357.pth', map_location=lambda storage, loc: storage))
num_ftrs = cnn_model.fc.in_features
cnn_model.fc = torch.nn.Linear(num_ftrs, int(num_ftrs/2))
cnn_model.cuda()

cnn_model2 =  models.vgg16(pretrained=False)
cnn_model2.load_state_dict(torch.load('../input/vgg16/vgg16.pth', map_location=lambda storage, loc: storage))
cnn_model2.cuda()


layer = cnn_model._modules.get('layer3')
layer2 = cnn_model2._modules.get('features')

def image_to_tensor_transform(image):
    mean=[0.5, 0.5, 0.5]
    std =[0.5, 0.5, 0.5]
    tensor = torch.from_numpy(image).float().div(255)
    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor.cuda()
batch_size = 50
train = PureImageDataset(train_image_path,train_df['PetID'].values)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
train_imgnet = np.zeros((len(train_df),1024+512))
#train_imgnet = np.zeros((len(train_df), 1024, 14, 14))
avgpool = nn.AvgPool2d(14)
avgpool2 = nn.AvgPool2d(7)
flattener = Flatten()
last = len(train_df)//batch_size
for i, (x_img) in tqdm(enumerate(train_loader),total=len(train_df)//batch_size):

    my_embedding = torch.zeros((x_img.shape[0], 1024, 14, 14))
    my_embedding2 = torch.zeros((x_img.shape[0], 512, 7, 7))
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    def copy_data2(m, i, o):
        my_embedding2.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    h2 = layer2.register_forward_hook(copy_data2)
    y_pred = cnn_model(x_img).detach().cpu().numpy()
    y_pred2 = cnn_model2(x_img).detach().cpu().numpy()
    h.remove()
    h2.remove()
    my_embedding = flattener(avgpool(my_embedding)).detach().cpu().numpy()
    my_embedding2 = flattener(avgpool2(my_embedding2)).detach().cpu().numpy()
    train_imgnet[i * batch_size:(i+1) * batch_size, :1024] = my_embedding
    train_imgnet[i * batch_size:(i+1) * batch_size, 1024:] = my_embedding2
test = PureImageDataset(test_image_path,test_df['PetID'].values)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
test_imgnet = np.zeros((len(test_df),1024+512))
for i, (x_img) in tqdm(enumerate(test_loader), total=len(test_df)//batch_size):
    my_embedding = torch.zeros((x_img.shape[0], 1024, 14, 14))
    my_embedding2 = torch.zeros((x_img.shape[0], 512, 7, 7))
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    def copy_data2(m, i, o):
        my_embedding2.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    h2 = layer2.register_forward_hook(copy_data2)
    y_pred = cnn_model(x_img).detach().cpu().numpy()
    y_pred2 = cnn_model2(x_img).detach().cpu().numpy()
    h.remove()
    h2.remove()
    my_embedding = flattener(avgpool(my_embedding)).detach().cpu().numpy()
    my_embedding2 = flattener(avgpool2(my_embedding2)).detach().cpu().numpy()
    test_imgnet[i * batch_size:(i+1) * batch_size, :1024] = my_embedding
    test_imgnet[i * batch_size:(i+1) * batch_size, 1024:] = my_embedding2
train_df.head()
from torchvision import transforms, datasets, models

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.cnn_model =  nn.Linear(1024+512, 500)
        self.cnn_model2 =  nn.Linear(500, 500)
        self.batchnorm_img = nn.BatchNorm1d(500)
        
        self.embedding = nn.Embedding(max_features, embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.emb_breed1   = nn.Embedding(len(tknzr_breed1), emb_breed1)
        self.linear_breed1 = nn.Linear(64, 10)
        self.emb_breed2   = nn.Embedding(len(tknzr_breed2), emb_breed2)
        self.linear_breed2 = nn.Linear(64, 10)
        self.emb_gen   = nn.Embedding(len(tknzr_gen), emb_gen)
        self.linear_gen = nn.Linear(2, 10)
        self.emb_col1   = nn.Embedding(len(tknzr_col1), emb_col1)
        self.linear_col1 = nn.Linear(4, 10)
        self.emb_col2   = nn.Embedding(len(tknzr_col2), emb_col2)
        self.linear_col2 = nn.Linear(4, 10)
        self.emb_col3   = nn.Embedding(len(tknzr_col3), emb_col3)
        self.linear_col3 = nn.Linear(3, 10)
        self.emb_state   = nn.Embedding(len(tknzr_state), emb_state)
        self.linear_state = nn.Linear(7, 10)
        self.emb_flatten = Flatten()
        self.emb_vac = nn.Embedding(len(tknzr_vac), emb_vac)
        self.linear_vac = nn.Linear(2, 2)
        self.emb_msize = nn.Embedding(len(tknzr_msize), emb_msize)
        self.linear_msize = nn.Linear(2, 2)
        self.emb_dworm = nn.Embedding(len(tknzr_dworm), emb_dworm)
        self.linear_dworm = nn.Linear(2, 2)
        self.emb_health = nn.Embedding(len(tknzr_health), emb_health)
        self.linear_health = nn.Linear(2, 2)
        self.emb_rid = nn.Embedding(len(tknzr_rid), emb_rid)
        self.linear_rid = nn.Linear(64, 100)
        self.emb_ster = nn.Embedding(len(tknzr_ster), emb_ster)
        self.linear_ster = nn.Linear(2, 2)
        self.emb_ty = nn.Embedding(len(tknzr_ty), emb_ty)
        self.linear_ty = nn.Linear(3, 5)
        self.emb_tx = nn.Embedding(len(tknzr_tx), emb_tx)
        self.linear_tx = nn.Linear(2, 2)
        self.emb_fl = nn.Embedding(len(tknzr_fl), emb_fl)
        self.linear_fl = nn.Linear(2, 2)
    
        self.embedding_dropout = nn.Dropout2d(0.1)
        
        self.lstm = nn.LSTM(embedding_matrix.shape[1], 256, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.lstm_name = nn.LSTM(embedding_matrix.shape[1], 20, bidirectional=True, batch_first=True)
        self.gru_name = nn.GRU(40, 20, bidirectional=True, batch_first=True)
        self.dense_rnn = nn.Linear(512+40, 256)
        self.batchnorm_rnn = nn.BatchNorm1d(256)
        self.dropout_rnn = nn.Dropout(0.2)
        
        self.linear_num = nn.Linear(19, 5000)
        self.linear_cat = nn.Linear(184, 5000)
        self.batchnorm_cat = nn.BatchNorm1d(5000)
        self.batchnorm_num = nn.BatchNorm1d(5000)
        
        
        self.linear_one = nn.Linear(10756, 500)
        self.batchnorm = nn.BatchNorm1d(500)
        self.linear_two = nn.Linear(500, 200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(200, num_classes)
    
    def forward(self, x_img, x, x_name, breed1, breed2, gen, col1, col2, col3, state,vac,msize,dworm, health,rid,ster,ty,tx,fl, numerical):
        
        img_feat = self.cnn_model(x_img)
        img_feat = self.dropout(img_feat)
        img_feat = self.cnn_model2(img_feat)
        img_feat = self.dropout(img_feat)
        #img_feat = self.batchnorm_img(img_feat)
        
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        h2_embedding = self.embedding(x_name)
        h2_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h2_embedding, 0)))
        
            
        f1 = self.emb_breed1(breed1)
        f1 = self.linear_breed1(f1)
        f2 = self.emb_breed2(breed2)
        f2 = self.linear_breed2(f2)
        f3 = self.emb_gen(gen)
        f3 = self.linear_gen(f3)
        f4 = self.emb_col1(col1)
        f4 = self.linear_col1(f4)
        f5 = self.emb_col2(col2)
        f5 = self.linear_col2(f5)  
        f6 = self.emb_col3(col3)
        f6 = self.linear_col3(f6)
        f7 = self.emb_state(state)
        f7 = self.linear_state(f7)
        f8 = self.emb_vac(vac)
        f8 = self.linear_vac(f8)
        f9 = self.emb_msize(msize)
        f9 = self.linear_msize(f9)
        f10 = self.emb_dworm(dworm)
        f10 = self.linear_dworm(f10)
        f11 = self.emb_health(health)
        f11 = self.linear_health(f11)
        f12 = self.emb_rid(rid)
        f12 = self.linear_rid(f12)
        f13 = self.emb_ster(ster)
        f13 = self.linear_ster(f13)
        f14 = self.emb_ty(ty)
        f14 = self.linear_ty(f14)
        f15 = self.emb_tx(tx)
        f15 = self.linear_tx(f15)
        f16 = self.emb_fl(fl)
        f16 = self.linear_fl(f16)
        cat_conc = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f15, f16), -1)
        #cat_conc = self.emb_flatten(cat_conc)
        cat_conc = self.linear_cat(cat_conc)
        #cat_conc = self.batchnorm_cat(cat_conc)
        cat_conc = self.dropout(cat_conc)
        
        h_lstm, _ = self.lstm(h_embedding)
        h_gru, (ht, ct) = self.gru(h_lstm)
        
        
        h2_lstm, _ = self.lstm_name(h2_embedding)
        h2_gru, (ht2, ct2) = self.gru_name(h2_lstm)
        
        # global average pooling
        #avg_pool = torch.mean(h_gru, 1)
        #avg_pool2 = torch.mean(h2_gru, 1)
        # global max pooling
        #max_pool, _ = torch.max(h_gru, 1)
        #max_pool2, _ = torch.max(h2_gru, 1)
        # last state
        last_state = torch.cat([ht, ct], 1)
        last_state2 = torch.cat([ht2, ct2], 1)
        final_state = torch.cat([last_state, last_state2], 1)
        final_state = self.dense_rnn(final_state)
        #final_state = self.batchnorm_rnn(final_state)
        final_state = self.dropout_rnn(final_state)
        
        
        numerical = self.linear_num(numerical)
        #numerical = self.batchnorm_num(numerical)
        numerical = self.dropout(numerical)
        
        conc = torch.cat((numerical, cat_conc, final_state, img_feat), 1)
        #conc = self.batchnorm(conc)
        conc = self.relu(self.linear_one(conc))
        #conc = self.batchnorm(conc)
        conc = self.dropout(conc)
        conc = self.relu(self.linear_two(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        
        return out
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
num_classes = len(np.unique(y))
batch_size = 516
n_folds=5
epochs=30
train_preds = np.zeros((len(train_df),num_classes))
test_preds = np.zeros((len(test_df),num_classes))

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


#test_pet_img = PetImageDataset(test_image_path,test_df['PetID'].values, aug='False')
x_img_test_cuda = torch.tensor(test_imgnet, dtype=torch.float32).cuda()
x_test_cuda = torch.tensor(te_desc_pad, dtype=torch.long).cuda()
x_test_name_cuda = torch.tensor(te_name_pad, dtype=torch.long).cuda()
te_breed1_cuda = torch.tensor(te_breed1, dtype=torch.long).cuda()
te_breed2_cuda = torch.tensor(te_breed2, dtype=torch.long).cuda()
te_gen_cuda = torch.tensor(te_gen, dtype=torch.long).cuda()
te_col1_cuda = torch.tensor(te_col1, dtype=torch.long).cuda()
te_col2_cuda = torch.tensor(te_col2, dtype=torch.long).cuda()
te_col3_cuda = torch.tensor(te_col3, dtype=torch.long).cuda()
te_state_cuda = torch.tensor(te_state, dtype=torch.long).cuda()
te_vac_cuda = torch.tensor(te_vac, dtype=torch.long).cuda()
te_msize_cuda = torch.tensor(te_msize, dtype=torch.long).cuda()
te_dworm_cuda = torch.tensor(te_dworm, dtype=torch.long).cuda()
te_health_cuda = torch.tensor(te_health, dtype=torch.long).cuda()
te_rid_cuda = torch.tensor(te_rid, dtype=torch.long).cuda()
te_ster_cuda = torch.tensor(te_ster, dtype=torch.long).cuda()
te_ty_cuda = torch.tensor(te_ty, dtype=torch.long).cuda()
te_tx_cuda = torch.tensor(te_tx, dtype=torch.long).cuda()
te_fl_cuda = torch.tensor(te_fl, dtype=torch.long).cuda()

x_test_feats_cuda = torch.tensor(test_features, dtype=torch.float32).cuda()

test = torch.utils.data.TensorDataset(x_img_test_cuda, x_test_cuda,x_test_name_cuda, te_breed1_cuda, te_breed2_cuda,
                                      te_gen_cuda, te_col1_cuda, te_col2_cuda, te_col3_cuda, te_state_cuda,te_vac_cuda,te_msize_cuda,
                                      te_dworm_cuda,te_health_cuda,te_rid_cuda,te_ster_cuda,te_ty_cuda,te_tx_cuda,te_fl_cuda, x_test_feats_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(tr_desc_pad, y)) # shuffle or no?

for fold_id, (train_idx, valid_idx) in enumerate(splits):
    
    x_img_train_fold = torch.tensor(train_imgnet[train_idx], dtype=torch.float32).cuda()
    x_train_fold = torch.tensor(tr_desc_pad[train_idx], dtype=torch.long).cuda()
    x_name_train_fold = torch.tensor(tr_name_pad[train_idx], dtype=torch.long).cuda()
    tr_breed1_fold = torch.tensor(tr_breed1[train_idx], dtype=torch.long).cuda()
    tr_breed2_fold = torch.tensor(tr_breed2[train_idx], dtype=torch.long).cuda()
    tr_gen_fold = torch.tensor(tr_gen[train_idx], dtype=torch.long).cuda()
    tr_col1_fold = torch.tensor(tr_col1[train_idx], dtype=torch.long).cuda()
    tr_col2_fold = torch.tensor(tr_col2[train_idx], dtype=torch.long).cuda()
    tr_col3_fold = torch.tensor(tr_col3[train_idx], dtype=torch.long).cuda()
    tr_state_fold = torch.tensor(tr_state[train_idx], dtype=torch.long).cuda()
    tr_vac_fold = torch.tensor(tr_vac[train_idx], dtype=torch.long).cuda()
    tr_msize_fold = torch.tensor(tr_msize[train_idx], dtype=torch.long).cuda()
    tr_dworm_fold = torch.tensor(tr_dworm[train_idx], dtype=torch.long).cuda()
    tr_health_fold = torch.tensor(tr_health[train_idx], dtype=torch.long).cuda()
    tr_rid_fold = torch.tensor(tr_rid[train_idx], dtype=torch.long).cuda()
    tr_ster_fold = torch.tensor(tr_ster[train_idx], dtype=torch.long).cuda()
    tr_ty_fold = torch.tensor(tr_ty[train_idx], dtype=torch.long).cuda()
    tr_tx_fold = torch.tensor(tr_tx[train_idx], dtype=torch.long).cuda()
    tr_fl_fold = torch.tensor(tr_fl[train_idx], dtype=torch.long).cuda()
    x_feats_train_fold = torch.tensor(train_features[train_idx], dtype=torch.float32).cuda()
    y_train_fold = torch.tensor(y[train_idx], dtype=torch.long).cuda()
    #train = PetImageDataset(train_image_path,train_df['PetID'].values[train_idx],x_train_fold,x_name_train_fold, tr_breed1_fold, tr_breed2_fold, tr_gen_fold, tr_col1_fold, tr_col2_fold,
    #                                       tr_col3_fold, tr_state_fold, x_feats_train_fold, y_train_fold, aug='True')
    
    x_img_val_fold = torch.tensor(train_imgnet[valid_idx], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(tr_desc_pad[valid_idx], dtype=torch.long).cuda()
    x_name_val_fold = torch.tensor(tr_name_pad[valid_idx], dtype=torch.long).cuda()
    val_breed1_fold = torch.tensor(tr_breed1[valid_idx], dtype=torch.long).cuda()
    val_breed2_fold = torch.tensor(tr_breed2[valid_idx], dtype=torch.long).cuda()
    val_gen_fold = torch.tensor(tr_gen[valid_idx], dtype=torch.long).cuda()
    val_col1_fold = torch.tensor(tr_col1[valid_idx], dtype=torch.long).cuda()
    val_col2_fold = torch.tensor(tr_col2[valid_idx], dtype=torch.long).cuda()
    val_col3_fold = torch.tensor(tr_col3[valid_idx], dtype=torch.long).cuda()
    val_state_fold = torch.tensor(tr_state[valid_idx], dtype=torch.long).cuda()
    val_vac_fold = torch.tensor(tr_vac[valid_idx], dtype=torch.long).cuda()
    val_msize_fold = torch.tensor(tr_msize[valid_idx], dtype=torch.long).cuda()
    val_dworm_fold = torch.tensor(tr_dworm[valid_idx], dtype=torch.long).cuda()
    val_health_fold = torch.tensor(tr_health[valid_idx], dtype=torch.long).cuda()
    val_rid_fold = torch.tensor(tr_rid[valid_idx], dtype=torch.long).cuda()
    val_ster_fold = torch.tensor(tr_ster[valid_idx], dtype=torch.long).cuda()
    val_ty_fold = torch.tensor(tr_ty[valid_idx], dtype=torch.long).cuda()
    val_tx_fold = torch.tensor(tr_tx[valid_idx], dtype=torch.long).cuda()
    val_fl_fold = torch.tensor(tr_fl[valid_idx], dtype=torch.long).cuda()
    x_feats_val_fold = torch.tensor(train_features[valid_idx], dtype=torch.float32).cuda()
    y_val_fold = torch.tensor(y[valid_idx], dtype=torch.long).cuda()
    #valid = PetImageDataset(train_image_path,train_df['PetID'].values[valid_idx], x_val_fold,x_name_val_fold, val_breed1_fold, val_breed2_fold, val_gen_fold, val_col1_fold, val_col2_fold,
    #                                       val_col3_fold, val_state_fold, x_feats_val_fold, y_val_fold, aug='False')

    model = NeuralNet()   
    model.cuda()   

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train = torch.utils.data.TensorDataset(x_img_train_fold, x_train_fold,x_name_train_fold, tr_breed1_fold, tr_breed2_fold, tr_gen_fold, tr_col1_fold, tr_col2_fold,
                                           tr_col3_fold, tr_state_fold,tr_vac_fold,tr_msize_fold, tr_dworm_fold,tr_health_fold,tr_rid_fold,
                                           tr_ster_fold,tr_ty_fold,tr_tx_fold,tr_fl_fold,x_feats_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_img_val_fold, x_val_fold,x_name_val_fold, val_breed1_fold, val_breed2_fold, val_gen_fold, val_col1_fold, val_col2_fold,
                                           val_col3_fold, val_state_fold,val_vac_fold,val_msize_fold,val_dworm_fold,val_health_fold,val_rid_fold,
                                           val_ster_fold,val_ty_fold,val_tx_fold,val_fl_fold,x_feats_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # start training on fold
    print(f'Fold {fold_id + 1}')

    best_valid_loss = 1e10

    for epoch in range(epochs):
        start_time = time.time()

        # train
        model.train()
        avg_loss = 0.
        for i, (x_img, x_batch,x_name_batch,tr_breed1_batch, tr_breed2_batch, tr_gen_batch, tr_col1_batch, tr_col2_batch,
                    tr_col3_batch, tr_state_batch,tr_vac_batch, tr_msize_batch,tr_dworm_batch,tr_health_batch,tr_rid_batch,tr_ster_batch,
                    tr_ty_batch,tr_tx_batch,tr_fl_batch,x_feats_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x_img, x_batch,x_name_batch,tr_breed1_batch, tr_breed2_batch, tr_gen_batch, tr_col1_batch, tr_col2_batch,
                    tr_col3_batch, tr_state_batch,tr_vac_batch,tr_msize_batch,tr_dworm_batch,tr_health_batch,tr_rid_batch,tr_ster_batch,
                    tr_ty_batch,tr_tx_batch,tr_fl_batch,x_feats_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_idx)

        # evaluate
        model.eval()
        valid_preds_fold = np.zeros((x_val_fold.size(0), num_classes))
        test_preds_fold = np.zeros((len(te_desc_pad),num_classes))
        avg_val_loss = 0.
        for i, (x_img, x_batch,x_name_batch,tr_breed1_batch, tr_breed2_batch, tr_gen_batch, tr_col1_batch, tr_col2_batch,
                    tr_col3_batch, tr_state_batch,tr_vac_batch,tr_msize_batch,tr_dworm_batch,tr_health_batch,tr_rid_batch,tr_ster_batch,
                    tr_ty_batch,tr_tx_batch,tr_fl_batch,x_feats_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_img, x_batch,x_name_batch,tr_breed1_batch, tr_breed2_batch, tr_gen_batch, tr_col1_batch, tr_col2_batch,
                        tr_col3_batch, tr_state_batch,tr_vac_batch,tr_msize_batch,tr_dworm_batch,tr_health_batch,tr_rid_batch,tr_ster_batch,
                        tr_ty_batch,tr_tx_batch,tr_fl_batch, x_feats_batch).detach()

            val_loss = loss_fn(y_pred, y_batch).item() 
            avg_val_loss += val_loss / len(valid_idx)
            valid_preds_fold[i * batch_size:(i+1) * batch_size] = softmax(y_pred.cpu().numpy())
        qwk = quadratic_weighted_kappa(y[valid_idx], np.argmax(valid_preds_fold, axis=1))
        elapsed_time = time.time() - start_time 
        print('quadratic_weighted_kappa: {}'.format(qwk))
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

        # checkpoint
        if avg_val_loss < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = avg_val_loss
            save_checkpoint('malaysia.pth', model, optimizer)

    # create predictions
    # val_preds['fold{}'.format(fold_id+1)] = [valid_preds_fold, training_labels[valid_idx]]

    load_checkpoint('malaysia.pth', model, optimizer)            
    for i,(x_img, x_batch,x_name_batch, tr_breed1_batch, tr_breed2_batch, tr_gen_batch, tr_col1_batch, tr_col2_batch,
                    tr_col3_batch, tr_state_batch,tr_vac_batch,tr_msize_batch,tr_dworm_batch,tr_health_batch,tr_rid_batch,tr_ster_batch,
                    tr_ty_batch,tr_tx_batch,tr_fl_batch,x_feats_batch, ) in enumerate(test_loader):
        y_pred = model(x_img, x_batch,x_name_batch,tr_breed1_batch, tr_breed2_batch, tr_gen_batch, tr_col1_batch, tr_col2_batch,
                        tr_col3_batch, tr_state_batch,tr_vac_batch,tr_msize_batch,tr_dworm_batch,tr_health_batch,tr_rid_batch,tr_ster_batch,
                        tr_ty_batch,tr_tx_batch,tr_fl_batch,x_feats_batch).detach()
        test_preds_fold[i * batch_size:(i+1) * batch_size] = softmax(y_pred.cpu().numpy())

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)
final_predictions = np.argmax(test_preds, axis=1)
submit_df = pd.read_csv(f'{FILE_DIR}/test/sample_submission.csv')
submit_df['AdoptionSpeed'] = final_predictions
submit_df.to_csv("submission.csv", index=False)



