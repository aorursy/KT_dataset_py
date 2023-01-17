# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import json, glob, cv2

from math import copysign, log10

from PIL import Image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

from catboost import CatBoostClassifier, FeaturesData, Pool

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import ParameterGrid

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from joblib import Parallel, delayed

from tqdm import tqdm, tqdm_notebook



palette = sns.color_palette("Paired")

sns.set()

sns.set_palette(palette)



split_char = '/'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("./"))



# Any results you write to the current directory are saved as output.
# local

# trainPath = './train.csv'

# testPath = './test.csv'

# trainSentimentPath = './train_sentiment/'

# testSentimentPath = './test_sentiment/'

# trainMetadataPath = './train_metadata/'

# testMetadataPath = './test_metadata/'

# trainImagePath = './train_images/'

# testImagePath = './test_images/'

# breedPath = './breed_labels.csv'

# colorPath = './color_labels.csv'

# statePath = 'state_labels.csv'

# kaggle kernel

trainPath = '../input/petfinder-adoption-prediction/train/train.csv'

testPath = '../input/petfinder-adoption-prediction/test/test.csv'

trainSentimentPath = '../input/petfinder-adoption-prediction/train_sentiment/'

testSentimentPath = '../input/petfinder-adoption-prediction/test_sentiment/'

trainMetadataPath = '../input/petfinder-adoption-prediction/train_metadata/'

testMetadataPath = '../input/petfinder-adoption-prediction/test_metadata/'

trainImagePath = '../input/petfinder-adoption-prediction/train_images/'

testImagePath = '../input/petfinder-adoption-prediction/test_images/'

breedPath = '../input/petfinder-adoption-prediction/breed_labels.csv'

colorPath = '../input/petfinder-adoption-prediction/color_labels.csv'

statePath = '../input/petfinder-adoption-prediction/state_labels.csv'

trainPrecomputedPath = '../input/precomputedfeaturespetfinder/train_precomputed.csv'

testPrecomputedPath = '../input/precomputedfeaturespetfinder/test_precomputed.csv'
categoricalFeatures = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',

                       'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',

                       'Health', 'State', 'RescuerID', 'AdoptionSpeed']

numericalFeatures = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt']
#  local

# train = pd.read_csv(trainPath)

# test = pd.read_csv(testPath)

# kaggle kernel

train = pd.read_csv(trainPath)

test = pd.read_csv(testPath)

train.info()
import cv2

import os

from keras.applications.densenet import preprocess_input, DenseNet121
def resize_to_square(im):

    old_size = im.shape[:2]

    ratio = float(img_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = img_size - new_size[1]

    delta_h = img_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)

    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

    return new_im



def load_image(path, pet_id):

    image = cv2.imread(f'{path}{pet_id}-1.jpg')

    new_image = resize_to_square(image)

    new_image = preprocess_input(new_image)

    return new_image
img_size = 256

batch_size = 256
from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D

import keras.backend as K



# denseNetPath = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'

denseNetPath = '../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'



inp = Input((256,256,3))

backbone = DenseNet121(input_tensor = inp, 

                       weights=denseNetPath,

                       include_top = False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)
pet_ids = train['PetID'].values

n_batches = len(pet_ids) // batch_size + 1



features = {}

for b in tqdm(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

    for i,pet_id in enumerate(batch_pets):

        try:

            batch_images[i] = load_image(trainImagePath, pet_id)

        except:

            pass

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]
train_feats = pd.DataFrame.from_dict(features, orient='index')

train_feats.columns = [f'pic_{i}' for i in range(train_feats.shape[1])]
pet_ids = test['PetID'].values

n_batches = len(pet_ids) // batch_size + 1



features = {}

for b in tqdm(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

    for i,pet_id in enumerate(batch_pets):

        try:

            batch_images[i] = load_image(testImagePath, pet_id)

        except:

            pass

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]
test_feats = pd.DataFrame.from_dict(features, orient='index')

test_feats.columns = [f'pic_{i}' for i in range(test_feats.shape[1])]
train_feats = train_feats.reset_index()

train_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)



test_feats = test_feats.reset_index()

test_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)
all_ids = pd.concat([train, test], axis=0, ignore_index=True, sort=False)[['PetID']]

all_ids.shape
n_components = 32

svd_ = TruncatedSVD(n_components=n_components, random_state=1337)



features_df = pd.concat([train_feats, test_feats], axis=0)

features = features_df[[f'pic_{i}' for i in range(256)]].values



svd_col = svd_.fit_transform(features)

svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('IMG_SVD_')



img_features = pd.concat([all_ids, svd_col], axis=1)
img_features.info()
img_features.head()
labels_breed = pd.read_csv(breedPath)

labels_state = pd.read_csv(colorPath)

labels_color = pd.read_csv(statePath)
train_image_files = sorted(glob.glob(trainImagePath + '*.jpg'))

train_metadata_files = sorted(glob.glob(trainMetadataPath + '*.json'))

train_sentiment_files = sorted(glob.glob(trainSentimentPath + '*.json'))



print(f'num of train images files: {len(train_image_files)}')

print(f'num of train metadata files: {len(train_metadata_files)}')

print(f'num of train sentiment files: {len(train_sentiment_files)}')





test_image_files = sorted(glob.glob(testImagePath + '*.jpg'))

test_metadata_files = sorted(glob.glob(testMetadataPath + '*.json'))

test_sentiment_files = sorted(glob.glob(testSentimentPath + '*.json'))



print(f'num of test images files: {len(test_image_files)}')

print(f'num of test metadata files: {len(test_metadata_files)}')

print(f'num of test sentiment files: {len(test_sentiment_files)}')
# Images:

train_df_ids = train[['PetID']]

print(train_df_ids.shape)



# Metadata:

train_df_ids = train[['PetID']]

train_df_metadata = pd.DataFrame(train_metadata_files)

train_df_metadata.columns = ['metadata_filename']

train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])

train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)

print(len(train_metadata_pets.unique()))



pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))

print(f'fraction of pets with metadata: {pets_with_metadatas / train_df_ids.shape[0]:.3f}')



# Sentiment:

train_df_ids = train[['PetID']]

train_df_sentiment = pd.DataFrame(train_sentiment_files)

train_df_sentiment.columns = ['sentiment_filename']

train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])

train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)

print(len(train_sentiment_pets.unique()))



pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))

print(f'fraction of pets with sentiment: {pets_with_sentiments / train_df_ids.shape[0]:.3f}')
# Images:

test_df_ids = test[['PetID']]

print(test_df_ids.shape)



# Metadata:

test_df_metadata = pd.DataFrame(test_metadata_files)

test_df_metadata.columns = ['metadata_filename']

test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])

test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)

print(len(test_metadata_pets.unique()))



pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))

print(f'fraction of pets with metadata: {pets_with_metadatas / test_df_ids.shape[0]:.3f}')



# Sentiment:

test_df_sentiment = pd.DataFrame(test_sentiment_files)

test_df_sentiment.columns = ['sentiment_filename']

test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split(split_char)[-1].split('.')[0])

test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)

print(len(test_sentiment_pets.unique()))



pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))

print(f'fraction of pets with sentiment: {pets_with_sentiments / test_df_ids.shape[0]:.3f}')
class PetFinderParser(object):

    

    def __init__(self, debug=False):

        

        self.debug = debug

        self.sentence_sep = ' '

        

        self.extract_sentiment_text = False

    

    def open_json_file(self, filename):

        with open(filename, 'r', encoding='utf-8') as f:

            json_file = json.load(f)

        return json_file

        

    def parse_sentiment_file(self, file):

        """

        Parse sentiment file. Output DF with sentiment features.

        """

        

        file_sentiment = file['documentSentiment']

        file_entities = [x['name'] for x in file['entities']]

        file_entities = self.sentence_sep.join(file_entities)

        

        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        

        file_sentences_sentiment = pd.DataFrame.from_dict(

            file_sentences_sentiment, orient='columns')

        file_sentences_sentiment_df = pd.DataFrame(

            {

                'magnitude_sum': file_sentences_sentiment['magnitude'].sum(axis=0),

                'score_sum': file_sentences_sentiment['score'].sum(axis=0),

                'magnitude_mean': file_sentences_sentiment['magnitude'].mean(axis=0),

                'score_mean': file_sentences_sentiment['score'].mean(axis=0),

                'magnitude_var': file_sentences_sentiment['magnitude'].var(axis=0),

                'score_var': file_sentences_sentiment['score'].var(axis=0),

            }, index=[0]

        )

        

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T

        df_sentiment = pd.concat([df_sentiment, file_sentences_sentiment_df], axis=1)

            

        df_sentiment['entities'] = file_entities

        df_sentiment = df_sentiment.add_prefix('sentiment_')

        

        return df_sentiment

    

    def parse_metadata_file(self, file):

        """

        Parse metadata file. Output DF with metadata features.

        """

        

        file_keys = list(file.keys())

        

        if 'labelAnnotations' in file_keys:

            file_annots = file['labelAnnotations']

            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()

            file_top_desc = [x['description'] for x in file_annots]

        else:

            file_top_score = np.nan

            file_top_desc = ['']

        

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']

        file_crops = file['cropHintsAnnotation']['cropHints']



        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()

        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()



        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()

        

        if 'importanceFraction' in file_crops[0].keys():

            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()

        else:

            file_crop_importance = np.nan



        df_metadata = {

            'annots_score': file_top_score,

            'color_score': file_color_score,

            'color_pixelfrac': file_color_pixelfrac,

            'crop_conf': file_crop_conf,

            'crop_importance': file_crop_importance,

            'annots_top_desc': self.sentence_sep.join(file_top_desc)

        }

        

        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T

        df_metadata = df_metadata.add_prefix('metadata_')

        

        return df_metadata

    



def extract_additional_features(pet_id, mode='train'):

    

    sentiment_filename = f'../input/petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json'

    try:

        sentiment_file = pet_parser.open_json_file(sentiment_filename)

        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)

        df_sentiment['PetID'] = pet_id

    except FileNotFoundError:

        df_sentiment = []



    dfs_metadata = []

    metadata_filenames = sorted(glob.glob(f'../input/petfinder-adoption-prediction/{mode}_metadata/{pet_id}*.json'))

    if len(metadata_filenames) > 0:

        for f in metadata_filenames:

            metadata_file = pet_parser.open_json_file(f)

            df_metadata = pet_parser.parse_metadata_file(metadata_file)

            df_metadata['PetID'] = pet_id

            dfs_metadata.append(df_metadata)

        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)

    dfs = [df_sentiment, dfs_metadata]

    

    return dfs





pet_parser = PetFinderParser()
debug = False

train_pet_ids = train.PetID.unique()

test_pet_ids = test.PetID.unique()



if debug:

    train_pet_ids = train_pet_ids[:1000]

    test_pet_ids = test_pet_ids[:500]





dfs_train = Parallel(n_jobs=-1, verbose=1)(

    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)



train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]

train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]



train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)

train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)



print(train_dfs_sentiment.shape, train_dfs_metadata.shape)





dfs_test = Parallel(n_jobs=-1, verbose=1)(

    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)



test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]

test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]



test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)

test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)



print(test_dfs_sentiment.shape, test_dfs_metadata.shape)
aggregates = ['sum', 'mean', 'var']

sent_agg = ['sum']





# Train

train_metadata_desc = train_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()

train_metadata_desc = train_metadata_desc.reset_index()

train_metadata_desc[

    'metadata_annots_top_desc'] = train_metadata_desc[

    'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))



prefix = 'metadata'

train_metadata_gr = train_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)

for i in train_metadata_gr.columns:

    if 'PetID' not in i:

        train_metadata_gr[i] = train_metadata_gr[i].astype(float)

train_metadata_gr = train_metadata_gr.groupby(['PetID']).agg(aggregates)

train_metadata_gr.columns = pd.Index([f'{c[0]}_{c[1].upper()}' for c in train_metadata_gr.columns.tolist()])

train_metadata_gr = train_metadata_gr.reset_index()





train_sentiment_desc = train_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()

train_sentiment_desc = train_sentiment_desc.reset_index()

train_sentiment_desc[

    'sentiment_entities'] = train_sentiment_desc[

    'sentiment_entities'].apply(lambda x: ' '.join(x))



prefix = 'sentiment'

train_sentiment_gr = train_dfs_sentiment.drop(['sentiment_entities'], axis=1)

for i in train_sentiment_gr.columns:

    if 'PetID' not in i:

        train_sentiment_gr[i] = train_sentiment_gr[i].astype(float)

train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(sent_agg)

train_sentiment_gr.columns = pd.Index([f'{c[0]}' for c in train_sentiment_gr.columns.tolist()])

train_sentiment_gr = train_sentiment_gr.reset_index()





# Test

test_metadata_desc = test_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()

test_metadata_desc = test_metadata_desc.reset_index()

test_metadata_desc[

    'metadata_annots_top_desc'] = test_metadata_desc[

    'metadata_annots_top_desc'].apply(lambda x: ' '.join(x))



prefix = 'metadata'

test_metadata_gr = test_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)

for i in test_metadata_gr.columns:

    if 'PetID' not in i:

        test_metadata_gr[i] = test_metadata_gr[i].astype(float)

test_metadata_gr = test_metadata_gr.groupby(['PetID']).agg(aggregates)

test_metadata_gr.columns = pd.Index([f'{c[0]}_{c[1].upper()}' for c in test_metadata_gr.columns.tolist()])

test_metadata_gr = test_metadata_gr.reset_index()





test_sentiment_desc = test_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()

test_sentiment_desc = test_sentiment_desc.reset_index()

test_sentiment_desc[

    'sentiment_entities'] = test_sentiment_desc[

    'sentiment_entities'].apply(lambda x: ' '.join(x))



prefix = 'sentiment'

test_sentiment_gr = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)

for i in test_sentiment_gr.columns:

    if 'PetID' not in i:

        test_sentiment_gr[i] = test_sentiment_gr[i].astype(float)

test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(sent_agg)

test_sentiment_gr.columns = pd.Index([f'{c[0]}' for c in test_sentiment_gr.columns.tolist()])

test_sentiment_gr = test_sentiment_gr.reset_index()
# Train merges:

train_proc = train.copy()

train_proc = train_proc.merge(

    train_sentiment_gr, how='left', on='PetID')

train_proc = train_proc.merge(

    train_metadata_gr, how='left', on='PetID')

train_proc = train_proc.merge(

    train_metadata_desc, how='left', on='PetID')

train_proc = train_proc.merge(

    train_sentiment_desc, how='left', on='PetID')



# Test merges:

test_proc = test.copy()

test_proc = test_proc.merge(

    test_sentiment_gr, how='left', on='PetID')

test_proc = test_proc.merge(

    test_metadata_gr, how='left', on='PetID')

test_proc = test_proc.merge(

    test_metadata_desc, how='left', on='PetID')

test_proc = test_proc.merge(

    test_sentiment_desc, how='left', on='PetID')



print(train_proc.shape, test_proc.shape)

assert train_proc.shape[0] == train.shape[0]

assert test_proc.shape[0] == test.shape[0]
train_breed_main = train_proc[['Breed1']].merge(

    labels_breed, how='left',

    left_on='Breed1', right_on='BreedID',

    suffixes=('', '_main_breed'))



train_breed_main = train_breed_main.iloc[:, 2:]

train_breed_main = train_breed_main.add_prefix('main_breed_')



train_breed_second = train_proc[['Breed2']].merge(

    labels_breed, how='left',

    left_on='Breed2', right_on='BreedID',

    suffixes=('', '_second_breed'))



train_breed_second = train_breed_second.iloc[:, 2:]

train_breed_second = train_breed_second.add_prefix('second_breed_')





train_proc = pd.concat(

    [train_proc, train_breed_main, train_breed_second], axis=1)





test_breed_main = test_proc[['Breed1']].merge(

    labels_breed, how='left',

    left_on='Breed1', right_on='BreedID',

    suffixes=('', '_main_breed'))



test_breed_main = test_breed_main.iloc[:, 2:]

test_breed_main = test_breed_main.add_prefix('main_breed_')



test_breed_second = test_proc[['Breed2']].merge(

    labels_breed, how='left',

    left_on='Breed2', right_on='BreedID',

    suffixes=('', '_second_breed'))



test_breed_second = test_breed_second.iloc[:, 2:]

test_breed_second = test_breed_second.add_prefix('second_breed_')





test_proc = pd.concat(

    [test_proc, test_breed_main, test_breed_second], axis=1)



print(train_proc.shape, test_proc.shape)
X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
X_temp = X.copy()



text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']

categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']



to_drop_columns = ['PetID', 'Name', 'RescuerID']
rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()

rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']



X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')
for i in categorical_columns:

    X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]
X_text = X_temp[text_columns]



for i in X_text.columns:

    X_text.loc[:, i] = X_text.loc[:, i].fillna('none')
X_temp['Length_Description'] = X_text['Description'].map(len)

X_temp['Length_metadata_annots_top_desc'] = X_text['metadata_annots_top_desc'].map(len)

X_temp['Lengths_sentiment_entities'] = X_text['sentiment_entities'].map(len)
n_components = 16

text_features = []



# Generate text features:

for i in X_text.columns:

    

    # Initialize decomposition methods:

    print(f'generating features from: {i}')

    tfv = TfidfVectorizer(min_df=2,  max_features=None,

                          strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',

                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)

    svd_ = TruncatedSVD(

        n_components=n_components, random_state=1337)

    

    tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)

    

    svd_col = svd_.fit_transform(tfidf_col)

    svd_col = pd.DataFrame(svd_col)

    svd_col = svd_col.add_prefix('TFIDF_{}_'.format(i))

    

    text_features.append(svd_col)

    

text_features = pd.concat(text_features, axis=1)



X_temp = pd.concat([X_temp, text_features], axis=1)



for i in X_text.columns:

    X_temp = X_temp.drop(i, axis=1)
X_temp = X_temp.merge(img_features, how='left', on='PetID')
from PIL import Image

train_df_ids = train[['PetID']]

test_df_ids = test[['PetID']]



train_df_imgs = pd.DataFrame(train_image_files)

train_df_imgs.columns = ['image_filename']

train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])



test_df_imgs = pd.DataFrame(test_image_files)

test_df_imgs.columns = ['image_filename']

test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])



train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)



def getSize(filename):

    st = os.stat(filename)

    return st.st_size



def getDimensions(filename):

    img_size = Image.open(filename).size

    return img_size 



train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)

train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)

train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x : x[0])

train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x : x[1])

train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)



test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)

test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)

test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x : x[0])

test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x : x[1])

test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)



aggs = {

    'image_size': ['sum', 'mean', 'var'],

    'width': ['sum', 'mean', 'var'],

    'height': ['sum', 'mean', 'var'],

}



agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)

new_columns = [

    k + '_' + agg for k in aggs.keys() for agg in aggs[k]

]

agg_train_imgs.columns = new_columns

agg_train_imgs = agg_train_imgs.reset_index()



agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)

new_columns = [

    k + '_' + agg for k in aggs.keys() for agg in aggs[k]

]

agg_test_imgs.columns = new_columns

agg_test_imgs = agg_test_imgs.reset_index()



agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
X_temp = X_temp.merge(agg_imgs, how='left', on='PetID')
# X_temp = X_temp.drop(to_drop_columns, axis=1)
X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]

X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]



# X_test = X_test.drop(['AdoptionSpeed'], axis=1)



assert X_train.shape[0] == train.shape[0]

assert X_test.shape[0] == test.shape[0]



train_cols = X_train.columns.tolist()

# train_cols.remove('AdoptionSpeed')



test_cols = X_test.columns.tolist()



assert np.all(train_cols == test_cols)
# save final train and test (with NA)

X_train.to_csv('train_precomputed.csv', index=False)

X_test.to_csv('test_precomputed.csv', index=False)
X_train_non_null = X_train.fillna(-1)

X_test_non_null = X_test.fillna(-1)
X_train_non_null.isnull().any().any(), X_test_non_null.isnull().any().any()
X_train_non_null.shape, X_test_non_null.shape
import scipy as sp



from collections import Counter

from functools import partial

from math import sqrt



from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix





# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features



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
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']
breedDf = pd.read_csv(breedPath)

breedDf.head()
colorDf = pd.read_csv(colorPath)

colorDf.head()
stateDf = pd.read_csv(statePath)

stateDf.head()
def cleanTransformDataset(dataset, categoricalFeatures):

    breedDf2 = breedDf.set_index('BreedID')

    idx = breedDf2.to_dict()

    dataset.Breed1 = dataset.Breed1.map(idx['BreedName'])

    dataset.Breed2 = dataset.Breed2.map(idx['BreedName'])

    

    colorDf2 = colorDf.set_index('ColorID')

    idx = colorDf2.to_dict()

    dataset.Color1 = dataset.Color1.map(idx['ColorName'])

    dataset.Color2 = dataset.Color2.map(idx['ColorName'])

    dataset.Color3 = dataset.Color3.map(idx['ColorName'])

    

    stateDf2 = stateDf.set_index('StateID')

    idx = stateDf2.to_dict()

    dataset.State = dataset.State.map(idx['StateName'])

    

    # 1 = Dog, 2 = Cat

    dataset.Type = dataset.Type.map({1: 'Dog', 2: 'Cat'})

    # 1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets

    dataset.Gender = dataset.Gender.map({1: 'Male', 2: 'Female', 3: 'Mixed'})

    # 1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified

    dataset.MaturitySize = dataset.MaturitySize.map({1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large', 0: 'Not Specified'})

    # 1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified

    dataset.FurLength = dataset.FurLength.map({1: 'Short', 2: 'Medium', 3: 'Long', 0: 'Not Specified'})

    # 1 = Yes, 2 = No, 3 = Not Sure

    dataset.Vaccinated = dataset.Vaccinated.map({1: 'Yes', 2: 'No', 3: 'Not Sure'})

    # 1 = Yes, 2 = No, 3 = Not Sure

    dataset.Dewormed = dataset.Dewormed.map({1: 'Yes', 2: 'No', 3: 'Not Sure'})

    # 1 = Yes, 2 = No, 3 = Not Sure

    dataset.Sterilized = dataset.Sterilized.map({1: 'Yes', 2: 'No', 3: 'Not Sure'})

    # 1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified

    dataset.Health = dataset.Health.map({1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury', 0: 'Not Specified'})

    # transform to categorical

    dataset[categoricalFeatures] = dataset[categoricalFeatures].astype('category')

    return dataset
train = cleanTransformDataset(train, categoricalFeatures)

test = cleanTransformDataset(test, list(set(categoricalFeatures) - set(['AdoptionSpeed'])))
def extraFeatures(train):

    # Color (Create a Flag pet has 1 color, 2 colors, 3 colors)

    train['L_Color1'] = (pd.isnull(train['Color3']) & pd.isnull(train['Color2']) & pd.notnull(train['Color1'])).astype(int)

    train['L_Color2'] = (pd.isnull(train['Color3']) & pd.notnull(train['Color2']) & pd.notnull(train['Color1'])).astype(int)

    train['L_Color3'] = (pd.notnull(train['Color3']) & pd.notnull(train['Color2']) & pd.notnull(train['Color1'])).astype(int)



    # Breed (create a flag if the pet has 1 breed or 2)

    train['L_Breed1'] = (pd.isnull(train['Breed2']) & pd.notnull(train['Breed1'])).astype(int)

    train['L_Breed2'] = (pd.notnull(train['Breed2']) & pd.notnull(train['Breed1'])).astype(int)



    #Name (create a flag if the name is missing, with less than two letters)

    train['Name_Length']= train['Name'].str.len()

    train['L_Name_missing'] = (pd.isnull(train['Name'])).astype(int)



    # Breed create columns

    train['L_Breed1_Siamese'] =(train['Breed1']=='Siamese').astype(int)

    train['L_Breed1_Persian']=(train['Breed1']=='Persian').astype(int)

    train['L_Breed1_Labrador_Retriever']=(train['Breed1']=='Labrador Retriever').astype(int)

    train['L_Breed1_Terrier']=(train['Breed1']=='Terrier').astype(int)

    train['L_Breed1_Golden_Retriever ']=(train['Breed1']=='Golden Retriever').astype(int)



    #Description 

    train['Description_Length']=train['Description'].str.len() 



    # Fee Amount

    train['L_Fee_Free'] =  (train['Fee']==0).astype(int)



    #Add the Number of Pets per Rescuer 

    pets_total = train.groupby(['RescuerID']).size().reset_index(name='N_pets_total')

    train= pd.merge(train, pets_total, left_on='RescuerID', right_on='RescuerID', how='inner')

    train.count()



    # No photo

    train['L_NoPhoto'] =  (train['PhotoAmt']==0).astype(int)



    #No Video

    train['L_NoVideo'] =  (train['VideoAmt']==0).astype(int)



    #Log Age 

    train['Log_Age']= np.log(train.Age + 1) 



    #Quantity Amount >5

    train.loc[train['Quantity'] > 5, 'Quantity'] = 5

    return train
train = extraFeatures(train)

test = extraFeatures(test)
train.describe()
train.info()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numericalFeatures = list(train.select_dtypes(include=numerics).columns)
train.isna().sum()
msno.matrix(train)
numColumns = train.select_dtypes(include='number').columns.tolist()
len(numColumns)
i = 0

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(6, 4, figsize=(28,38))

for column in numColumns:

    i += 1

    plt.subplot(6, 4, i)

    sns.distplot(train[column].dropna())

    sns.distplot(test[column].dropna())

    plt.legend(title=column, loc='upper left', labels=['train', 'test'])

plt.show()
i = 0

sns.set_style('whitegrid')

plt.figure()

fig, ax = plt.subplots(6, 4, figsize=(28,38))

for column in numColumns:

    i += 1

    plt.subplot(6, 4, i)

    sns.kdeplot(train[column], bw=0.5)

    sns.kdeplot(test[column], bw=0.5)

    plt.legend(title=column, loc='upper left', labels=['train', 'test'])

plt.show()
speeds = train.AdoptionSpeed.unique()

def plotDistributionPerTarget(data, num_rows, num_columns, size=(28,38)):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(num_rows, num_columns, figsize=size)

    for column in data.columns:

        if column == 'AdoptionSpeed':

            continue

        i += 1

        plt.subplot(num_rows, num_columns, i)

        for speed in speeds:

            sns.kdeplot(data[data['AdoptionSpeed'] == speed][column], bw=0.5)

        plt.legend(title=column, loc='upper left', labels=speeds)

    plt.show()



plotDistributionPerTarget(train[['AdoptionSpeed'] + numColumns], 6, 4)
sns.catplot(x="AdoptionSpeed", kind="count", data=train)
fig, ax = plt.subplots(figsize=(4,8))

sns.boxplot(y=train.Age)
fig, ax = plt.subplots(figsize=(10,6))

sns.distplot(train.Age)
sns.barplot(x="AdoptionSpeed", y="Age", data=train)
sns.catplot(y="AdoptionSpeed", x="Age", data=train, orient="h", kind="box")
train['AgeInterval'] = pd.Series(['0-3', '3-6', '6-12', '12-24', '24-48', '48-120', '>120'], dtype='category')

train.loc[(train['Age'] >= 0) & (train['Age'] <= 3),'AgeInterval'] = '0-3'

train.loc[(train['Age'] > 3) & (train['Age'] <= 6),'AgeInterval'] = '3-6'

train.loc[(train['Age'] > 6) & (train['Age'] <= 12),'AgeInterval'] = '6-12'

train.loc[(train['Age'] > 12) & (train['Age'] <= 24),'AgeInterval'] = '12-24'

train.loc[(train['Age'] > 24) & (train['Age'] <= 48),'AgeInterval'] = '24-48'

train.loc[(train['Age'] > 48) & (train['Age'] <= 120),'AgeInterval'] = '48-120'

train.loc[train['Age'] > 120,'AgeInterval'] = '>120'
fig, ax = plt.subplots(figsize=(10,6))

sns.countplot(x="AdoptionSpeed", hue="AgeInterval", data=train)
total = train[train.AgeInterval == '0-3'].size

interval1 = [

 train[(train.AdoptionSpeed == 0) & (train.AgeInterval == '0-3')].size,

 train[(train.AdoptionSpeed == 1) & (train.AgeInterval == '0-3')].size,

 train[(train.AdoptionSpeed == 2) & (train.AgeInterval == '0-3')].size,

 train[(train.AdoptionSpeed == 3) & (train.AgeInterval == '0-3')].size,

 train[(train.AdoptionSpeed == 4) & (train.AgeInterval == '0-3')].size

] / total



percents = np.append(interval1, [])

types = (['0-3'] * 5)



feePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })

fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 1, hue='type', style='type', markers=True, data=feePercentDf)
fig, ax = plt.subplots(figsize=(10,6))

sns.distplot(train.Quantity)
sns.jointplot(x="Quantity", y="AdoptionSpeed", data=train, kind="reg", height=8)
sns.barplot(x="AdoptionSpeed", y="Quantity", data=train)
fig, ax = plt.subplots(figsize=(10,6))

sns.distplot(train.VideoAmt)
sns.jointplot(x="VideoAmt", y="AdoptionSpeed", data=train, kind="reg", height=8)
sns.barplot(x="AdoptionSpeed", y="VideoAmt", data=train)
fig, ax = plt.subplots(figsize=(10,6))

sns.distplot(train.PhotoAmt)
sns.jointplot(x="PhotoAmt", y="AdoptionSpeed", data=train, kind="reg", height=8)
sns.barplot(x="AdoptionSpeed", y="PhotoAmt", data=train)
train['PhotoAmtInterval'] = pd.Series(['0', '1', '2', '3', '4', '5', '>5'], dtype='category')

train.loc[train['PhotoAmt'] == 0 ,'PhotoAmtInterval'] = '0'

train.loc[train['PhotoAmt'] == 1 ,'PhotoAmtInterval'] = '1'

train.loc[train['PhotoAmt'] == 2 ,'PhotoAmtInterval'] = '2'

train.loc[train['PhotoAmt'] == 3 ,'PhotoAmtInterval'] = '3'

train.loc[train['PhotoAmt'] == 4 ,'PhotoAmtInterval'] = '4'

train.loc[train['PhotoAmt'] == 5 ,'PhotoAmtInterval'] = '5'

train.loc[train['PhotoAmt'] > 5 ,'PhotoAmtInterval'] = '>5'



fig, ax = plt.subplots(figsize=(10,6))

sns.countplot(x='AdoptionSpeed', hue='PhotoAmtInterval', data=train)
fig, ax = plt.subplots(figsize=(10,6))

sns.distplot(train.Fee)
train = train[train.Fee <= 1500]
sns.jointplot(x='Fee', y='AdoptionSpeed', data=train, kind='reg', height=8)
sns.barplot(x='AdoptionSpeed', y='Fee', data=train)
train['FeeInterval'] = pd.Series(['Free', 'Paid'], dtype='category')

train.loc[train['Fee'] == 0 ,'FeeInterval'] = 'Free'

train.loc[train['Fee'] > 0 ,'FeeInterval'] = 'Paid'



fig, ax = plt.subplots(figsize=(10,6))

sns.countplot(x='AdoptionSpeed', hue='FeeInterval', data=train)
total = train[train.Fee == 0].size

feeFreePercent = [

 train[(train.AdoptionSpeed == 0) & (train.Fee == 0)].size,

 train[(train.AdoptionSpeed == 1) & (train.Fee == 0)].size,

 train[(train.AdoptionSpeed == 2) & (train.Fee == 0)].size,

 train[(train.AdoptionSpeed == 3) & (train.Fee == 0)].size,

 train[(train.AdoptionSpeed == 4) & (train.Fee == 0)].size

] / total



total = train[train.Fee > 0].size

feePaidPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Fee > 0)].size,

 train[(train.AdoptionSpeed == 1) & (train.Fee > 0)].size,

 train[(train.AdoptionSpeed == 2) & (train.Fee > 0)].size,

 train[(train.AdoptionSpeed == 3) & (train.Fee > 0)].size,

 train[(train.AdoptionSpeed == 4) & (train.Fee > 0)].size

] / total



percents = np.append(feeFreePercent, feePaidPercent)

types = (['free'] * 5) + (['paid'] * 5)



feePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })

fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 2, hue='type', style='type', markers=True, data=feePercentDf)
# 1 = Dog, 2 = Cat

sns.catplot(x="Type", kind="count", data=train)
fig, ax = plt.subplots(figsize=(10,6))

sns.countplot(x='AdoptionSpeed', hue='Type', data=train)
total = train[train.Type == 'Dog'].size

dogPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Type == 'Dog')].size,

 train[(train.AdoptionSpeed == 1) & (train.Type == 'Dog')].size,

 train[(train.AdoptionSpeed == 2) & (train.Type == 'Dog')].size,

 train[(train.AdoptionSpeed == 3) & (train.Type == 'Dog')].size,

 train[(train.AdoptionSpeed == 4) & (train.Type == 'Dog')].size

] / total



total = train[train.Type == 'Cat'].size

catPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Type == 'Cat')].size,

 train[(train.AdoptionSpeed == 1) & (train.Type == 'Cat')].size,

 train[(train.AdoptionSpeed == 2) & (train.Type == 'Cat')].size,

 train[(train.AdoptionSpeed == 3) & (train.Type == 'Cat')].size,

 train[(train.AdoptionSpeed == 4) & (train.Type == 'Cat')].size

] / total



percents = np.append(dogPercent, catPercent)

types = (['dog'] * 5) + (['cat'] * 5)



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })

fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 2, hue='type', style='type', markers=True, data=typePercentDf)
dogCounts = train[train.Type == 'Dog'].Breed1.value_counts()

fig, ax = plt.subplots(figsize=(8,8))

dogCounts.nlargest(15).plot(kind='bar')
catCounts = train[train.Type == 'Cat'].Breed1.value_counts()

fig, ax = plt.subplots(figsize=(8,8))

catCounts.nlargest(15).plot(kind='bar')
strayCat = ['Domestic Short Hair', 'Domestic Medium Hair', 'Domestic Long Hair']

strayDog = ['Mixed Breed']



train.loc[(train.Breed1.isin(strayDog)) & (train.Type == 'Dog'), 'Breed1Type'] = 'Stray'

train.loc[(train.Breed1.isin(strayCat)) & (train.Type == 'Cat'), 'Breed1Type'] = 'Stray'

train.loc[(~train.Breed1.isin(strayDog)) & (train.Type == 'Dog'), 'Breed1Type'] = 'Breed'

train.loc[(~train.Breed1.isin(strayCat)) & (train.Type == 'Cat'), 'Breed1Type'] = 'Breed'
sns.countplot(x='AdoptionSpeed', hue='Breed1Type', data=train[train.Type == 'Dog'])
sns.countplot(x='AdoptionSpeed', hue='Breed1Type', data=train[train.Type == 'Cat'])
total = train[(train.Type == 'Dog') & (train.Breed1Type == 'Stray')].size

dogPercentStray = [

 train[(train.AdoptionSpeed == 0) & (train.Type == 'Dog') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 1) & (train.Type == 'Dog') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 2) & (train.Type == 'Dog') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 3) & (train.Type == 'Dog') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 4) & (train.Type == 'Dog') & (train.Breed1Type == 'Stray')].size

] / total



total = train[(train.Type == 'Dog') & (train.Breed1Type == 'Breed')].size

dogPercentBreed = [

 train[(train.AdoptionSpeed == 0) & (train.Type == 'Dog') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 1) & (train.Type == 'Dog') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 2) & (train.Type == 'Dog') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 3) & (train.Type == 'Dog') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 4) & (train.Type == 'Dog') & (train.Breed1Type == 'Breed')].size

] / total



total = train[(train.Type == 'Cat') & (train.Breed1Type == 'Stray')].size

catPercentStray = [

 train[(train.AdoptionSpeed == 0) & (train.Type == 'Cat') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 1) & (train.Type == 'Cat') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 2) & (train.Type == 'Cat') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 3) & (train.Type == 'Cat') & (train.Breed1Type == 'Stray')].size,

 train[(train.AdoptionSpeed == 4) & (train.Type == 'Cat') & (train.Breed1Type == 'Stray')].size

] / total



total = train[(train.Type == 'Cat') & (train.Breed1Type == 'Breed')].size

catPercentBreed = [

 train[(train.AdoptionSpeed == 0) & (train.Type == 'Cat') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 1) & (train.Type == 'Cat') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 2) & (train.Type == 'Cat') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 3) & (train.Type == 'Cat') & (train.Breed1Type == 'Breed')].size,

 train[(train.AdoptionSpeed == 4) & (train.Type == 'Cat') & (train.Breed1Type == 'Breed')].size

] / total



percents = np.append(dogPercentStray, dogPercentBreed)

percents = np.append(percents, catPercentStray)

percents = np.append(percents, catPercentBreed)

types = (['dog-stray'] * 5 + ['dog-breed'] * 5 + ['cat-stray'] * 5 + ['cat-breed'] * 5)



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 4, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
sns.catplot(x='Color1', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='Color1', data=train)
sns.catplot(x='MaturitySize', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='MaturitySize', data=train)
total = train[(train.MaturitySize == 'Large')].size

largePercent = [

 train[(train.AdoptionSpeed == 0) & (train.MaturitySize == 'Large')].size,

 train[(train.AdoptionSpeed == 1) & (train.MaturitySize == 'Large')].size,

 train[(train.AdoptionSpeed == 2) & (train.MaturitySize == 'Large')].size,

 train[(train.AdoptionSpeed == 3) & (train.MaturitySize == 'Large')].size,

 train[(train.AdoptionSpeed == 4) & (train.MaturitySize == 'Large')].size

] / total



total = train[(train.MaturitySize == 'Medium')].size

mediumPercent = [

 train[(train.AdoptionSpeed == 0) & (train.MaturitySize == 'Medium')].size,

 train[(train.AdoptionSpeed == 1) & (train.MaturitySize == 'Medium')].size,

 train[(train.AdoptionSpeed == 2) & (train.MaturitySize == 'Medium')].size,

 train[(train.AdoptionSpeed == 3) & (train.MaturitySize == 'Medium')].size,

 train[(train.AdoptionSpeed == 4) & (train.MaturitySize == 'Medium')].size

] / total



total = train[(train.MaturitySize == 'Small')].size

smallPercent = [

 train[(train.AdoptionSpeed == 0) & (train.MaturitySize == 'Small')].size,

 train[(train.AdoptionSpeed == 1) & (train.MaturitySize == 'Small')].size,

 train[(train.AdoptionSpeed == 2) & (train.MaturitySize == 'Small')].size,

 train[(train.AdoptionSpeed == 3) & (train.MaturitySize == 'Small')].size,

 train[(train.AdoptionSpeed == 4) & (train.MaturitySize == 'Small')].size

] / total



percents = np.append(largePercent, mediumPercent)

percents = np.append(percents, smallPercent)

types = (['Large'] * 5 + ['Medium'] * 5 + ['Small'] * 5 )



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 3, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
sns.catplot(x='FurLength', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='FurLength', data=train)
total = train[(train.FurLength == 'Long')].size

longPercent = [

 train[(train.AdoptionSpeed == 0) & (train.FurLength == 'Long')].size,

 train[(train.AdoptionSpeed == 1) & (train.FurLength == 'Long')].size,

 train[(train.AdoptionSpeed == 2) & (train.FurLength == 'Long')].size,

 train[(train.AdoptionSpeed == 3) & (train.FurLength == 'Long')].size,

 train[(train.AdoptionSpeed == 4) & (train.FurLength == 'Long')].size

] / total



total = train[(train.FurLength == 'Medium')].size

mediumPercent = [

 train[(train.AdoptionSpeed == 0) & (train.FurLength == 'Medium')].size,

 train[(train.AdoptionSpeed == 1) & (train.FurLength == 'Medium')].size,

 train[(train.AdoptionSpeed == 2) & (train.FurLength == 'Medium')].size,

 train[(train.AdoptionSpeed == 3) & (train.FurLength == 'Medium')].size,

 train[(train.AdoptionSpeed == 4) & (train.FurLength == 'Medium')].size

] / total



total = train[(train.FurLength == 'Short')].size

shortPercent = [

 train[(train.AdoptionSpeed == 0) & (train.FurLength == 'Short')].size,

 train[(train.AdoptionSpeed == 1) & (train.FurLength == 'Short')].size,

 train[(train.AdoptionSpeed == 2) & (train.FurLength == 'Short')].size,

 train[(train.AdoptionSpeed == 3) & (train.FurLength == 'Short')].size,

 train[(train.AdoptionSpeed == 4) & (train.FurLength == 'Short')].size

] / total



percents = np.append(longPercent, mediumPercent)

percents = np.append(percents, shortPercent)

types = (['Long'] * 5 + ['Medium'] * 5 + ['Short'] * 5 )



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 3, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
sns.catplot(x='Vaccinated', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='Vaccinated', data=train)
total = train[(train.Vaccinated == 'Yes')].size

yesPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Vaccinated == 'Yes')].size,

 train[(train.AdoptionSpeed == 1) & (train.Vaccinated == 'Yes')].size,

 train[(train.AdoptionSpeed == 2) & (train.Vaccinated == 'Yes')].size,

 train[(train.AdoptionSpeed == 3) & (train.Vaccinated == 'Yes')].size,

 train[(train.AdoptionSpeed == 4) & (train.Vaccinated == 'Yes')].size

] / total



total = train[(train.Vaccinated == 'No')].size

noPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Vaccinated == 'No')].size,

 train[(train.AdoptionSpeed == 1) & (train.Vaccinated == 'No')].size,

 train[(train.AdoptionSpeed == 2) & (train.Vaccinated == 'No')].size,

 train[(train.AdoptionSpeed == 3) & (train.Vaccinated == 'No')].size,

 train[(train.AdoptionSpeed == 4) & (train.Vaccinated == 'No')].size

] / total



total = train[(train.Vaccinated == 'Not Sure')].size

notSurePercent = [

 train[(train.AdoptionSpeed == 0) & (train.Vaccinated == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 1) & (train.Vaccinated == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 2) & (train.Vaccinated == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 3) & (train.Vaccinated == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 4) & (train.Vaccinated == 'Not Sure')].size

] / total



percents = np.append(yesPercent, noPercent)

percents = np.append(percents, notSurePercent)

types = (['Yes'] * 5 + ['No'] * 5 + ['Not Sure'] * 5 )



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 3, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
sns.catplot(x='Dewormed', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='Dewormed', data=train)
total = train[(train.Dewormed == 'Yes')].size

yesPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Dewormed == 'Yes')].size,

 train[(train.AdoptionSpeed == 1) & (train.Dewormed == 'Yes')].size,

 train[(train.AdoptionSpeed == 2) & (train.Dewormed == 'Yes')].size,

 train[(train.AdoptionSpeed == 3) & (train.Dewormed == 'Yes')].size,

 train[(train.AdoptionSpeed == 4) & (train.Dewormed == 'Yes')].size

] / total



total = train[(train.Dewormed == 'No')].size

noPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Dewormed == 'No')].size,

 train[(train.AdoptionSpeed == 1) & (train.Dewormed == 'No')].size,

 train[(train.AdoptionSpeed == 2) & (train.Dewormed == 'No')].size,

 train[(train.AdoptionSpeed == 3) & (train.Dewormed == 'No')].size,

 train[(train.AdoptionSpeed == 4) & (train.Dewormed == 'No')].size

] / total



total = train[(train.Dewormed == 'Not Sure')].size

notSurePercent = [

 train[(train.AdoptionSpeed == 0) & (train.Dewormed == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 1) & (train.Dewormed == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 2) & (train.Dewormed == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 3) & (train.Dewormed == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 4) & (train.Dewormed == 'Not Sure')].size

] / total



percents = np.append(yesPercent, noPercent)

percents = np.append(percents, notSurePercent)

types = (['Yes'] * 5 + ['No'] * 5 + ['Not Sure'] * 5 )



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 3, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
sns.catplot(x='Sterilized', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='Sterilized', data=train)
total = train[(train.Sterilized == 'Yes')].size

yesPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Sterilized == 'Yes')].size,

 train[(train.AdoptionSpeed == 1) & (train.Sterilized == 'Yes')].size,

 train[(train.AdoptionSpeed == 2) & (train.Sterilized == 'Yes')].size,

 train[(train.AdoptionSpeed == 3) & (train.Sterilized == 'Yes')].size,

 train[(train.AdoptionSpeed == 4) & (train.Sterilized == 'Yes')].size

] / total



total = train[(train.Sterilized == 'No')].size

noPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Sterilized == 'No')].size,

 train[(train.AdoptionSpeed == 1) & (train.Sterilized == 'No')].size,

 train[(train.AdoptionSpeed == 2) & (train.Sterilized == 'No')].size,

 train[(train.AdoptionSpeed == 3) & (train.Sterilized == 'No')].size,

 train[(train.AdoptionSpeed == 4) & (train.Sterilized == 'No')].size

] / total



total = train[(train.Sterilized == 'Not Sure')].size

notSurePercent = [

 train[(train.AdoptionSpeed == 0) & (train.Sterilized == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 1) & (train.Sterilized == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 2) & (train.Sterilized == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 3) & (train.Sterilized == 'Not Sure')].size,

 train[(train.AdoptionSpeed == 4) & (train.Sterilized == 'Not Sure')].size

] / total



percents = np.append(yesPercent, noPercent)

percents = np.append(percents, notSurePercent)

types = (['Yes'] * 5 + ['No'] * 5 + ['Not Sure'] * 5 )



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 3, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
sns.catplot(x='Health', kind='count', data=train)
fig, ax = plt.subplots(figsize=(8,8))

sns.countplot(x='AdoptionSpeed', hue='Health', data=train)
total = train[(train.Health == 'Healthy')].size

healthyPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Health == 'Healthy')].size,

 train[(train.AdoptionSpeed == 1) & (train.Health == 'Healthy')].size,

 train[(train.AdoptionSpeed == 2) & (train.Health == 'Healthy')].size,

 train[(train.AdoptionSpeed == 3) & (train.Health == 'Healthy')].size,

 train[(train.AdoptionSpeed == 4) & (train.Health == 'Healthy')].size

] / total



total = train[(train.Health == 'Minor Injury')].size

minorPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Health == 'Minor Injury')].size,

 train[(train.AdoptionSpeed == 1) & (train.Health == 'Minor Injury')].size,

 train[(train.AdoptionSpeed == 2) & (train.Health == 'Minor Injury')].size,

 train[(train.AdoptionSpeed == 3) & (train.Health == 'Minor Injury')].size,

 train[(train.AdoptionSpeed == 4) & (train.Health == 'Minor Injury')].size

] / total



total = train[(train.Health == 'Serious Injury')].size

seriousPercent = [

 train[(train.AdoptionSpeed == 0) & (train.Health == 'Serious Injury')].size,

 train[(train.AdoptionSpeed == 1) & (train.Health == 'Serious Injury')].size,

 train[(train.AdoptionSpeed == 2) & (train.Health == 'Serious Injury')].size,

 train[(train.AdoptionSpeed == 3) & (train.Health == 'Serious Injury')].size,

 train[(train.AdoptionSpeed == 4) & (train.Health == 'Serious Injury')].size

] / total



percents = np.append(healthyPercent, minorPercent)

percents = np.append(percents, seriousPercent)

types = (['Healthy'] * 5 + ['Minor Injury'] * 5 + ['Serious Injury'] * 5 )



typePercentDf = pd.DataFrame({ 'percent': percents, 'type': types })



fig, ax = plt.subplots(figsize=(8,8))

sns.lineplot(y='percent', x=[0, 1, 2, 3, 4] * 3, hue='type', style='type', markers=True, data=typePercentDf, palette='coolwarm')
target = 'AdoptionSpeed'

# categoricalFeatures = list(set(categoricalFeatures) - set([target, 'State', 'RescuerID', 'PetID', 'Color2', 'Color3', 'Breed2']))

#numericalFeatures = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt']

# catFeaturesIndex = list(range(0, len(categoricalFeatures)))



# cateogrical features to train

categorical = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']



features = categorical + numericalFeatures

data = train[features + [target]].dropna()



# X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.25, random_state=42)
# train_data = FeaturesData(

#     num_feature_data=X_train[numericalFeatures].astype('float32').values,

#     cat_feature_data=X_train[categoricalFeatures].__array__(dtype=object)

# )



# train_labels = y_train.astype('int').values



# clf = CatBoostClassifier(loss_function='MultiClass', verbose=True, depth=10, iterations= 100, l2_leaf_reg= 9, learning_rate= 0.15)

# clf.fit(train_data, train_labels)
# test_data = FeaturesData(

#     num_feature_data=X_test[numericalFeatures].astype('float32').values,

#     cat_feature_data=X_test[categoricalFeatures].__array__(dtype=object)

# )



# test_labels = y_test.astype('int').values

# y_predicted = clf.predict(test_data)



def generateConfusionMatrix(y_real, y_predicted):

    cm = pd.DataFrame()

    cm['Satisfaction'] = y_real

    cm['Predict'] = y_predicted

    mappingSatisfaction = {0:'Same Day', 1: 'First Week', 2: 'First Month', 3: '2-3 Month', 4: 'Non-Adopted >100'}

    mappingPredict = {0.0:'Same Day', 1.0: 'First Week', 2.0: 'First Month', 3.0: '2-3 Month', 4.0: 'Non-Adopted >100'}

    cm = cm.replace({'Satisfaction': mappingSatisfaction, 'Predict': mappingPredict})

    return pd.crosstab(cm['Satisfaction'], cm['Predict'], margins=True)



# generateConfusionMatrix(y_test, y_predicted)
# clf.score(test_data, test_labels)
def plotMostRelevantFeatures(indexes, model, train_data, train_labels, title='Feature Importance Ranking'):

    feature_score = pd.DataFrame(list(zip(indexes, model.get_feature_importance(Pool(train_data, label=train_labels)))),

                columns=['Feature','Score'])

    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

    plt.rcParams["figure.figsize"] = (12,7)

    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')

    ax.set_title(title, fontsize = 14)

    ax.set_xlabel('')



    rects = ax.patches



    # get feature score as labels round to 2 decimal

    labels = feature_score['Score'].round(2)



    for rect, label in zip(rects, labels):

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')



    plt.show()



# plotMostRelevantFeatures(X_train.dtypes.index, clf, train_data, train_labels)
# bestCatFeatures = ['Sterilized', 'Breed1', 'Type', 'MaturitySize', 'FurLength', 'Gender', 'Health', 'Color1', 'Fee', 'Vaccinated']

# bestNumFeatures = ['Age', 'Quantity', 'PhotoAmt']



# train_data = FeaturesData(

#     num_feature_data=X_train[bestNumFeatures].astype('float32').values,

#     cat_feature_data=X_train[bestCatFeatures].__array__(dtype=object)

# )



# train_labels = y_train.astype('int').values



# clf = CatBoostClassifier(loss_function='MultiClass', verbose=False, depth=10, iterations= 100, l2_leaf_reg= 9, learning_rate= 0.15)

# clf.fit(data[mostImportantFeatures], data[target], cat_features= mostImportantCatIndex, plot=False)

# predictions = clf.predict(test[mostImportantFeatures])

# predictions
# test['AdoptionSpeed'] = predictions

# test.AdoptionSpeed = test['AdoptionSpeed'].map({0.0: '0', 1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4'})

# test[['PetID', 'AdoptionSpeed']].to_csv('submission.csv', index=False)
def calculateClassificationScores(y_true, y_predicted, model, X_test, average='macro'):

    accuracy = accuracy_score(y_true, y_predicted)

    f1 = f1_score(y_true, y_predicted, average=average)

    precision = precision_score(y_true, y_predicted, average=average)

    recall = recall_score(y_true, y_predicted, average=average)

    if model and isinstance(model, CatBoostClassifier):

        score = model.get_best_score()

        if score.get('validation_0'):

            multiclass = score['validation_0']['MultiClass']

        else:

            multiclass = model.score(X_test, y_true)

        return (accuracy, f1, precision, recall, multiclass)

    

    return (accuracy, f1, precision, recall)



def crossValidation(params, X, y):

    f1_scores = []

    accuracy_scores = []

    precision_scores = []

    recall_scores = []

    multiclass_scores = []



    for train_index, val_index in skf.split(X.values, y.values):

        

        X_train = X[X.index.isin(train_index)]

        X_train = FeaturesData(

            num_feature_data=X_train[numericalFeatures].astype('float32').values,

            cat_feature_data=X_train[categoricalFeatures].__array__(dtype=object)

        )

        y_train = y[y.index.isin(train_index)].astype('int').values

        

        X_valid = X[X.index.isin(val_index)]

        X_valid = FeaturesData(

            num_feature_data=X_valid[numericalFeatures].astype('float32').values,

            cat_feature_data=X_valid[categoricalFeatures].__array__(dtype=object)

        )

        y_valid = y[y.index.isin(val_index)].astype('int').values

        

        pool_test = Pool(X_valid, label=y_valid)

        

        clf = CatBoostClassifier(

            loss_function='MultiClass',

            verbose=False,

            depth=params['depth'],

            iterations=params['iterations'],

            l2_leaf_reg=params['l2_leaf_reg'],

            learning_rate=params['learning_rate'],

            task_type='CPU'

        )

        

        clf.fit(X_train, y_train, eval_set=pool_test, use_best_model=True)

        

        y_pred = clf.predict(X_valid)

        

        # calculateClassificationScores(y_test, y_predicted, clfSentiment, X_test_pool)

        

        (accuracy, f1, precision, recall, multiclass) = calculateClassificationScores(y_valid, y_pred, clf, X_valid)

        

        multiclass_scores.append(multiclass)

        accuracy_scores.append(accuracy)

        f1_scores.append(f1)

        precision_scores.append(precision)

        recall_scores.append(recall)

        

    return (multiclass_scores, accuracy_scores, f1_scores, precision_scores, recall_scores)

    



def searchBestParams(grid, X, y):

    catboostDf = pd.DataFrame({

        'model':[],

        'multiclass_score_mean':[],

        'multiclass_score_std':[],

        'f1_score_mean':[],

        'f1_score_std':[],

        'accuracy_score_mean':[],

        'accuracy_score_std':[],

        'precision_score_mean':[],

        'precision_score_std':[],

        'recall_score_mean':[],

        'recall_score_std':[],

        'params': []}

    )

    for params in grid:

        print(params)

        (multiclass_scores, accuracy_scores, f1_scores, precision_scores, recall_scores) = crossValidation(params, X, y)

        catboostDf = catboostDf.append({

            'multiclass_score_mean': np.mean(multiclass_scores),

            'multiclass_score_std': np.std(multiclass_scores),

            'f1_score_mean': np.mean(f1_scores),

            'f1_score_std': np.std(f1_scores),

            'accuracy_score_mean': np.mean(accuracy_scores),

            'accuracy_score_std': np.std(accuracy_scores),

            'precision_score_mean': np.mean(precision_scores),

            'precision_score_std': np.std(precision_scores),

            'recall_score_mean': np.mean(recall_scores),

            'recall_score_std': np.std(recall_scores),

            'params': params

        }, ignore_index=True)



    return catboostDf
# params = {

#     'depth':[6, 8, 10, 14, 20],

#     'iterations':[100, 150, 200, 300, 500, 1000],

#     'learning_rate':[0.15], 

#     'l2_leaf_reg':[12]

# }



# print(numericalFeatures + categoricalFeatures)



# grid = ParameterGrid(params)



# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)



# bestScores = []

# X = data[numericalFeatures + categoricalFeatures]

# y = data[target]



# catboostDf = searchBestParams(grid, X, y)



# run only if you want to check a new combination of params

# catboostDf = searchBestParams(grid, X, y)

# bestCatModel = catboostDf[catboostDf.f1_score_mean == catboostDf.f1_score_mean.max()]

# bestCatModel.params.values[0]
# catboostDf.sort_values(by=['f1_score_mean'])
# categoricalFeatures = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',

#                        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',

#                        'Health', 'State', 'RescuerID', 'AdoptionSpeed']



# target = 'AdoptionSpeed'

# categoricalFeatures = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',

#                        'Health']

# numericalFeatures = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt']
def trainCatboost(numericalFeatures, categoricalFeatures, target, data, params=None):

    print(numericalFeatures)

    print(categoricalFeatures)

    print(target)

    print(data.size)

    X_train, X_test, y_train, y_test = train_test_split(

        data[numericalFeatures + categoricalFeatures],

        data[target],

        test_size=0.25,

        random_state=42

    )



    X_train_pool = FeaturesData(

        num_feature_data=X_train[numericalFeatures].astype('float32').values,

        cat_feature_data=X_train[categoricalFeatures].__array__(dtype=object)

    )

    y_train_pool = y_train.astype('int').values



    X_test_pool = FeaturesData(

        num_feature_data=X_test[numericalFeatures].astype('float32').values,

        cat_feature_data=X_test[categoricalFeatures].__array__(dtype=object)

    )

    y_test_pool = y_test.astype('int').values

    

    model = None

    

    if params:

        model = CatBoostClassifier(

            loss_function='MultiClass',

            verbose=False,

            depth=params['depth'],

            iterations=params['iterations'],

            l2_leaf_reg=params['l2_leaf_reg'],

            learning_rate=params['learning_rate'],

            task_type='GPU',

            class_weights=[4, 1, 1, 1, 1]

        )

    else:

        model = CatBoostClassifier(

            loss_function='MultiClass',

            verbose=False,

            depth=8,

            iterations=140,

            l2_leaf_reg=12,

            learning_rate=0.15,

            task_type='GPU',

            class_weights=[4, 1, 1, 1, 1]

        )



    model.fit(X_train_pool, y_train_pool,logging_level='Silent')

    

    y_predicted = model.predict(X_test_pool)

    return (model, y_predicted, X_train, y_train, X_train_pool, X_test_pool, y_test)
# params = {

#     'depth':8,

#     'iterations': 140,

#     'learning_rate': 0.15, 

#     'l2_leaf_reg': 12

# }



# (bestClf, 

#  y_predicted, 

#  X_train, 

#  y_train, 

#  X_train_pool,

#  X_test_pool,

#  y_test) = trainCatboost(numericalFeatures, categorical, target, data, params)
# scores = calculateClassificationScores(y_test, y_predicted, bestClf, X_test_pool)

# print('accuracy: %f f1: %f precision: %f recall: %f multiclass: %f' % scores )
# generateConfusionMatrix(y_test, y_predicted)
# np.sum(y_predicted == 0.0)
# plotMostRelevantFeatures(X_train.dtypes.index, bestClf, X_train_pool, y_train.astype('int').values)
# targetSentiment = 'AdoptionSpeed'

# categoricalFeaturesSentiment = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',

#                        'Health']

# numericalFeaturesSentiment = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt', 'sentiment_document_score', 'sentiment_document_magnitude']



# trying new set of features from different models

# categoricalFeaturesSentiment = ['Breed1', 'Sterilized', 'Vaccinated', 'MaturitySize', 'Gender']

# numericalFeaturesSentiment = ['sentiment_document_magnitude', 'Age', 'PhotoAmt', 'Quantity']
# msno.matrix(train[categoricalFeaturesSentiment + numericalFeaturesSentiment])
# params = {'depth': 6, 'iterations': 300, 'l2_leaf_reg': 12, 'learning_rate': 0.15}



# (clfSentiment, 

#  y_predicted, 

#  X_train, 

#  y_train, 

#  X_train_pool,

#  X_test_pool,

#  y_test) = trainCatboost(numericalFeaturesSentiment, categoricalFeaturesSentiment, targetSentiment, train[categoricalFeaturesSentiment + numericalFeaturesSentiment + [targetSentiment]].dropna(), params)
# scores = calculateClassificationScores(y_test, y_predicted, clfSentiment, X_test_pool)

# print('accuracy: %f f1: %f precision: %f recall: %f multiclass: %f' % scores )
# generateConfusionMatrix(y_test, y_predicted)
# plotMostRelevantFeatures(X_train.dtypes.index, bestClf, X_train_pool, y_train.astype('int').values)
# search best params for new features

# params = {

#     'depth':[6, 8, 10, 12],

#     'iterations':[100, 150, 300],

#     'learning_rate':[0.15], 

#     'l2_leaf_reg':[12]

# }



# grid = ParameterGrid(params)



# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)



#dataS = train[numericalFeaturesSentiment + categoricalFeaturesSentiment + [targetSentiment]].dropna()



#bestScores = []

#X = dataS[numericalFeaturesSentiment + categoricalFeaturesSentiment]

#y = dataS[targetSentiment]



#catboostDf = searchBestParams(grid, X, y)

#bestCatModel = catboostDf[catboostDf.f1_score_mean == catboostDf.f1_score_mean.max()]

#bestCatModel.params.values[0]
#bestCatModel.f1_score_mean
# data['AgeInterval'] = pd.Series(['0-3', '3-6', '6-12', '12-24', '24-48', '48-120', '>120'], dtype='category')

# data.loc[(data['Age'] >= 0) & (data['Age'] <= 3),'AgeInterval'] = '0-3'

# data.loc[(data['Age'] > 3) & (data['Age'] <= 6),'AgeInterval'] = '3-6'

# data.loc[(data['Age'] > 6) & (data['Age'] <= 12),'AgeInterval'] = '6-12'

# data.loc[(data['Age'] > 12) & (data['Age'] <= 24),'AgeInterval'] = '12-24'

# data.loc[(data['Age'] > 24) & (data['Age'] <= 48),'AgeInterval'] = '24-48'

# data.loc[(data['Age'] > 48) & (data['Age'] <= 120),'AgeInterval'] = '48-120'

# data.loc[data['Age'] > 120,'AgeInterval'] = '>120'
# target = 'AdoptionSpeed'

# categoricalFeatures = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',

#                        'Health', 'AgeInterval']

# numericalFeatures = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt']



# (model, 

#  y_predicted, 

#  X_train, 

#  y_train, 

#  X_train_pool, 

#  X_test_pool) = trainCatboost(numericalFeatures, categoricalFeatures, target, data)
# scores = calculateClassificationScores(y_test, y_predicted, model, X_test_pool)

# print('accuracy: %f f1: %f precision: %f recall: %f multiclass: %f' % scores )
# plotMostRelevantFeatures(X_train.dtypes.index, bestClf, X_train_pool, y_train.astype('int').values)
# dataS = train[numericalFeatures + categorical + [target]].dropna()



# X = dataS[numericalFeatures + categorical]

# y = dataS[target]



# prepare data for catboost

# X_train = FeaturesData(

#     num_feature_data=X[numericalFeatures].astype('float32').values,

#     cat_feature_data=X[categorical].__array__(dtype=object)

# )

# y_train = y.astype('int').values



# best model is with sentiment data {'depth': 6, 'iterations': 300, 'l2_leaf_reg': 12, 'learning_rate': 0.15}

# bestClf = CatBoostClassifier(

#             loss_function='MultiClass',

#             verbose=False,

#             depth=8,

#             iterations=140,

#             l2_leaf_reg=12,

#             learning_rate=0.15,

#             task_type='CPU',

#             class_weights=[4, 1, 1, 1, 1]

#         )

        

# bestClf.fit(X_train, y_train,logging_level='Silent')
# transforming test as we transformed train

# testClean = cleanTransformDataset(test)

# testClean.head()
def generateSubmissionCatboost(model, numericalFeatures, categoricalFeatures, test, fileName):

    X = test[numericalFeatures + categoricalFeatures]

    X = FeaturesData(

            num_feature_data=X[numericalFeatures].astype('float32').values,

            cat_feature_data=X[categoricalFeatures].__array__(dtype=object)

        )

    predictions = model.predict(X)

    test['AdoptionSpeed'] = predictions

    test.AdoptionSpeed = test['AdoptionSpeed'].map({0.0: '0', 1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4'})

    test[['PetID', 'AdoptionSpeed']].to_csv(fileName + '.csv', index=False)
# generateSubmissionCatboost(bestClf, numericalFeatures, categorical, 'submission')
# X_train, X_test, y_train, y_test = train_test_split(

#     pd.get_dummies(X),

#     y.astype('int64'),

#     test_size=0.25,

#     random_state=42

# )



# rf = RandomForestClassifier()



# rf.fit(X_train, y_train)

# y_predicted = rf.predict(X_test)
# scores = calculateClassificationScores(y_test, y_predicted, rf, X_test)

# print('accuracy: %f f1: %f precision: %f recall: %f' % scores )
# Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 400, num = 10)]

# Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(6, 110, num = 11)]

# max_depth.append(None)

# Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

# bootstrap = [True, False]

# Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters

# First create the base model to tune

# rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

# rf_random.fit(X_train, y_train)
# y_predicted = rf_random.predict(X_test)

# scores = calculateClassificationScores(y_test, y_predicted, rf_random, X_test)

# print('accuracy: %f f1: %f precision: %f recall: %f' % scores )
# rf_random.best_params_
# generateConfusionMatrix(y_test, y_predicted)
# ab = AdaBoostClassifier()

# ab.fit(X_train, y_train)
# y_predicted = ab.predict(X_test)

# scores = calculateClassificationScores(y_test, y_predicted, ab, X_test)

# print('accuracy: %f f1: %f precision: %f recall: %f' % scores )
# gb = GradientBoostingClassifier()

# gb.fit(X_train, y_train)
# y_predicted = gb.predict(X_test)

# scores = calculateClassificationScores(y_test, y_predicted, gb, X_test)

# print('accuracy: %f f1: %f precision: %f recall: %f' % scores )
# param = {'max_depth':8, 'eta':0.15, 'silent':1, 'objective':'multi:softmax' }

# num_round = 140

# xgbModel = xgb.XGBClassifier(max_depth=8, n_estimators=140, learning_rate=0.15)

# xgbModel.fit(X_train, y_train)
# vc = VotingClassifier(estimators=[('cb', bestClf), ('rf', rf_random), ('ab', ab), ('gb', gb) ], voting='hard')

# vc = vc.fit(X_train, y_train)
# y_predicted = vc.predict(X_test)

# scores = calculateClassificationScores(y_test, y_predicted, vc, X_test)

# print('accuracy: %f f1: %f precision: %f recall: %f' % scores )  
# X_train_non_null = trainPrecomputed.fillna(-1)

# X_test_non_null = testPrecomputed.fillna(-1)
X_train_non_null.head()
train.columns
# newFeatures = [

#     'metadata_topicality_max',

#     'metadata_topicality_mean',

#     'metadata_topicality_min',

#     'metadata_topicality_0_mean',

#     'metadata_topicality_0_max',

#     'metadata_topicality_0_min',

#     'L_metadata_0_cat_sum',

#     'L_metadata_0_dog_sum',

#     'L_metadata_any_cat_sum',

#     'L_metadata_any_dog_sum',

#     'blur_max',

#     'blur_sum',

#     'huMoments0',

#     'huMoments1',

#     'huMoments2',

#     'huMoments3',

#     'huMoments4',

#     'huMoments5',

#     'huMoments6',

#     'state_gdp',

#     'state_population',

#     'state_area',

#     'state_unemployment',

#     'state_birth_rate',

#     'L_Fee_Free',

#     'N_pets_total',

#     'L_NoPhoto',

#     'L_NoVideo',

#     'Log_Age',

#     'L_scoreneg',

#     'PetID'

# ]

# X_train_non_null = X_train_non_null.join(train[newFeatures].set_index('PetID'), 'PetID')

# X_test_non_null = X_test_non_null.join(test[newFeatures].set_index('PetID'), 'PetID')
to_drop_columns = ['PetID', 'Name', 'RescuerID']

X_train_non_null = X_train_non_null.drop(to_drop_columns, axis=1)
testIds = X_test_non_null['PetID']

X_test_non_null = X_test_non_null.drop(to_drop_columns, axis=1)
X_test_non_null = X_test_non_null.drop(['AdoptionSpeed'], axis=1)
import scipy as sp



from collections import Counter

from functools import partial

from math import sqrt



from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix





# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features



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
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold



xgb_params = {

    'eval_metric': 'rmse',

    'seed': 1337,

    'eta': 0.0123,

    'subsample': 0.8,

    'colsample_bytree': 0.85,

    'tree_method': 'gpu_hist',

    'device': 'gpu',

    'silent': 1,

}
def run_xgb(params, X_train, X_test):

    n_splits = 10

    verbose_eval = 1000

    num_rounds = 60000

    early_stop = 500



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)



    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], n_splits))



    i = 0



    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]



        y_tr = X_tr['AdoptionSpeed'].values

        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)



        y_val = X_val['AdoptionSpeed'].values

        X_val = X_val.drop(['AdoptionSpeed'], axis=1)



        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)



        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,

                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test
model, oof_train, oof_test = run_xgb(xgb_params, X_train_non_null, X_test_non_null)
def plot_pred(pred):

    sns.distplot(pred, kde=True, hist_kws={'range': [0, 5]})
plot_pred(oof_train)
plot_pred(oof_test.mean(axis=1))
optR = OptimizedRounder()

optR.fit(oof_train, X_train_non_null['AdoptionSpeed'].values)

coefficients = optR.coefficients()

valid_pred = optR.predict(oof_train, coefficients)

qwk = quadratic_weighted_kappa(X_train_non_null['AdoptionSpeed'].values, valid_pred)

print("QWK = ", qwk)
coefficients_ = coefficients.copy()

coefficients_[0] = 1.66

coefficients_[1] = 2.13

coefficients_[3] = 2.85

train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)

print(f'train pred distribution: {Counter(train_predictions)}')

test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)

print(f'test pred distribution: {Counter(test_predictions)}')
Counter(train_predictions)
Counter(test_predictions)
X_test_non_null.shape
len(test_predictions)
submission = pd.DataFrame({'PetID': testIds.values, 'AdoptionSpeed': test_predictions})

submission.to_csv('submission.csv', index=False)

submission.head()
# X_train_cb = X_train_non_null.copy()





# target = 'AdoptionSpeed'

# categoricalFeatures = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',

#                        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized',

#                        'Health', 'State', 'AdoptionSpeed']



# X_train_cb[categoricalFeatures] = X_train_cb[categoricalFeatures].astype('category')



# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# numericalFeatures = list(X_train_cb.select_dtypes(include=numerics).columns)



# X_train_cb[numericalFeatures] = X_train_cb[numericalFeatures].fillna(-1)



# categoricalFeaturesTest = list(set(categoricalFeatures) - set([target]))

# X_test_cb = X_test_non_null.copy()

# X_test_cb[categoricalFeaturesTest] = X_test_cb[categoricalFeaturesTest].astype('category')

# X_test_cb[numericalFeatures] = X_test_cb[numericalFeatures].fillna(-1)



# # cleanTransformDataset(X_train_cb, categoricalFeatures)



# (model, 

#  y_predicted, 

#  X_train, 

#  y_train, 

#  X_train_pool, 

#  X_test_pool,

#  y_test) = trainCatboost(numericalFeatures, categoricalFeatures, target, X_train_cb)
# scores = calculateClassificationScores(y_test, y_predicted, model, X_test_pool)

# print('accuracy: %f f1: %f precision: %f recall: %f multiclass: %f' % scores )
# generateConfusionMatrix(y_test, y_predicted)
# plotMostRelevantFeatures(X_train.dtypes.index, model, X_train_pool, y_train.astype('int').values)
# X = X_train_cb[numericalFeatures + categoricalFeatures]

# y = X_train_cb[target]



# # prepare data for catboost

# X_train = FeaturesData(

#     num_feature_data=X[numericalFeatures].astype('float32').values,

#     cat_feature_data=X[categoricalFeatures].__array__(dtype=object)

# )

# y_train = y.astype('int').values



# # best model is with sentiment data {'depth': 6, 'iterations': 300, 'l2_leaf_reg': 12, 'learning_rate': 0.15}

# bestClf = CatBoostClassifier(

#             loss_function='MultiClass',

#             verbose=False,

#             depth=8,

#             iterations=140,

#             l2_leaf_reg=12,

#             learning_rate=0.15,

#             task_type='CPU',

#             class_weights=[4, 1, 1, 1, 1]

#         )

        

# bestClf.fit(X_train, y_train,logging_level='Silent')
# generateSubmissionCatboost(bestClf, numericalFeatures, categoricalFeatures, X_test_cb, 'submission')