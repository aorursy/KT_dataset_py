import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Conv2D, Reshape

from keras.layers import MaxPool2D, PReLU, AvgPool2D

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.optimizers import Adam

from keras.preprocessing import text, sequence

from gensim.models.keyedvectors import KeyedVectors

import tensorflow as tf

from functools import partial

import scipy as sp

from PIL import Image

from sklearn.model_selection import KFold

from catboost import CatBoostRegressor

from sklearn.metrics import cohen_kappa_score

def kappa(y_true, y_pred):

    return cohen_kappa_score(y_true, y_pred, weights='quadratic')



def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn



import os

#print(os.listdir("../input"))



import gc

import glob

import json

import matplotlib.pyplot as plt

import pprint

from joblib import Parallel, delayed

from tqdm import tqdm

from PIL import Image

# fastai

import numpy as np # linear algebra

import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



from matplotlib import pyplot as plt



from fastai import *

from fastai.tabular import *

from fastai.basic_data import DataBunch

from tqdm import tqdm_notebook

%matplotlib inline



pd.options.display.max_rows = 128

pd.options.display.max_columns = 128
plt.rcParams['figure.figsize'] = (12, 9)
os.listdir('../input/')
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')

labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
#data_path = "../input/petfinder-adoption-prediction/"

#train_df = pd.read_csv(data_path + "train/train.csv")

#test_df = pd.read_csv(data_path + "test/test.csv")



# https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn

train_img = pd.read_csv("../input/extract-image-features-from-pretrained-nn/train_img_features.csv")

test_img = pd.read_csv("../input/extract-image-features-from-pretrained-nn/test_img_features.csv")

train_img.rename(columns=lambda i: f"img_{i}" ,inplace=True)

test_img.rename(columns=lambda i: f"img_{i}" ,inplace=True)



train = pd.concat([train, train_img], axis=1)

test = pd.concat([test, test_img], axis=1)
# add features from ratings 

with open('../input/cat-and-dog-breeds-parameters/rating.json', 'r') as f:

        ratings = json.load(f)
breed_id = {}

for id,name in zip(labels_breed.BreedID,labels_breed.BreedName):

    breed_id[id] = name
breed_ratings = json.load(open("../input/cat-and-dog-breeds-parameters/rating.json",'r'))

species_keys = list(breed_ratings.keys())



# Spread species dictionary into a single dictionary.

breed_ratings_dict = {**breed_ratings['cat_breeds']}

#breed_ratings_dog = {**breed_ratings['dog_breeds']}

breed_ratings_dict = {**breed_ratings_dict,**breed_ratings['dog_breeds']}



# Convert into a dataframe, transpose the table so the Breed is the index, rename the index.

breed_score_df = pd.DataFrame(breed_ratings_dict).T.reset_index().rename(columns={'index':'breed_name'})

#breed_score_cat = pd.DataFrame(breed_ratings_cat).T.reset_index().rename(columns={'index':'breed_name'})

#train_ = train_.merge(breed_score_df,how='left',left_on='breed_name',right_on='breed_name')
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



state_area ={

    41336:19102,

41325:9500,

41367:15099,

41401:243,

41415:91,

41324:1664,

41332:6686,

41335:36137,

41330:21035,

41380:821,

41327:1048,

41345:73631,

41342:124450,

41326:8104,

41361:13035}



train["state_gdp"] = train.State.map(state_gdp)

train["state_population"] = train.State.map(state_population)

test["state_gdp"] = test.State.map(state_gdp)

test["state_population"] = test.State.map(state_population)

train["state_area"] = train.State.map(state_area)

test["state_area"] = test.State.map(state_area)
train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))

train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))

train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))



print('num of train images files: {}'.format(len(train_image_files)))

print('num of train metadata files: {}'.format(len(train_metadata_files)))

print('num of train sentiment files: {}'.format(len(train_sentiment_files)))





test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))

test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))

test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))



print('num of test images files: {}'.format(len(test_image_files)))

print('num of test metadata files: {}'.format(len(test_metadata_files)))

print('num of test sentiment files: {}'.format(len(test_sentiment_files)))
plt.rcParams['figure.figsize'] = (12, 9)

plt.style.use('ggplot')





# Images:

train_df_ids = train[['PetID']]

print(train_df_ids.shape)



train_df_imgs = pd.DataFrame(train_image_files)

train_df_imgs.columns = ['image_filename']

train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

print(len(train_imgs_pets.unique()))



pets_with_images = len(np.intersect1d(train_imgs_pets.unique(), train_df_ids['PetID'].unique()))

print('fraction of pets with images: {:.3f}'.format(pets_with_images / train_df_ids.shape[0]))



# Metadata:

train_df_ids = train[['PetID']]

train_df_metadata = pd.DataFrame(train_metadata_files)

train_df_metadata.columns = ['metadata_filename']

train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)

print(len(train_metadata_pets.unique()))



pets_with_metadatas = len(np.intersect1d(train_metadata_pets.unique(), train_df_ids['PetID'].unique()))

print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / train_df_ids.shape[0]))



# Sentiment:

train_df_ids = train[['PetID']]

train_df_sentiment = pd.DataFrame(train_sentiment_files)

train_df_sentiment.columns = ['sentiment_filename']

train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])

train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)

print(len(train_sentiment_pets.unique()))



pets_with_sentiments = len(np.intersect1d(train_sentiment_pets.unique(), train_df_ids['PetID'].unique()))

print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiments / train_df_ids.shape[0]))
# Images:

test_df_ids = test[['PetID']]

print(test_df_ids.shape)



test_df_imgs = pd.DataFrame(test_image_files)

test_df_imgs.columns = ['image_filename']

test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

print(len(test_imgs_pets.unique()))



pets_with_images = len(np.intersect1d(test_imgs_pets.unique(), test_df_ids['PetID'].unique()))

print('fraction of pets with images: {:.3f}'.format(pets_with_images / test_df_ids.shape[0]))





# Metadata:

test_df_ids = test[['PetID']]

test_df_metadata = pd.DataFrame(test_metadata_files)

test_df_metadata.columns = ['metadata_filename']

test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)

print(len(test_metadata_pets.unique()))



pets_with_metadatas = len(np.intersect1d(test_metadata_pets.unique(), test_df_ids['PetID'].unique()))

print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadatas / test_df_ids.shape[0]))



# Sentiment:

test_df_ids = test[['PetID']]

test_df_sentiment = pd.DataFrame(test_sentiment_files)

test_df_sentiment.columns = ['sentiment_filename']

test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])

test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)

print(len(test_sentiment_pets.unique()))



pets_with_sentiments = len(np.intersect1d(test_sentiment_pets.unique(), test_df_ids['PetID'].unique()))

print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiments / test_df_ids.shape[0]))





# are distributions the same?

print('images and metadata distributions the same? {}'.format(

    np.all(test_metadata_pets == test_imgs_pets)))
class PetFinderParser(object):

    

    def __init__(self, debug=False):

        

        self.debug = debug

        self.sentence_sep = ' '

        

        # Does not have to be extracted because main DF already contains description

        self.extract_sentiment_text = False

        

        

    def open_metadata_file(self, filename):

        """

        Load metadata file.

        """

        with open(filename, 'r') as f:

            metadata_file = json.load(f)

        return metadata_file

            

    def open_sentiment_file(self, filename):

        """

        Load sentiment file.

        """

        with open(filename, 'r') as f:

            sentiment_file = json.load(f)

        return sentiment_file

            

    def open_image_file(self, filename):

        """

        Load image file.

        """

        image = np.asarray(Image.open(filename))

        return image

        

    def parse_sentiment_file(self, file):

        """

        Parse sentiment file. Output DF with sentiment features.

        """

        

        file_sentiment = file['documentSentiment']

        file_entities = [x['name'] for x in file['entities']]

        file_entities = self.sentence_sep.join(file_entities)



        if self.extract_sentiment_text:

            file_sentences_text = [x['text']['content'] for x in file['sentences']]

            file_sentences_text = self.sentence_sep.join(file_sentences_text)

        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        

        file_sentences_sentiment = pd.DataFrame.from_dict(

            file_sentences_sentiment, orient='columns').sum()

        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

        

        file_sentiment.update(file_sentences_sentiment)

        

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T

        if self.extract_sentiment_text:

            df_sentiment['text'] = file_sentences_text

            

        df_sentiment['entities'] = file_entities

        df_sentiment = df_sentiment.add_prefix('sentiment_')

        

        return df_sentiment

    

    def parse_metadata_file(self, file):

        """

        Parse metadata file. Output DF with metadata features.

        """

        

        file_keys = list(file.keys())

        

        if 'labelAnnotations' in file_keys:

            file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.3)]

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

    



# Helper function for parallel data processing:

def extract_additional_features(pet_id, mode='train'):

    

    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)

    try:

        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)

        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)

        df_sentiment['PetID'] = pet_id

    except FileNotFoundError:

        df_sentiment = []



    dfs_metadata = []

    metadata_filenames = sorted(glob.glob('../input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(mode, pet_id)))

    if len(metadata_filenames) > 0:

        for f in metadata_filenames:

            metadata_file = pet_parser.open_metadata_file(f)

            df_metadata = pet_parser.parse_metadata_file(metadata_file)

            df_metadata['PetID'] = pet_id

            dfs_metadata.append(df_metadata)

        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)

    dfs = [df_sentiment, dfs_metadata]

    

    return dfs





pet_parser = PetFinderParser()
# Unique IDs from train and test:

debug = False

train_pet_ids = train.PetID.unique()

test_pet_ids = test.PetID.unique()



if debug:

    train_pet_ids = train_pet_ids[:1000]

    test_pet_ids = test_pet_ids[:500]





# Train set:

# Parallel processing of data:

dfs_train = Parallel(n_jobs=6, verbose=1)(

    delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)



# Extract processed data and format them as DFs:

train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]

train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]



train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)

train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)



print(train_dfs_sentiment.shape, train_dfs_metadata.shape)





# Test set:

# Parallel processing of data:

dfs_test = Parallel(n_jobs=6, verbose=1)(

    delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)



# Extract processed data and format them as DFs:

test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]

test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]



test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)

test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)



print(test_dfs_sentiment.shape, test_dfs_metadata.shape)


def impact_coding(data, feature, target='y'):

    '''

    In this implementation we get the values and the dictionary as two different steps.

    This is just because initially we were ignoring the dictionary as a result variable.

    

    In this implementation the KFolds use shuffling. If you want reproducibility the cv 

    could be moved to a parameter.

    '''

    n_folds = 20

    n_inner_folds = 10

    impact_coded = pd.Series()

    

    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)

    kf = KFold(n_splits=n_folds, shuffle=True)

    oof_mean_cv = pd.DataFrame()

    split = 0

    for infold, oof in kf.split(data[feature]):

            impact_coded_cv = pd.Series()

            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)

            inner_split = 0

            inner_oof_mean_cv = pd.DataFrame()

            oof_default_inner_mean = data.iloc[infold][target].mean()

            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):

                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)

                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()

                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(

                            lambda x: oof_mean[x[feature]]

                                      if x[feature] in oof_mean.index

                                      else oof_default_inner_mean

                            , axis=1))



                # Also populate mapping (this has all group -> mean for all inner CV folds)

                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')

                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)

                inner_split += 1



            # Also populate mapping

            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')

            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)

            split += 1

            

            impact_coded = impact_coded.append(data.iloc[oof].apply(

                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()

                                      if x[feature] in inner_oof_mean_cv.index

                                      else oof_default_mean

                            , axis=1))



    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean  

    

    

def frequency_encoding(df, col_name):

    new_name = "{}_counts".format(col_name)

    new_col_name = "{}_freq".format(col_name)

    grouped = df.groupby(col_name).size().reset_index(name=new_name)

    df = df.merge(grouped, how = "left", on = col_name)

    df[new_col_name] = df[new_name]/df[new_name].count()

    del df[new_name]

    return df

# Extend aggregates and improve column naming

aggregates = ['mean', 'sum', 'var']





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

train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(

            prefix, c[0], c[1].upper()) for c in train_metadata_gr.columns.tolist()])

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

train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(aggregates)

train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(

            prefix, c[0], c[1].upper()) for c in train_sentiment_gr.columns.tolist()])

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

test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(

            prefix, c[0], c[1].upper()) for c in test_metadata_gr.columns.tolist()])

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

test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(aggregates)

test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(

            prefix, c[0], c[1].upper()) for c in test_sentiment_gr.columns.tolist()])

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
train_proc = train_proc.merge(breed_score_df,how='left',left_on='main_breed_BreedName',right_on='breed_name')

test_proc = test_proc.merge(breed_score_df,how='left',left_on='main_breed_BreedName',right_on='breed_name')
train_proc.head()
X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)

print('NaN structure:\n{}'.format(np.sum(pd.isnull(X))))
column_types = X.dtypes



int_cols = column_types[column_types == 'int']

float_cols = column_types[column_types == 'float']

cat_cols = column_types[column_types == 'object']



print('\tinteger columns:\n{}'.format(int_cols))

print('\n\tfloat columns:\n{}'.format(float_cols))

print('\n\tto encode categorical columns:\n{}'.format(cat_cols))
# Copy original X DF for easier experimentation,

# all feature engineering will be performed on this one:

X_temp = X.copy()





# Select subsets of columns:

text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']

categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']



# Names are all unique, so they can be dropped by default

# Same goes for PetID, it shouldn't be used as a feature

to_drop_columns = ['PetID', 'Name', 'RescuerID']

# RescuerID will also be dropped, as a feature based on this column will be extracted independently
# Count RescuerID occurrences:

rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()

rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']



# Merge as another feature onto main DF:

X_temp = X_temp.merge(rescuer_count, how='left', on='RescuerID')
# Factorize categorical columns:

for i in categorical_columns:

    X_temp.loc[:, i] = pd.factorize(X_temp.loc[:, i])[0]
# Subset text features:

X_text = X_temp[text_columns]



for i in X_text.columns:

    X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF



n_components = 5

text_features = []





# Generate text features:

for i in X_text.columns:

    

    # Initialize decomposition methods:

    print('generating features from: {}'.format(i))

    svd_ = TruncatedSVD(

        n_components=n_components, random_state=1337)

    nmf_ = NMF(

        n_components=n_components, random_state=1337)

    

    tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)

    svd_col = svd_.fit_transform(tfidf_col)

    svd_col = pd.DataFrame(svd_col)

    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))

    

    nmf_col = nmf_.fit_transform(tfidf_col)

    nmf_col = pd.DataFrame(nmf_col)

    nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))

    

    text_features.append(svd_col)

    text_features.append(nmf_col)



    

# Combine all extracted features:

text_features = pd.concat(text_features, axis=1)



# Concatenate with main DF:

X_temp = pd.concat([X_temp, text_features], axis=1)



# Remove raw text columns:

for i in X_text.columns:

    X_temp = X_temp.drop(i, axis=1)
# Remove unnecessary columns:

X_temp = X_temp.drop(to_drop_columns, axis=1)



# Check final df shape:

print('X shape: {}'.format(X_temp.shape))
# Split into train and test again:

X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]

X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]



# Remove missing target column from test:

X_test = X_test.drop(['AdoptionSpeed'], axis=1)





print('X_train shape: {}'.format(X_train.shape))

print('X_test shape: {}'.format(X_test.shape))



assert X_train.shape[0] == train.shape[0]

assert X_test.shape[0] == test.shape[0]





# Check if columns between the two DFs are the same:

train_cols = X_train.columns.tolist()

train_cols.remove('AdoptionSpeed')



test_cols = X_test.columns.tolist()



assert np.all(train_cols == test_cols)
categorical_features = ["Type", "Breed1", "Breed2", "Color1" ,"Color2", "Color3", "State"]



impact_coding_map = {}

for f in categorical_features:

        print("Impact coding for {}".format(f))

        X_train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(X_train, f, target="AdoptionSpeed")

        impact_coding_map[f] = (impact_coding_mapping, default_coding)

        mapping, default_mean = impact_coding_map[f]

        X_test["impact_encoded_{}".format(f)] = X_test.apply(lambda x: mapping[x[f]] if x[f] in mapping

                                                         else default_mean, axis=1)



for cat in categorical_features:

        X_train = frequency_encoding(X_train, cat)

        X_test = frequency_encoding(X_test, cat)
X_train.head()
np.sum(pd.isnull(X_test))
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

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = quadratic_weighted_kappa(y, X_p)

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']

    

def rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))
import lightgbm as lgb





params = {'application': 'regression',

          'boosting': 'gbdt',

          'metric': 'rmse',

          'num_leaves': 70,

          'max_depth': 9,

          'learning_rate': 0.01,

          'bagging_fraction': 0.85,

          'feature_fraction': 0.8,

          'min_split_gain': 0.02,

          'min_child_samples': 150,

          'min_child_weight': 0.02,

          'lambda_l2': 0.0475,

          'verbosity': -1,

          'data_random_seed': 17}



# Additional parameters:

early_stop = 1000

verbose_eval = 100

num_rounds = 10000

n_splits = 5
X_train = X_train.drop('img_Unnamed: 0',axis=1)

X_test = X_test.drop('img_Unnamed: 0',axis=1)
X_train = X_train.drop('breed_name',axis=1)

X_test = X_test.drop('breed_name',axis=1)
from sklearn.model_selection import StratifiedKFold





kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)





oof_train_lgb = np.zeros((X_train.shape[0]))

oof_test_lgb = np.zeros((X_test.shape[0], n_splits))

qwk_scores = []



i = 0

for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):

    

    X_tr = X_train.iloc[train_index, :]

    X_val = X_train.iloc[valid_index, :]

    

    y_tr = X_tr['AdoptionSpeed'].values

    X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

    

    y_val = X_val['AdoptionSpeed'].values

    X_val = X_val.drop(['AdoptionSpeed'], axis=1)

    

    print('\ny_tr distribution: {}'.format(Counter(y_tr)))

    

    d_train = lgb.Dataset(X_tr, label=y_tr)

    d_valid = lgb.Dataset(X_val, label=y_val)

    watchlist = [d_train, d_valid]

    

    print('training LGB:')

    model = lgb.train(params,

                      train_set=d_train,

                      num_boost_round=num_rounds,

                      valid_sets=watchlist,

                      verbose_eval=verbose_eval,

                      early_stopping_rounds=early_stop)

    

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    optR = OptimizedRounder()

    optR.fit(val_pred, y_val)

    coefficients = optR.coefficients()

    coefficients[0] = 1.645

    coefficients[1] = 2.115

    coefficients[3] = 2.84



    pred_val_y_k = optR.predict(val_pred, coefficients)

    print("Valid Counts = ", Counter(y_val))

    print("Predicted Counts = ", Counter(pred_val_y_k))

    #print("Coefficients = ", coefficients)

    qwk = quadratic_weighted_kappa(y_val, pred_val_y_k)

    qwk_scores.append(qwk)

    print("QWK = ", qwk)

    

    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    

    oof_train_lgb[valid_index] = val_pred

    oof_test_lgb[:, i] = test_pred

    

    i += 1

    

print('{} cv mean QWK score : {}'.format('LGBM', np.mean(qwk_scores)))
importance_type= "split" # "gain"

idx_sort = np.argsort(model.feature_importance(importance_type=importance_type))[::-1]

names_sorted = np.array(model.feature_name())[idx_sort]

imports_sorted = model.feature_importance(importance_type=importance_type)[idx_sort]

for n, im in zip(names_sorted, imports_sorted):

    print(n, im)
# Compute QWK based on OOF train predictions:

optR = OptimizedRounder()

optR.fit(oof_train_lgb, X_train['AdoptionSpeed'].values)

coefficients = optR.coefficients()

pred_test_y_k = optR.predict(oof_train_lgb, coefficients)

print("\nValid Counts = ", Counter(X_train['AdoptionSpeed'].values))

print("Predicted Counts = ", Counter(pred_test_y_k))

print("Coefficients = ", coefficients)

qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, pred_test_y_k)

print("QWK = ", qwk)
# Manually adjusted coefficients:



coefficients_ = coefficients.copy()



coefficients_[0] = 1.645

coefficients_[1] = 2.115

coefficients_[3] = 2.84



train_predictions_lgb = optR.predict(oof_train_lgb, coefficients_).astype(int)

print('train pred distribution: {}'.format(Counter(train_predictions_lgb)))

#print("Train Predictions Counts = ", Counter(train_predictions))



test_predictions_lgb = optR.predict(oof_test_lgb.mean(axis=1), coefficients_)

#print("Test Predictions Counts = ", Counter(test_predictions))

print('test pred distribution: {}'.format(Counter(test_predictions_lgb)))
# Distribution inspection of original target and predicted train and test:



print("True Distribution:")

print(pd.value_counts(X_train['AdoptionSpeed'], normalize=True).sort_index())

print("\nTrain Predicted Distribution:")

print(pd.value_counts(train_predictions_lgb, normalize=True).sort_index())

print("\nTest Predicted Distribution:")

print(pd.value_counts(test_predictions_lgb, normalize=True).sort_index())
# Generate submission:



submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions_lgb.astype(np.int32)})

submission.head()

submission.to_csv('submission.csv', index=False)
submission.head()