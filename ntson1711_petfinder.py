# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import glob

import zipfile

import os

import gc

import json

import seaborn as sns

#import parallel

from collections import Counter

from functools import partial

import scipy as sp

from PIL import Image

from joblib import Parallel, delayed

from scipy.stats import skew 

from scipy.stats import norm

from wordcloud import WordCloud, STOPWORDS



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF
#from google.colab import drive

#drive.mount('/content/drive')
#breeds = pd.read_csv('/content/drive/My Drive/PetFinder.my Adoption Prediction/PetFinder.my Adoption Prediction/breed_labels.csv')

#colors = pd.read_csv('/content/drive/My Drive/PetFinder.my Adoption Prediction/PetFinder.my Adoption Prediction/color_labels.csv')

#states = pd.read_csv('/content/drive/My Drive/PetFinder.my Adoption Prediction/PetFinder.my Adoption Prediction/state_labels.csv')
breeds = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/breed_labels.csv')

colors = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/color_labels.csv')

states = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/state_labels.csv')
print(breeds.head())

print(colors.head())

print(states.head())
train = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')

test = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv')

submission = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/sample_submission.csv')
train_images = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/train_images/*.jpg'))

train_metadata = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/train_metadata/*.json'))

train_sentiment = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/train_sentiment/*.json'))



test_images = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/test_images/*.jpg'))

test_metadata = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/test_metadata/*.json'))

test_sentiment = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/test_sentiment/*.json'))
print('train_images: ', len(train_images))

print('train_metadata: ', len(train_metadata))

print('train_sentiment: ', len(train_sentiment))



print('test_images: ', len(test_images))

print('test_metadata: ', len(test_metadata))

print('test_sentiment: ', len(test_sentiment))
print(train.shape)

print(submission.shape)

print(test.shape)
train.head()
print('Data Submission')

print(submission.head())

print('Data Test')

print(test.head())
# Images



train_df_ids = train['PetID']

print('shape of train_df_ids: ', train_df_ids.shape)



train_df_imgs = pd.DataFrame(train_images, columns=['image_filename'])

train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_imgs = train_df_imgs.assign(PetID = train_imgs_pets)

print('len of unique train_imgs_pet', len(train_imgs_pets.unique()))

print(train_df_imgs.head())



pets_with_images = len(np.intersect1d(train_imgs_pets.unique(), train_df_imgs['PetID'].unique()))

print('fraction of pets with images: {:.3f}'.format(pets_with_images/train_df_ids.shape[0]))
# Metadata



train_df_ids = train['PetID']

print('shape of train_df_ids: ', train_df_ids.shape)



train_df_metadata = pd.DataFrame(train_metadata, columns=['metadata_filename'])

train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_metadata = train_df_metadata.assign(PetID = train_metadata_pets)

print('len of unique train_metadata_pets', len(train_metadata_pets.unique()))

print(train_df_metadata.head())



pets_with_metadata = len(np.intersect1d(train_metadata_pets.unique(), train_df_metadata['PetID'].unique()))

print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadata/train_df_ids.shape[0]))
# Sentiment



train_df_ids = train['PetID']

print('shape of train_df_ids: ', train_df_ids.shape)



train_df_sentiment = pd.DataFrame(train_sentiment, columns=['sentiment_filename'])

train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_sentiment = train_df_sentiment.assign(PetID = train_sentiment_pets)

print('len of unique train_sentiment_pets', len(train_sentiment_pets.unique()))

print(train_df_sentiment.head())



pets_with_sentiment = len(np.intersect1d(train_sentiment_pets.unique(), train_df_sentiment['PetID'].unique()))

print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiment/train_df_ids.shape[0]))
# Check some descriptions that the Google API could not analyze

a = train_df_sentiment['PetID'].apply(lambda x: x.split('.json')[0])

b = {}

for i in train.PetID:

    if i not in list(a):

        b.update(train[train.PetID == i]['Description'])
b
c = {}

for i in list(a):

    c.update(train[train.PetID == i]['Description'])
c
# Images



test_df_ids = test['PetID']

print('shape of test_df_ids: ', test_df_ids.shape)



test_df_imgs = pd.DataFrame(test_images, columns=['image_filename'])

test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

test_df_imgs = test_df_imgs.assign(PetID = test_imgs_pets)

print('len of unique test_imgs_pet', len(test_imgs_pets.unique()))

print(test_df_imgs.head())



pets_with_images = len(np.intersect1d(test_imgs_pets.unique(), test_df_imgs['PetID'].unique()))

print('fraction of pets with images: {:.3f}'.format(pets_with_images/test_df_ids.shape[0]))
# Metadata



test_df_ids = test['PetID']

print('shape of test_df_ids: ', test_df_ids.shape)



test_df_metadata = pd.DataFrame(test_metadata, columns=['metadata_filename'])

test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

test_df_metadata = test_df_metadata.assign(PetID = test_metadata_pets)

print('len of unique train_metadata_pets', len(test_metadata_pets.unique()))

print(test_df_metadata.head())



pets_with_metadata = len(np.intersect1d(test_metadata_pets.unique(), test_df_metadata['PetID'].unique()))

print('fraction of pets with metadata: {:.3f}'.format(pets_with_metadata/test_df_ids.shape[0]))
# Sentiment



test_df_ids = test['PetID']

print('shape of test_df_ids: ', test_df_ids.shape)



test_df_sentiment = pd.DataFrame(test_sentiment, columns=['sentiment_filename'])

test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

test_df_sentiment = test_df_sentiment.assign(PetID = test_sentiment_pets)

print('len of unique train_sentiment_pets', len(test_sentiment_pets.unique()))

print(test_df_sentiment.head())



pets_with_sentiment = len(np.intersect1d(test_sentiment_pets.unique(), test_df_sentiment['PetID'].unique()))

print('fraction of pets with sentiment: {:.3f}'.format(pets_with_sentiment/test_df_ids.shape[0]))
class petfinder(object):

    def __init__(self, debug=False):

        self.debug = debug

        self.sentence_sep = ' '

        self.extract_sentiment_text = False

        

    def open_metadata_file(self, filename):

        

    # Load metadata file

        

        with open(filename, 'r') as f:

            metadata_file = json.load(f)

        return metadata_file

    

    def open_sentiment_file(self, filename):

        

    # Load sentiment file

        

        with open(filename, 'r') as f:

            sentiment_file = json.load(f)

        return sentiment_file

    

    def open_image_file(self, filename):

        

    # Load image file

        

        image = np.asarray(Image.open(filename))

        return image

    

    def parse_sentiment_file(self, file):

        

    # Parse sentiment file. Output is dataframe with sentiment feature

        

        file_sentiment = file['documentSentiment']

        file_entities = [x['name'] for x in file['entities']]

        file_entities = self.sentence_sep.join(file_entities)



        if self.extract_sentiment_text:

            file_sentences_text = [x['text']['content'] for x in file['sentences']]

            file_sentences_text = ' '.join(file_sentence_text)

    

        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(file_sentences_sentiment, orient='columns').sum()

        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()



        file_sentiment.update(file_sentences_sentiment)

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T



        if self.extract_sentiment_text:

            df_sentiment['test'] = file_sentences_text

    

        df_sentiment['entities'] = file_entities

        df_sentiment = df_sentiment.add_prefix('sentiment_')

        

        return df_sentiment

    

    def parse_metadata_file(self, file):

        

    # Parse metadata file. Output is Dataframe with metadata features

    

        file_keys = list(file.keys())



        if 'labelAnnotations' in file_keys:

            file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.3)]

            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()

            file_top_description = [x['description'] for x in file_annots]

            file_top_description = self.sentence_sep.join(file_top_description)

        else:

            file_top_score = np.nan

            file_top_description = ['']

    

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']

        file_crops = file['cropHintsAnnotation']['cropHints']



        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()

        file_color_pixelfraction = np.asarray([x['pixelFraction'] for x in file_colors]).mean()



        file_crop_confidence = np.asarray([x['confidence'] for x in file_crops]).mean()



        if 'importanceFraction' in file_crops[0].keys():

            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()

        else:

            file_crop_importance = np.nan

    

        df_metadata = pd.DataFrame.from_dict({'annots_score': file_top_score,

                                             'color_score': file_color_score,

                                             'color_pixelfraction': file_color_pixelfraction,

                                             'crop_confidence': file_crop_confidence,

                                             'crop_importance': file_crop_importance,

                                             'annots_top_description': file_top_description}, orient='index')

        df_metadata = df_metadata.T

        df_metadata = df_metadata.add_prefix('metadata_')

    

        return df_metadata
file = petfinder().open_metadata_file(train_sentiment[1])

file1 = petfinder().open_metadata_file(train_metadata[1])

print(petfinder().parse_sentiment_file(file))

print(petfinder().parse_metadata_file(file1))
print(file)

print('   ')

print(train_sentiment[1])

print('   ')

print(train[train.PetID == '000a290e4'].Description)
print(file1)

print('   ')

print(train_metadata[1])

print('  ')

print(train[train.PetID == '0008c5398'].Description)
# Create function for parallel data processing



def extract_additional_features(pet_id, mode='train'):

    sentiment_filename = '/kaggle/input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)

    try:

        sentiment_file = petfinder.open_sentiment_file(sentiment_filename)

        df_sentiment = petfinder.parse_sentiment_file(sentiment_file)

        df_sentiment['PetID'] = pet_id

    except FileNotFoundError:

        df_sentiment = []

    

    dfs_metadata = []

    metadata_filenames = sorted(glob.glob('/kaggle/input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(mode, pet_id)))

    if len(metadata_filenames) > 0:

        for f in metadata_filenames:

            metadata_file = petfinder.open_metadata_file(f)

            df_metadata = petfinder.parse_metadata_file(metadata_file)

            df_metadata['PetID'] = pet_id

            dfs_metadata.append(df_metadata)

        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)

    

    dfs = [df_sentiment, dfs_metadata]

    

    return dfs
petfinder = petfinder()

debug = False

train_pet_ids = train.PetID.unique()

test_pet_ids = test.PetID.unique()



if debug:

    train_pet_ids = train_pet_ids[:1000]

    test_pet_ids = test_pet_ids[:1000]

    

dfs_train = Parallel(n_jobs=6, verbose=1)(delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

dfs_test = Parallel(n_jobs=6, verbose=1)(delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)
train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]

train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]



test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]

test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]
test_dfs_sentiment
train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)

train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)



test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)

test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)
print('shape of train_dfs sentiment & metadata:')

print(train_dfs_sentiment.shape, train_dfs_metadata.shape)

print('shape of test_dfs sentiment & metadata:')

print(test_dfs_sentiment.shape, test_dfs_metadata.shape)
# extend aggregates and improve column naming

aggregates = ['mean', 'sum']
# Train



prefix = 'metadata'

train_metadata_group = train_dfs_metadata.drop(['metadata_annots_top_description'], axis=1)



for i in train_metadata_group.columns:

    if 'PetID' not in i:

        train_metadata_group[i] = train_metadata_group[i].astype(float)

train_metadata_group = train_metadata_group.groupby(['PetID']).agg(aggregates)

train_metadata_group.columns = pd.Index(['{}_{}_{}'.format(prefix, c[0], c[1].upper()) for c in train_metadata_group.columns.tolist()])

train_metadata_group = train_metadata_group.reset_index()
train_sentiment_description = train_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()

train_sentiment_description = train_sentiment_description.reset_index()

train_sentiment_description['sentiment_entities'] = train_sentiment_description['sentiment_entities'].apply(lambda x: ' '.join(x))



prefix = 'sentiment'

train_sentiment_group = train_dfs_sentiment.drop(['sentiment_entities'], axis=1)



for i in train_sentiment_group.columns:

    if 'PetID' not in i:

        train_sentiment_group[i] = train_sentiment_group[i].astype(float)

train_sentiment_group = train_sentiment_group.groupby(['PetID']).agg(aggregates)

train_sentiment_group.columns = pd.Index(['{}_{}_{}'.format(prefix, c[0], c[1].upper()) for c in train_sentiment_group.columns.tolist()])

train_sentiment_group = train_sentiment_group.reset_index()
# Test



prefix = 'metadata'

test_metadata_group = test_dfs_metadata.drop(['metadata_annots_top_description'], axis=1)



for i in test_metadata_group.columns:

    if 'PetID' not in i:

        test_metadata_group[i] = test_metadata_group[i].astype(float)

test_metadata_group = test_metadata_group.groupby(['PetID']).agg(aggregates)

test_metadata_group.columns = pd.Index(['{}_{}_{}'.format(prefix, c[0], c[1].upper()) for c in test_metadata_group.columns.tolist()])

test_metadata_group = test_metadata_group.reset_index()
test_sentiment_description = test_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()

test_sentiment_description = test_sentiment_description.reset_index()

test_sentiment_description['sentiment_entities'] = test_sentiment_description['sentiment_entities'].apply(lambda x: ' '.join(x))



prefix = 'sentiment'

test_sentiment_group = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)



for i in test_sentiment_group.columns:

    if 'PetID' not in i:

        test_sentiment_group[i] = test_sentiment_group[i].astype(float)

test_sentiment_group = test_sentiment_group.groupby(['PetID']).agg(aggregates)

test_sentiment_group.columns = pd.Index(['{}_{}_{}'.format(prefix, c[0], c[1].upper()) for c in test_sentiment_group.columns.tolist()])

test_sentiment_group = test_sentiment_group.reset_index()
# Train merge

train_process = train.copy()

train_process = train_process.merge(train_sentiment_group, how='left', on='PetID')

train_process = train_process.merge(train_metadata_group, how='left', on='PetID')



# Test merge

test_process = test.copy()

test_process = test_process.merge(test_sentiment_group, how='left', on='PetID')

test_process = test_process.merge(test_metadata_group, how='left', on='PetID')
print(train_process.shape, test_process.shape)
# Breed Mapping



# Train



train_breed_main = train_process[['Breed1']].merge(breeds, how='left', left_on='Breed1', right_on='BreedID', suffixes=('', '_main_breed'))

train_breed_main = train_breed_main.iloc[:, 2:]

train_breed_main = train_breed_main.add_prefix('main_breed_')



train_breed_second = train_process[['Breed2']].merge(breeds, how='left', left_on='Breed2', right_on='BreedID', suffixes=('', '_second_breed'))

train_breed_second = train_breed_second.iloc[:, 2:]

train_breed_second = train_breed_second.add_prefix('second_breed_')



train_process = pd.concat([train_process, train_breed_main, train_breed_second], axis = 1)



# Test



test_breed_main = test_process[['Breed1']].merge(breeds, how='left', left_on='Breed1', right_on='BreedID', suffixes=('', '_main_breed'))

test_breed_main = test_breed_main.iloc[:, 2:]

test_breed_main = test_breed_main.add_prefix('main_breed_')



test_breed_second = test_process[['Breed2']].merge(breeds, how='left', left_on='Breed2', right_on='BreedID', suffixes=('', '_second_breed'))

test_breed_second = test_breed_second.iloc[:, 2:]

test_breed_second = test_breed_second.add_prefix('second_breed_')



test_process = pd.concat([test_process, test_breed_main, test_breed_second], axis = 1)
print(train_process.shape, test_process.shape)
# Concat Train & Test set



X = pd.concat([train_process, test_process], ignore_index=True, sort=False)

print(X.shape)

print('NaN structure:\n{}'.format(X.isnull().sum()))
column_types = X.dtypes



int_cols = column_types[column_types == 'int']

float_cols = column_types[column_types == 'float']

cate_cols = column_types[column_types == 'object']



print('\tinteger columns:\n{}'.format(int_cols))

print('\tfloat columns:\n{}'.format(float_cols))

print('\tcategorical columns:\n{}'.format(cate_cols))
X_new = X.copy()



text_columns = ['Description']

categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']



to_drop_columns = ['PetID', 'Name', 'RescuerID']
# Count RescuerID

rescuer_count = X.groupby(['RescuerID'])['PetID'].count().reset_index()

rescuer_count.columns = ['RescuerID', 'RescuerID_count']



# Merge as another feature

X_new = X_new.merge(rescuer_count, how='left', on='RescuerID')
# factorize categorical columns

for i in categorical_columns:

    X_new.loc[:, i] = pd.factorize(X_new.loc[:, i])[0]
# subset text features



X_text = X_new[text_columns]



for i in X_text.columns:

    X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')
n_components = 5

text_features = []



# Generate text features



for i in X_text.columns:

    # initialize decomposition methods

    

    svd = TruncatedSVD(n_components=n_components, random_state=42)

    nmf = NMF(n_components=n_components, random_state=42)

    

    tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)

    svd_col = svd.fit_transform(tfidf_col)

    svd_col = pd.DataFrame(svd_col)

    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))

    

    nmf_col = nmf.fit_transform(tfidf_col)

    nmf_col = pd.DataFrame(nmf_col)

    nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))

    

    text_features.append(svd_col)

    text_features.append(nmf_col)
# combine all extracted features

text_features = pd.concat(text_features, axis=1)



# Concatenate with main DF

X_new = pd.concat([X_new, text_features], axis=1)



# remove raw text columns

X_new = X_new.drop(text_columns, axis=1)
X_new = X_new.drop(to_drop_columns, axis=1)

X_new.shape
# split into train & test again

X_train = X_new.loc[np.isfinite(X_new.AdoptionSpeed), :]

X_test = X_new.loc[~np.isfinite(X_new.AdoptionSpeed), :]



# remove target value from test

X_test = X_test.drop(['AdoptionSpeed'], axis=1)
print('X_train shape: {}'.format(X_train.shape))

print('X_test shape: {}'.format(X_test.shape))



assert X_train.shape[0] == train.shape[0]

assert X_test.shape[0] == test.shape[0]
# check columns between the 2 DF train & test are the same

train_cols = X_train.columns.tolist()

train_cols.remove('AdoptionSpeed')



test_cols = X_test.columns.tolist()



assert np.all(train_cols == test_cols)
X_train.select_dtypes(include=(np.number))
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



parama = {'application': 'regression',

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

         'lambda_12': 0.0465,

         'verboaity': -1,

         'data_random_seed': 17}



early_stop = 500

verbose_eval = 100

num_rounds = 10000

n_splits = 5
from sklearn.model_selection import StratifiedKFold



kfold = StratifiedKFold(n_splits=n_splits, random_state=42)



oof_train = np.zeros((X_train.shape[0]))

oof_test = np.zeros((X_test.shape[0], n_splits))



i = 0

for train_index, val_index in kfold.split(X_train, X_train['AdoptionSpeed'].values):

    

    X_tr = X_train.iloc[train_index, :]

    X_val = X_train.iloc[val_index, :]

    

    y_tr = X_tr['AdoptionSpeed'].values

    y_val = X_val['AdoptionSpeed'].values

    

    X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

    X_val = X_val.drop(['AdoptionSpeed'], axis=1)

    

    print('\ny_train distribution: {}'.format(Counter(y_tr)))

    

    d_train = lgb.Dataset(X_tr, label=y_tr)

    d_val = lgb.Dataset(X_val, label=y_val)

    watchlist = [d_train, d_val]

    

    print('training LGB:')

    model = lgb.train(parama,

                      train_set=d_train,

                      num_boost_round=num_rounds,

                      valid_sets=watchlist,

                      verbose_eval=verbose_eval,

                      early_stopping_rounds=early_stop)

    

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    

    oof_train[val_index] = val_pred

    oof_test[:, i] = test_pred

    

    i += 1
plt.hist(oof_train)
optR = OptimizedRounder()

optR.fit(oof_train, X_train['AdoptionSpeed'].values)

coefficients = optR.coefficients()

pred_test_y_k = optR.predict(oof_train, coefficients)

print('\nValid Counts = ', Counter(X_train['AdoptionSpeed'].values))

print('Predicted Counts = ', Counter(pred_test_y_k))

print('Coefficients = ', coefficients)

qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, pred_test_y_k)

print('QWK = ', qwk)
coefficients_ = coefficients.copy()



coefficients_[0] = 1.645

coefficients_[1] = 2.115

coefficients_[3] = 2.84



train_predictions = optR.predict(oof_train, coefficients_).astype(int)

print('train pred distribution: {}'.format(Counter(train_predictions)))



test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_)

print('test pred distribution: {}'.format(Counter(test_predictions)))
print('True Distribution:')

print(pd.value_counts(X_train['AdoptionSpeed'], normalize=True).sort_index())

print('\nTrain Predicted Distribution:')

print(pd.value_counts(train_predictions, normalize=True).sort_index())

print('\nTest Predicted Distribution:')

print(pd.value_counts(test_predictions, normalize=True).sort_index())
submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions.astype(np.int32)})

submission.head()

submission.to_csv('submission.csv', index=False)
print(submission)