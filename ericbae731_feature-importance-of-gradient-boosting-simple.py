# The score of V7 was 0.327.

# V7 code uses Gradient Boosting (sklearn) and verify feature importance as input data.



# This version will add more features (images, image metadata, and sentiment data).

# Reference: https://www.kaggle.com/wrosinski/baselinemodeling

# Our goal: Integration of more features will enable us to possibly improve the score.
import gc

import glob

import os

import json

import matplotlib.pyplot as plt

import pprint



import numpy as np

import pandas as pd



from joblib import Parallel, delayed

from tqdm import tqdm

from PIL import Image



%matplotlib inline



pd.options.display.max_rows = 128

pd.options.display.max_columns = 128

plt.rcParams['figure.figsize'] = (12, 9)
# load core DFs (train and test):

train = pd.read_csv('../input/train/train.csv')

print('train shape:', train.shape)

test = pd.read_csv('../input/test/test.csv')

print('test shape:', test.shape)

sample_submission = pd.read_csv('../input/test/sample_submission.csv')



# load mapping dictionaries:

labels_breed = pd.read_csv('../input/breed_labels.csv')

labels_state = pd.read_csv('../input/color_labels.csv')

labels_color = pd.read_csv('../input/state_labels.csv')



# add additional features (The type is LIST and is has directory of files):

train_image_files = sorted(glob.glob('../input/train_images/*.jpg'))

train_metadata_files = sorted(glob.glob('../input/train_metadata/*.json'))

train_sentiment_files = sorted(glob.glob('../input/train_sentiment/*.json'))

print('num of train images files: {}'.format(len(train_image_files)))

print('num of train metadata files: {}'.format(len(train_metadata_files)))

print('num of train sentiment files: {}'.format(len(train_sentiment_files)))

test_image_files = sorted(glob.glob('../input/test_images/*.jpg'))

test_metadata_files = sorted(glob.glob('../input/test_metadata/*.json'))

test_sentiment_files = sorted(glob.glob('../input/test_sentiment/*.json'))

print('num of test images files: {}'.format(len(test_image_files)))

print('num of test metadata files: {}'.format(len(test_metadata_files)))

print('num of test sentiment files: {}'.format(len(test_sentiment_files)))
# plt.rcParams['figure.figsize'] = (12, 9)

plt.style.use('ggplot')



# Images:

train_df_ids = train[['PetID']]

print('length of train data is', train_df_ids.shape[0])

print()

train_df_imgs = pd.DataFrame(train_image_files)

train_df_imgs.columns = ['image_filename']

train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

print('# of train data that has images is', len(train_imgs_pets.unique()))

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
# for test

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

print()

# are distributions the same?

print('images and metadata distributions the same? {}'.format(

    np.all(test_metadata_pets == test_imgs_pets)))
# data parsing & feature extraction:

# After taking a look at the data, we know its structure and can use it to extract additional features and concatenate them with basic train/test DFs.

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

    

    sentiment_filename = '../input/{}_sentiment/{}.json'.format(mode, pet_id)

    try:

        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)

        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)

        df_sentiment['PetID'] = pet_id

    except FileNotFoundError:

        df_sentiment = []



    dfs_metadata = []

    metadata_filenames = sorted(glob.glob('../input/{}_metadata/{}*.json'.format(mode, pet_id)))

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
train_dfs_metadata.head(1)
train_dfs_sentiment.head(1)
# group extracted features by PetID:

# Extend aggregates and improve column naming

aggregates = ['mean', 'sum']



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
# merge processed DFs with base train/test DF:

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
train_proc.head(1)
# add breed mapping:

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
train_proc.head(1)
# concatenate train & test:

# Inspect NaN structure of the processed data: AdoptionSpeed is the target column.

X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)

# print('NaN structure:\n{}'.format(np.sum(pd.isnull(X))))



# extract different column types:

# integer columns are usually categorical features, which do not need encoding

# float columns are numerical features

# object columns are categorical features, which should be encoded

column_types = X.dtypes

int_cols = column_types[column_types == 'int']

float_cols = column_types[column_types == 'float']

cat_cols = column_types[column_types == 'object']

print('\tinteger columns:\n{}'.format(int_cols))

print('\n\tfloat columns:\n{}'.format(float_cols))

print('\n\tto encode categorical columns:\n{}'.format(cat_cols))
print(X.shape)

X.main_breed_Type[0:10] == X.Type[0:10]

# same feature?!
# feature engineering:



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
print(X_temp.shape)

# print(X_temp.keys())

print(X_temp.main_breed_Type[0:10] == X_temp.Type[0:10])

X_temp.head(1)

# X.Breed1[0:1]

# X.main_breed_BreedName[0:1]

# X.Breed2[0:1]

# X.second_breed_BreedName[0:1]

# still same feature?!
X_temp_column_types = X_temp.dtypes



X_temp_int_cols = X_temp_column_types[X_temp_column_types == 'int']

X_temp_float_cols = X_temp_column_types[X_temp_column_types == 'float']

X_temp_cat_cols = X_temp_column_types[X_temp_column_types == 'object']



print('\tinteger columns:\n{}'.format(X_temp_int_cols))

print('\n\tfloat columns:\n{}'.format(X_temp_float_cols))

print('\n\tto encode categorical columns:\n{}'.format(X_temp_cat_cols))
X_temp.tail(5)
# train/test split:



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
# at this time, simply drop columns with missing values for GBM

train_data = X_train.dropna(axis=1)

test_data = X_test.dropna(axis=1)

test_data = test_data.drop(['main_breed_Type'], axis=1)

print(train_data.shape)

print(test_data.shape)

# train.keys() == test.keys()
# applying GBM

# Gradient Boosting Classifier https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

from sklearn.datasets import make_hastie_10_2 

from sklearn.ensemble import GradientBoostingClassifier



# Plot

import matplotlib.pyplot as plt
# splitting step - training set and validation set 

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_data, test_size = .2)

print (train_df.shape)

print (val_df.shape)

train_df.head(5)
# setting up targets (labels)

tr_y = train_df["AdoptionSpeed"]

tr_x = train_df.drop(['AdoptionSpeed'], axis=1)

val_y = val_df["AdoptionSpeed"]

val_x = val_df.drop(['AdoptionSpeed'], axis=1)

print(tr_x.shape)

print(tr_y.shape)
# # np.sum(pd.isnull(tr_x))

# np.sum(pd.isnull(test))
# # test_temp = test

# # data_without_missing_values = test_temp.dropna(axis=1)

# np.sum(pd.isnull(test))
# parameters setting for model



# GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, 

#                            subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, 

#                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 

#                            min_impurity_decrease=0.0, min_impurity_split=None, init=None, 

#                            random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 

#                            warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)



clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0)

clf.fit(tr_x, tr_y)
# prediction for validation data

val_prediction = clf.predict(val_x)
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# Plot feature importance https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

# plt.subplot(1, 2, 2)

plt.figure(figsize=(8, 18))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, tr_x.keys()[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# tr_x.keys()

# tr_x.main_breed_BreedName.head(5)
# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# print(os.listdir("../input"))



# # Any results you write to the current directory are saved as output.

# import warnings

# warnings.filterwarnings("ignore")



# # Gradient Boosting Classifier https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

# from sklearn.datasets import make_hastie_10_2 

# from sklearn.ensemble import GradientBoostingClassifier



# # Plot

# import matplotlib.pyplot as plt

# # read data

# breeds = pd.read_csv('../input/breed_labels.csv')

# colors = pd.read_csv('../input/color_labels.csv')

# states = pd.read_csv('../input/state_labels.csv')



# train = pd.read_csv('../input/train/train.csv')

# test = pd.read_csv('../input/test/test.csv')

# sub = pd.read_csv('../input/test/sample_submission.csv')



# train['dataset_type'] = 'train'

# test['dataset_type'] = 'test'

# all_data = pd.concat([train, test])
# # given shape

# print(train.shape)

# train.head(1)
# # drop some features

# list_to_drop = ["Name", "RescuerID", "Description", "PetID", 'dataset_type']

# train.drop(list_to_drop, axis = 1, inplace = True)

# print(train.shape)

# train.head(1)
# # splitting step - training set and validation set 

# from sklearn.model_selection import train_test_split

# train_df, val_df = train_test_split(train, test_size = .2)

# print (train_df.shape)

# print (val_df.shape)
# train_df.head(1)
# # setting up targets (labels)

# tr_y = train_df["AdoptionSpeed"]

# tr_x = train_df.iloc[:,0:19]

# val_y = val_df["AdoptionSpeed"]

# val_x = val_df.iloc[:,0:19]
# # parameters setting for model



# # GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, 

# #                            subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, 

# #                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 

# #                            min_impurity_decrease=0.0, min_impurity_split=None, init=None, 

# #                            random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 

# #                            warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)



# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0)

# clf.fit(tr_x, tr_y)
# # prediction for validation data

# val_prediction = clf.predict(val_x)
# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# # Plot feature importance https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

# feature_importance = clf.feature_importances_

# # make importances relative to max importance

# feature_importance = 100.0 * (feature_importance / feature_importance.max())

# sorted_idx = np.argsort(feature_importance)

# pos = np.arange(sorted_idx.shape[0]) + .5

# # plt.subplot(1, 2, 2)

# plt.figure(figsize=(12, 6))

# plt.barh(pos, feature_importance[sorted_idx], align='center')

# plt.yticks(pos, tr_x.keys()[sorted_idx])

# plt.xlabel('Relative Importance')

# plt.title('Variable Importance')

# plt.show()
# for submission, we use all data (train + validation)

all_x = pd.concat([tr_x, val_x])

all_y = pd.concat([tr_y, val_y])

print (all_x.shape)

print (all_y.shape)
# train again for submission purpose based on all data 

clf_submit = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0)

clf_submit.fit(all_x, all_y)
# # make test data as input features

# test.drop(list_to_drop, axis = 1, inplace = True)

# # see input features

# print(test.shape)

# test.head(1)
# # predction for all data

# prediction = clf_submit.predict(test)

# # Create submission data

# submission = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})

# print(submission.head())

# # Create submission file

# submission.to_csv('submission.csv', index=False)
# test.keys()

# X_test.keys()
# predction for all data

prediction = clf_submit.predict(test_data)

# Create submission data

submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': [int(i) for i in prediction]})

print(submission.head())

# Create submission file

submission.to_csv('submission.csv', index=False)
## in case for xgboost

# import xgboost as xgb

# # read in data

# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')

# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')

# # specify parameters via map

# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }

# num_round = 2

# bst = xgb.train(param, dtrain, num_round)

# # make prediction

# preds = bst.predict(dtest)

# import xgboost as xgb

# dtrain = xgb.DMatrix(tr_x, tr_y)

# params = {

#     'booster': "gbtree",

#     'objective': 'multi:softmax',

#     'eval_metric': 'merror',

#     'eta' : 0.01,

#     'lambda': 2.0,

#     'alpha': 1.0,

#     'lambda_bias': 6.0,

#     'num_class': 5,

# #     'n_jobs' : 4,

#     'silent': 1,

# #     'n_estimators':100

#     'max_depth': 12

# }



# %time booster = xgb.train(params, dtrain, num_boost_round=100)
# dtest = xgb.DMatrix(val_x)

# result = booster.predict(dtest)

# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, result, weights = "quadratic")
# n_estimators=1000, learning_rate=0.01, max_depth=7

# 0.32785083936122394

# n_estimators=500, learning_rate=0.01, max_depth=5

# 0.32597593350337895

# 0.30082779006306737

# n_estimators=500, learning_rate=0.01, max_depth=10

# 0.3031026793030307

# n_estimators=100, learning_rate=0.05, max_depth=5

# 0.33081184760966265

# 0.3408957404611054

# n_estimators=500, learning_rate=0.01, max_depth=7

# 0.33095254066990065

# 0.3202743639855826



# 0.32387427192480744 (random state x)

# 0.32099448579490575 (random state 1)

# 0.32543273785368976 (random state 1 with 1000)
# # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=7, random_state=0)

# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# # clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5, max_depth=5, random_state=0)

# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# # clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.5, max_depth=5, random_state=0)

# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# # clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.5, max_depth=12, random_state=0)

# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=5, random_state=0)

# from sklearn.metrics import cohen_kappa_score

# cohen_kappa_score(val_y, val_prediction, weights = "quadratic")
# # Plot feature importance https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

# feature_importance = clf.feature_importances_

# # make importances relative to max importance

# feature_importance = 100.0 * (feature_importance / feature_importance.max())

# sorted_idx = np.argsort(feature_importance)

# pos = np.arange(sorted_idx.shape[0]) + .5

# # plt.subplot(1, 2, 2)

# plt.figure(figsize=(12, 6))

# plt.barh(pos, feature_importance[sorted_idx], align='center')

# plt.yticks(pos, tr_x.keys()[sorted_idx])

# plt.xlabel('Relative Importance')

# plt.title('Variable Importance')

# plt.show()
# # for submission, we use all data (train + validation)

# all_x = pd.concat([tr_x, val_x])

# all_y = pd.concat([tr_x, val_y])

# print (all_x)

# print (all_y)
# # train again for submission purpose based on all data 

# clf_submit = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, max_depth=7, random_state=1)

# clf_submit.fit(tr_x, tr_y)
# # make test data as input features

# test.drop(list_to_drop, axis = 1, inplace = True)
# # see input features

# test.head(1)
# # prediction

# # prediction = clf.predict(test)

# predction for all data

# prediction = clf_submit.predict(test)
# # Create submission data

# submission = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})

# submission.head()
# # Create submission file

# submission.to_csv('submission.csv', index=False)
# print(train.iloc[:,[0,2,3,4,5,6,7,8,9,11,12,13,14,19,22]].shape)

# train.iloc[:,[0,2,3,4,5,6,7,8,9,11,12,13,14,19,22]][0:10]

# # tr_y = (train.iloc[:,[23]]).values
# print(train.shape)

# pr = int(train.shape[0]*0.8)

# print(pr, '+', train.shape[0] - pr, '=', train.shape[0])

# tr = train[0:pr]

# val = train[pr:]
# # tr_y = (train.iloc[:,[23]]).values

# train.Breed1.values

# train.Breed2.values

# # just go for breed1

# TR_BR = train.iloc[:,[0,3,23]].values

# TR_BR



# # TR_BR[0][1]

# # for i in range(len(TR_BR)):

# #     for j in range(len(BRS_cn)):

# #         if (TR_BR[i][0] == BRS_cn[j][1]) and (TR_BR[i][1] == BRS_cn[j][0]):

# #             BRS_cn[j][TR_BR[i][2]+2] += 1
# target encoding....
# breeds[210:270]

# breeds[0:10]

# BRS = breeds.iloc[:,[0,1]].values

# # BRS.shape

# # np.zeros([BRS.shape[0]]).shape

# BRS_cn = np.concatenate((BRS,np.zeros([BRS.shape[0],5])), axis=1)

# print(BRS_cn.shape)

# BRS_cn
# for i in range(len(TR_BR)):

#     for j in range(len(BRS_cn)):

#         if (TR_BR[i][0] == BRS_cn[j][1]) and (TR_BR[i][1] == BRS_cn[j][0]):

#             BRS_cn[j][TR_BR[i][2]+2] += 1
# MT[0][0]*0 + MT[0][1]*1 + MT[0][2]*2 + MT[0][3]*3 + MT[0][4]*4

# sum
# Mean = np.zeros([BRS_cn[:,2:].shape[0]])

# MT = BRS_cn[:,2:]

# MT

# for k in range(len(MT)):

#     if (MT[k][0] + MT[k][1] + MT[k][2] + MT[k][3] + MT[k][4]) == 0:

#         Mean[k] = 2

#     else:

#         Mean[k] = (MT[k][0]*0 + MT[k][1]*1 + MT[k][2]*2 + MT[k][3]*3 + MT[k][4]*4) / (MT[k][0] + MT[k][1] + MT[k][2] + MT[k][3] + MT[k][4])

    



# # for i in range(len(BRS_cn)):

# #     for j in range

# #     if BRS_cn[i][0] == 
# Mean.reshape(Mean.shape[0],1)

# # BRS_1 = np.concatenate((BRS,Mean), axis=1)

# BRS_1 = np.concatenate((BRS,Mean.reshape(Mean.shape[0],1)), axis=1)

# BRS_1
# # TR_BR.shape[0]

# BR_1 = np.zeros([TR_BR.shape[0],1])

# for i in range(len(TR_BR)):

#     for j in range(len(BRS_1)):

#         if (TR_BR[i][0] == BRS_1[j][1]) and (TR_BR[i][1] == BRS_1[j][0]):

#             BR_1[i] = BRS_1[j][2]

# #             BRS_cn[j][TR_BR[i][2]+2] += 1
# print(BR_1.shape)

# BR_1[0:100]
# import matplotlib.pyplot as plt

# # rng = np.random.RandomState(10)  # deterministic random data

# # a = np.hstack((BR_1.normal(size=1000),BR_1.normal(loc=5, scale=2, size=1000)))

# plt.figure(figsize=(15, 6))

# plt.hist(BR_1, bins='auto')  # arguments are passed to np.histogram

# plt.title("Histogram with 'auto' bins")

# plt.show()
# import matplotlib.pyplot as plt

# # rng = np.random.RandomState(10)  # deterministic random data

# # a = np.hstack((BR_1.normal(size=1000),BR_1.normal(loc=5, scale=2, size=1000)))

# plt.figure(figsize=(15, 6))

# plt.hist(train.iloc[:,[3]].values, bins='auto')  # arguments are passed to np.histogram

# plt.title("Histogram with 'auto' bins")

# plt.show()
# new_train = np.concatenate((train.iloc[:,[2,4,5,6,7,8,9,11,12,13,14,19,22]].values,BR_1), axis=1)

# # print(new_train.shape)

# # print(new_train[0])

# # train.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,14,19,22]].head(1)

# ## train.iloc[:,[0,2,3,4,5,6,7,8,9,11,12,13,14,19,22]].shape
# print(train.shape)

# pr = int(train.shape[0]*0.8)

# print(pr, '+', train.shape[0] - pr, '=', train.shape[0])

# tr = train[0:pr]

# val = train[pr:]



# new_tr = train[0:pr]

# new_val = train[pr:]
# # train input data (x) and train labels (y), at this time no seperation for validating, no normalization

# # tr_x = train.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,14,19,22]].values

# tr_y = (train.iloc[:,[23]]).values



# # features as input

# # 1. Age - Age of pet when listed, in months

# # 2. Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)

# # 3. Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)

# # 4. Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)

# # 5. Color1 - Color 1 of pet (Refer to ColorLabels dictionary)

# # 6. Color2 - Color 2 of pet (Refer to ColorLabels dictionary)

# # 7. Color3 - Color 3 of pet (Refer to ColorLabels dictionary)

# # 8. MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)

# # 9. Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)

# # 10. Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)

# # 11. Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)

# # 12. Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)

# # 13. VideoAmt - Total uploaded videos for this pet

# # 14. PhotoAmt - Total uploaded photos for this pet
# # Training for Gradient Boosting Classifier

# # clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.5, max_depth=12, random_state=0).fit(tr_x, tr_y)

# clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.5, max_depth=12, random_state=0).fit(new_train, tr_y)
# # Plot feature importance https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

# feature_importance = clf.feature_importances_

# # make importances relative to max importance

# feature_importance = 100.0 * (feature_importance / feature_importance.max())

# sorted_idx = np.argsort(feature_importance)

# pos = np.arange(sorted_idx.shape[0]) + .5

# # plt.subplot(1, 2, 2)

# plt.figure(figsize=(12, 6))

# plt.barh(pos, feature_importance[sorted_idx], align='center')

# plt.yticks(pos, train.iloc[:,[2,4,5,6,7,8,9,11,12,13,14,19,22,3]].keys()[sorted_idx])

# plt.xlabel('Relative Importance')

# plt.title('Variable Importance')

# plt.show()
# train.iloc[:,[2,4,5,6,7,8,9,11,12,13,14,19,22,3]].keys()
# >>> from sklearn.metrics import cohen_kappa_score

# >>> y_true = [2, 0, 2, 2, 0, 1]

# >>> y_pred = [0, 0, 2, 2, 0, 2]

# >>> cohen_kappa_score(y_true, y_pred)

# 0.4285714285714286
# This is for Target Encoding Results [only for Breed1]
# # # TR_BR.shape[0]

# # BR_1 = np.zeros([TR_BR.shape[0],1])

# # for i in range(len(TR_BR)):

# #     for j in range(len(BRS_1)):

# #         if (TR_BR[i][0] == BRS_1[j][1]) and (TR_BR[i][1] == BRS_1[j][0]):

# #             BR_1[i] = BRS_1[j][2]

# # #             BRS_cn[j][TR_BR[i][2]+2] += 1



# # TR_BR.shape[0]

# BR_1 = np.zeros([TR_BR.shape[0],1])

# for i in range(len(TR_BR)):

#     for j in range(len(BRS_1)):

#         if (TR_BR[i][0] == BRS_1[j][1]) and (TR_BR[i][1] == BRS_1[j][0]):

#             BR_1[i] = BRS_1[j][2]

# #             BRS_cn[j][TR_BR[i][2]+2] += 1
# # TE_BR = test.iloc[:,[0,3,23]].values

# # TE_BR

# TE_BR = test.iloc[:,[0,3]].values

# TE_BR_1 = np.zeros([TE_BR.shape[0],1])

# for i in range(len(TE_BR)):

#     for j in range(len(BRS_1)):

#         if (TE_BR[i][0] == BRS_1[j][1]) and (TE_BR[i][1] == BRS_1[j][0]):

#             TE_BR_1[i] = BRS_1[j][2]
# TE_BR_1[0:1000]
# new_test = np.concatenate((test.iloc[:,[2,4,5,6,7,8,9,11,12,13,14,19,22]].values,TE_BR_1), axis=1)

# prediction = clf.predict(new_test)

# # clf.score(tr_x, tr_y) 
# # For submission

# test_x = test.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,14,19,22]].values

# prediction = clf.predict(test_x)

# clf.score(tr_x, tr_y) 
# # Create submission data

# submission = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})

# submission.head()
# # Create submission file

# submission.to_csv('submission.csv', index=False)