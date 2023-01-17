import gc

import glob

import os

import json

import matplotlib.pyplot as plt

import seaborn as sns

import pprint

import string

import warnings

import time

import imageio

import random

from scipy.stats import itemfreq



import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import Ridge

from sklearn.decomposition import TruncatedSVD

from joblib import Parallel, delayed

import operator

from tqdm import tqdm, tqdm_notebook

from collections import defaultdict

from PIL import Image

from skimage import feature as SKImageFeature

from IPython.core.display import HTML

from IPython.display import Image as Image2

from contextlib import contextmanager

from sklearn.pipeline import FeatureUnion

from scipy.sparse import hstack, csr_matrix



from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor



%matplotlib inline



np.random.seed(seed=1337)

warnings.filterwarnings('ignore')

# progress bar

tqdm.pandas(tqdm_notebook)



from sklearn.preprocessing import Imputer





split_char = '/'
os.listdir('../input')
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
@contextmanager

def timer(task_name="timer"):

    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

    print("----{} started".format(task_name))

    t0 = time.time()

    yield

    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))
debug = False

if debug:

    train = train[:500]

    test = test[:200]
train.columns
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

inp = Input((256,256,3))

backbone = DenseNet121(input_tensor = inp, 

                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",

                       include_top = False)

x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)
pet_ids = train['PetID'].values

n_batches = int(np.ceil(len(pet_ids) / batch_size))



features = {}

for b in tqdm_notebook(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

    for i,pet_id in enumerate(batch_pets):

        try:

            batch_images[i] = load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)

        except:

            pass

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]
train_feats = pd.DataFrame.from_dict(features, orient='index')

train_feats.columns = [f'pic_{i}' for i in range(train_feats.shape[1])]
pet_ids = test['PetID'].values

n_batches = int(np.ceil(len(pet_ids) / batch_size))



features = {}

for b in tqdm(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),img_size,img_size,3))

    for i,pet_id in enumerate(batch_pets):

        try:

            batch_images[i] = load_image("../input/petfinder-adoption-prediction/test_images/", pet_id)

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
labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')

labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))

train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))

train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))



if debug:

    train_image_files = train_image_files[:1000]

    train_metadata_files = train_metadata_files[:1000]

    train_sentiment_files = train_sentiment_files[:1000]



print(f'num of train images files: {len(train_image_files)}')

print(f'num of train metadata files: {len(train_metadata_files)}')

print(f'num of train sentiment files: {len(train_sentiment_files)}')





test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))

test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))

test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))



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

    for ind in range(1,200):

        metadata_filename = '../input/petfinder-adoption-prediction/{}_metadata/{}-{}.json'.format(mode, pet_id, ind)

        try:

            metadata_file = pet_parser.open_json_file(metadata_filename)

            df_metadata = pet_parser.parse_metadata_file(metadata_file)

            df_metadata['PetID'] = pet_id

            dfs_metadata.append(df_metadata)

        except FileNotFoundError:

            break

    if dfs_metadata:

        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)

    dfs = [df_sentiment, dfs_metadata]

    return dfs





pet_parser = PetFinderParser()
train_pet_ids = train.PetID.unique()

test_pet_ids = test.PetID.unique()



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





X_temp["state_gdp"] = X_temp.State.map(state_gdp)

X_temp["state_population"] = X_temp.State.map(state_population)
X_temp = X_temp.merge(img_features, how='left', on='PetID')
len(np.asarray(np.array([[1,2,3],[4,5,6]])).shape)
def color_analysis(img):

    # obtain the color palatte of the image 

    palatte = defaultdict(int)

    for pixel in img.getdata():

        palatte[pixel] += 1

    

    # sort the colors present in the image 

    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)

    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25

    for i, x in enumerate(sorted_x[:pixel_limit]):

        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 

            dark_shade += x[1]

        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 

            light_shade += x[1]

        shade_count += x[1]

        

    light_percent = round((float(light_shade)/shade_count)*100, 2)

    dark_percent = round((float(dark_shade)/shade_count)*100, 2)

    return light_percent, dark_percent



def perform_color_analysis(im, flag):

    # cut the images into two halves as complete average may give bias results

    size = im.size

    halves = (size[0]/2, size[1]/2)

    im1 = im.crop((0, 0, size[0], halves[1]))

    im2 = im.crop((0, halves[1], size[0], size[1]))



    try:

        light_percent1, dark_percent1 = color_analysis(im1)

        light_percent2, dark_percent2 = color_analysis(im2)

    except Exception as e:

        return np.NaN



    light_percent = (light_percent1 + light_percent2)/2 

    dark_percent = (dark_percent1 + dark_percent2)/2 

    if flag == 'black':

        return dark_percent

    elif flag == 'white':

        return light_percent

    else:

        return np.NaN



def get_blurrness_score(image):

    image = np.asarray(image)

    if len(image.shape) == 3:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fm = cv2.Laplacian(image, cv2.CV_64F).var()

    return fm



def average_pixel_width(im):

    im_array = np.asarray(im.convert(mode='L'))

    edges_sigma1 = SKImageFeature.canny(im_array, sigma=3)

    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))

    return apw * 100



def getSize(filename):

    st = os.stat(filename)

    return st.st_size



def getDimensions(img):

    return img.size



def get_brightness(img_path):

    try:

        img = cv2.imread(img_path, 0)

        return np.mean(img)

    except Exception as e:

        return np.NaN

    

def get_dominant_color(img):

    img = np.asarray(img)

    arr = np.float32(img)

    pixels = arr.reshape((-1, 3))



    n_colors = 5

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)

    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)



    palette = np.uint8(centroids)

    quantized = palette[labels.flatten()]

    quantized = quantized.reshape(img.shape)



    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

    return dominant_color
from PIL import Image

replace_backslash = lambda x: x.replace('\\', '/')

vectorized_replace_backslash = np.vectorize(replace_backslash)



aggs = {

    'image_blurness': ['sum', 'mean', 'var'],

    'image_size': ['sum', 'mean', 'var'],

    'image_width': ['sum', 'mean', 'var'],

    'image_height': ['sum', 'mean', 'var'],

}
def get_image_features(img_info):

    img_info["image_size"] = getSize(img_info['image_filename'])

    

    try:

        im = Image.open(img_info['image_filename'])

    except Exception as e:

        im = np.NaN

    

    if im != np.NaN:

        dimension = getDimensions(im)

        img_info["image_width"] = dimension[0]

        img_info["image_height"] = dimension[1]

        img_info["image_blurness"] = get_blurrness_score(im)

    else:

        img_info["image_width"] = np.NaN

        img_info["image_height"] = np.NaN

        img_info["image_blurness"] = np.NaN

    

    return img_info
train_df_ids = train[['PetID']]

train_image_files = vectorized_replace_backslash(train_image_files)

train_df_imgs = pd.DataFrame(train_image_files)

train_df_imgs.columns = ['image_filename']

train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])



train_image_features = train_df_imgs.assign(PetID=train_imgs_pets)



with timer("getting image extra image features for train: "):

    train_image_features = train_image_features.apply(get_image_features, axis=1)
train_image_features
agg_train_imgs = train_image_features.groupby('PetID').agg(aggs)

new_columns = [

    k + '_' + agg for k in aggs.keys() for agg in aggs[k]

]

agg_train_imgs.columns = new_columns

agg_train_imgs = agg_train_imgs.reset_index()
test_df_ids = test[['PetID']]

test_image_files = vectorized_replace_backslash(test_image_files)

test_df_imgs = pd.DataFrame(test_image_files)

test_df_imgs.columns = ['image_filename']

test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split(split_char)[-1].split('-')[0])

test_image_features = test_df_imgs.assign(PetID=test_imgs_pets)



with timer("getting image extra image features for test: "):

    test_image_features = test_image_features.apply(get_image_features, axis=1)
agg_test_imgs = test_image_features.groupby('PetID').agg(aggs)

new_columns = [

    k + '_' + agg for k in aggs.keys() for agg in aggs[k]

]

agg_test_imgs.columns = new_columns

agg_test_imgs = agg_test_imgs.reset_index()
agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)



agg_imgs_columns = list(agg_imgs.columns)

agg_imgs_columns.remove('PetID')
agg_imgs.head(5)
X_temp = X_temp.merge(agg_imgs, how='left', on='PetID')
X_text = X_temp[text_columns]



for i in X_text.columns:

    X_text.loc[:, i] = X_text.loc[:, i].fillna('none')
n_components = 32 # increase num components to 32

text_features = []



# Generate text features:

for i in X_text.columns:

    

    # Initialize decomposition methods:

    print(f'generating features from: {i}')

    tfv = TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',

                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)

    

    svd_ = TruncatedSVD(n_components=n_components, random_state=1337)

    

    tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)

    

    svd_col = svd_.fit_transform(tfidf_col)

    svd_col = pd.DataFrame(svd_col)

    svd_col = svd_col.add_prefix('TFIDF_{}_'.format(i))



    current_text = X_temp[i].str.lower().astype(str)

    X_temp[i] = X_temp[i].astype(str)

    X_temp[i + '_num_chars'] = current_text.apply(len) # Count number of Characters

    X_temp[i + '_num_words'] = current_text.apply(lambda comment: len(comment.split())) # Count number of Words

    X_temp[i + '_num_unique_words'] = current_text.apply(lambda comment: len(set(w for w in comment.split())))

    X_temp[i + '_words_vs_unique'] = X_temp[i + '_num_unique_words'] / X_temp[i+'_num_words'] * 100 # Count Unique Words

    X_temp[i + '_count_letters'] = current_text.apply(lambda x: len(str(x)))

    X_temp[i + "_count_punctuations"] = current_text.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    X_temp[i + "_mean_word_len"] = current_text.apply(lambda x: np.mean([len(w) for w in str(x).split()]))

        

    text_features.append(svd_col)

    

X_temp['words_vs_unique_description'] = X_temp['Description_num_unique_words'] / X_temp['Description_num_words'] * 100

X_temp['words_vs_unique_description'] = X_temp['Description_num_unique_words'] / X_temp['Description_num_words'] * 100

X_temp['len_description'] = X_temp['Description'].apply(lambda x: len(x)) 

X_temp['average_description_word_length'] = X_temp['len_description'] / X_temp['Description_num_words']
embeddings_index = {}

time_start_load_embedding = time.time()

with open('../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec', encoding='utf8') as f:

    for line in f:

        values = line.rstrip().rsplit(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    print("finished loading embeddings file in: ", time.time() - time_start_load_embedding)
def sum_embeddings_vector(x):

    sum_bag_of_words = np.zeros(300)

    words = x.split(' ')

    count = 0

    for w in words:

        if w in embeddings_index:

            sum_bag_of_words += embeddings_index.get(w)

            count += 1

    if count > 0:

        sum_bag_of_words /= count

    return pd.Series(sum_bag_of_words)
svd_ = TruncatedSVD(n_components=n_components, random_state=1337)

for i in X_text.columns:

    emb_feat = X_text[i].apply(sum_embeddings_vector)

    emb_feat = svd_.fit_transform(emb_feat)

    emb_feat = pd.DataFrame(emb_feat)

    emb_feat = emb_feat.add_prefix('word_embedding_' + i + '_')

    text_features.append(emb_feat)
text_features = pd.concat(text_features, axis=1)



X_temp = pd.concat([X_temp, text_features], axis=1)
for i in X_text.columns:

    X_temp = X_temp.drop(i, axis=1)
rescuer_ids = X_temp['RescuerID'].values
X_temp = X_temp.drop(to_drop_columns, axis=1)
X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]

X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]



X_test = X_test.drop(['AdoptionSpeed'], axis=1)



assert X_train.shape[0] == train.shape[0]

assert X_test.shape[0] == test.shape[0]



train_cols = X_train.columns.tolist()

train_cols.remove('AdoptionSpeed')



test_cols = X_test.columns.tolist()



assert np.all(train_cols == test_cols)

rescuer_ids = rescuer_ids[:len(X_train)]
assert len(rescuer_ids) == len(X_train)
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
# put numerical value to one of bins

def to_bins(x, borders):

    for i in range(len(borders)):

        if x <= borders[i]:

            return i

    return len(borders)



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _loss(self, coef, X, y, idx):

        X_p = np.array([to_bins(pred, coef) for pred in X])

        ll = -quadratic_weighted_kappa(y, X_p)

        return ll



    def fit(self, X, y):

        coef = [1.5, 2.0, 2.5, 3.0]

        golden1 = 0.618

        golden2 = 1 - golden1

        ab_start = [(1, 2), (1.5, 2.5), (2, 3), (2.5, 3.5)]

        for it1 in range(10):

            for idx in range(4):

                # golden section search

                a, b = ab_start[idx]

                # calc losses

                coef[idx] = a

                la = self._loss(coef, X, y, idx)

                coef[idx] = b

                lb = self._loss(coef, X, y, idx)

                for it in range(20):

                    # choose value

                    if la > lb:

                        a = b - (b - a) * golden1

                        coef[idx] = a

                        la = self._loss(coef, X, y, idx)

                    else:

                        b = b - (b - a) * golden2

                        coef[idx] = b

                        lb = self._loss(coef, X, y, idx)

        self.coef_ = {'x': coef}



    def predict(self, X, coef):

        X_p = np.array([to_bins(pred, coef) for pred in X])

        return X_p



    def coefficients(self):

        return self.coef_['x']
import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, GroupKFold
ntrain = X_train_non_null.shape[0]

ntest = X_test_non_null.shape[0]

NFOLDS = 10
def stratified_group_k_fold(X, y, groups, k, seed=None):

    labels_num = np.max(y) + 1

    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

    y_distr = Counter()

    for label, g in zip(y, groups):

        y_counts_per_group[g][label] += 1

        y_distr[label] += 1



    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

    groups_per_fold = defaultdict(set)



    def eval_y_counts_per_fold(y_counts, fold):

        y_counts_per_fold[fold] += y_counts

        std_per_label = []

        for label in range(labels_num):

            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])

            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts

        return np.mean(std_per_label)

    

    groups_and_y_counts = list(y_counts_per_group.items())

    random.Random(seed).shuffle(groups_and_y_counts)



    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):

        best_fold = None

        min_eval = None

        for i in range(k):

            fold_eval = eval_y_counts_per_fold(y_counts, i)

            if min_eval is None or fold_eval < min_eval:

                min_eval = fold_eval

                best_fold = i

        y_counts_per_fold[best_fold] += y_counts

        groups_per_fold[best_fold].add(g)



    all_groups = set(groups)

    for i in range(k):

        train_groups = all_groups - groups_per_fold[i]

        test_groups = groups_per_fold[i]



        train_indices = [i for i, g in enumerate(groups) if g in train_groups]

        test_indices = [i for i, g in enumerate(groups) if g in test_groups]



        yield train_indices, test_indices
def get_oof(clf, X, y, X_test, groups):    

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(stratified_group_k_fold(X, y, rescuer_ids, NFOLDS, 1337)):

        print('Training for fold: ', i + 1)

        

        x_tr = X.iloc[train_index, :]

        y_tr = y[train_index]

        x_te = X.iloc[test_index, :]

        y_te = y[test_index]



        clf.train(x_tr, y_tr, x_val=x_te, y_val=y_te)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(X_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
class SklearnWrapper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train, **kwargs):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    

class XgbWrapper(object):

    def __init__(self, params=None):

        self.param = params

        self.nrounds = params.pop('nrounds', 60000)

        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)



    def train(self, x_train, y_train, **kwargs):

        dtrain = xgb.DMatrix(x_train, label=y_train)

        dvalid = xgb.DMatrix(data=kwargs['x_val'], label=kwargs['y_val'])

        

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        

        self.model = xgb.train(dtrain=dtrain, num_boost_round=self.nrounds, evals=watchlist, early_stopping_rounds=self.early_stop_rounds, 

                               verbose_eval=1000, params=self.param)



    def predict(self, x):

        return self.model.predict(xgb.DMatrix(x), ntree_limit=self.model.best_ntree_limit)



    

class LGBWrapper(object):

    def __init__(self, params=None):

        self.param = params

        self.num_rounds = params.pop('nrounds', 60000)

        self.early_stop_rounds = params.pop('early_stop_rounds', 2000)



    def train(self, x_train, y_train, **kwargs):

        dtrain = lgb.Dataset(x_train, label=y_train)

        dvalid = lgb.Dataset(kwargs['x_val'], label=kwargs['y_val'])



        watchlist = [dtrain, dvalid]

        

        print('training LightGBM with params: ', self.param)

        self.model = lgb.train(

                  self.param,

                  train_set=dtrain,

                  num_boost_round=self.num_rounds,

                  valid_sets=watchlist,

                  verbose_eval=1000,

                  early_stopping_rounds=self.early_stop_rounds

        )



    def predict(self, x):

        return self.model.predict(x, num_iteration=self.model.best_iteration)
# LightGBM



lgbm_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'nrounds': 50000,

    'early_stop_rounds': 2000,

    # trainable params

    'max_depth': 4,

    'num_leaves': 46,

    'feature_fraction': 0.6,

    'bagging_fraction': 0.9,

    'bagging_freq': 8,

    'learning_rate': 0.019,

    'verbose': 0

}

lgb_wrapper = LGBWrapper(lgbm_params)



with timer('Training LightGBM'):

    lgb_oof_train, lgb_oof_test = get_oof(lgb_wrapper, X_train_non_null.drop(['AdoptionSpeed'], axis=1), X_train_non_null['AdoptionSpeed'].values.astype(int), X_test_non_null, groups=rescuer_ids)
# Random Forrest

rf_params = {

    'n_jobs': -1,

    'n_estimators': 150,

    'max_features': 'auto',

    'max_depth': 20,

    'min_samples_leaf': 2,

}



rf = SklearnWrapper(clf=RandomForestRegressor, seed=1337, params=rf_params)

with timer('Training Random Forest'):

    rf_oof_train, rf_oof_test = get_oof(rf, X_train_non_null.drop(['AdoptionSpeed'], axis=1), X_train_non_null['AdoptionSpeed'].values.astype(int), X_test_non_null, groups=rescuer_ids)
# extra trees

et_params = {

    'n_jobs': -1,

    'n_estimators': 150,

    'max_features': 'auto',

    'max_depth': 50,

    'min_samples_leaf': 2,

}

et = SklearnWrapper(clf=ExtraTreesRegressor, seed=1337, params=et_params)



with timer('Training Extra trees'):

    et_oof_train, et_oof_test = get_oof(et, X_train_non_null.drop(['AdoptionSpeed'], axis=1), X_train_non_null['AdoptionSpeed'].values.astype(int), X_test_non_null, groups=rescuer_ids)
# # xgboost

xgb_params = {

    'eval_metric': 'rmse',

    'tree_method': 'gpu_hist',

    'device': 'gpu',

    'silent': 1,

    'seed': 1337,

    'nrounds': 60000,

    'early_stop_rounds': 2000,

    # trainable params

    'eta': 0.025,

    'subsample': 0.8,

    'colsample_bytree': 0.6000000000000001,

    'gamma': 0.65,

    'max_depth': 4,

    'min_child_weight': 5.0,

    'n_estimators': 1000,

}



xgbWrapper = XgbWrapper(xgb_params)



with timer('Training Xgboost'):

    xgb_oof_train, xgb_oof_test = get_oof(xgbWrapper, X_train_non_null.drop(['AdoptionSpeed'], axis=1), X_train_non_null['AdoptionSpeed'].values.astype(int), X_test_non_null, groups=rescuer_ids)
x_train_ensemble = np.concatenate((et_oof_train, rf_oof_train, lgb_oof_train, xgb_oof_train), axis=1)

x_test_ensemble = np.concatenate((et_oof_test, rf_oof_test, lgb_oof_test, xgb_oof_test), axis=1)
# Stack, combine and train ridge regressor



ridge_params = {

    'alpha':50.0, 

    'fit_intercept':True, 

    'normalize':False, 

    'copy_X':True,

    'max_iter':None, 

    'tol':0.001, 

    'solver':'auto', 

    'random_state':1337

}

ridge = SklearnWrapper(clf=Ridge, seed=1337, params=ridge_params)



final_oof_train = np.zeros((ntrain,))

final_oof_test = np.zeros((ntest,))

final_oof_test_skf = np.empty((NFOLDS, ntest))



for i, (train_index, test_index) in enumerate(stratified_group_k_fold(x_train_ensemble, X_train_non_null['AdoptionSpeed'].values.astype(int), rescuer_ids, NFOLDS, 1337)):

    print('Training ridge regressor for fold: ', i + 1)

    

    x_tr = x_train_ensemble[train_index]

    y_tr = X_train_non_null['AdoptionSpeed'].values[train_index]

    x_te = x_train_ensemble[test_index]

    y_te = X_train_non_null['AdoptionSpeed'].values[test_index]

    

    print(x_tr.shape, y_tr.shape, x_te.shape, y_te.shape)



    ridge.train(x_tr, y_tr, x_val=x_te, y_val=y_te)

    

    final_oof_train[test_index] = ridge.predict(x_te)

    final_oof_test_skf[i, :] = ridge.predict(x_test_ensemble)



final_oof_test[:] = final_oof_test_skf.mean(axis=0)

final_oof_train = final_oof_train.reshape(-1, 1)

final_oof_test = final_oof_test.reshape(-1, 1)
def plot_pred(pred):

    sns.distplot(pred, kde=True, hist_kws={'range': [0, 5]})
plot_pred(final_oof_train)
plot_pred(final_oof_test.mean(axis=1))
optR = OptimizedRounder()

optR.fit(final_oof_train, X_train['AdoptionSpeed'].values)

coefficients = optR.coefficients()

valid_pred = optR.predict(final_oof_train, coefficients)

qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)

print("QWK = ", qwk)
gc.collect()
coefficients_ = coefficients.copy()

train_predictions = optR.predict(final_oof_train, coefficients_).astype(np.int8)

print(f'train pred distribution: {Counter(train_predictions)}')

test_predictions = optR.predict(final_oof_test, coefficients_).astype(np.int8)

print(f'test pred distribution: {Counter(test_predictions)}')
Counter(train_predictions)
Counter(test_predictions)
submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})

submission.to_csv('submission.csv', index=False)

submission.head()