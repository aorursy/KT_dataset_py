from keras import backend as K

K.tensorflow_backend._get_available_gpus()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

from collections import Counter

import glob

import os

import json

import matplotlib.pyplot as plt

import seaborn as sns

import pprint

import warnings



import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from joblib import Parallel, delayed

from tqdm import tqdm, tqdm_notebook



import scipy as sp 

from functools import partial

from math import sqrt



from sklearn.metrics import cohen_kappa_score



%matplotlib inline



np.random.seed(seed=1337)

warnings.filterwarnings('ignore')



split_char = '/'



pd.set_option('display.max_columns', 1000)



from sklearn.preprocessing import LabelEncoder



from sklearn.preprocessing import PolynomialFeatures



import string



import spacy
pd.set_option('display.max_columns', 50)
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

sample_submission = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

labels_color = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')

labels_state = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
df_all = pd.concat([train, test], ignore_index=True, sort=False)



df_all.loc[np.isfinite(df_all.AdoptionSpeed), 'is_train'] = 1

df_all.loc[~np.isfinite(df_all.AdoptionSpeed), 'is_train'] = 0
all_columns = []
train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))

train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))

train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))





test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))

test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))

test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))
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
aggregates = ['sum', 'mean', 'max', 'min', 'std'] # enis: I added 'max', 'min', 'std'

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
all_sentiment_gr = pd.concat([train_sentiment_gr, test_sentiment_gr], ignore_index=True, sort=False)

all_metadata_gr = pd.concat([train_metadata_gr, test_metadata_gr], ignore_index=True, sort=False)

all_metadata_desc = pd.concat([train_metadata_desc, test_metadata_desc], ignore_index=True, sort=False)

all_sentiment_desc = pd.concat([train_sentiment_desc, test_sentiment_desc], ignore_index=True, sort=False)
# merges:

df_all = df_all.merge(

    all_sentiment_gr, how='left', on='PetID')

df_all = df_all.merge(

    all_metadata_gr, how='left', on='PetID')

df_all = df_all.merge(

    all_metadata_desc, how='left', on='PetID')

df_all = df_all.merge(

    all_sentiment_desc, how='left', on='PetID')





all_columns.extend(list(all_sentiment_gr.columns[1:]) +list(all_metadata_gr.columns[1:]))
text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']



X_text = df_all[text_columns].copy()



for i in X_text.columns:

    X_text.loc[:, i] = X_text.loc[:, i].fillna('none')
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
df_all = pd.concat([df_all, text_features.reset_index(drop=True)], axis=1)



all_columns.extend(text_features.columns)
train_image_nums = Counter([i.split('/')[-1].split('-')[0] for i in train_image_files])

test_image_nums = Counter([i.split('/')[-1].split('-')[0] for i in test_image_files])



train['image_num'] = train.PetID.apply(lambda x: train_image_nums[x] if x in train_image_nums else 0)

test['image_num'] = test.PetID.apply(lambda x: test_image_nums[x] if x in test_image_nums else 0)
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

    'image_size': ['sum', 'mean', 'max', 'min', 'std'], # enis: I added 'max', 'min', 'std'

    'width': ['sum', 'mean', 'max', 'min', 'std'], # enis: I added 'max', 'min', 'std'

    'height': ['sum', 'mean', 'max', 'min', 'std'], # enis: I added 'max', 'min', 'std'

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
df_all = pd.merge(df_all,agg_imgs,on='PetID',how='left')



all_columns.extend(new_columns + ['image_num'])
breed_job = df_all[['Breed1','Breed2']]

breed_job['mixed_breed'] = ((breed_job[['Breed1','Breed2']]==307).sum(axis=1)>0).astype('int')

breed_job.loc[breed_job['Breed1']==307,'Breed1'] = 0

breed_job.loc[breed_job['Breed2']==307,'Breed2'] = 0



breed_job.loc[breed_job['mixed_breed']==1,'Breed1'] = breed_job.loc[breed_job['mixed_breed']==1,'Breed1']+breed_job.loc[breed_job['mixed_breed']==1,'Breed2']

breed_job.loc[breed_job['mixed_breed']==1,'Breed2'] = 0



breed_job.loc[(breed_job['mixed_breed']==0)&(breed_job['Breed2']!=0),'extra_mixed'] = 1



breed_job.loc[breed_job['extra_mixed']==1,'Breed2'] = 0



breed_job['mixed_breed'] = breed_job['mixed_breed'] + breed_job['extra_mixed'].fillna(0)



df_all = df_all.drop(['Breed1', 'Breed2',],axis=1)



df_all = pd.concat(

    [df_all, breed_job[['Breed1','mixed_breed']]], axis=1)
df_all['Type'] = df_all['Type'].map({1:1,2:0})



df_all = df_all.rename(columns={'Type':'is_dog'})
df_all_breed_names = df_all[['Breed1']].merge(

    labels_breed[['BreedID','BreedName']], how='left',

    left_on='Breed1', right_on='BreedID',

    suffixes=('', '_main_breed'))
from keras.preprocessing import image                  



def path_to_tensor(img_path):

    img = image.load_img(img_path, target_size=(299, 299))

    x = image.img_to_array(img)

    return np.expand_dims(x, axis=0)



def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]

    return np.vstack(list_of_tensors)
predicted_df = df_all[(df_all.Breed1 == 0) & (df_all.mixed_breed == 1) & df_all.is_dog][['PetID', 'is_train']].copy().reset_index(drop=True)

predicted_df['img_path'] = predicted_df.apply(lambda x: f"../input/petfinder-adoption-prediction/{'train' if x['is_train'] else 'test'}_images/{x['PetID']}-1.jpg", axis=1)

                                              

predicted_df = predicted_df[predicted_df.img_path.map(os.path.exists)].reset_index(drop=True)

                                              

predicted_df['pred_breed'] = 0

                                              

fns = predicted_df.img_path.values
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True 



def create_generator(batch_size):

    while True:

        for i in range(0, len(fns), batch_size):

            chunk = fns[i:i+batch_size]                            

            x = paths_to_tensor(chunk).astype('float32')/255

            

            yield x
batchsize = 256



pred_generator = create_generator(batchsize)
from keras.models import Model, Input



import keras.backend as K



from keras.applications.inception_resnet_v2 import InceptionResNetV2



from keras.layers import GlobalAveragePooling2D, Dense, Dropout



inp_ten = Input(shape=(299, 299, 3))

x = InceptionResNetV2(

    weights='../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', 

    include_top=False, 

    pooling=None

)(inp_ten)

x = GlobalAveragePooling2D()(x)

out = Dense(133, activation='softmax')(x)



model = Model(inp_ten, out)



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()



model.load_weights('../input/dog-identification-pretrained/dog_identification_model.hdf5')
breed_dict = {i:j 

              for i, j in zip(

                  labels_breed[labels_breed.Type == 1].BreedName.apply(lambda x: x.lower().replace(' ', '_')).values,

                  labels_breed[labels_breed.Type == 1].BreedID.values)}
dog_names = ['in/001.Affenpinscher', 'in/002.Afghan_hound', 'in/003.Airedale_terrier', 'in/004.Akita', 'in/005.Alaskan_malamute', 'in/006.American_eskimo_dog', 'in/007.American_foxhound', 'in/008.American_staffordshire_terrier', 'in/009.American_water_spaniel', 'in/010.Anatolian_shepherd_dog', 'in/011.Australian_cattle_dog', 'in/012.Australian_shepherd', 'in/013.Australian_terrier', 'in/014.Basenji', 'in/015.Basset_hound', 'in/016.Beagle', 'in/017.Bearded_collie', 'in/018.Beauceron', 'in/019.Bedlington_terrier', 'in/020.Belgian_malinois', 'in/021.Belgian_sheepdog', 'in/022.Belgian_tervuren', 'in/023.Bernese_mountain_dog', 'in/024.Bichon_frise', 'in/025.Black_and_tan_coonhound', 'in/026.Black_russian_terrier', 'in/027.Bloodhound', 'in/028.Bluetick_coonhound', 'in/029.Border_collie', 'in/030.Border_terrier', 'in/031.Borzoi', 'in/032.Boston_terrier', 'in/033.Bouvier_des_flandres', 'in/034.Boxer', 'in/035.Boykin_spaniel', 'in/036.Briard', 'in/037.Brittany', 'in/038.Brussels_griffon', 'in/039.Bull_terrier', 'in/040.Bulldog', 'in/041.Bullmastiff', 'in/042.Cairn_terrier', 'in/043.Canaan_dog', 'in/044.Cane_corso', 'in/045.Cardigan_welsh_corgi', 'in/046.Cavalier_king_charles_spaniel', 'in/047.Chesapeake_bay_retriever', 'in/048.Chihuahua', 'in/049.Chinese_crested', 'in/050.Chinese_shar-pei', 'in/051.Chow_chow', 'in/052.Clumber_spaniel', 'in/053.Cocker_spaniel', 'in/054.Collie', 'in/055.Curly-coated_retriever', 'in/056.Dachshund', 'in/057.Dalmatian', 'in/058.Dandie_dinmont_terrier', 'in/059.Doberman_pinscher', 'in/060.Dogue_de_bordeaux', 'in/061.English_cocker_spaniel', 'in/062.English_setter', 'in/063.English_springer_spaniel', 'in/064.English_toy_spaniel', 'in/065.Entlebucher_mountain_dog', 'in/066.Field_spaniel', 'in/067.Finnish_spitz', 'in/068.Flat-coated_retriever', 'in/069.French_bulldog', 'in/070.German_pinscher', 'in/071.German_shepherd_dog', 'in/072.German_shorthaired_pointer', 'in/073.German_wirehaired_pointer', 'in/074.Giant_schnauzer', 'in/075.Glen_of_imaal_terrier', 'in/076.Golden_retriever', 'in/077.Gordon_setter', 'in/078.Great_dane', 'in/079.Great_pyrenees', 'in/080.Greater_swiss_mountain_dog', 'in/081.Greyhound', 'in/082.Havanese', 'in/083.Ibizan_hound', 'in/084.Icelandic_sheepdog', 'in/085.Irish_red_and_white_setter', 'in/086.Irish_setter', 'in/087.Irish_terrier', 'in/088.Irish_water_spaniel', 'in/089.Irish_wolfhound', 'in/090.Italian_greyhound', 'in/091.Japanese_chin', 'in/092.Keeshond', 'in/093.Kerry_blue_terrier', 'in/094.Komondor', 'in/095.Kuvasz', 'in/096.Labrador_retriever', 'in/097.Lakeland_terrier', 'in/098.Leonberger', 'in/099.Lhasa_apso', 'in/100.Lowchen', 'in/101.Maltese', 'in/102.Manchester_terrier', 'in/103.Mastiff', 'in/104.Miniature_schnauzer', 'in/105.Neapolitan_mastiff', 'in/106.Newfoundland', 'in/107.Norfolk_terrier', 'in/108.Norwegian_buhund', 'in/109.Norwegian_elkhound', 'in/110.Norwegian_lundehund', 'in/111.Norwich_terrier', 'in/112.Nova_scotia_duck_tolling_retriever', 'in/113.Old_english_sheepdog', 'in/114.Otterhound', 'in/115.Papillon', 'in/116.Parson_russell_terrier', 'in/117.Pekingese', 'in/118.Pembroke_welsh_corgi', 'in/119.Petit_basset_griffon_vendeen', 'in/120.Pharaoh_hound', 'in/121.Plott', 'in/122.Pointer', 'in/123.Pomeranian', 'in/124.Poodle', 'in/125.Portuguese_water_dog', 'in/126.Saint_bernard', 'in/127.Silky_terrier', 'in/128.Smooth_fox_terrier', 'in/129.Tibetan_mastiff', 'in/130.Welsh_springer_spaniel', 'in/131.Wirehaired_pointing_griffon', 'in/132.Xoloitzcuintli', 'in/133.Yorkshire_terrier']
preds = model.predict_generator(pred_generator, steps=len(fns) // batchsize + 1, verbose=1)



preds = np.argmax(preds, axis=1)



mapping = {

    'american_foxhound': 'foxhound',

    'anatolian_shepherd_dog': 'anatolian_shepherd',

    'australian_cattle_dog': 'australian_cattle_dog/blue_heeler',

    'belgian_malinois': 'belgian_shepherd_malinois',

    'belgian_sheepdog': 'belgian_shepherd_dog_sheepdog',

    'belgian_tervuren': 'belgian_shepherd_tervuren',

    'bouvier_des_flandres': 'bouvier_des_flanders',

    'brittany': 'brittany_spaniel',

    'bulldog': 'american_bulldog',

    'cane_corso': 'cane_corso_mastiff',

    'cardigan_welsh_corgi': 'welsh_corgi',

    'chinese_crested': 'chinese_crested_dog',

    'chinese_shar-pei': 'shar_pei',

    'dandie_dinmont_terrier': 'dandi_dinmont_terrier',

    'entlebucher_mountain_dog': 'entlebucher',

    'icelandic_sheepdog': 'shetland_sheepdog_sheltie',

    'irish_red_and_white_setter': 'irish_setter',

    'miniature_schnauzer': 'schnauzer',

    'newfoundland': 'newfoundland_dog',

    'nova_scotia_duck_tolling_retriever': 'nova_scotia_duck-tolling_retriever',

    'parson_russell_terrier': 'jack_russell_terrier',

    'pembroke_welsh_corgi': 'welsh_corgi',

    'plott': 'plott_hound',

    'wirehaired_pointing_griffon': 'german_wirehaired_pointer',

    'xoloitzcuintli': 'xoloitzcuintle/mexican_hairless',

    'yorkshire_terrier': 'yorkshire_terrier_yorkie'

}



def get_breed_id(prediction):

    dog_name = dog_names[prediction].split('.')[-1].lower()

    dog_name = dog_name if dog_name not in mapping else mapping[dog_name]

    return breed_dict[dog_name]



predicted_df['pred_breed'] = [get_breed_id(x) for x in preds]



del model; gc.collect()



K.clear_session()
predicted_df = predicted_df[['PetID', 'pred_breed']]



df_all = df_all.merge(predicted_df, on='PetID', how='left')



del predicted_df; gc.collect()



df_all.pred_breed = df_all.pred_breed.fillna(0)

get_values = lambda i, j: int(j) if j != 0 else i

df_all.pred_breed = [get_values(i, j) for i,j in zip(df_all.Breed1.values, df_all.pred_breed.values)]
for e in ['hair','domestic', 'short', 'medium', 'retriever', 'terrier', 'tabby', 'long']:

    df_all_breed_names['Breed_{}'.format(e)] = (df_all_breed_names.BreedName.apply(lambda x: e in str(x).lower())).astype(int)



breed_detail_features = ['Breed_{}'.format(e) for e in ['hair','domestic', 'short', 'medium', 'retriever', 'terrier', 'tabby', 'long']]



df_all = pd.concat(

    [df_all, df_all_breed_names[breed_detail_features]], axis=1)
le = LabelEncoder()

df_all['Breed1_le'] = le.fit_transform(df_all['Breed1'])
breed_columns = ['is_dog', 'Breed1_le', 'mixed_breed']



breed_columns.extend(breed_detail_features)
all_columns = []



all_columns.extend(breed_columns)
df_all['Age_bins'] = pd.cut(np.log1p(df_all.Age),9,labels=['Age_{}'.format(e) for e in range(9)])

# df_all['age_bin_1'] = pd.cut((df_all.Age),10,labels=['age_{}'.format(e) for e in range(10)])



# df_all.groupby('Age_bins').Age.describe()



le = LabelEncoder()

df_all['Age_bins'] = le.fit_transform(df_all['Age_bins'])



age_columns = ['Age_bins']
all_columns.extend(age_columns)
# na features

df_all['Name_isna'] = df_all['Name'].isna().astype(int)

# name lens

df_all.loc[~df_all['Name'].isna(),'Name_len'] = df_all.loc[~df_all['Name'].isna(),'Name'].apply(lambda x : len(str(x).split()))

df_all['Name_len'] = df_all['Name_len'].fillna(0)

# name with numbers

name_with_numbers = [e for e in df_all.Name.unique() if len(set([str(t) for t in range(10)]).intersection(set(str(e).split())))>0]

df_all['Name_with_numbers'] = (df_all.Name.isin(name_with_numbers)).astype(int)
name_columns = ['Name_len','Name_isna',"Name_with_numbers"]
all_columns.extend(name_columns)
df_all['Color_range'] = (df_all[['Color1','Color2','Color3',]]>0).sum(axis=1)
color_columns = ['Color_range']



all_columns.extend(color_columns)

df_all['Fee_bins']= pd.cut(np.log1p(df_all.Fee),5,labels=['Fee_{}'.format(e) for e in range(5)])



le = LabelEncoder()

df_all['Fee_bins'] = le.fit_transform(df_all['Fee_bins'])

fee_columns = ['Fee_bins']
all_columns.extend(fee_columns)
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
# External Data from Wiki

df_all["state_gdp"] = df_all.State.map(state_gdp)

df_all["state_population"] = df_all.State.map(state_population)

# label encoding states

le = LabelEncoder()

df_all['State_le'] = le.fit_transform(df_all['State'])
state_columns = ['state_gdp',

        'state_population',

        'State_le',]
all_columns.extend(state_columns)
# any video?

df_all['is_VideoAmt'] =(df_all.VideoAmt>0).astype('int')

# bin 

df_all['PhotoAmt_bins']= pd.cut(np.log1p(df_all.PhotoAmt),5,labels=['PhotoAmt_{}'.format(e) for e in range(5)])



# df_all.groupby(['PhotoAmt_bins'])['PhotoAmt'].describe()



le = LabelEncoder()

df_all['PhotoAmt_bins'] = le.fit_transform(df_all['PhotoAmt_bins'])
video_photo_columns = ['is_VideoAmt','PhotoAmt_bins']
all_columns.extend(video_photo_columns)
# na features

df_all['Description_isna'] = df_all['Description'].isna().astype(int)

# no of words 

df_all['len_words_Description'] = df_all.Description.apply(lambda x: len(str(x).split()))

# create bins

n = 6

df_all['len_words_Description_bins']= pd.cut(np.log1p(df_all.len_words_Description),n,labels=['Description_{}'.format(e) for e in range(n)])

# df_all.groupby(['len_words_Description_bins'])['len_words_Description'].describe()

le = LabelEncoder()

df_all['len_words_Description_bins'] = le.fit_transform(df_all['len_words_Description_bins'])



# number of uppercase 

df_all['len_isupper_Description'] = df_all.Description.apply(lambda x: len([c for c in str(x) if c.isupper()]))

# number of lowercase

df_all['len_islower_Description'] = df_all.Description.apply(lambda x: len([c for c in str(x) if c.islower()]))



# some numbers 

count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))



df_all['len_punctuation_Description'] = df_all.Description.apply(lambda x: count(str(x), string.punctuation))

df_all['len_letters_Description'] = df_all.Description.apply(lambda x: count(str(x), string.ascii_letters))

df_all['len_characters_Description'] = df_all.Description.apply(lambda x: len(str(x)))



# ratios to words

df_all['len_isupper_Description_to_words'] = df_all['len_isupper_Description'] / df_all['len_words_Description']

df_all['len_islower_Description_to_words'] = df_all['len_islower_Description'] / df_all['len_words_Description']

df_all['len_punctuation_Description_to_words'] = df_all['len_punctuation_Description'] / df_all['len_words_Description']

df_all['len_letters_Description_to_words'] = df_all['len_letters_Description'] / df_all['len_words_Description']

df_all['len_characters_Description_to_words'] = df_all['len_characters_Description'] / df_all['len_words_Description']



# ratios to letters

df_all['len_isupper_Description_to_characters'] = df_all['len_isupper_Description'] / df_all['len_characters_Description']

df_all['len_islower_Description_to_characters'] = df_all['len_islower_Description'] / df_all['len_characters_Description']

df_all['len_punctuation_Description_to_characters'] = df_all['len_punctuation_Description'] / df_all['len_characters_Description']

df_all['len_letters_Description_to_characters'] = df_all['len_letters_Description'] / df_all['len_characters_Description']
description_columns = [

    'Description_isna',

    'len_words_Description',

    'len_words_Description_bins',

    'len_isupper_Description',

    'len_islower_Description',

    'len_punctuation_Description',

    'len_letters_Description',

    'len_characters_Description',

    'len_isupper_Description_to_words',

    'len_islower_Description_to_words',

    'len_punctuation_Description_to_words',

    'len_letters_Description_to_words',

    'len_isupper_Description_to_characters',

    'len_islower_Description_to_characters',

    'len_punctuation_Description_to_characters',

    'len_letters_Description_to_characters',

    'len_characters_Description_to_words',

                      ]
all_columns.extend(description_columns)
nlp = spacy.load('../input/spacyen-vectors-web-lg/spacy-en_vectors_web_lg/en_vectors_web_lg')



desc_embd_cols = ['Describe_spacy_{}'.format(e) for e in range(300)]



desc_embd = pd.DataFrame(np.vstack(df_all[['Description']].dropna().apply(lambda x: nlp(x['Description']).vector,axis=1).values),

                         index=df_all[['Description']].dropna().index)



desc_embd.columns= desc_embd_cols

del nlp; gc.collect()
Describe_SVD = True



if Describe_SVD:

    n_components = 32



    svd_ = TruncatedSVD(n_components=n_components, random_state=1881)

    svd_col = svd_.fit_transform(desc_embd)

    svd_col_df = pd.DataFrame(svd_col,index=df_all[['Description']].dropna().index)

    svd_col_df = svd_col_df.add_prefix('Describe_spacy_SVD')



    df_all  = df_all.join(svd_col_df)

    all_columns.extend(svd_col_df.columns)

    

else:

    

    df_all  = df_all.join(desc_embd)

    all_columns.extend(desc_embd_cols)
del svd_col_df

del desc_embd; gc.collect()
images_df = df_all[['PetID', 'is_train', 'is_dog']].copy()



images_df['img_path'] = images_df.apply(lambda x: f"../input/petfinder-adoption-prediction/{'train' if x['is_train'] else 'test'}_images/{x['PetID']}-1.jpg", axis=1)

                                           

images_df = images_df[images_df.img_path.map(os.path.exists)].reset_index(drop=True)
from keras.preprocessing import image                  



def load_image(img_path, preprocess_fn, img_size):

    image = cv2.imread(img_path)

    image = cv2.resize(image, (img_size, img_size))

    image = preprocess_fn(image)

    return image
images_df = images_df.set_index('PetID')
from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D



from keras.applications.densenet import preprocess_input, DenseNet121



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
import cv2



batch_size = 256



pet_ids = images_df.index.values

n_batches = len(pet_ids) // batch_size + 1



features = {}

for b in tqdm(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),256,256,3))

    for i,pet_id in enumerate(batch_pets):

        image_ins = images_df.loc[pet_id]

        batch_images[i] = load_image(image_ins['img_path'], preprocess_input, 256)

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]
img_feats = pd.DataFrame.from_dict(features, orient='index')

img_feats.columns = [f'pic_{i}' for i in range(img_feats.shape[1])]



img_feats = img_feats.reset_index()

img_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)
all_ids = img_feats[['PetID']]

all_ids.shape
n_components = 32

svd_ = TruncatedSVD(n_components=n_components, random_state=1337)



features = img_feats[[f'pic_{i}' for i in range(len(img_feats.columns) - 1)]].values



svd_col = svd_.fit_transform(features)

svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('IMG_SVD_Dense_')



img_feats = pd.concat([all_ids, svd_col], axis=1)
df_all = df_all.merge(img_feats, on='PetID', how='left')



all_columns.extend(img_feats.columns.values)



del img_feats



del m; gc.collect()

K.clear_session()
from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D



from keras.applications.inception_v3 import InceptionV3, preprocess_input



inp = Input(shape=(224, 224, 3))

backbone = InceptionV3(

    input_tensor=inp, 

    weights='../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 

    include_top=False, 

    pooling=None,

)



x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)
batch_size = 256



pet_ids = images_df.index.values

n_batches = len(pet_ids) // batch_size + 1



features = {}

for b in tqdm(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),224, 224,3))

    for i,pet_id in enumerate(batch_pets):

        image_ins = images_df.loc[pet_id]

        batch_images[i] = load_image(image_ins['img_path'], preprocess_input, 224)

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]
img_feats = pd.DataFrame.from_dict(features, orient='index')

img_feats.columns = [f'pic_inception{i}' for i in range(img_feats.shape[1])]



img_feats = img_feats.reset_index()

img_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)
all_ids = img_feats[['PetID']]

all_ids.shape
n_components = 32

svd_ = TruncatedSVD(n_components=n_components, random_state=1337)



features = img_feats[[f'pic_inception{i}' for i in range(len(img_feats.columns) - 1)]].values



svd_col = svd_.fit_transform(features)

svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('IMG_SVD_Incep_')



img_feats = pd.concat([all_ids, svd_col], axis=1)
df_all = df_all.merge(img_feats, on='PetID', how='left')



all_columns.extend(img_feats.columns.values)



del img_feats



del m; gc.collect()

K.clear_session()
from keras.models import Model

from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D



from keras.applications.xception import Xception, preprocess_input



inp = Input(shape=(224, 224, 3))

backbone = Xception(

    input_tensor=inp, 

    weights='../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', 

    include_top=False, 

    pooling=None,

)



x = backbone.output

x = GlobalAveragePooling2D()(x)

x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)

x = AveragePooling1D(4)(x)

out = Lambda(lambda x: x[:,:,0])(x)



m = Model(inp,out)
batch_size = 256



pet_ids = images_df.index.values

n_batches = len(pet_ids) // batch_size + 1



features = {}

for b in tqdm(range(n_batches)):

    start = b*batch_size

    end = (b+1)*batch_size

    batch_pets = pet_ids[start:end]

    batch_images = np.zeros((len(batch_pets),224, 224,3))

    for i,pet_id in enumerate(batch_pets):

        image_ins = images_df.loc[pet_id]

        batch_images[i] = load_image(image_ins['img_path'], preprocess_input, 224)

    batch_preds = m.predict(batch_images)

    for i,pet_id in enumerate(batch_pets):

        features[pet_id] = batch_preds[i]
img_feats = pd.DataFrame.from_dict(features, orient='index')

img_feats.columns = [f'pic_xception{i}' for i in range(img_feats.shape[1])]



img_feats = img_feats.reset_index()

img_feats.rename({'index': 'PetID'}, axis='columns', inplace=True)
all_ids = img_feats[['PetID']]

all_ids.shape
n_components = 32

svd_ = TruncatedSVD(n_components=n_components, random_state=1337)



features = img_feats[[f'pic_xception{i}' for i in range(len(img_feats.columns) - 1)]].values



svd_col = svd_.fit_transform(features)

svd_col = pd.DataFrame(svd_col)

svd_col = svd_col.add_prefix('IMG_SVD_Xcep_')



img_feats = pd.concat([all_ids, svd_col], axis=1)
df_all = df_all.merge(img_feats, on='PetID', how='left')



all_columns.extend(img_feats.columns.values)



del img_feats

del images_df

del m; gc.collect()

K.clear_session()
df_external_1 = pd.read_csv('../input/petfindercomexternal/petfinder.com_external.csv')

df_external_1 = df_external_1.drop('attr_other_names',axis=1)

df_external_1 = df_external_1.rename(columns={'BreedID':'pred_breed'})
df_all = df_all.merge(df_external_1,how='left',on='pred_breed')
all_columns.extend(df_external_1.columns[:].values)
# add features from ratings 

with open('../input/cat-and-dog-breeds-parameters/rating.json', 'r') as f:

        ratings = json.load(f)

cat_ratings = ratings['cat_breeds']

dog_ratings = ratings['dog_breeds']



breed_id = {}

for id_,name in zip(labels_breed.BreedID,labels_breed.BreedName):

    breed_id[name] = id_



breed_names_1 = [i for i in cat_ratings.keys()]

breed_names_2 = [i for i in dog_ratings.keys()]
df_cat_breeds = pd.DataFrame(cat_ratings).T

df_dog_breeds = pd.DataFrame(dog_ratings).T

df_detail_breeds = pd.concat([df_cat_breeds,df_dog_breeds],axis=1)

df_detail_breeds = df_detail_breeds.reset_index().rename(columns={'index':'Breed1'})
df_detail_breeds = df_detail_breeds.append({'Breed1':'Terrier',

    'Affectionate with Family': 4.606060606060606,

 'Friendly Toward Strangers': 3.757575757575758,

 'General Health': 4.0606060606060606,

 'Intelligence': 4.090909090909091,

 ' Adaptability': 3.212121212121212,

 ' All Around Friendliness': 4.121212121212121,

 ' Exercise Needs': 4.454545454545454,

 ' Health Grooming': 2.8181818181818183,

 ' Trainability': 3.606060606060606,

 'Adapts Well to Apartment Living': 3.9696969696969697,

 'Amount Of Shedding': 2.515151515151515,

 'Dog Friendly': 3.1818181818181817,

 'Drooling Potential': 1.2424242424242424,

 'Easy To Groom': 3.1515151515151514,

 'Easy To Train': 3.3636363636363638,

 'Energy Level': 4.212121212121212,

 'Exercise Needs': 4.090909090909091,

 'Good For Novice Owners': 3.0606060606060606,

 'Incredibly Kid Friendly Dogs': 4.212121212121212,

 'Intensity': 4.03030303030303,

 'Potential For Mouthiness': 2.8484848484848486,

 'Potential For Playfulness': 4.666666666666667,

 'Potential For Weight Gain': 3.3333333333333335,

 'Prey Drive': 3.5757575757575757,

 'Sensitivity Level': 3.5454545454545454,

 'Size': 2.0,

 'Tendency To Bark Or Howl': 3.212121212121212,

 'Tolerates Being Alone': 2.1818181818181817,

 'Tolerates Cold Weather': 3.0303030303030303,

 'Tolerates Hot Weather': 3.242424242424242,

 'Wanderlust Potential': 3.6363636363636362}, ignore_index=True)
breed_mapper = {"White German Shepherd":"German Shepherd Dog",

    "Jack Russell Terrier (Parson Russell Terrier)": "Jack Russell Terrier",

"Belgian Shepherd Dog Sheepdog": "Belgian Sheepdog",

"Shetland Sheepdog Sheltie": "Shetland Sheepdog",

"English Pointer": "Pointer",

"Appenzell Mountain Dog": "Bernese Mountain Dog",

"Yorkshire Terrier Yorkie": "Yorkshire Terrier",

"Thai Ridgeback": "Rhodesian Ridgeback",

"Spitz": "Finnish Spitz",

"Standard Poodle": "Poodle",

"Havana": "Havana Brown",

"Munsterlander": "Small Munsterlander Pointer",

"English Bulldog": "Bulldog",

"Husky": "Siberian Husky",

"Entlebucher": "Entlebucher Mountain Dog",

"Wire-haired Pointing Griffon": "Wirehaired Pointing Griffon",

"Chocolate Labrador Retriever": "Labrador Retriever",

"Belgian Shepherd Malinois": "Belgian Sheepdog",

"German Spitz": "Finnish Spitz",

"German Spitz": "Japanese Spitz",

"Smooth Fox Terrier": "Fox Terrier",

"Belgian Shepherd Tervuren": "Belgian Tervuren",

"Wirehaired Terrier": "German Wirehaired Pointer",

"Galgo Spanish Greyhound": "Greyhound",

"Welsh Corgi": "Cardigan Welsh Corgi",

"Eskimo Dog": "American Eskimo Dog",

"Sheep Dog": "Old English Sheepdog",

"American Hairless Terrier": "American Staffordshire Terrier",

"Retriever": "Labrador Retriever",

"Caucasian Sheepdog (Caucasian Ovtcharka)": "Caucasian Shepherd Dog",

"Oriental Tabby": "Oriental",

"Flat-coated Retriever": "Flat-Coated Retriever",

"Oriental Short Hair": "Oriental",

"Exotic Shorthair": "American Shorthair",

"Spaniel": "Cocker Spaniel",

"Wire Fox Terrier": "Fox Terrier",

"Oriental Long Hair": "Oriental",

"Chinese Crested Dog": "Chinese Crested",

"Applehead Siamese": "Siamese Cat",

"Klee Kai": "Alaskan Klee Kai",

"Dandi Dinmont Terrier": "Dandie Dinmont Terrier",

"Yellow Labrador Retriever": "Labrador Retriever",

"Bobtail": "American Bobtail",

"Anatolian Shepherd": "Anatolian Shepherd Dog",

"Cane Corso Mastiff": "Cane Corso",

"Bengal": "Bengal Cats",

"Pit Bull Terrier": "American Pit Bull Terrier",

"Shepherd": "German Shepherd Dog",

"Scottish Terrier Scottie": "Scottish Terrier",

"Mountain Dog": "Bernese Mountain Dog",

"Jindo": "Korean Jindo Dog",

"Foxhound": "American Foxhound",

"Bouvier des Flanders": "Bouvier des Flandres",

"Schnauzer": "Standard Schnauzer",

"Newfoundland Dog": "Newfoundland",

"Cattle Dog": "Australian Cattle Dog",

"West Highland White Terrier Westie": "West Highland White Terrier",

"Australian Cattle Dog/Blue Heeler": "Australian Cattle Dog",

"Maremma Sheepdog": "Belgian Sheepdog",

"Ragdoll": "Ragdoll Cats",

"Wheaten Terrier": "Soft Coated Wheaten Terrier",

"Setter": "English Setter",

"Siamese": "Siamese Cat",

"Black Labrador Retriever": "Labrador Retriever",

"Norwegian Forest Cat": "Norwegian Forest",

"English Coonhound": "American English Coonhound",

"Coonhound": "American English Coonhound",

"English Shepherd": "Old English Sheepdog",

"Plott Hound": "Plott",

"Brittany Spaniel": "Brittany",

"Corgi": "Cardigan Welsh Corgi",

"Illyrian Sheepdog": "Old English Sheepdog",

"Patterdale Terrier (Fell Terrier)": "Terrier",

"Nova Scotia Duck-Tolling Retriever": "Nova Scotia Duck Tolling Retriever",

"Hound": "Afghan Hound",

"Belgian Shepherd Laekenois": "Belgian Sheepdog",

"Sphynx (hairless cat)": "Sphynx",

"Mountain Cur": "Bernese Mountain Dog",

"Kai Dog": "Alaskan Klee Kai",}
labels_breed_temp = labels_breed.copy()



choose = labels_breed_temp.BreedName.isin(list(breed_mapper.keys()))

labels_breed_temp['NewBreedName'] = labels_breed_temp['BreedName']

labels_breed_temp.loc[choose,'NewBreedName'] = labels_breed_temp.loc[choose,'BreedName'].map(breed_mapper)



breed_mapper_all = labels_breed_temp.set_index('BreedID')['NewBreedName'].to_dict()
df_all_temp = df_all.copy()



df_all_temp.Breed1 = df_all_temp.pred_breed.map(breed_mapper_all)



df_all_temp = df_all_temp.merge(df_detail_breeds,on='Breed1',how='left')



df_detail_breeds_cols = df_detail_breeds.columns[1:]
df_all = pd.concat((df_all,df_all_temp[df_detail_breeds_cols]),axis=1)



all_columns.extend(list(df_detail_breeds.columns.values[1:]))
df_external_3_cat = pd.read_csv('../input/pet-breed-characteristics/cat_breed_characteristics.csv')

df_external_3_dog = pd.read_csv('../input/pet-breed-characteristics/dog_breed_characteristics.csv')
cat_dog_cols = list(df_external_3_dog.columns)+list(set(df_external_3_cat.columns)- set(df_external_3_dog.columns))



df_external_3_all = pd.concat((df_external_3_dog,df_external_3_cat),axis=0)[cat_dog_cols]
breed_id = {}

for id_,name in zip(labels_breed.BreedID,labels_breed.BreedName):

    breed_id[name] = id_
df_external_3_all['BreedName']= df_external_3_all.BreedName.map(breed_id)
df_external_3_all = df_external_3_all.rename(columns={"BreedName":'pred_breed'})
df_all = df_all.merge(df_external_3_all,how='left',on='pred_breed')



# all_columns.extend(list(df_external_3_all.columns.values[1:]))
df_all_temp = df_all[['Temperment']].copy()



df_all_temp['Temperment'] = df_all_temp['Temperment'].apply(lambda x: [e.strip(',') for e in str(x).split()])



df_all_temp = df_all_temp.drop('Temperment', 1).join(df_all_temp.Temperment.str.join('|').str.get_dummies())





df_all_temp = df_all_temp.add_prefix('Temperment_')



df_all = pd.concat((df_all,df_all_temp),axis=1)



del df_all['Temperment']
rescuer_count = df_all.groupby(['RescuerID'])['PetID'].count().reset_index()

rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']



df_all = df_all.merge(rescuer_count, how='left', on='RescuerID')
groupby_columns = [['is_dog','Breed1'],

                   ['State'],

                   ['is_dog','Gender','Age_bins'],

                   ['is_dog','MaturitySize'],

                   ['is_dog','FurLength'],

                   ['is_dog','Breed1','Vaccinated', 'Dewormed', 'Sterilized', 'Health',]



                ]



aggregate_columns = ['Age',

    'Name_len',

    'Color1',

    'Color2',

    'Color3',

    'MaturitySize',

    'FurLength',

    'Vaccinated',

    'Dewormed',

    'Sterilized',

    'Health',

    'Quantity',

    'Fee',

    'VideoAmt',

    'PhotoAmt',

    'state_gdp',

    'state_population',

    'len_words_Description', 

    'len_isupper_Description_to_characters',

    'len_islower_Description_to_characters',

    'len_punctuation_Description_to_characters',

    'len_letters_Description_to_characters',

    'len_characters_Description_to_words',

    'RescuerID_COUNT',

                    ]
for _groupby in groupby_columns:

    

    _aggregate = [e for e in aggregate_columns if e not in _groupby]

    _aggregate = {e:['mean','std','min','max'] for e in _aggregate}



    gr = df_all.groupby(_groupby).agg(_aggregate)

    gr.columns = ['{}_{}_{}'.format("_".join(_groupby),e[0],e[1]) for e in gr.columns]

    all_columns.extend(gr.columns)

    gr = gr.reset_index()

    df_all = pd.merge(df_all,gr,on=_groupby,how='left')
other_columns =[

        'PetID', 

        'AdoptionSpeed',

        'Age',

        'Color1',

        'Color2',

        'Color3',

        'Dewormed',

        'Fee',

        'FurLength',

        'Gender',

        'Health',

        'MaturitySize',

        'PhotoAmt',

        'Quantity',

        'Sterilized',

        'Vaccinated',

        'VideoAmt',]
categorical_columns = ['Gender', 'Color1', 'Color2', 'Color3','Vaccinated', 'Dewormed',

       'Sterilized', 'State','Breed1_le', 'RescuerID']

    

freq_categorical_columns = []

for c in tqdm(categorical_columns):

    df_all[c+'_freq'] = df_all[c].map(df_all.groupby(c).size() / df_all.shape[0])

    freq_categorical_columns.append(c+'_freq')

    indexer = pd.factorize(df_all[c], sort=True)[1]

    df_all[c] = indexer.get_indexer(df_all[c])
del train 

del test



gc.collect()
excl_columns = list(df_all.dtypes[df_all.dtypes == 'object'].index.values)
df_all = df_all.loc[:, ~df_all.columns.isin(excl_columns)]
train = df_all.loc[df_all.is_train==1, ~df_all.columns.duplicated()].drop('is_train', 1)

test =  df_all.loc[df_all.is_train==0, ~df_all.columns.duplicated()].drop('is_train', 1)
# train.to_csv('27_03_lite_feature_engineered_train.csv',index=False)

# test.to_csv('27_03_lite_feature_engineered_test.csv',index=False)
# train = pd.read_csv('27_03_lite_feature_engineered_train.csv')

# test = pd.read_csv('27_03_lite_feature_engineered_test.csv')
feat_cols = list(train.columns[

    (train.columns.values != 'AdoptionSpeed')

].values)

y_col = 'AdoptionSpeed'
X_train = train[feat_cols]

X_test = test[feat_cols]



y_train = train[y_col]
print(f"X_train shape: {X_train.shape} \nX_test shape: {X_test.shape}")
from sklearn.metrics import cohen_kappa_score



class OptimizedRounder(object):

        def __init__(self):

            self.coef_ = 0



        def _loss(self, coef, X, y, idx):

            X_p = np.array([to_bins(pred, coef) for pred in X])

            ll = -cohen_kappa_score(y, X_p, weights='quadratic')

            return ll



        def fit(self, X, y):

            coef = [1.5, 2.0, 2.5, 3.0]

            golden1 = 0.618

            golden2 = 1 - golden1

            ab_start = [(1, 2), (1.8, 2.5), (2, 2.8), (2.5, 3.0)]

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

        

# put some numerical values to bins

def to_bins(x, borders):

    for i in range(len(borders)):

        if x <= borders[i]:

            return i

    return len(borders)
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold



from bayes_opt import BayesianOptimization
# dtrain = xgb.DMatrix(X_train, label=y_train)

# dtest = xgb.DMatrix(X_test)



# def xgb_evaluate(max_depth, gamma, colsample_bytree, eta):

#     params = {'eval_metric': 'rmse',

#                 'max_depth': int(max_depth),

#                 'subsample': 0.8,

#                 'eta': eta,

#                 'gamma': gamma,

#                 'tree_method': 'gpu_hist',

#                 'device': 'gpu',

#                 'colsample_bytree': colsample_bytree}

#     # Used around 1000 boosting rounds in the full model

#     cv_result = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5)    

    

#     # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE

#     return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

# xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 

#                                              'gamma': (0, 1),

#                                              'colsample_bytree': (0.6, 1.0),

#                                             'eta': (0.1, 0.005)})

# # Use the expected improvement acquisition function to handle negative numbers

# # Optimally needs quite a few more initiation points and number of iterations

# xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
# best_params = xgb_bo.max['params']

# best_params['max_depth'] = int(best_params['max_depth'])



# best_params
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold



xgb_params = {

    'eval_metric': 'rmse',

    'seed': 1337,

    'subsample': 0.8,

    'tree_method': 'gpu_hist',

    'device': 'gpu',

    'colsample_bytree': 0.8152926875727884,

    'eta': 0.03402795645208036,

    'gamma': 0.9209917577458885,

    'max_depth': 6,

    'silent': 1,

}
def run_xgb(params, X_train, X_test):

    n_splits = 5

    verbose_eval = 1000

    num_rounds = 60000

    early_stop = 500



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)



    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], n_splits))

    oof_coefficients = []





    i = 0



    for train_idx, valid_idx in kf.split(X_train, y_train.values):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]



        y_tr = y_train.values[train_idx]

        y_val = y_train.values[valid_idx]



        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)



        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,

                         early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        

        optR = OptimizedRounder()

        optR.fit(valid_pred, y_val)

        coefficients = optR.coefficients()

#         coefficients = [1.6287721, 2.09219494, 2.480343, 2.8293866]

        oof_coefficients.append(coefficients)

        pred_val_y_k = optR.predict(valid_pred, coefficients)

        qwk = cohen_kappa_score(y_val, pred_val_y_k, weights='quadratic')



        print("QWK = ", qwk)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test, oof_coefficients
model, oof_train, oof_test, oof_coefficients = run_xgb(xgb_params, X_train, X_test)
optR = OptimizedRounder()

mean_coeffs = np.mean(np.array(oof_coefficients), axis=0)

valid_pred = optR.predict(oof_train, mean_coeffs)

qwk = cohen_kappa_score(y_train.values, valid_pred, weights='quadratic')

print("QWK = ", qwk)
# from lightgbm import LGBMRegressor

# import lightgbm as lgb

# from sklearn.model_selection import StratifiedKFold

# from sklearn.metrics import  mean_squared_error





# def lgbm_evaluate(**params_range):  

#     params_range['num_leaves'] = int(params_range['num_leaves'])

#     params_range['max_depth'] = int(params_range['max_depth'])

#     params_range['application'] = 'regression'

#     params_range['metric'] = 'rmse'

    

#     clf = LGBMRegressor(**params_range, n_estimators=1000, n_jobs=-1, )

        

#     mse_scores = np.zeros(5)

    

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)



#     for i, (f_ind, outf_ind) in enumerate(kf.split(X_train, y_train.values)):



#         X_tr = X_train.loc[f_ind].copy()

#         X_val = X_train.loc[outf_ind].copy()



#         y_tr, y_val = y_train[f_ind], y_train[outf_ind]

        

#         clf.fit(X_tr,

#                 y_tr,

#                eval_set=[(X_val, y_val)],

#                eval_metric='rmse',

#                verbose=False)



#         val_preds = clf.predict(X_val, num_iteration = clf.best_iteration_)

#         mse_scores[i] = mean_squared_error(y_val, val_preds)

#         gc.collect()



#     return -np.mean(mse_scores)

# params_range = {

#     'learning_rate': (.01, .001), 

#     'num_leaves': (60, 90), 

#     'subsample': (0.8, 1), 

#     'max_depth': (7, 18), 

#     'reg_alpha': (.03, .05), 

#     'reg_lambda': (.06, .08), 

#     'min_split_gain': (.01, .03),

#     'min_child_weight': (38, 50),

#     'feature_fraction': (0.7, 1.),

#     'bagging_fraction': (0.8, 1)

# }
# from bayes_opt import BayesianOptimization





# bo = BayesianOptimization(lgbm_evaluate, params_range)

# bo.maximize(init_points=3, n_iter=5)
# import operator

# results = bo.res

# results.sort(key=operator.itemgetter('target'), reverse=True)



# best_params = results[0]['params']

# best_params['num_leaves'] = int(best_params['num_leaves'])

# best_params['max_depth'] = int(best_params['max_depth'])



# best_params
import lightgbm as lgb



params = {

    'application': 'regression',

    'boosting': 'gbdt',

    'metric': 'rmse',

    'num_leaves': 60,

    'max_depth': 11,

    'learning_rate': 0.0076,

    'bagging_fraction': 0.9208175607947006,

    'feature_fraction': 0.7512774855381017,

    'learning_rate': 0.01,

    'max_depth': 17,

    'min_child_weight': 49.90252232871143,

    'min_split_gain': 0.021286271431087116,

    'num_leaves': 76,

    'reg_alpha': 0.043051815788046566,

    'reg_lambda': 0.07902389530425666,

    'subsample': 0.9253685895179343,

    'verbosity': -1,

    'data_random_seed': 17

}



# Additional parameters:

early_stop = 500



verbose_eval = 1000

num_rounds = 10000

n_splits = 5
oof_train_lgb = np.zeros((X_train.shape[0]))

oof_test_lgb = np.zeros((X_test.shape[0], n_splits))

qwk_scores = []

oof_coefficients_lgb = []



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)



for i, (f_ind, outf_ind) in enumerate(kf.split(X_train, y_train.values)):



    X_tr = X_train.loc[f_ind].copy()

    X_val = X_train.loc[outf_ind].copy()



    y_tr, y_val = y_train[f_ind], y_train[outf_ind]

    

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

    oof_coefficients_lgb.append(coefficients)

    

    pred_val_y_k = optR.predict(val_pred, coefficients)

    print("Valid Counts = ", Counter(y_val))

    print("Predicted Counts = ", Counter(pred_val_y_k))

    print("Coefficients = ", coefficients)

    qwk = cohen_kappa_score(y_val, pred_val_y_k, weights='quadratic')

    qwk_scores.append(qwk)

    print("QWK = ", qwk)

    

    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    

    oof_train_lgb[outf_ind] = val_pred

    oof_test_lgb[:, i] = test_pred

    

print('{} cv mean QWK score : {}, coeffs {}'.format('LGBM', np.mean(qwk_scores)

                                                    ,np.mean(np.array(oof_coefficients_lgb), axis=0)))
optR = OptimizedRounder()

mean_coeffs = np.mean(np.array(oof_coefficients_lgb), axis=0)

valid_pred = optR.predict(oof_train_lgb, mean_coeffs)

qwk = cohen_kappa_score(y_train.values, valid_pred, weights='quadratic')

print("QWK = ", qwk)
mean_coeff_m = (np.array(oof_coefficients_lgb) + np.array(oof_coefficients)) / 2
optR = OptimizedRounder()

mean_coeffs = np.mean(np.array(mean_coeff_m), axis=0)

valid_pred = optR.predict((oof_train_lgb + oof_train) / 2, mean_coeffs)

qwk = cohen_kappa_score(y_train.values, valid_pred, weights='quadratic')

print("QWK = ", qwk)
train_predictions = optR.predict((oof_train_lgb + oof_train) / 2, mean_coeffs).astype(np.int8)

print(f'train pred distribution: {Counter(train_predictions)}')

test_predictions = optR.predict((oof_test_lgb.mean(axis=1) + oof_test.mean(axis=1)) / 2, mean_coeffs).astype(np.int8)

print(f'test pred distribution: {Counter(test_predictions)}')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv', usecols=['PetID'])



submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})

submission.to_csv('submission.csv', index=False)

submission.head()