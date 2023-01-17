# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import json
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
np.random.seed = 42
breeds_df = pd.read_csv('../input/petfinder-adoption-prediction/BreedLabels.csv')
colors_df = pd.read_csv('../input/petfinder-adoption-prediction/ColorLabels.csv')
states_df = pd.read_csv('../input/petfinder-adoption-prediction/StateLabels.csv')
train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
img_size = 256
batch_size = 256
pet_ids = train_df['PetID'].values
n_batches = len(pet_ids) // batch_size + 1
from keras.applications.densenet import preprocess_input, DenseNet121
from tqdm import tqdm, tqdm_notebook
def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
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
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
inp = Input((256,256,3))
backbone = DenseNet121(input_tensor = inp,
                       weights='../input/petfinder-densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)
m = Model(inp,out)
features = []
for b in range(n_batches):
    start = b * batch_size
    end = (b + 1) * batch_size
    batch_pets = pet_ids[start: end]
    batch_images= np.zeros((len(batch_pets), img_size, img_size, 3))
    for i,pet_id in enumerate(batch_pets):
        try:
            batch_images[i]= load_image("../input/petfinder-adoption-prediction/train_images/", pet_id)
        except:
            pass
    batch_preds= m.predict(batch_images)
    for i, pet_id in enumerate(batch_pets):
        features.append([pet_id] + list(batch_preds[i]))

X = pd.DataFrame(features, columns=["PetID"] + ["dense121_2_{}".format(i) for i in
                                                        range(batch_preds.shape[1])])
TRAIN_METADATA_PATH = '../input/petfinder-adoption-prediction/train_metadata/'
TRAIN_SENTIMENT_PATH = '../input/petfinder-adoption-prediction/train_sentiment/'
TEST_METADATA_PATH = '../input/petfinder-adoption-prediction/test_metadata/'
TEST_SENTIMENT_PATH = '../input/petfinder-adoption-prediction/test_sentiment/'
sentiment_dict= {}
for filename in os.listdir(TRAIN_SENTIMENT_PATH):
    with open(TRAIN_SENTIMENT_PATH + filename, 'r') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}

    salience = [float(x['salience']) for x in sentiment['entities'] if 'salience' in x]
    if salience:
        sentiment_dict[pet_id]['entities_salience_var'] = np.var(salience)
        sentiment_dict[pet_id]['entities_salience_sum'] = np.sum(salience)
        sentiment_dict[pet_id]['entities_salience_mean'] = np.mean(salience)
        sentiment_dict[pet_id]['entities_salience_min'] = np.min(salience)
        sentiment_dict[pet_id]['entities_salience_max'] = np.max(salience)
    else:
        sentiment_dict[pet_id]['entities_salience_var'] = 0
        sentiment_dict[pet_id]['entities_salience_sum'] = 0
        sentiment_dict[pet_id]['entities_salience_mean'] = 0
        sentiment_dict[pet_id]['entities_salience_min'] = 0
        sentiment_dict[pet_id]['entities_salience_max'] = 0

    file_sentiment = ([x['sentiment'] for x in sentiment['sentences'] if 'sentiment' in x])
    magnitude = ([float(x['magnitude']) for x in file_sentiment if 'magnitude' in x])
    score = ([float(x['score']) for x in file_sentiment if 'score' in x])
    
    sentiment_dict[pet_id]['magnitude_var'] = np.var(magnitude)
    sentiment_dict[pet_id]['magnitude_sum'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['magnitude_mean'] = np.mean(magnitude)
    sentiment_dict[pet_id]['magnitude_min'] = np.min(magnitude)
    sentiment_dict[pet_id]['magnitude_max'] = np.max(magnitude)
    
    sentiment_dict[pet_id]['score_var'] = np.var(score)
    sentiment_dict[pet_id]['score_sum'] = np.sum(score)
    sentiment_dict[pet_id]['score_mean'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['score_min'] = np.min(score)
    sentiment_dict[pet_id]['score_max'] = np.max(score)
    
    sentiment_dict[pet_id]['lang'] = sentiment['language']

train_sentiment_df = pd.DataFrame()
train_sentiment_df['PetID'] = train_df['PetID']

train_sentiment_df['magnitude_var'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_var'] if x in sentiment_dict else 0)
train_sentiment_df['magnitude_sum'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_sum'] if x in sentiment_dict else 0)
train_sentiment_df['magnitude_mean'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_mean'] if x in sentiment_dict else 0)
train_sentiment_df['magnitude_min'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_min'] if x in sentiment_dict else 0)
train_sentiment_df['magnitude_max'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_max'] if x in sentiment_dict else 0)

train_sentiment_df['score_var'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['score_var'] if x in sentiment_dict else 0)
train_sentiment_df['score_sum'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['score_sum'] if x in sentiment_dict else 0)
train_sentiment_df['score_mean'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['score_mean'] if x in sentiment_dict else 0)
train_sentiment_df['score_min'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['score_min'] if x in sentiment_dict else 0)
train_sentiment_df['score_max'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['score_max'] if x in sentiment_dict else 0)

train_sentiment_df['entities_salience_var'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_var'] if x in sentiment_dict else 0)
train_sentiment_df['entities_salience_sum'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_sum'] if x in sentiment_dict else 0)
train_sentiment_df['entities_salience_mean'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_mean'] if x in sentiment_dict else 0)
train_sentiment_df['entities_salience_min'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_min'] if x in sentiment_dict else 0)
train_sentiment_df['entities_salience_max'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_max'] if x in sentiment_dict else 0)

train_sentiment_df['lang'] = train_df['PetID'].apply(lambda x: sentiment_dict[x]['lang'] if x in sentiment_dict else 0)

train_sentiment_df.set_index('PetID', inplace=True)
sentiment_dict= {}
for filename in os.listdir(TEST_SENTIMENT_PATH):
    with open(TEST_SENTIMENT_PATH + filename, 'r') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}

    salience = ([float(x['salience']) for x in sentiment['entities'] if 'salience' in x])
    if salience:
        sentiment_dict[pet_id]['entities_salience_var'] = np.var(salience)
        sentiment_dict[pet_id]['entities_salience_sum'] = np.sum(salience)
        sentiment_dict[pet_id]['entities_salience_mean'] = np.mean(salience)
        sentiment_dict[pet_id]['entities_salience_min'] = np.min(salience)
        sentiment_dict[pet_id]['entities_salience_max'] = np.max(salience)
    else:
        sentiment_dict[pet_id]['entities_salience_var'] = 0
        sentiment_dict[pet_id]['entities_salience_sum'] = 0
        sentiment_dict[pet_id]['entities_salience_mean'] = 0
        sentiment_dict[pet_id]['entities_salience_min'] = 0
        sentiment_dict[pet_id]['entities_salience_max'] = 0

    file_sentiment = [x['sentiment'] for x in sentiment['sentences'] if 'sentiment' in x]
    magnitude = [float(x['magnitude']) for x in file_sentiment if 'magnitude' in x]
    score = [float(x['score']) for x in file_sentiment if 'score' in x]
    
    sentiment_dict[pet_id]['magnitude_var'] = np.var(magnitude)
    sentiment_dict[pet_id]['magnitude_sum'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['magnitude_mean'] = np.mean(magnitude)
    sentiment_dict[pet_id]['magnitude_min'] = np.min(magnitude)
    sentiment_dict[pet_id]['magnitude_max'] = np.max(magnitude)
    
    sentiment_dict[pet_id]['score_var'] = np.var(score)
    sentiment_dict[pet_id]['score_sum'] = np.sum(score)
    sentiment_dict[pet_id]['score_mean'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['score_min'] = np.min(score)
    sentiment_dict[pet_id]['score_max'] = np.max(score)
    
    sentiment_dict[pet_id]['lang'] = sentiment['language']

test_sentiment_df = pd.DataFrame()
test_sentiment_df['PetID'] = test_df['PetID']

test_sentiment_df['magnitude_var'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_var'] if x in sentiment_dict else 0)
test_sentiment_df['magnitude_sum'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_sum'] if x in sentiment_dict else 0)
test_sentiment_df['magnitude_mean'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_mean'] if x in sentiment_dict else 0)
test_sentiment_df['magnitude_min'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_min'] if x in sentiment_dict else 0)
test_sentiment_df['magnitude_max'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['magnitude_max'] if x in sentiment_dict else 0)

test_sentiment_df['score_var'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['score_var'] if x in sentiment_dict else 0)
test_sentiment_df['score_sum'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['score_sum'] if x in sentiment_dict else 0)
test_sentiment_df['score_mean'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['score_mean'] if x in sentiment_dict else 0)
test_sentiment_df['score_min'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['score_min'] if x in sentiment_dict else 0)
test_sentiment_df['score_max'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['score_max'] if x in sentiment_dict else 0)

test_sentiment_df['entities_salience_var'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_var'] if x in sentiment_dict else 0)
test_sentiment_df['entities_salience_sum'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_sum'] if x in sentiment_dict else 0)
test_sentiment_df['entities_salience_mean'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_mean'] if x in sentiment_dict else 0)
test_sentiment_df['entities_salience_min'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_min'] if x in sentiment_dict else 0)
test_sentiment_df['entities_salience_max'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['entities_salience_max'] if x in sentiment_dict else 0)

test_sentiment_df['lang'] = test_df['PetID'].apply(lambda x: sentiment_dict[x]['lang'] if x in sentiment_dict else 0)

test_sentiment_df.set_index('PetID', inplace=True)
metadata_dict = {}

for filename in os.listdir(TRAIN_METADATA_PATH):
    with open(TRAIN_METADATA_PATH + filename, 'r') as f:
        metadata = json.load(f)
    pet_id = filename.split('-')[0]
    filename = filename.split('.')[0]
    if filename.endswith('1') == False:
        continue
    metadata_dict[pet_id] = {}
    if 'labelAnnotations' in metadata:
        annot_score = [float(x['score']) for x in metadata['labelAnnotations'] if 'score' in x]
    else:
        annot_score = []        
    if 'imagePropertiesAnnotation'in metadata:
        colors = metadata['imagePropertiesAnnotation']['dominantColors']['colors']
        color_score = [float(x['score']) for x in colors if 'score' in x]
        pixel_frac = [float(x['pixelFraction']) for x in colors if 'pixelFraction' in x]
    else:
        color_score = []
        pixel_frac = []
    if 'cropHintsAnnotation' in metadata:
        crop_hints = metadata['cropHintsAnnotation']['cropHints']
        crop_confidence = [x['confidence'] for x in crop_hints if 'confidence' in x]
    else:
        crop_confidence = []

    if len(annot_score) > 0:
        metadata_dict[pet_id]['annot_score_var'] = np.var(annot_score)
        metadata_dict[pet_id]['annot_score_sum'] = sum(annot_score)
        metadata_dict[pet_id]['annot_score_mean'] = np.mean(annot_score)
        metadata_dict[pet_id]['annot_score_min'] = min(annot_score)
        metadata_dict[pet_id]['annot_score_max'] = max(annot_score)
    else:
        metadata_dict[pet_id]['annot_score_var'] = 0
        metadata_dict[pet_id]['annot_score_sum'] = 0
        metadata_dict[pet_id]['annot_score_mean'] = 0
        metadata_dict[pet_id]['annot_score_min'] = 0
        metadata_dict[pet_id]['annot_score_max'] = 0

    if len(color_score) > 0:
        metadata_dict[pet_id]['color_score_var'] = np.var(color_score)
        metadata_dict[pet_id]['color_score_sum'] = sum(color_score)
        metadata_dict[pet_id]['color_score_mean'] = np.mean(color_score)
        metadata_dict[pet_id]['color_score_min'] = min(color_score)
        metadata_dict[pet_id]['color_score_max'] = max(color_score)
    else:
        metadata_dict[pet_id]['color_score_var'] = 0
        metadata_dict[pet_id]['color_score_sum'] = 0
        metadata_dict[pet_id]['color_score_mean'] = 0
        metadata_dict[pet_id]['color_score_min'] = 0
        metadata_dict[pet_id]['color_score_max'] = 0

    if len(pixel_frac) > 0:
        metadata_dict[pet_id]['pixel_frac_var'] = np.var(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_sum'] = sum(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_mean'] = np.mean(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_min'] = min(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_max'] = max(pixel_frac)
    else:
        metadata_dict[pet_id]['pixel_frac_var'] = 0
        metadata_dict[pet_id]['pixel_frac_sum'] = 0
        metadata_dict[pet_id]['pixel_frac_mean'] = 0
        metadata_dict[pet_id]['pixel_frac_min'] = 0
        metadata_dict[pet_id]['pixel_frac_max'] = 0
    
    metadata_dict[pet_id]['crop_confidence'] = crop_confidence[0]

train_metadata_df = pd.DataFrame()
train_metadata_df['PetID'] = train_df['PetID']

train_metadata_df['annot_score_var'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_var'] if x in metadata_dict else 0)
train_metadata_df['annot_score_sum'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_sum'] if x in metadata_dict else 0)
train_metadata_df['annot_score_mean'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_mean'] if x in metadata_dict else 0)
train_metadata_df['annot_score_min'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_min'] if x in metadata_dict else 0)
train_metadata_df['annot_score_max'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_max'] if x in metadata_dict else 0)

train_metadata_df['color_score_var'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_var'] if x in metadata_dict else 0)
train_metadata_df['color_score_sum'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_sum'] if x in metadata_dict else 0)
train_metadata_df['color_score_mean'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_mean'] if x in metadata_dict else 0)
train_metadata_df['color_score_min'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_min'] if x in metadata_dict else 0)
train_metadata_df['color_score_max'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_max'] if x in metadata_dict else 0)

train_metadata_df['pixel_frac_var'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_var'] if x in metadata_dict else 0)
train_metadata_df['pixel_frac_sum'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_sum'] if x in metadata_dict else 0)
train_metadata_df['pixel_frac_mean'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_mean'] if x in metadata_dict else 0)
train_metadata_df['pixel_frac_min'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_min'] if x in metadata_dict else 0)
train_metadata_df['pixel_frac_max'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_max'] if x in metadata_dict else 0)

train_metadata_df['crop_confidence'] = train_df['PetID'].apply(lambda x: metadata_dict[x]['crop_confidence'] if x in metadata_dict else 0)

train_metadata_df.set_index('PetID', inplace=True)
train_metadata_df.head()
metadata_dict = {}

for filename in os.listdir(TEST_METADATA_PATH):
    with open(TEST_METADATA_PATH + filename, 'r') as f:
        metadata = json.load(f)
    pet_id = filename.split('-')[0]
    filename = filename.split('.')[0]
    if filename.endswith('1') == False:
        continue
    metadata_dict[pet_id] = {}
    if 'labelAnnotations' in metadata:
        annot_score = [float(x['score']) for x in metadata['labelAnnotations'] if 'score' in x]
    else:
        annot_score = []        
    if 'imagePropertiesAnnotation'in metadata:
        colors = metadata['imagePropertiesAnnotation']['dominantColors']['colors']
        color_score = [float(x['score']) for x in colors if 'score' in x]
        pixel_frac = [float(x['pixelFraction']) for x in colors if 'pixelFraction' in x]
    else:
        color_score = []
        pixel_frac = []
    if 'cropHintsAnnotation' in metadata:
        crop_hints = metadata['cropHintsAnnotation']['cropHints']
        crop_confidence = [x['confidence'] for x in crop_hints if 'confidence' in x]
    else:
        crop_confidence = []

    if len(annot_score) > 0:
        metadata_dict[pet_id]['annot_score_var'] = np.var(annot_score)
        metadata_dict[pet_id]['annot_score_sum'] = sum(annot_score)
        metadata_dict[pet_id]['annot_score_mean'] = np.mean(annot_score)
        metadata_dict[pet_id]['annot_score_min'] = min(annot_score)
        metadata_dict[pet_id]['annot_score_max'] = max(annot_score)
    else:
        metadata_dict[pet_id]['annot_score_var'] = 0
        metadata_dict[pet_id]['annot_score_sum'] = 0
        metadata_dict[pet_id]['annot_score_mean'] = 0
        metadata_dict[pet_id]['annot_score_min'] = 0
        metadata_dict[pet_id]['annot_score_max'] = 0

    if len(color_score) > 0:
        metadata_dict[pet_id]['color_score_var'] = np.var(color_score)
        metadata_dict[pet_id]['color_score_sum'] = sum(color_score)
        metadata_dict[pet_id]['color_score_mean'] = np.mean(color_score)
        metadata_dict[pet_id]['color_score_min'] = min(color_score)
        metadata_dict[pet_id]['color_score_max'] = max(color_score)
    else:
        metadata_dict[pet_id]['color_score_var'] = 0
        metadata_dict[pet_id]['color_score_sum'] = 0
        metadata_dict[pet_id]['color_score_mean'] = 0
        metadata_dict[pet_id]['color_score_min'] = 0
        metadata_dict[pet_id]['color_score_max'] = 0

    if len(pixel_frac) > 0:
        metadata_dict[pet_id]['pixel_frac_var'] = np.var(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_sum'] = sum(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_mean'] = np.mean(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_min'] = min(pixel_frac)
        metadata_dict[pet_id]['pixel_frac_max'] = max(pixel_frac)
    else:
        metadata_dict[pet_id]['pixel_frac_var'] = 0
        metadata_dict[pet_id]['pixel_frac_sum'] = 0
        metadata_dict[pet_id]['pixel_frac_mean'] = 0
        metadata_dict[pet_id]['pixel_frac_min'] = 0
        metadata_dict[pet_id]['pixel_frac_max'] = 0
    
    metadata_dict[pet_id]['crop_confidence'] = crop_confidence[0]

test_metadata_df = pd.DataFrame()
test_metadata_df['PetID'] = test_df['PetID']
test_metadata_df['annot_score_var'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_var'] if x in metadata_dict else 0)
test_metadata_df['annot_score_sum'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_sum'] if x in metadata_dict else 0)
test_metadata_df['annot_score_mean'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_mean'] if x in metadata_dict else 0)
test_metadata_df['annot_score_min'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_min'] if x in metadata_dict else 0)
test_metadata_df['annot_score_max'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['annot_score_max'] if x in metadata_dict else 0)

test_metadata_df['color_score_var'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_var'] if x in metadata_dict else 0)
test_metadata_df['color_score_sum'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_sum'] if x in metadata_dict else 0)
test_metadata_df['color_score_mean'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_mean'] if x in metadata_dict else 0)
test_metadata_df['color_score_min'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_min'] if x in metadata_dict else 0)
test_metadata_df['color_score_max'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['color_score_max'] if x in metadata_dict else 0)

test_metadata_df['pixel_frac_var'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_var'] if x in metadata_dict else 0)
test_metadata_df['pixel_frac_sum'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_sum'] if x in metadata_dict else 0)
test_metadata_df['pixel_frac_mean'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_mean'] if x in metadata_dict else 0)
test_metadata_df['pixel_frac_min'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_min'] if x in metadata_dict else 0)
test_metadata_df['pixel_frac_max'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['pixel_frac_max'] if x in metadata_dict else 0)

test_metadata_df['crop_confidence'] = test_df['PetID'].apply(lambda x: metadata_dict[x]['crop_confidence'] if x in metadata_dict else 0)

test_metadata_df.set_index('PetID', inplace=True)
train_sentiment_metadata_df = pd.DataFrame()
train_sentiment_metadata_df['PetID'] = train_df['PetID']
train_sentiment_metadata_df.set_index('PetID', inplace=True)

train_sentiment_metadata_df = pd.merge(train_sentiment_metadata_df, train_metadata_df, how='left', on='PetID')
train_sentiment_metadata_df = pd.merge(train_sentiment_metadata_df, train_sentiment_df, how='left', on='PetID')
test_sentiment_metadata_df = pd.DataFrame()
test_sentiment_metadata_df['PetID'] = test_df['PetID']
test_sentiment_metadata_df.set_index('PetID', inplace=True)

test_sentiment_metadata_df = pd.merge(test_sentiment_metadata_df, test_metadata_df, how='left', on='PetID')
test_sentiment_metadata_df = pd.merge(test_sentiment_metadata_df, test_sentiment_df, how='left', on='PetID')
train_df = pd.merge(train_df, train_sentiment_metadata_df, how='left', on='PetID')
test_df = pd.merge(test_df, test_sentiment_metadata_df, how='left', on='PetID')
state_population = {
    41336:3.497,
    41325:2.072,
    41367:2.001,
    41401:1.808,
    41415:0.0993,
    41324:0.485,
    41332:1.098,
    41335:1.623,
    41330:2.447,
    41380:0.253,
    41327:0.222,
    41345:3.54,
    41342:2.619,
    41326:5.79,
    41361:1.125
}
state_gdp = {
    41336:36.394,
    41325:21.410,
    41367:13.668,
    41401:121.293,
    41415:74.337,
    41324:47.960,
    41332:43.047,
    41335:35.554,
    41330:30.303,
    41380:24.442,
    41327:52.937,
    41345:25.861,
    41342:52.301,
    41326:51.528,
    41361:30.216
}
train_df['state_population'] = train_df['State'].map(state_population)
train_df['state_gdp'] = train_df['State'].map(state_gdp)
train_df = pd.merge(train_df, X, how='left', on='PetID')
train_df.drop(train_df[train_df['Fee'] >= 1000].index, inplace = True)
train_df.drop(train_df[train_df['Age'] > 144].index, inplace = True)
train_df['Age'] = np.abs(stats.zscore(train_df['Age'], ddof=1))
train_df.drop(train_df[train_df['Age'] > 3].index, inplace = True)
train_df.drop(train_df[train_df['Age'] < -3].index, inplace = True)
train_df['Fee'] = np.abs(stats.zscore(train_df['Fee'], ddof=1))
train_df.drop(train_df[train_df['Fee'] > 3].index, inplace = True)
train_df.drop(train_df[train_df['Fee'] < -3].index, inplace = True)
for i in train_df.index:
    if train_df.at[i, 'Breed1'] == 0 and train_df.at[i, 'Breed2'] != 0:
        train_df.at[i, 'Breed1'] = train_df.at[i, 'Breed2']
    
    if train_df.at[i, 'Breed1'] == 307 and train_df.at[i, 'Breed2'] == 0:
        train_df.at[i, 'Breed2'] = train_df.at[i, 'Breed1']
y = pd.concat(g for _, g in train_df.groupby("Breed1") if len(g) <= 2)
x = pd.concat(g for _, g in train_df.groupby("Breed2") if len(g) <= 2)
temp_train = pd.concat([x, y])
temp_train = temp_train.drop_duplicates()
train_df.drop(['Name', 'PetID', 'RescuerID', 'Description', 'lang'], axis=1, inplace=True)
temp_train.drop(['Name', 'PetID', 'RescuerID', 'Description', 'lang'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(
    train_df.drop('AdoptionSpeed', axis=1), train_df['AdoptionSpeed'], test_size=0.20)
temp_df = pd.concat([temp_train.drop('AdoptionSpeed', axis=1), x_test])
temp_df = pd.concat([temp_df, temp_train.drop('AdoptionSpeed', axis=1)])
x_test = temp_df.drop_duplicates(keep=False)
temp_series = pd.Series(temp_train['AdoptionSpeed'], index=temp_train.index)
temp_series = temp_series.append(temp_series)
temp_series = temp_series.append(y_test)

y_test = temp_series[~temp_series.index.duplicated(keep=False)]
error = [x for x in y_test.index if x not in x_test.index]
y_test = y_test[~y_test.index.isin(error)]
temp_df = pd.concat([temp_train.drop('AdoptionSpeed', axis=1), x_train])
x_train = temp_df.drop_duplicates(keep='first')
temp_series = temp_series.append(y_train)

y_train = temp_series[~temp_series.index.duplicated(keep='first')]
error = [x for x in y_train.index if x not in x_train.index]
y_train = y_train[~y_train.index.isin(error)]
categorical_features = [
     'Breed1',
     'Breed2',
     'Color1',
     'Color2',
     'Color3',
     'Dewormed',
     'FurLength',
     'Gender',
     'Health',
     'MaturitySize',
     'State',
     'Sterilized',
     'Type',
     'Vaccinated'
]
numerical_features = [x for x in list(x_train.columns) if x not in categorical_features]
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
    


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(
    [
        ("select_numeric", DataFrameSelector(numerical_features)),
        ("imputer", SimpleImputer(strategy="median"))
    ]
)
num_pipeline.fit_transform(x_train)
from sklearn.preprocessing import OneHotEncoder

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

cat_pipeline = Pipeline(
    [
        ("select_cat", DataFrameSelector(categorical_features)),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ]
)
cat_pipeline.fit_transform(x_train)
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
x_train = preprocess_pipeline.fit_transform(x_train)
x_test = preprocess_pipeline.transform(x_test)
from sklearn.svm import SVC

svm_clf = SVC(C=2, gamma="auto", break_ties=True, random_state=777)
svm_clf.fit(x_train, y_train)
y_pred = svm_clf.predict(x_test)
print(classification_report(y_test, y_pred))
from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, x_train, y_train, cv=10)
svm_scores.mean()
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=777)
forest_clf.fit(x_train, y_train)
y_pred = forest_clf.predict(x_test)
print(classification_report(y_test, y_pred))
forest_scores = cross_val_score(forest_clf, x_train, y_train, cv=10)
forest_scores.mean()
from sklearn.tree import DecisionTreeClassifier


tree_clf = DecisionTreeClassifier(max_depth=9, random_state=777)
tree_clf.fit(x_train, y_train)
y_pred = tree_clf.predict(x_test)
tree_score = cross_val_score(tree_clf, x_train, y_train, cv=10)
tree_score.mean()
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
target_col = ['AdoptionSpeed']
predictors = list(set(list(train_df.columns)) - set(target_col))
train_df[predictors]
train_df.reset_index(drop=True, inplace=True)
train_df.info()
X = train_df[predictors].values
y = train_df[target_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_train.shape); print(X_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)
from keras.layers import Layer
from keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=len(predictors)))
model.add(Dense(100, activation='relu'))
model.add(RBFLayer(50, 0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=[keras.metrics.CategoricalAccuracy()])
model.fit(X_train, y_train, epochs=500)
pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1])) 
scores
