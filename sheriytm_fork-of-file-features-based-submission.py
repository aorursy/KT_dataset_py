import matplotlib.pylab as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import os

CAT_COLUMN = 'type_cat'
# read in the file paths

train_df = pd.DataFrame([{'path': c_path, 

                           'image_name': os.path.basename(c_path),

                          CAT_COLUMN: os.path.basename(os.path.dirname(c_path))}

              for c_path in glob('../input/train/*/*')])

print('Total Training Data',train_df.shape[0])

print('Sample Summary\n', pd.value_counts(train_df['type_cat']))

train_df.sample(3)
test_df = pd.DataFrame([dict(path = c_path, 

                           image_name = os.path.basename(c_path)) 

              for c_path in glob('../input/test/*')])

print('Total Testing',test_df.shape[0])

test_df.sample(3)
from skimage.io import imread

def safe_image_read(in_path):

    try:

        return imread(in_path)

    except:

        return np.zeros((1,1)) 

def generate_feature_vector(in_df):

    current_df = in_df.copy()

    current_df['file_size'] = in_df['path'].map(lambda x: os.stat(x).st_size)

    current_df['creation_time'] = in_df['path'].map(lambda x: os.stat(x).st_ctime)

    current_df['pixel_count'] = in_df['path'].map(lambda x: np.prod(safe_image_read(x).shape))

    current_df['bits_per_pixel'] = current_df['file_size']/current_df['pixel_count']

    keep_cols = ['image_name', 

                 'file_size', 

                 'creation_time',

                 'pixel_count',

                 'bits_per_pixel',

                 CAT_COLUMN]

    return current_df[[ccol for ccol in current_df.columns if ccol in keep_cols]]
%%time

# generate the features for the training set

ftrain_df = generate_feature_vector(train_df)

# generate the features for the test set

ftest_df = generate_feature_vector(test_df)
ftrain_df.sample(3)
ftrain_df.head()
from tpot import TPOTClassifier

auto_classifier = TPOTClassifier(generations=2, population_size=8, verbosity=2)
y_train = ftrain_df[CAT_COLUMN]

x_train = ftrain_df[[ccol for ccol in ftrain_df.columns if ccol not in [CAT_COLUMN, 'image_name']]]

auto_classifier.fit(x_train, y_train)
x_test = ftest_df[[ccol for ccol in ftrain_df.columns if ccol not in [CAT_COLUMN, 'image_name']]]

# we need access to the pipeline to get the probabilities

test_prob = auto_classifier._fitted_pipeline.predict_proba(x_test)

guess_df = test_df[['image_name']]

for i, class_name in enumerate(auto_classifier._fitted_pipeline.classes_):

    guess_df[class_name] = test_prob[:,i]

guess_df.sample(3)
guess_df.to_csv('guess_03_24th.csv', index = False)