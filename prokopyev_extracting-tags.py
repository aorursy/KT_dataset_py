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



from datetime import datetime



import warnings

warnings.filterwarnings('ignore')



# Set seed for reproducibility. I chose '1337' for good luck. See details about 1337-speak on KnowYourMeme.



myseed = 42

np.random.seed(myseed)
# Load in data

train = pd.read_csv('../input/mldub-comp1/train_data.csv')

test = pd.read_csv('../input/mldub-comp1/test_data.csv')

sample_sub = pd.read_csv('/kaggle/input/mldub-comp1/sample_sub.csv')
# Join training and testing features

train_features = train.drop(['target_variable'], axis=1)

test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)
# Create features from tags

from sklearn.feature_extraction.text import CountVectorizer

corpus = features['tags'].astype('U').values



vectorizer = CountVectorizer(min_df = 0.05) # Setting a minumum % of "documents" to contain each "tag"

tags = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())



tags = pd.DataFrame(tags.toarray(),index=features.index)

tags.columns = vectorizer.get_feature_names()

tags = tags.add_prefix('tags_')

# tags
# Add tags to our dataframe with features

features = pd.concat([features, tags], ignore_index=False, axis=1, sort=False).reset_index(drop=True)
# Enjoy new features

quantitative = [f for f in features.columns if features.dtypes[f] != 'object']

qualitative = [f for f in features.columns if features.dtypes[f] == 'object']

print("quantitative \n", quantitative)

print("qualitative \n", qualitative)