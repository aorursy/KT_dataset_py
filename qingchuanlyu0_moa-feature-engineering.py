#import libraries

import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder

#onehot encoder only takes numerical values, so need label encoder first

from sklearn.preprocessing import LabelEncoder 

#need to standardization features before using PCA

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

from sklearn.decomposition import PCA
train_feature = pd.read_csv("../input/lish-moa/train_features.csv")

test_feature = pd.read_csv("../input/lish-moa/test_features.csv")

train_targets_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")

submision = pd.read_csv("../input/lish-moa/sample_submission.csv")
#shape of each given dataset

train_feature.shape, test_feature.shape, train_targets_scored.shape, train_targets_nonscored.shape, submision.shape
train_feature.head()
train_targets_scored.head()
train_targets_scored[train_targets_scored['11-beta-hsd1_inhibitor']>0]
#missing values

train_feature[train_feature['g-1'].isnull()==True]
train_feature['cp_type']
#With 876 training features and 402 additional training features, I need to reduce dimensions: PCA

#First I need to scale data

#Before that, I need to encode categorical string data: cp_type and cp_dose both only have two categories. I can use one-hot encoder

### integer mapping using LabelEncoder

label_encoder = LabelEncoder()

onehot_encoder = OneHotEncoder(sparse=False)

for cat in ('cp_type', 'cp_dose'):

    for s in (train_feature, test_feature):

        #label enoding

        integer_encoded = label_encoder.fit_transform(s[cat])

        #print(integer_encoded)

        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        ### One hot encoding

    

        onehot_feature = onehot_encoder.fit_transform(integer_encoded)

        s[cat]=onehot_feature
onehot_feature
train_feature['cp_type']
test_feature['cp_dose']
#drop id column "sig_id" to prepare for standardization scale

train_feature=train_feature.drop(['sig_id'], axis=1)

train_feature.head()

test_feature=test_feature.drop(['sig_id'], axis=1)
#Fit standard scale on training features only.

scaler.fit(train_feature)

#Apply transform to both the training features and the test features.

train_scl = scaler.transform(train_feature)

test_scl = scaler.transform(test_feature)
#Make an instance of the Model

pca = PCA(.95)

#Fit PCA on the scaled training features

pca.fit(train_scl)

#Apply the mapping (transform) to both the scaled training features and the scaled test features

train_pca = pca.transform(train_scl)

test_pca = pca.transform(test_scl)
train_pca.shape

#got rid of about 300 features!