import numpy as np

import pandas as pd

from load_data import load_dataset_indonesian

from extract_feature import FeatureExtractor
train, test = load_dataset_indonesian(data_name='toxic')
FE = FeatureExtractor(model_name='xlm-r')
train['text'] = train['text'].apply(lambda x: FE.extract_features(x))

test['text'] = test['text'].apply(lambda x: FE.extract_features(x))

train.head()
np.save("train_text.npy", train['text'].values)

np.save("test_text.npy", test['text'].values)
train['label'].to_csv('train_label.csv', index=False, header=['label'])

test['label'].to_csv('test_label.csv', index=False, header=['label'])