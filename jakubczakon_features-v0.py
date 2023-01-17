import pandas as pd



train = pd.read_csv('../input/baseline-features/train_features.csv')

test = pd.read_csv('../input/baseline-features/test_features.csv')



train.to_csv('train_features_v0.csv', index=None)

test.to_csv('test_features_v0.csv', index=None)