import pandas as pd
import numpy as np
# get data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#extract ID
ids = test_data.values[:,0]

#drop columns
to_delete = ['ID']

train_data.drop(to_delete, axis=1, inplace=True)
test_data.drop(to_delete, axis=1, inplace=True)
#numerical & categorical features
numerical = train_data._get_numeric_data().columns
categorical = [item for item in train_data.columns if item not in numerical]
categorical_indexes = [train_data.columns.get_loc(x) for x in categorical] 
categorical
#LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for x in categorical:
    train_data[x] = le.fit_transform(train_data[x])
    test_data[x] = le.fit_transform(test_data[x])
