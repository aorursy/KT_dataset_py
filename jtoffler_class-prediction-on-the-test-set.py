import pandas as pd

import numpy as np

import os

import pickle

import gc

import xgboost as xgb

import numpy as np

import re

import pandas as pd

from sklearn.model_selection import train_test_split



max_num_features = 10

pad_size = 1

boundary_letter = -1

space_letter = 0

#max_data_size = 320000



out_path = r'.'

df = pd.read_csv(r'en_test_2.csv')



x_data = []

#y_data =  pd.factorize(df['class'])

#labels = y_data[1]

#y_data = y_data[0]

gc.collect()

for x in df['before'].values:

    x_row = np.ones(max_num_features, dtype=int) * space_letter

    for xi, i in zip(list(str(x)), np.arange(max_num_features)):

        x_row[i] = ord(xi)

    x_data.append(x_row)



def context_window_transform(data, pad_size):

    pre = np.zeros(max_num_features)

    pre = [pre for x in np.arange(pad_size)]

    data = pre + data + pre

    neo_data = []

    for i in np.arange(len(data) - pad_size * 2):

        row = []

        for x in data[i : i + pad_size * 2 + 1]:

            row.append([boundary_letter])

            row.append(x)

        row.append([boundary_letter])

        neo_data.append([int(x) for y in row for x in y])

    return neo_data



#x_data = x_data[:max_data_size]

#y_data = y_data[:max_data_size]

x_data = np.array(context_window_transform(x_data, pad_size))

gc.collect()

x_data = np.array(x_data)

#y_data = np.array(y_data)



print('Total number of samples:', len(x_data))

#print('Use: ', max_data_size)

#x_data = np.array(x_data)

#y_data = np.array(y_data)



print('x_data sample:')

print(x_data[0])

#print('y_data sample:')

#print(y_data[0])

#print('labels:')

#print(labels)



model = xgb.Booster({'nthread': 4})

model.load_model('xgb_model')



labels = [u'PLAIN', u'PUNCT', u'DATE', u'LETTERS', u'CARDINAL', u'VERBATIM',

       u'DECIMAL', u'MEASURE', u'MONEY', u'ORDINAL', u'TIME', u'ELECTRONIC',

       u'DIGIT', u'FRACTION', u'TELEPHONE', u'ADDRESS']



dtest = xgb.DMatrix(x_data)

preds = model.predict(dtest)

preds2 = [labels[int(x)] for x in preds]

preds3 = np.array(preds2)

preds4 = pd.DataFrame(preds3.reshape(len(preds3), 1))

preds4.to_csv(os.path.join(out_path, 'classes.csv'))


