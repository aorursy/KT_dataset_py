import pandas as pd

import numpy as np

import os

import pickle



max_num_features = 20



out_path = r'.'

df = pd.read_csv(r'../input/en_train.csv')

exclude_classes = ['PLAIN', 'VERBATIM', 'LETTERS', 'PUNCT']

df = df.loc[df['class'].isin(exclude_classes) == False]
max_size = 200000

x_data = []

y_data = pd.factorize(df['class'])

labels = y_data[1]

y_data = y_data[0]

for x in df['before'].values:

    x_row = np.zeros(max_num_features, dtype=int)

    for xi, i in zip(list(str(x)), np.arange(max_num_features)):

        x_row[i] = ord(xi) - ord('a')

    x_data.append(x_row)



print('Total number of samples:', len(x_data))

print('Use: ', max_size)

#x_data = np.array(x_data)

#y_data = np.array(y_data)

x_data = np.array(x_data[:max_size])

y_data = np.array(y_data[:max_size])



print('x_data sample:')

print(x_data[0])

print('y_data sample:')

print(y_data[0])

print('labels:')

print(labels)



del df
import xgboost as xgb

import numpy as np

import pickle

import os

import re

import pandas as pd

from sklearn.model_selection import train_test_split



out_path = r'.'



x_train = x_data

y_train = y_data

del x_data

del y_data



x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train,

                                                      test_size=0.1, random_state=2017)

num_class = len(labels)

dtrain = xgb.DMatrix(x_train, label=y_train)

dvalid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(dvalid, 'valid'), (dtrain, 'train')]



param = {'objective':'multi:softmax',

         'eta':'0.3', 'max_depth':10,

         'silent':1, 'nthread':-1,

         'num_class':num_class,

         'eval_metric':'merror'}

model = xgb.train(param, dtrain, 60, watchlist, early_stopping_rounds=20,

                  verbose_eval=10)
pred = model.predict(dvalid)

pred = [labels[int(x)] for x in pred]

y_valid = [labels[x] for x in y_valid]

x_valid = [ [ chr(x + ord('a')) for x in y] for y in x_valid]

x_valid = [''.join(x) for x in x_valid]

x_valid = [re.sub('a+$', '', x) for x in x_valid]



df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])

df_pred['data'] = x_valid

df_pred['predict'] = pred

df_pred['target'] = y_valid

df_pred.to_csv(os.path.join(out_path, 'pred.csv'))



df_errors = df_pred.loc[df_pred['predict'] != df_pred['target']]

df_errors.to_csv(os.path.join(out_path, 'errors.csv'))



model.save_model(os.path.join(out_path, 'xgb_model'))
df_pred[:10]
df_errors[:10]