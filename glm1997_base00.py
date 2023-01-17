import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression



import scikitplot as skplt

import matplotlib

import matplotlib.pyplot as plt



from imblearn.over_sampling import SMOTE

from itertools import groupby



import pickle

import os
# print(os.listdir('../input/bases-rn/novas-bases/novas-bases'))

###################### CLASSE 01

pickle_off = open('../input/bases-rn/X1_train0.pickle', 'rb')

X1_train0 = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/base0/y1_train0.pickle', 'rb')

y1_train0 = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/X1_val0.pickle', 'rb')

X1_val0 = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/base0/y1_val0.pickle', 'rb')

y1_val0 = pickle.load(pickle_off)



###################### CLASSE 02

pickle_off = open('../input/bases-rn/X2_train0.pickle', 'rb')

X2_train0 = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/base0/y2_train0.pickle', 'rb')

y2_train0 = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/X2_val0.pickle', 'rb')

X2_val0 = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/base0/y2_val0.pickle', 'rb')

y2_val0 = pickle.load(pickle_off)



###################### TESTE



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/X_test.pickle', 'rb')

X_test = pickle.load(pickle_off)



pickle_off = open('../input/bases-rn/novas-bases/novas-bases/y_test.pickle', 'rb')

y_test = pickle.load(pickle_off)
df_X1_train = pd.DataFrame(data=X1_train0)

df_X1_train['243'] = y1_train0

df_X1_train.head()
df_X1_val = pd.DataFrame(data=X1_val0)

df_X1_val['243'] = y1_val0

df_X1_val.head()
df_X2_train = pd.DataFrame(data=X2_train0)

df_X2_train['243'] = y2_train0

df_X2_train.head()
df_X2_val = pd.DataFrame(data=X2_val0)

df_X2_val['243'] = y2_val0

df_X2_val.head()
print('####CLASSE 01####')

print('##TREINO')

print(df_X1_train.shape)

print('##VALIDAÇÃO')

print(df_X1_val.shape)

print()

print('####CLASSE 02####')

print('##TREINO')

print(df_X2_train.shape)

print('##VALIDAÇÃO')

print(df_X2_val.shape)
df_train = pd.concat([df_X1_train, df_X2_train])

df_val = pd.concat([df_X1_val, df_X2_val])
print('####CLASSE 01####')

print(df_train.shape)

print()

print('####CLASSE 02####')

print(df_val.shape)
df_train.head()
df_treino = df_train.sample(frac=1).reset_index(drop=True)

df_treino.head()
df_val.head()
df_validacao = df_val.sample(frac=1).reset_index(drop=True)

df_validacao.head()
X_train0 = df_treino.iloc[:, 0:-1].values

y_train0 = df_treino.iloc[:, -1].values



X_val0 = df_validacao.iloc[:, 0:-1].values

y_val0 = df_validacao.iloc[:, -1].values
print('## y_train0 ##')

print(pd.value_counts(y_train0, normalize=True))

print('## y_val0 ##')

print(pd.value_counts(y_val0, normalize=True))
scaler = MinMaxScaler()

X_train0 = scaler.fit_transform(X_train0)

X_val0 = scaler.fit_transform(X_val0)
smt = SMOTE(random_state=2, ratio = 1)

X_train0, y_train0 = smt.fit_sample(X_train0, y_train0)

X_val0, y_val0 = smt.fit_sample(X_val0, y_val0)
X_train0s = pickle.dump(X_train0, open('X_train0s.pickle', 'wb'))

y_train0s = pickle.dump(y_train0, open('y_train0s.pickle', 'wb'))

X_val0s = pickle.dump(X_val0, open('X_val0s.pickle', 'wb'))

y_val0s = pickle.dump(y_val0, open('y_val0s.pickle', 'wb'))
from IPython.display import HTML



def create_download_link(title = "Download CSV file", filename = "data.pickle"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
print('BASE 0')

create_download_link(filename='X_train0s.pickle')
create_download_link(filename='y_train0s.pickle')
create_download_link(filename='X_val0s.pickle')
create_download_link(filename='y_val0s.pickle')