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



from itertools import groupby



import pickle



import os
# print(os.listdir('../input/novas-bases/novas-bases/novas-bases/base4'))

###################### CLASSE 01

pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/X1_train4.pickle', 'rb')

X1_train4 = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/y1_train4.pickle', 'rb')

y1_train4 = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/X1_val4.pickle', 'rb')

X1_val4 = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/y1_val4.pickle', 'rb')

y1_val4 = pickle.load(pickle_off)



###################### CLASSE 02

pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/X2_train4.pickle', 'rb')

X2_train4 = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/y2_train4.pickle', 'rb')

y2_train4 = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/X2_val4.pickle', 'rb')

X2_val4 = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/base4/y2_val4.pickle', 'rb')

y2_val4 = pickle.load(pickle_off)



###################### TESTE



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/X_test.pickle', 'rb')

X_test = pickle.load(pickle_off)



pickle_off = open('../input/novas-bases/novas-bases/novas-bases/y_test.pickle', 'rb')

y_test = pickle.load(pickle_off)
df_classe1_treino = pd.DataFrame(data=X1_train4)

df_classe1_treino['243'] = y1_train4

df_classe1_treino.head()
df_classe1_validacao = pd.DataFrame(data=X1_val4)

df_classe1_validacao['243'] = y1_val4

df_classe1_validacao.head()
df_classe2_treino = pd.DataFrame(data=X2_train4)

df_classe2_treino['243'] = y2_train4

df_classe2_treino.head()
df_classe2_validacao = pd.DataFrame(data=X2_val4)

df_classe2_validacao['243'] = y2_val4

df_classe2_validacao.head()
print('####CLASSE 01####')

print('##TREINO')

print(df_classe1_treino.shape)

print('##VALIDAÇÃO')

print(df_classe1_validacao.shape)

print()

print('####CLASSE 02####')

print('##TREINO')

print(df_classe2_treino.shape)

print('##VALIDAÇÃO')

print(df_classe2_validacao.shape)
df_classe1_treino_copy = df_classe1_treino.copy()

df_classe1_train = pd.concat([df_classe1_treino, df_classe1_treino_copy])



df_classe1_validacao_copy = df_classe1_validacao.copy()

df_classe1_val = pd.concat([df_classe1_validacao, df_classe1_validacao_copy])
print('####CLASSE 01####')

print('##TREINO')

print(df_classe1_train.shape)

print('##VALIDAÇÃO')

print(df_classe1_val.shape)

print()

print('####CLASSE 02####')

print('##TREINO')

print(df_classe2_treino.shape)

print('##VALIDAÇÃO')

print(df_classe2_validacao.shape)
classe1_train = df_classe1_train[:127548]

classe1_val = df_classe1_val[:63775]
print('####CLASSE 01####')

print('##TREINO')

print(classe1_train.shape)

print('##VALIDAÇÃO')

print(classe1_val.shape)

print()

print('####CLASSE 02####')

print('##TREINO')

print(df_classe2_treino.shape)

print('##VALIDAÇÃO')

print(df_classe2_validacao.shape)
df_train_all = pd.concat([classe1_train, df_classe2_treino])

df_val_all = pd.concat([classe1_val, df_classe2_validacao])
df_train_all.head()
df_train = df_train_all.sample(frac=1).reset_index(drop=True)

df_train.head()
df_val_all.head()
df_val = df_val_all.sample(frac=1).reset_index(drop=True)

df_val.head()
X_train4 = df_train.iloc[:, 0:-1].values

y_train4 = df_train.iloc[:, -1].values



X_val4 = df_val.iloc[:, 0:-1].values

y_val4 = df_val.iloc[:, -1].values
print('## y_train4 ##')

print(pd.value_counts(y_train4, normalize=True))

print('## y_val4 ##')

print(pd.value_counts(y_val4, normalize=True))
scaler = MinMaxScaler()

X_train4 = scaler.fit_transform(X_train4)

X_val4 = scaler.fit_transform(X_val4)
X_train4s = pickle.dump(X_train4, open('X_train4s.pickle', 'wb'))

y_train4s = pickle.dump(y_train4, open('y_train4s.pickle', 'wb'))

X_val4s = pickle.dump(X_val4, open('X_val4s.pickle', 'wb'))

y_val4s = pickle.dump(y_val4, open('y_val4s.pickle', 'wb'))
from IPython.display import HTML



def create_download_link(title = "Download CSV file", filename = "data.pickle"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)

print('BASE 4')

create_download_link(filename='X_train4s.pickle')

create_download_link(filename='y_train4s.pickle')

create_download_link(filename='X_val4s.pickle')

create_download_link(filename='y_val4s.pickle')