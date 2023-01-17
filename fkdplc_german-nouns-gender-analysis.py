import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt

import seaborn as sns



file_name = '/kaggle/input/german-words-genders/deutsch.csv'



data = pd.read_csv(file_name)

data.sample(frac=1)
data.isna().sum()
ohe = pd.get_dummies(data['gender'])

data = pd.concat([data, ohe], axis=1)

data.drop(columns=['gender'], inplace=True)



data.columns
masc_cnt = data['m'].sum()

fem_cnt = data['f'].sum()

neut_cnt = data['n'].sum()



plt.figure(1, figsize=(6, 6))

plt.pie([masc_cnt, fem_cnt, neut_cnt], labels=['masculine', 'feminine', 'neutral'], autopct='%.1f%%')

plt.show()
from sklearn.metrics import accuracy_score



def acc(y_true_ohe, y_pred_ohe):

    y_true = np.argmax(y_true_ohe, axis=1)

    y_pred = np.argmax(y_pred_ohe, axis=1)

    return accuracy_score(y_true, y_pred)



dummy_pred = np.zeros((data.shape[0], 3))

dummy_pred[:, 0] = 1

print(f'Dummy majority: score = {acc(dummy_pred, data[["f", "m", "n"]].to_numpy()):.3f}')



guess_pred = np.zeros((data.shape[0], 3))

for i in range(guess_pred.shape[0]):

    pos = np.random.randint(3)

    guess_pred[i, pos] = 1

print(f'Random guesser: score = {acc(guess_pred, data[["f", "m", "n"]].to_numpy()):.3f}')
data['word'] = data['word'].apply(lambda word: word.lower())



data['word'][np.random.randint(data.shape[0])]
data['word_len'] = data['word'].apply(lambda word: len(word))



min_len = data['word_len'].min()

max_len = data['word_len'].max()

lens = [i for i in range(min_len, max_len + 1)]



masc_cnt_per_len = []

fem_cnt_per_len = []

neut_cnt_per_len = []

total_cnt_per_len = []



for word_len in lens:

    d = data[data['word_len'] == word_len]

    

    total = d.shape[0]

    total_cnt_per_len.append(total)

    masc_cnt_per_len.append(d['m'].sum() / total)

    fem_cnt_per_len.append(d['f'].sum() / total)

    neut_cnt_per_len.append(d['n'].sum() / total)



fig = plt.figure(1, figsize=(16, 6))

ax = fig.add_axes([0, 0, 1, 1])

ax.bar(lens, masc_cnt_per_len, 1, color='r')

ax.bar(lens, fem_cnt_per_len, 1, bottom=masc_cnt_per_len, color='b')

ax.bar(lens, neut_cnt_per_len, 1, bottom=[masc_cnt_per_len[i] + fem_cnt_per_len[i] for i in range(len(lens))], color='g')

ax.set_ylabel('word lengths')

ax.set_xticks(lens)

ax.set_yticks([0, .2, .4, .6, .8, 1])

ax.legend(labels=['masculine', 'feminine', 'neutral'])

plt.legend()

plt.show()
for i in range(5):

    ln = i + 1

    data[f'postfix_{ln}'] = data['word'].apply(lambda word: word[-ln:] if len(word) >= ln else word)



postfix_freq_threshold = 1000



frequent_postfixes = []

for i in range(5):

    postfixes, counts = np.unique(data[f'postfix_{i + 1}'], return_counts=True)

    freq = np.argsort(counts)

    counts = counts[freq]

    postfixes = postfixes[freq]

    counts = counts[counts > postfix_freq_threshold]

    postfixes = postfixes[-counts.shape[0]:]

    frequent_postfixes += list(postfixes)



print(frequent_postfixes)
postfix_distrib = {}

for postfix in frequent_postfixes:

    tmp = data[data[f'postfix_{len(postfix)}'] == postfix]

    total = tmp.shape[0]

    

    masc_cnt = tmp['m'].sum() / total

    fem_cnt = tmp['f'].sum() / total

    neut_cnt = tmp['n'].sum() / total



    mx = max(masc_cnt, fem_cnt, neut_cnt)

    postfix_distrib[postfix] = {'m': masc_cnt, 'f': fem_cnt, 'n': neut_cnt, 'total': total, 'max': mx}



postfix_distrib = {k: v for k, v in sorted(postfix_distrib.items(), key=lambda item: -item[1]['max'])}

for postfix, distrib in postfix_distrib.items():

    print(f'{postfix}: m: {distrib["m"]:.3f}, f: {distrib["f"]:.3f}, n: {distrib["n"]:.3f} -- {distrib["total"]}')
for postfix in frequent_postfixes:

    data[f'postfix_{postfix}'] = data[f'postfix_{len(postfix)}'].apply(lambda post: 1 if post == postfix else 0)



data.drop(columns=[f'postfix_{i+1}' for i in range(5)], inplace=True)

data.columns
target = data[['f', 'm', 'n']]

data.drop(columns=['word', 'f', 'm', 'n'], inplace=True)



data.shape, target.shape
from sklearn.decomposition import PCA



pca = PCA(n_components=0.995, random_state=289)

pca.fit(data)



data_pca = pca.transform(data)

data_pca.shape
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



data_s = StandardScaler().fit_transform(data)



model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(data.shape[1], )))

model.add(Dropout(0.5))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))



model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['acc']

)



x_train, x_test, y_train, y_test = train_test_split(data_s, target, test_size=0.2, random_state=289)



rlr = ReduceLROnPlateau(patience=2, verbose=1)

es = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)



history = model.fit(

    x_train,

    y_train,

    epochs=30,

    batch_size=64,

    verbose=0,

    callbacks=[rlr, es],

    validation_data=(x_test, y_test)

)
h = history.history

epochs = range(len(h['loss']))



plt.figure(1, figsize=(20, 6))



plt.subplot(121)

plt.xlabel('epochs')

plt.ylabel('loss')

plt.plot(epochs, h['loss'], label='train')

plt.plot(epochs, h['val_loss'], label='val')

plt.legend()



plt.subplot(122)

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.plot(h[f'acc'], label='train')

plt.plot(h[f'val_acc'], label='val')

plt.legend()



plt.show()