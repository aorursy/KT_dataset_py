from urllib.request import urlopen

from json import loads

import pandas as pd

from itertools import chain

from dask import bag

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
drink_df = pd.read_csv('../input/all_drinks.csv')

drink_df.sample(3)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer = 'char_wb')

cv.fit(drink_df['strDrink'].values)

new_vocab_dict = {id: word for word,id in cv.vocabulary_.items()}
cv_mat = cv.transform(drink_df['strDrink'].values)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
str_vec = drink_df['strDrink'].str.lower()

MAX_NB_WORDS, MAX_SEQUENCE_LENGTH = 5000, 30

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=True)

tokenizer.fit_on_texts(str_vec)

train_sequences = tokenizer.texts_to_sequences(str_vec)

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
plt.matshow(train_data)
def isempty(x):

    try:

        if x is None: 

            return True

        elif len(x)<1:

            return True

        else:

            return False

    except:

        # floating point nans

        return True

all_ingred = drink_df[[x for x in drink_df.columns 

                       if 'Ingredient' in x]].apply(lambda c_row: [v.lower() for k,v in c_row.items() if not isempty(v)],1)

all_ingred[0:3]
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

ingred_label = LabelEncoder()

ingred_label.fit(list(chain(*all_ingred.values)))

print('Found', len(ingred_label.classes_), 'unique ingredients, ', ingred_label.classes_[0:3])
y_vec = np.stack(all_ingred.map(lambda x: np.sum(to_categorical(ingred_label.transform(x), 

                                        num_classes=len(ingred_label.classes_)),0)),0).clip(0,1)

plt.matshow(y_vec)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(range(y_vec.shape[0]), 

                                                    random_state = 12345,

                                                   train_size = 0.7)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(cv_mat[train_idx], y_vec[train_idx])

pred_vec = rf.predict(cv_mat[test_idx])



print('Mean Error %2.2f%%' % (100*mean_absolute_error(y_vec[test_idx], pred_vec)))
print('Input Name:', drink_df['strDrink'].values[test_idx[0]])

print('Real Ingredients', all_ingred.values[test_idx[0]])



proc_pred = lambda out_pred: sorted([(ingred_label.inverse_transform(idx), out_pred[idx])

                              for idx in np.where(out_pred>0)[0]], key = lambda x: -x[1])



print('Predicted Ingredients')

for (i,j) in proc_pred(pred_vec[0]):

    print('%25s\t\t%2.2f%%' % (i,100*j))
print('Input Name:', drink_df['strDrink'].values[test_idx[1]])

print('Real Ingredients', all_ingred.values[test_idx[1]])



proc_pred = lambda out_pred: sorted([(ingred_label.inverse_transform(idx), out_pred[idx])

                              for idx in np.where(out_pred>0)[0]], key = lambda x: -x[1])



print('Predicted Ingredients')

for (i,j) in proc_pred(pred_vec[1]):

    print('%25s\t\t%2.2f%%' % (i,100*j))