import numpy as np

import pandas as pd

import itertools

from keras.preprocessing.text import text_to_word_sequence
all_drinks = pd.read_csv('../input/all_drinks.csv')

for i in range(13, 16):

    all_drinks = all_drinks.drop(['strIngredient{}'.format(i), 'strMeasure{}'.format(i)], axis=1)

all_drinks = all_drinks.drop(['strVideo', 'dateModified', 'idDrink', 'strDrinkThumb', 'strIBA', 'Unnamed: 0'], axis=1)



all_ingredients = set(itertools.chain.from_iterable([all_drinks['strIngredient{}'.format(i)].values.tolist() for i in range(1, 13)]))

all_ingredients.remove(np.nan)

all_ingredients = list(all_ingredients)

ingredient_to_index = {k: v for v, k in enumerate(all_ingredients)}

index_to_igredient = {v: k for v, k in enumerate(all_ingredients)}



name_tokens = sorted(set(itertools.chain.from_iterable([text_to_word_sequence(drink) for drink in all_drinks['strDrink']])))

name_token_to_index = {k: v for v, k in enumerate(name_tokens)}

index_to_name_token = {v: k for v, k in enumerate(name_tokens)}



print('{} drinks'.format(len(all_drinks)))

print('{} name tokens'.format(len(name_tokens)))

print('{} Ingredients'.format(len(all_ingredients)))
from sklearn.model_selection import train_test_split



data_X = np.zeros(shape=(len(all_drinks), len(name_tokens)), dtype=np.float)

data_y = np.zeros(shape=(len(all_drinks), len(all_ingredients)), dtype=np.float)

for i in range(len(all_drinks)):

    for token in text_to_word_sequence(all_drinks['strDrink'][i]):

        data_X[i][name_token_to_index[token]] = 1

    for ig in range(1, 13):

        ingredient = all_drinks['strIngredient{}'.format(ig)][i]

        if ingredient is not np.nan:

            data_y[i][ingredient_to_index[ingredient]] = 1



train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.1)



print('Training: {} --> {}'.format(train_X.shape, train_y.shape))

print('Testing: {} --> {}'.format(test_X.shape, test_y.shape))
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



model = Sequential()

model.add(Dense(len(all_ingredients), input_shape=(len(name_tokens),), activation='sigmoid'))

model.compile(loss='mae', optimizer=Adam(lr=0.01))
from keras.callbacks import EarlyStopping



model.fit(

    train_X, train_y,

    epochs=12345, batch_size=256,

    validation_data=(test_X, test_y),

    callbacks=[EarlyStopping()])
num_preds = 10

pred_indices = np.random.choice(

    test_X.shape[0], size=num_preds, replace=False)

predictions = model.predict(test_X[pred_indices, :])

for i in range(num_preds):

    drink_name = []

    for n in range(len(name_tokens)):

        if test_X[pred_indices[i]][n] == 1.0:

            drink_name.append(index_to_name_token[n])

    real_ingredients = []

    for n in range(len(all_ingredients)):

        if test_y[pred_indices[i]][n] == 1.0:

            real_ingredients.append(index_to_igredient[n])

    pred_ingredients = []

    pmin = predictions[i].min()

    pmax = predictions[i].max()

    preds = (predictions[i] - pmin) / (pmax - pmin)

    for n in range(len(all_ingredients)):

        if preds[n] > 0.9:

            pred_ingredients.append(index_to_igredient[n])

    print('\nname: {}\nreal: {}\npred: {}\n'.format(drink_name, real_ingredients, pred_ingredients))
