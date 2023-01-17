import numpy as np

import pandas as pd



import re

from nltk.stem import PorterStemmer

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression, Ridge, Lasso
data = pd.read_csv('../input/ice-cream-dataset/combined/products.csv')
data
data = data.drop(['key', 'name', 'subhead', 'description'], axis=1)
data
data = data.drop(data.query('rating_count < 10').index, axis=0).reset_index(drop=True)
data = data.drop('rating_count', axis=1)
data
def process_ingredients(ingredients):

    ps = PorterStemmer()

    new_ingredients = re.sub(r'\(.*?\)', '', ingredients)

    new_ingredients = re.sub(r'CONTAINS:.*$', '', new_ingredients)

    new_ingredients = re.sub(r'\..*?:', ',', new_ingredients)

    new_ingredients = re.sub(r'( AND/OR )', ',', new_ingredients)

    new_ingredients = re.sub(r'( AND )', ',', new_ingredients)

    new_ingredients = new_ingredients.split(',')

    for i in range(len(new_ingredients)):

        new_ingredients[i] = new_ingredients[i].replace('†', '').replace('*', ' ').replace(')', '').replace('/', ' ')

        new_ingredients[i] = re.sub(r'^.+:', '', new_ingredients[i])

        new_ingredients[i] = ps.stem(new_ingredients[i].strip())

        if new_ingredients[i] == 'milk fat':

            new_ingredients[i] = 'milkfat'

    return new_ingredients
# Add all unique ingredients to all_ingredients



all_ingredients = set()



for row in data.iterrows():

    ingredients = process_ingredients(data.loc[row[0], 'ingredients'])

    for ingredient in ingredients:

        if ingredient not in all_ingredients:

            all_ingredients.add(ingredient)



all_ingredients.remove('')
all_ingredients
data
y = data.loc[:, 'rating']

X = data.drop('rating', axis=1)
X
def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
X = onehot_encode(X, 'brand', 'b')
X
X['ingredients'] = X['ingredients'].apply(process_ingredients)
X
ingredients_df = X['ingredients']

ingredients_df
mlb = MultiLabelBinarizer()



ingredients_df = pd.DataFrame(mlb.fit_transform(ingredients_df))
ingredients_df
X = pd.concat([X, ingredients_df], axis=1)

X = X.drop('ingredients', axis=1)
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
model = LinearRegression()



model.fit(X_train, y_train)
model.score(X_test, y_test)
l2_model = Ridge(alpha=1000.0)



l2_model.fit(X_train, y_train)
l2_model.score(X_test, y_test)
l1_model = Lasso(alpha=0.1)



l1_model.fit(X_train, y_train)
l1_model.score(X_test, y_test)