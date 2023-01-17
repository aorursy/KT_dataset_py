import numpy as np

import pandas as pd



import re

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
one_star_df = pd.read_csv('../input/michelin-restaurants/one-star-michelin-restaurants.csv')

two_star_df = pd.read_csv('../input/michelin-restaurants/two-stars-michelin-restaurants.csv')

three_star_df = pd.read_csv('../input/michelin-restaurants/three-stars-michelin-restaurants.csv')
one_star_df.head()
two_star_df.head()
three_star_df.head()
one_star_df['stars'] = pd.Series(0, index=one_star_df.index)

two_star_df['stars'] = pd.Series(1, index=two_star_df.index)

three_star_df['stars'] = pd.Series(2, index=three_star_df.index)



combined_df = pd.concat([one_star_df, two_star_df, three_star_df], axis=0).sample(frac=1.0).reset_index(drop=True)
combined_df
y = combined_df['stars'].copy()

X = combined_df.drop('stars', axis=1)
X = X.drop(['name', 'zipCode', 'url'], axis=1)
X
X.isna().sum()
X['price'].value_counts()
X['price'] = X['price'].fillna(X['price'].mode().values[0])
X.isna().sum()
{column: list(X[column].unique()) for column in X.columns if X.dtypes[column] == 'object'}
price_ordering = ['$', '$$', '$$$', '$$$$', '$$$$$']



X['price'] = X['price'].apply(lambda price: price_ordering.index(price))
X
# Removing zip codes from city column

X['city'] = X['city'].apply(lambda city: re.sub(r' - \d+$', '', city) if str(city) != 'nan' else city)
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
X = onehot_encode(

    X,

    ['city', 'region', 'cuisine'],

    ['CI', 'RE', 'CU']

)
X
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=40)
models = []

Cs = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]



for i in range(len(Cs)):

    model = LogisticRegression(C=Cs[i])

    model.fit(X_train, y_train)

    models.append(model)
model_acc = [model.score(X_test, y_test) for model in models]



print(f"Model Accuracy (C={Cs[0]}):", model_acc[0])

print(f" Model Accuracy (C={Cs[1]}):", model_acc[1])

print(f"  Model Accuracy (C={Cs[2]}):", model_acc[2])

print(f"   Model Accuracy (C={Cs[3]}):", model_acc[3])

print(f"   Model Accuracy (C={Cs[4]}):", model_acc[4])

print(f"  Model Accuracy (C={Cs[5]}):", model_acc[3])

print(f" Model Accuracy (C={Cs[6]}):", model_acc[4])