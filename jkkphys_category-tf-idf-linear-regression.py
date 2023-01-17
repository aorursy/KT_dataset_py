from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_log_error

from sklearn.linear_model import SGDRegressor

from sklearn.pipeline import Pipeline



import pandas as pd

import numpy as np

import time

import re



seed = 101
def tokenizer(text):

    if text:

        result = re.findall('[a-z]{2,}', text.lower())

    else:

        result = []

    return result
df = pd.read_csv('../input/train.tsv', sep='\t')

df.head()
df['item_description'].fillna(value='Missing', inplace=True)

X = (df['name'] + ' ' + df['item_description']).values

y = np.log1p(df['price'].values)



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=seed)
vect = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')

start = time.time()

X_train_vect = vect.fit_transform(X_train)

end = time.time()

print('Time to train vectorizer and transform training text: %0.2fs' % (end - start))
# I was using a LinearRegression previously, but with the wider vocab it's too slow. 

# Let's use the SGDRegressor with ordinary least squares.

# Also, using mean squared error as the eval metric, since negative values crash mean squared log error.



model = SGDRegressor(loss='squared_loss', penalty='l2', random_state=seed, max_iter=5)

params = {'penalty':['none','l2','l1'],

          'alpha':[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}

gs = GridSearchCV(estimator=model,

                  param_grid=params,

                  scoring='neg_mean_squared_error',

                  n_jobs=1,

                  cv=5,

                  verbose=3)

start = time.time()

gs.fit(X_train_vect, y_train)

end = time.time()

print('Time to train model: %0.2fs' % (end -start))
model = gs.best_estimator_

print(gs.best_params_)

print(gs.best_score_)
pipe = Pipeline([('vect',vect),('model',model)])

start = time.time()

y_pred = pipe.predict(X_test)

end = time.time()

print('Time to generate predictions on test set: %0.2fs' % (end - start))
# Replace negative values with zero for the time being.

print(np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(y_pred)-1)))
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df_test.head()
df_test['item_description'].fillna('Missing', inplace=True)

df_test['price'] = np.exp(pipe.predict((df_test['name'] + ' ' + df_test['item_description']).values))-1

df_test.head()
df_test[['test_id','price']].to_csv('output.csv', index=False)