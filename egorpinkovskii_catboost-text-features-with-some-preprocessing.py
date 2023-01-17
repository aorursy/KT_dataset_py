import re

import pandas as pd

import numpy as np

import urllib.parse as up



from catboost import CatBoostClassifier, cv, Pool
hashtags = re.compile('#[^ ]+')

mentions = re.compile('@[^ ]+')





def extract_matches(string, regex):

    matches = []

    match = regex.search(string)

    while match is not None:

        matches.append(match.group())

        match = regex.search(string, match.span()[1])

    return matches





def extract_and_combine_matches(string, regex):

    matches = extract_matches(string, regex)

    return ' '.join(map(lambda m: m[1:], matches))





def extract_hashtags(string):

    return extract_and_combine_matches(string, hashtags)





def extract_mentions(string):

    return extract_and_combine_matches(string, mentions)
def etl(dataset):

    dataset = dataset.fillna('')

    dataset['mentions'] = dataset.text.apply(extract_mentions)

    dataset['hashtags'] = dataset.text.apply(extract_hashtags)

    dataset['keyword'] = dataset.keyword.apply(lambda k: up.unquote(k))

    return dataset.fillna('')





def extract_x(dataset):

    return dataset[['keyword', 'location', 'text', 'mentions', 'hashtags']]





def extract_y(dataset):

    return dataset['target']





def split_x_y(dataset):

    return extract_x(dataset), extract_y(dataset)
train = pd.read_csv('../input/nlp-getting-started/train.csv')

train
train.text.apply(extract_mentions).value_counts()
train.text.apply(extract_hashtags).value_counts()
train_x, train_y = split_x_y(

    etl(train)

)

train_x
cv_dataset = Pool(

    data=train_x,

    label=train_y,

    text_features=['keyword', 'location', 'text', 'mentions', 'hashtags'],

)



cv(

    pool=cv_dataset,

    params={

        'iterations': 500,

        'learning_rate': 0.1,

        'loss_function': 'Logloss',

        'depth': 6,

        'verbose': False,

        'use_best_model': True,

    },

    fold_count=5,

    as_pandas=True,

    plot=True,

    seed=0,

)
model = CatBoostClassifier(

    iterations=100,

    depth=6,

    learning_rate=0.1,

    loss_function='Logloss',

    verbose=False,

    random_seed=0,

)

model.fit(

    train_x, train_y,

    text_features=['keyword', 'location', 'text', 'mentions', 'hashtags'],

    plot=True

)
test = pd.read_csv('../input/nlp-getting-started/test.csv')

test_x = extract_x(etl(test))

test_y = model.predict(test_x)

test['target'] = model.predict(test_x)

test.to_csv('submission.csv', columns=['id', 'target'], index=False)