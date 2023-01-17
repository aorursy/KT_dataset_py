# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/wy4vnuumx7y294p/train.csv')

test = pd.read_csv('/kaggle/input/wy4vnuumx7y294p/test.csv')

subm = pd.read_csv('/kaggle/input/wy4vnuumx7y294p/sample_submission.csv')



# Any results you write to the current directory are saved as output.

# subm.to_csv('/kaggle/working/submission.csv', index=False)
train.head()
subm.head()
print(train.values[0])

print(train.isnull().values.any())

print(test.isnull().values.any())
s1_lens = train.sentence1.str.split().str.len()

s2_lens = train.sentence2.str.split().str.len()

print(len(train))

print(s1_lens.mean(), s1_lens.std(), s1_lens.min(), s1_lens.max())

print(s2_lens.mean(), s2_lens.std(), s2_lens.min(), s2_lens.max())

s1_lens.hist()

s2_lens.hist()
s1_lens = test.sentence1.str.split().str.len()

s2_lens = test.sentence2.str.split().str.len()

print(len(test))

print(s1_lens.mean(), s1_lens.std(), s1_lens.min(), s1_lens.max())

print(s2_lens.mean(), s2_lens.std(), s2_lens.min(), s2_lens.max())

s1_lens.hist()

s2_lens.hist()
corpus = pd.concat(

    [train.sentence1,train.sentence2, test.sentence1, test.sentence2],

    ignore_index=True)
# corpus = [

#     '17244 28497 16263',

#     '5464 4053 14577 8272 15775 3437 20163 8711',

#     '24645 8554 25911',

#     '14080 15907 25964 3099 26989 26797 3397 9553',

#     '14313 2348 4875 23364',

# ]

vectorizer = TfidfVectorizer(

    lowercase=False,

    strip_accents=None,

    tokenizer=lambda x: x.split(),

    preprocessor=lambda x: x,

    ngram_range=(1,3),

    min_df=3, max_df=0.9, 

    use_idf=1, smooth_idf=1, 

    sublinear_tf=1)

vectorizer.fit(corpus)

vectorizer.get_feature_names()

len(vectorizer.get_feature_names())
train_x = vectorizer.transform(train.sentence1) + vectorizer.transform(train.sentence2)

train_y = train.label.values

test_x = vectorizer.transform(test.sentence1) + vectorizer.transform(test.sentence2)
model = LogisticRegressionCV(dual=True, solver='liblinear', max_iter=100)

# model.scores_

# model = LogisticRegression(C=1, dual=True, solver='liblinear')

model.fit(train_x, train_y)

train_preds = model.predict(train_x)

sum(train_y == train_preds) / len(train_y)
test_preds = model.predict(test_x)

submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(test_preds, columns=['label'])], axis=1)

submission.to_csv('/kaggle/working/submission_cv_trigram_sub.csv', index=False)
# Define a pipeline to search for the best combination of PCA truncation and classifier regularization.

pca = TruncatedSVD()

logistic = LogisticRegression(dual=True, solver='liblinear', max_iter=100)

pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])



# Parameters of pipelines can be set using ‘__’ separated parameter names:

param_grid = {

    'pca__n_components': [50, 100, 200, 300, 500, 1000],

    'logistic__C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

}

search = GridSearchCV(pipe, param_grid, n_jobs=-1)

search.fit(train_x, train_y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)

print(search.best_params_)