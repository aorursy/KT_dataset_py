import numpy as np

import pandas as pd

import os



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm_notebook

from sklearn.pipeline import Pipeline

print(os.listdir('../input/'))
path = '../input/aclimdb/aclImdb/'
train_text = []

test_text = []

train_label = []

test_label = []



for train_test in ['train','test']:

    for neg_pos in ['neg','pos']:

        file_path = path + train_test + '/' + neg_pos + '/'

        for file in tqdm_notebook(os.listdir(file_path)):

            with open(file_path + file, 'r') as f:

                if train_test == 'train':

                    train_text.append(f.read())

                    if neg_pos == 'neg':

                        train_label.append(0)

                    else:

                        train_label.append(1)

                else:

                    test_text.append(f.read())

                    if neg_pos == 'neg':

                        test_label.append(0)

                    else:

                        test_label.append(1)
X_train = pd.DataFrame()

X_train['review'] = train_text

X_train['label'] = train_label



X_test = pd.DataFrame()

X_test['review'] = test_text

X_test['label'] = test_label
X_train_train, X_train_valid, y_train_train, y_train_valid = train_test_split(X_train['review'].values, X_train['label'].values, test_size=0.3, random_state=19)
vect = TfidfVectorizer(ngram_range=(1,1), max_features=50000)

logit = LogisticRegression(C=1, random_state=19)



pipe = Pipeline([

    ('vect', vect),

    ('logit', logit)

])
pipe.fit(X_train_train, y_train_train)
preds_valid = pipe.predict(X_train_valid)
print('Accuracy score on the validation dataset: ',accuracy_score(y_train_valid, preds_valid))