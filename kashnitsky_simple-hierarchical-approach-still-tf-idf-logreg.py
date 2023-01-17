# some necessary imports

import os

import numpy as np

import pandas as pd

from pathlib import Path

import time

import pickle

from contextlib import contextmanager

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

import seaborn as sns

from matplotlib import pyplot as plt

%config InlineBackend.figure_format = 'retina'
# nice way to report running times

@contextmanager

def timer(name):

    t0 = time.time()

    yield

    print(f'[{name}] done in {time.time() - t0:.0f} s')
PATH_TO_DATA = Path('../input/hierarchical-text-classification/')
train_df = pd.read_csv(PATH_TO_DATA / 'train_40k.csv').fillna(' ')

val_df = pd.read_csv(PATH_TO_DATA / 'val_10k.csv').fillna(' ')
train_df.head()
train_df.loc[0, 'Text']
train_df.loc[0, 'Cat1'], train_df.loc[0, 'Cat2']
train_df['Cat1'].value_counts()
train_df['Cat1_Cat2'] = train_df['Cat1'] + '/' + train_df['Cat2']

val_df['Cat1_Cat2'] = val_df['Cat1'] + '/' + val_df['Cat2']
# put a limit on maximal number of features and minimal word frequency

tf_idf = TfidfVectorizer(max_features=50000, min_df=2)

# multinomial logistic regression a.k.a softmax classifier

logit = LogisticRegression(C=1e2, n_jobs=4, solver='lbfgs', 

                           random_state=17, verbose=0, 

                           multi_class='multinomial',

                           fit_intercept=True)

# sklearn's pipeline

base_model = Pipeline([('tf_idf', tf_idf), 

                       ('logit', logit)])
class TfIdfLogitPipelineHierarchical(BaseEstimator):

    

    def __init__(self, 

                 base_model, 

                 model_store_path,

                 class_separator = '/',

                 min_size_to_train=50

                ):

        """



        :param base_model: Sklearn model to train, one instance for level 1,

                           and several instances for level 2 

        :param model_store_path: where to store models as pickle files

        :param class_separator: separator between level 1 and level 2 class names

        :param min_size_to_train: do not train a model with less data

        """

        self.base_model = base_model

        self.model_store_path = Path(model_store_path)

        self.class_separator = class_separator

        self.min_size_to_train = min_size_to_train

        

        self.model_store_path.mkdir(exist_ok=True)

        

    def fit(self, X, y):

        

        lev1_classes = [label.split(self.class_separator)[0]

                        for label in y]

        

        with timer('Training level 1 model'):

            self.base_model.fit(X, lev1_classes)

            

            

            with open(self.model_store_path / 'level1_model.pkl', 'wb') as f:

                pickle.dump(self.base_model, f)

        

        

        for lev1_class in np.unique(lev1_classes):

            

            with timer(f'Training level 2 model for class: {lev1_class}'):

                curr_X = X.loc[y.str.startswith(lev1_class)]

                curr_y = y.loc[y.str.startswith(lev1_class)].apply(lambda s: s.split(self.class_separator)[1])

                

                if len(curr_X) < self.min_size_to_train:

                    print(f"Skipped class {lev1_class.replace(' ', '_')} due to a too small dataset size: {len(curr_X)}")

                    continue

                    

                self.base_model.fit(curr_X, curr_y)

                

                model_name = f"level2_model_{lev1_class.replace(' ', '_')}.pkl"

                

                with open(self.model_store_path / model_name, 'wb') as f:

                    pickle.dump(self.base_model, f)

    

    def predict(self, X):

        

        model_name =  'level1_model.pkl'

        with open(self.model_store_path / model_name, 'rb') as f:

            level1_model = pickle.load(f)

        

        level1_preds = level1_model.predict(X)

            

        level2_preds = np.zeros_like(level1_preds)

            

        for lev1_class in np.unique(level1_preds):

            

            idx = level1_preds == lev1_class

            curr_X = X.iloc[idx]

            

            model_name = f"level2_model_{lev1_class.replace(' ', '_')}.pkl"

            

            if Path(self.model_store_path / model_name).exists():

            

                with open(self.model_store_path / model_name, 'rb') as f:

                    level2_model = pickle.load(f)



                curr_level2_preds = level2_model.predict(curr_X)

                level2_preds[idx] = curr_level2_preds

            

            else:

                level2_preds[idx] = lev1_class

                

        return level1_preds, level2_preds    
model = TfIdfLogitPipelineHierarchical(

    base_model=base_model,

    model_store_path='models'

)
model.fit(train_df['Title'], train_df['Cat1_Cat2'])
level1_pred, level2_pred = model.predict(val_df['Title'])
f1_score(y_true=val_df['Cat1'], y_pred=level1_pred, average='micro').round(3),\

f1_score(y_true=val_df['Cat1'], y_pred=level1_pred, average='weighted').round(3)
f1_score(y_true=val_df['Cat2'], y_pred=level2_pred, average='micro').round(3),\

f1_score(y_true=val_df['Cat2'], y_pred=level2_pred, average='weighted').round(3)
print(classification_report(

    y_true=val_df['Cat1'], 

    y_pred=level1_pred)

)