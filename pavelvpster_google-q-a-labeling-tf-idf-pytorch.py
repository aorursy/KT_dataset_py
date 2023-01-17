import numpy as np

import pandas as pd



import warnings

warnings.simplefilter('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/google-quest-challenge/train.csv", index_col='qa_id')

train.shape
test = pd.read_csv("../input/google-quest-challenge/test.csv", index_col='qa_id')

test.shape
train.head(3).T
target_columns = [

    'question_asker_intent_understanding',

    'question_body_critical',

    'question_conversational',

    'question_expect_short_answer',

    'question_fact_seeking',

    'question_has_commonly_accepted_answer',

    'question_interestingness_others',

    'question_interestingness_self',

    'question_multi_intent',

    'question_not_really_a_question',

    'question_opinion_seeking',

    'question_type_choice',

    'question_type_compare',

    'question_type_consequence',

    'question_type_definition',

    'question_type_entity',

    'question_type_instructions',

    'question_type_procedure',

    'question_type_reason_explanation',

    'question_type_spelling',

    'question_well_written',

    'answer_helpful',

    'answer_level_of_information',

    'answer_plausible',

    'answer_relevance',

    'answer_satisfaction',

    'answer_type_instructions',

    'answer_type_procedure',

    'answer_type_reason_explanation',

    'answer_well_written'

]
y_train = train[target_columns].copy()

x_train = train.drop(target_columns, axis=1)

del train



x_test = test.copy()

del test
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
text_encoder = Pipeline([

    ('Text-TF-IDF', TfidfVectorizer(ngram_range=(1, 3))),

    ('Text-SVD', TruncatedSVD(n_components = 100))], verbose=True)
# from https://www.kaggle.com/abazdyrev/use-features-oof



from urllib.parse import urlparse

import re

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer

from category_encoders.one_hot import OneHotEncoder





# gives part of string (URL) before '.'

before_dot = re.compile('^[^.]*')



def transform_url(x):

    return x.apply(lambda v: re.findall(before_dot, urlparse(v).netloc)[0])



url_encoder = Pipeline([

    ('URL-transformer', FunctionTransformer(transform_url, validate=False)),

    ('URL-OHE', OneHotEncoder(drop_invariant=True))], verbose=True)
# https://contrib.scikit-learn.org/categorical-encoding/



from category_encoders.one_hot import OneHotEncoder





ohe = OneHotEncoder(cols='category', drop_invariant=True)
from sklearn.preprocessing import StandardScaler

import re





def counts(data):

    out = pd.DataFrame(index=data.index)

    for column in data.columns:

        out[column + '_sentences'] = data[column].apply(lambda x: str(x).count('\n') + 1)

        out[column + '_words'] = data[column].apply(lambda x: len(str(x).split()))

        out[column + '_letters'] = data[column].apply(lambda x: len(str(x)))

        out[column + '_unique_words'] = data[column].apply(lambda x: len(set(str(x).split())))

    return out



counters = Pipeline([

    ('Counters-transformer', FunctionTransformer(counts, validate=False)),

    ('Counters-std', StandardScaler())], verbose=True)
# counts(pd.DataFrame(data={'A': ['abc xyz \n\n abc 12345']})).head().T
preprocessor = ColumnTransformer([

    ('Q-T', text_encoder, 'question_title'),

    ('Q-B', text_encoder, 'question_body'),

    ('A', text_encoder, 'answer'),

    ('URL', url_encoder, 'url'),

    ('Categoty', ohe, 'category'),

    ('C', counters, ['question_body', 'answer'])], verbose=True)
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
x_train.shape
y_train = y_train.values
# from https://www.kaggle.com/c/google-quest-challenge/discussion/126778



from scipy.stats import spearmanr





def mean_spearmanr_correlation_score(y, y_pred):

    spearsum = 0

    cnt = 0 

    for col in range(y_pred.shape[1]):

        v = spearmanr(y_pred[:,col], y[:,col]).correlation

        if np.isnan(v):

            continue

        spearsum += v

        cnt += 1

    res = spearsum / cnt

    return res
trained_estimators = []

all_scores = []
from sklearn.linear_model import RidgeCV





ridge_grid = RidgeCV(alphas=np.linspace(0.1, 2.0, num=100)).fit(x_train, y_train)



best_Alpha = ridge_grid.alpha_

best_Alpha
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold

import math





n_splits = 10



scores = []



cv = KFold(n_splits=n_splits, random_state=42)

for train_idx, valid_idx in cv.split(x_train, y_train):

    

    x_train_train = x_train[train_idx]

    y_train_train = y_train[train_idx]

    x_train_valid = x_train[valid_idx]

    y_train_valid = y_train[valid_idx]

    

    estimator = Ridge(alpha=best_Alpha, random_state=42)

    estimator.fit(x_train_train, y_train_train)

    trained_estimators.append(estimator)

    

    oof_part = estimator.predict(x_train_valid)

    score = mean_spearmanr_correlation_score(y_train_valid, oof_part)

    print('Score:', score)

    scores.append(score)





print('Mean score:', np.mean(scores))

all_scores.extend(scores)
import torch

import torch.nn as nn



from torch.nn import Sequential

from torch.nn import Linear

from torch.nn import ReLU

from torch.nn.utils.weight_norm import weight_norm



from torch.nn import MSELoss

from torch.optim import Adam



import random





random.seed(42)

np.random.seed(42)

torch.manual_seed(42)





class PyTorch:

    

    def __init__(self, in_features, out_features, n_epochs, patience):

        self.in_features = in_features

        self.out_features = out_features

        self.n_epochs = n_epochs

        self.patience = patience

    

    

    def init_model(self):

        

        # define a model

        self.model = Sequential(

            weight_norm(Linear(self.in_features, 128)),

            ReLU(),

            weight_norm(Linear(128, 128)),

            ReLU(),

            weight_norm(Linear(128, self.out_features)))

        

        # initialize model

        for t in self.model:

            if isinstance(t, Linear):

                nn.init.kaiming_normal_(t.weight_v)

                nn.init.kaiming_normal_(t.weight_g)

                nn.init.constant_(t.bias, 0)

        

        # define loss function

        self.loss_func = MSELoss()

        

        # define optimizer

        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    

    

    def fit(self, x_train, y_train, x_valid, y_valid):

        

        validate = (x_valid is not None) & (y_valid is not None)

        

        self.init_model()

        

        x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)

        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)

        

        if validate:

            x_valid_tensor = torch.as_tensor(x_valid, dtype=torch.float32)

            y_valid_tensor = torch.as_tensor(y_valid, dtype=torch.float32)

        

        min_loss = np.inf

        counter = 0

        

        for epoch in range(self.n_epochs):

            

            self.model.train()

            y_pred = self.model(x_train_tensor)

            loss = self.loss_func(y_pred, y_train_tensor)

            

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

            

            current_loss = loss.item()

            # print('Epoch %5d / %5d. Loss = %.5f' % (epoch + 1, self.n_epochs, current_loss))



            if validate:

                # calculate loss for validation set

                self.model.eval()

                with torch.no_grad():

                    current_loss = self.loss_func(self.model(x_valid_tensor), y_valid_tensor).item()

                # print('Epoch %5d / %5d. Validation loss = %.5f' % (epoch + 1, self.n_epochs, current_loss))

            

            # early stopping

            if current_loss < min_loss:

                min_loss = current_loss

                counter = 0

            else:

                counter += 1

                # print('Early stopping: %i / %i' % (counter, self.patience))

                if counter >= self.patience:

                    # print('Early stopping at epoch', epoch + 1)

                    break

    

    

    def predict(self, x):

        x_tenson = torch.as_tensor(x, dtype=torch.float32)

        self.model.eval()

        with torch.no_grad():

            return self.model(x_tenson).numpy()
pytorch_params = {

    'in_features': x_train.shape[1],

    'out_features': y_train.shape[1],

    'n_epochs': 2500,

    'patience': 5

}
# estimator = PyTorch(**pytorch_params)

# estimator.fit(x_train, y_train, None, None)

# trained_estimators.append(estimator)
from sklearn.model_selection import KFold

import math





n_splits = 10



scores = []



cv = KFold(n_splits=n_splits, random_state=42)

for train_idx, valid_idx in cv.split(x_train, y_train):

    

    x_train_train = x_train[train_idx]

    y_train_train = y_train[train_idx]

    x_train_valid = x_train[valid_idx]

    y_train_valid = y_train[valid_idx]

    

    estimator = PyTorch(**pytorch_params)

    estimator.fit(x_train_train, y_train_train, x_train_valid, y_train_valid)

    trained_estimators.append(estimator)

    

    oof_part = estimator.predict(x_train_valid)

    score = mean_spearmanr_correlation_score(y_train_valid, oof_part)

    print('Score:', score)

    scores.append(score)





print('Mean score:', np.mean(scores))

all_scores.extend(scores)
len(trained_estimators)
y_pred = []

for estimator in trained_estimators:

    y_pred.append(estimator.predict(x_test))
sum_scores = sum(all_scores)

weights = [x / sum_scores for x in all_scores]
from scipy.stats import rankdata





def blend_by_ranking(data, weights):

    out = np.zeros(data.shape[0])

    for idx,column in enumerate(data.columns):

        out += weights[idx] * rankdata(data[column].values)

    out /= np.max(out)

    return out
submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv", index_col='qa_id')



out = pd.DataFrame(index=submission.index)

for column_idx,column in enumerate(target_columns):

    

    # collect all predictions for one column

    column_data = pd.DataFrame(index=submission.index)

    for prediction_idx,prediction in enumerate(y_pred):

        column_data[str(prediction_idx)] = prediction[:, column_idx]

    

    out[column] = blend_by_ranking(column_data, weights)
out.head()
out.to_csv("submission.csv")