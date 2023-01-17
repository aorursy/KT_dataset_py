# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls ../input
path_dir='/kaggle/input/google-quest-challenge/'
test_df=pd.read_csv(path_dir+'test.csv',index_col='qa_id')

train_df=pd.read_csv(path_dir+'train.csv',index_col='qa_id')

samp_sum_df=pd.read_csv(path_dir+'sample_submission.csv')
print('Train data ',train_df.shape)

print('Test data ',test_df.shape)

print('Sample submission data ',samp_sum_df.shape)
#test_df.describe()

test_df.info()
train_df.T
import matplotlib.pyplot as plt

%matplotlib inline
train_category = train_df['category'].value_counts()

test_category = test_df['category'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

train_category.plot(kind='bar', ax=axes[0])

axes[0].set_title('Train')

test_category.plot(kind='bar', ax=axes[1])

axes[1].set_title('Test')

print('Train/Test category distribution')
from wordcloud import WordCloud





def plot_wordcloud(text, ax, title=None):

    wordcloud = WordCloud(max_font_size=None, background_color='white',

                          width=1200, height=1000).generate(text_cat)

    ax.imshow(wordcloud)

    if title is not None:

        ax.set_title(title)

    ax.axis("off")
print('Training data Word Cloud')



fig, axes = plt.subplots(1, 3, figsize=(16, 18))



text_cat = ' '.join(train_df['question_title'].values)

plot_wordcloud(text_cat, axes[0], 'Question title')



text_cat = ' '.join(train_df['question_body'].values)

plot_wordcloud(text_cat, axes[1], 'Question body')



text_cat = ' '.join(train_df['answer'].values)

plot_wordcloud(text_cat, axes[2], 'Answer')



plt.tight_layout()

fig.show()
print('Test data Word Cloud')



fig, axes = plt.subplots(1, 3, figsize=(16, 18))



text_cat = ' '.join(test_df['question_title'].values)

plot_wordcloud(text_cat, axes[0], 'Question title')



text_cat = ' '.join(test_df['question_body'].values)

plot_wordcloud(text_cat, axes[1], 'Question body')



text_cat = ' '.join(test_df['answer'].values)

plot_wordcloud(text_cat, axes[2], 'Answer')



plt.tight_layout()

fig.show()
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
y_train = train_df[target_columns].copy()

x_train = train_df.drop(target_columns, axis=1)

#del train_df



x_test = test_df.copy()

#del test_df
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





def count_words(data):

    out = pd.DataFrame(index=data.index)

    for column in data.columns:

        out[column] = data[column].str.split().str.len()

    return out



word_counter = Pipeline([

    ('WordCounter-transformer', FunctionTransformer(count_words, validate=False)),

    ('WordCounter-std', StandardScaler())], verbose=True)
preprocessor = ColumnTransformer([

    ('Q-T', text_encoder, 'question_title'),

    ('Q-B', text_encoder, 'question_body'),

    ('A', text_encoder, 'answer'),

    ('URL', url_encoder, 'url'),

    ('Categoty', ohe, 'category'),

    ('W-C', word_counter, ['question_body', 'answer'])], verbose=True)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
x_train.shape
y_train = y_train.values
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
from scipy.stats import spearmanr





def mean_spearmanr_correlation_score(y_true, y_pred):

    return np.mean([spearmanr(y_pred[:, idx], y_true[:, idx]).correlation for idx in range(len(target_columns))])
pytorch_params = {

    'in_features': x_train.shape[1],

    'out_features': y_train.shape[1],

    'n_epochs': 2500,

    'patience': 5

}
trained_estimators = []

estimator = PyTorch(**pytorch_params)

estimator.fit(x_train, y_train, None, None)

trained_estimators.append(estimator)
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

    

    oof_part = estimator.predict(x_train_valid)

    score = mean_spearmanr_correlation_score(y_train_valid, oof_part)

    print('Score:', score)

    

    if not math.isnan(score):

        trained_estimators.append(estimator)

        scores.append(score)





print('Mean score:', np.mean(scores))
y_pred = []

for estimator in trained_estimators:

    y_pred.append(estimator.predict(x_test))
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

    

    out[column] = blend_by_ranking(column_data, np.ones(column_data.shape[1]))
out.to_csv("submission.csv")
submission_f=pd.read_csv('submission.csv')
submission_f.head()