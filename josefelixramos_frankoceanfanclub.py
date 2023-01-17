# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!cp -r ../input/vadersentiment/vaderSentiment-master/* ./

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import random

import math

from sklearn.linear_model import LinearRegression

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()



def substrings(n, x):

    return np.fromfunction(lambda i, j: x[i + j], (len(x) - n + 1, n), dtype=int)





# Importing the dataset

train_data = pd.read_csv('../input/tweet-sentiment-extraction/train.csv', delimiter=',')

test_data = pd.read_csv('../input/tweet-sentiment-extraction/test.csv', delimiter=',')

submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv', delimiter=',')

train_data = train_data.dropna(axis=0, how='any')



train_data["text_length"] = train_data["text"].str.split().str.len()

train_data["selected_text_length"] = train_data["selected_text"].str.split().str.len()



test_data["text_length"] = test_data["text"].str.split().str.len()



positive_data = train_data.loc[train_data['sentiment'] == 'positive']

neutral_data = train_data.loc[train_data['sentiment'] == 'neutral']

negative_data = train_data.loc[train_data['sentiment'] == 'negative']



positive_X = positive_data['text_length'].values.reshape(-1,1)

positive_y = positive_data['selected_text_length'].values.reshape(-1,1)

positive_regressor = LinearRegression()

positive_regressor.fit(positive_X, positive_y)



neutral_X = neutral_data['text_length'].values.reshape(-1,1)

neutral_y = neutral_data['selected_text_length'].values.reshape(-1,1)

neutral_regressor = LinearRegression()

neutral_regressor.fit(neutral_X, neutral_y)



negative_X = negative_data['text_length'].values.reshape(-1,1)

negative_y = negative_data['selected_text_length'].values.reshape(-1,1)

negative_regressor = LinearRegression()

negative_regressor.fit(negative_X, negative_y)



for i in range(test_data.shape[0]): 

    if test_data['sentiment'].iloc[i] == 'positive':

        size_selected_text = math.ceil((positive_regressor.coef_ * test_data['text_length'].iloc[i]) + positive_regressor.intercept_)

        if size_selected_text >= test_data['text_length'].iloc[i]:

            submission['selected_text'].iloc[i] = f"{test_data['text'].iloc[i]}"

        else:

            words = test_data['text'].iloc[i].split()

            substr = substrings(size_selected_text, np.asarray(words))

            index = 0

            current_score = 0.00000

            for j in range(substr.shape[0]):

                vadersenti = analyzer.polarity_scores(' '.join(substr[j]))

                if vadersenti['pos'] > current_score:

                    current_score = vadersenti['pos']

                    index = j

            submission['selected_text'].iloc[i] = f"{' '.join(substr[index])}"

       

    elif test_data['sentiment'].iloc[i] == 'neutral':

        size_selected_text = math.ceil((neutral_regressor.coef_ * test_data['text_length'].iloc[i]) + neutral_regressor.intercept_)

        if size_selected_text >= test_data['text_length'].iloc[i]:

            submission['selected_text'].iloc[i] = f"{test_data['text'].iloc[i]}"

        else:

            words = test_data['text'].iloc[i].split() 

            substr = substrings(size_selected_text, np.asarray(words))

            index = 0

            current_score = 0.00000

            for j in range(substr.shape[0]):

                vadersenti = analyzer.polarity_scores(' '.join(substr[j]))

                if vadersenti['neu'] > current_score:

                    current_score = vadersenti['neu']

                    index = j

            submission['selected_text'].iloc[i] = f"{' '.join(substr[index])}"

 

    elif test_data['sentiment'].iloc[i] == 'negative':

        size_selected_text = math.ceil((negative_regressor.coef_ * test_data['text_length'].iloc[i]) + negative_regressor.intercept_)

        if size_selected_text >= test_data['text_length'].iloc[i]:

            submission['selected_text'].iloc[i] = f"{test_data['text'].iloc[i]}"

        else:

            words = test_data['text'].iloc[i].split() 

            substr = substrings(size_selected_text, np.asarray(words))

            index = 0

            current_score = 0.00000

            for j in range(substr.shape[0]):

                vadersenti = analyzer.polarity_scores(' '.join(substr[j]))

                if vadersenti['neg'] > current_score:

                    current_score = vadersenti['neg']

                    index = j

            submission['selected_text'].iloc[i] = f"{' '.join(substr[index])}"

 

submission.to_csv('submission.csv', index=False)



print("DONE")
