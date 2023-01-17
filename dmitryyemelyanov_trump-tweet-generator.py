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
tweets = pd.read_csv("/kaggle/input/trump-tweets/trumptweets.csv")
tweets.head(50)
tweets = tweets[["content"]]
tweets.head(5)
tweets.shape
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space.

    # which in effect deletes the punctuation marks.

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks.

    return text.translate(translator)
tweets['text'] = tweets['content'].apply(remove_punctuation)

tweets.head(10)
tweets = tweets["text"]

tweets.head(10)
from fastai.text import *
data = pd.read_csv("/kaggle/input/trump-tweets/trumptweets.csv",  encoding='latin1')

data.head()
data = (TextList.from_df(data, cols='content')

                .split_by_rand_pct(0.1)

                .label_for_lm()  

                .databunch(bs=48))



data.show_batch()
# Create deep learning model

learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3, model_dir = '/tmp/work')



# select the appropriate learning rate

learn.lr_find()



# we typically find the point where the slope is steepest

learn.recorder.plot(skip_end=15)



# Fit the model based on selected learning rate

learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))



# Predict Tweets starting from the given words 

N_WORDS = 20





print(learn.predict("Clean energy will be", N_WORDS, temperature=0.75))
print(learn.predict("Russian hackers", N_WORDS, temperature=0.75))
print(learn.predict("Tesla", N_WORDS, temperature=0.75))
print(learn.predict("Clean energy will be", 2, temperature=0.75))
print(learn.predict("Clean energy will be", 10, temperature=0.75))
print(learn.predict("Global warming", 10, temperature=0.75))
print(learn.predict("Clean energy will be", 11, temperature=0.75))
print(learn.predict("Global warming", 11, temperature=0.75))
print(learn.predict("White house", 10, temperature=0.75))
print(learn.predict("I am", 10, temperature=0.75))
print(learn.predict("Deep fake", 10, temperature=0.75))
print(learn.predict("Calling", 10, temperature=0.75))
print(learn.predict("Putin", 10, temperature=0.75))
print(learn.predict("Russia", 10, temperature=0.75))
print(learn.predict("Nuclear war is", 10, temperature=0.75))
print(learn.predict("Iran is democratic", 10, temperature=0.75))
print(learn.predict("Global warming", 10, temperature=0.75))