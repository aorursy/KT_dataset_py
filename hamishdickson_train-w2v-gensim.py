# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train_1 = pd.read_csv('../input/cleaned-toxic-comments/train_preprocessed.csv')

df_test_1 = pd.read_csv('../input/cleaned-toxic-comments/test_preprocessed.csv')

df_train_2 = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

# df_test_2 = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
df_train_1.sample(3)
def basic_preprocess(data):

    data = str(data)

    data = data.lower()

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):        

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = clean_special_chars(data, punct)

    return data
df_train_1['clean'] = df_train_1['comment_text'].apply(basic_preprocess)

df_test_1['clean'] = df_test_1['comment_text'].apply(basic_preprocess)

df_train_2['clean'] = df_train_2['comment_text'].apply(basic_preprocess)

# df_test_2['clean'] = df_test_2['comment_text'].apply(basic_preprocess)
all_text = list(df_train_2['clean'] + df_train_1['clean'] + df_test_1['clean'])



all_text = list(map(lambda s: str(s).split(' '), all_text))

all_text = list(map(lambda s: list(filter(lambda w: str(w).strip() != "", s)), all_text))



all_text[:3]
import gensim
%%time



gensim_model = gensim.models.Word2Vec(all_text, size=300, min_count=1, iter=25, window=5)
gensim_model.wv.most_similar("khadr")
gensim_model.save('jigsaw_w2v_model')