# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df.head()
del df['id']

train = df.as_matrix()
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.neural_network import MLPClassifier

import xgboost
from sklearn.feature_extraction import text

st_wd = text.ENGLISH_STOP_WORDS

c_vector = CountVectorizer(stop_words = st_wd,min_df=.000001,lowercase=1)

c_vector.fit(pd.concat((df['text'],df_test['text']),axis=0))

X_train_counts = c_vector.transform(df['text'].values)
X_train_counts
dic = {'EAP':0,'HPL':1,'MWS':2}

df['author'] = df['author'].map(dic)

Y_train = df['author'].values
def prob_y(Y_train,num_class=3):

    p_y = np.zeros([num_class,])

    n_y = np.zeros([num_class,])

    d_y = Y_train.shape[0]

    for i in range(Y_train.shape[0]):

        n_y[Y_train[i]] = n_y[Y_train[i]]+1

    p_y = n_y/d_y

    return p_y
p_y = prob_y(Y_train,num_class=3)

p_y
def prob_xy(c_vec,train_df,Y_train,num_class=3):

    d_y = np.zeros([num_class,])+len(c_vec.vocabulary_)

    p_xy = np.zeros([num_class,len(c_vec.vocabulary_)])

    for i in np.unique(Y_train):

        temp_df = train_df[train_df['author']==i]

        temp_x = c_vec.transform(temp_df['text'].values)

        n_xy = np.sum(temp_x,axis=0)+1

        d_y[i] = d_y[i]+np.sum(temp_x)

        p_xy[i] = n_xy/d_y[i] 

    return p_xy
p_xy = prob_xy(c_vector,df,Y_train,3)

p_xy
def classify(c_vec,test_df,p_xy,p_y,num_class=3):

    pred = []

    pre_yx = []

    for doc in test_df['text'].values:

        temp_doc = (c_vec.transform([doc])).todense()

        temp_prob = np.zeros([num_class,])

        for i in range(num_class):

            temp_prob[i] = np.prod(np.power(p_xy[i],temp_doc))*p_y[i]

        pred.append(np.argmax(temp_prob))

    return pred
def accuracy(pred,Y):

    return np.sum(pred==Y)/Y.shape[0]
pred_train = classify(c_vector,df,p_xy,p_y,num_class=3)

print('Train Data Accuracy = '+str(accuracy(pred_train,Y_train)))