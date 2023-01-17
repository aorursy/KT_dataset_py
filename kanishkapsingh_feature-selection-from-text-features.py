# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from matplotlib import pyplot as plt

from subprocess import check_output

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from collections import Counter

from tqdm import tqdm

import nltk

from string import printable,punctuation

from sklearn.feature_selection import SelectKBest,f_regression

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.tsv',sep='\t')

def useable_words(x):

    temp=[]

    for i in nltk.word_tokenize(str(x)):

        if i not in nltk.corpus.stopwords.words('english'):

            clean_word="".join([j for j in i if j in printable and j not in punctuation])

            temp.append(clean_word)

    return " ".join(temp)

#lem=nltk.stem.wordnet.WordNetLemmatizer()

for col in tqdm(['brand_name','item_description','name','category_name']):

    train.loc[pd.isnull(train[col]),col]='Missing'

    #if col=='item_description':

    #    train[col].apply(useable_words)
# x=train.item_description

# for word in tqdm(nltk.corpus.stopwords.words('english')):

#     x=x.str.replace(pat=word,repl='')

train.head()
catlevels=train.category_name.astype('str').str.split('/',expand=True)

catlevels.rename(columns={0:'FirstLevelCat',1:'SecondLevelCat',2:'ThirdLevelCat',3:'FourthLevelCat',4:'FifthLevelCat'},inplace=True)

catlevels.fillna('Missing',inplace=True)

for col in tqdm(['FirstLevelCat','SecondLevelCat','ThirdLevelCat','FourthLevelCat','FifthLevelCat']):

    train[col]=catlevels[col]

print('Generated the following columns:',catlevels.columns)

del catlevels
def tokenize(x):

    return x.split('/')

for col in tqdm(['item_description']):

    CV=CountVectorizer(ngram_range=(1,1),min_df=10,binary=True)

    data_sparse=CV.fit_transform(train[col])

print('Generated the following sparse data based on ngrams:',data_sparse.shape)
print('Started working on the sparse data')

skbest=SelectKBest(score_func=f_regression,k=data_sparse.shape[1])

skbest.fit_transform(data_sparse,train['price'])

feat_imp=pd.Series(skbest.scores_,index=CV.get_feature_names())

pvalues=pd.Series(skbest.pvalues_,index=CV.get_feature_names())

print('Sparse data feature selection completed. Starting on other features')

for col in tqdm(['FirstLevelCat','SecondLevelCat','ThirdLevelCat','FourthLevelCat','FifthLevelCat']):

    dummy_sparse=pd.get_dummies(train[col],prefix=col,prefix_sep='_',sparse=True)

    skbest=SelectKBest(score_func=f_regression,k=dummy_sparse.shape[1])

    skbest.fit_transform(dummy_sparse,train['price'])

    feat_imp=feat_imp.append(pd.Series(skbest.scores_,index=dummy_sparse.columns))

    pvalues=pvalues.append(pd.Series(skbest.pvalues_,index=dummy_sparse.columns))

    del dummy_sparse

skbest=SelectKBest(score_func=f_regression,k=2)

skbest.fit_transform(train[['shipping','item_condition_id']],train['price'])

feat_imp=feat_imp.append(pd.Series(skbest.scores_,index=['shipping','item_condition_id']))

pvalues=pvalues.append(pd.Series(skbest.pvalues_,index=['shipping','item_condition_id']))

    

# skbest=SelectKBest(score_func=f_regression,k=dummy_sparse.shape[1])

# skbest.fit_transform(dummy_sparse,train['price'])

# feat_imp=feat_imp.append(pd.Series(skbest.scores_,index=dummy_sparse.columns))

feat_imp.sort_values(ascending=False,inplace=True)

pvalues.sort_values(ascending=True,inplace=True)
fig,ax=plt.subplots(1,1,figsize=(10,100))

sns.barplot(y=list(feat_imp[:200].index),x=feat_imp[:200],orient='h')