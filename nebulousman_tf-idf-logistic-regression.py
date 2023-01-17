#imports

import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
import os
import regex as re
from nltk.tokenize import casual_tokenize, word_tokenize
from sklearn import metrics
from string import punctuation
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import matplotlib.pyplot as plt 
import torch
# read in data

df = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)
df.head()
#df.dtypes
# define functions

def lowerCase(data, tokenizer = casual_tokenize):
    p_strip = lambda x: "".join(w for w in x if w not in punctuation)
    allcaps = re.findall(r"\b[A-Z][A-Z]+\b",data)
    to_lower = lambda l: " ".join( a if p_strip(a) in allcaps else a.lower() for a in l.split())
    data = to_lower(data)
    return data

def train_val_test(df, seed=7):
    np.random.seed(seed)
    idx=np.random.permutation(df.shape[0])
    test = mdf.iloc[idx[160677:]].reset_index()
    val = mdf.iloc[idx[130551:160677]].reset_index()
    train = mdf.iloc[idx[:130551]].reset_index()
    return train,val,test
# combine similar categories

df.category[df.category=='THE WORLDPOST'] = 'WORLDPOST'
df.category[df.category=='GREEN'] = 'ENVIRONMENT'
df.category[df.category=='CULTURE & ARTS'] = 'ARTS'
df.category[df.category=='COMEDY'] = 'ENTERTAINMENT'
df.category[df.category=='STYLE'] = 'STYLE & BEAUTY'
df.category[df.category=='ARTS & CULTURE'] = 'ARTS'
df.category[df.category=='COLLEGE'] = 'EDUCATION'
df.category[df.category=='SCIENCE'] = 'TECH'
df.category[df.category=='WEDDINGS'] = 'GOOD NEWS'
df.category[df.category=='TASTE'] = 'FOOD & DRINK'
df.category[(df.category=='PARENTING') | (df.category=='FIFTY')] = 'PARENTS'
df.category[df.category=='WORLD NEWS'] = 'WORLDPOST'
# clean up & create train test validation

df = df[(df.headline!='')|(df.short_description!='')]
df = df.reset_index()
df['head_short'] = df['headline'] +' '+df['short_description']
mdf = df[['category', 'head_short']]
mdf.category = pd.Categorical(mdf.category)
mdf['code'] = mdf.category.cat.codes
train,val,test=train_val_test(mdf)

train_text = [lowerCase(train.head_short[i]) for i in range(train.head_short.shape[0])]
val_text = [lowerCase(test.head_short[i]) for i in range(test.head_short.shape[0])]
# model fit

vect_word = TfidfVectorizer(max_features=60000, lowercase=False, analyzer='word',
                        tokenizer=word_tokenize,ngram_range=(1,3),dtype=np.float32)

tr_vect = vect_word.fit_transform(train_text)
vl_vect = vect_word.transform(val_text)

y_train = train.code
y_val = test.code

lr = LogisticRegression(solver='saga',multi_class='multinomial', max_iter=1000, C=1)

lr.fit(tr_vect, y_train)

y_pred = lr.predict(vl_vect)
# view true & predicted lable matrix

confmat = metrics.confusion_matrix(y_val, y_pred) 
fig, ax = plt.subplots(figsize=(20, 20)) 
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.4) 
for i in range(confmat.shape[0]): 
    for j in range(confmat.shape[1]): 
        ax.text(x=j, y=i, 
            s=confmat[i, j], 
                     va= 'center', ha='center') 
plt.xlabel('predicted label') 
plt.ylabel('true label') 
plt.show()
# % accuracy

sum(y_pred==y_val)/len(y_val)
# look at examples of classifications to understand/describe errors

#code2label = mdf[['code','category']].drop_duplicates()

#test['pred'] = y_pred
#test['predCat'] = pd.merge(test['pred'], code2label, how='left', left_on = 'pred', right_on = 'code')['category']

# can export to csv for viewing in spreadsheet apps
#test.to_csv(path+'logisticoutput.csv')
