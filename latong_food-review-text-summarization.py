import nltk

from bs4 import BeautifulSoup

import string

from nltk.tokenize import RegexpTokenizer

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
#load data

data= pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
data.head()
len(data) # Total rows 
#Drop the columns where at least one element is missing.

df=data.dropna()

# and check again how many rows left

len(df) 

#We need only three columns Score, Summary & Text. Let 's remove all the other columns.



df=df[['Score', 'Summary','Text']]

df.head()
#calculating length of lists in pandas dataframe column 

#ref. https://stackoverflow.com/questions/41340341/pythonic-way-for-calculating-length-of-lists-in-pandas-dataframe-column

sum=df['Summary'].str.len()

print(sum)
# let's check the length of summaries, the average length is 20 characters.

df['summary length'] = df['Summary'].apply(len)

df['summary length'].describe()
import seaborn as sns

sns.boxplot(x='Score', y=df['summary length'], data=df)
# let's do the same to the text, the avearage length is 302 characters.

df['Text length'] = df['Text'].apply(len)

df['Text length'].describe()
import seaborn as sns

import matplotlib.pyplot as plt

sns.boxplot(x='Score', y=df['Text length'], data=df)

plt.ylim(0, 900)
# Let 's keep the max length of summary to 30 characters and Text to 300 characters. 

df= df[df['summary length'] <=30]

df= df[df['Text length'] <=300]



len(df)

df.head()
df=df[['Score', 'Summary','Text']]

df.head()

#len(df) the length of updated dataset is 239868
df = df.reset_index(drop=True)# we need to reindex the dataset after removed some rows

df.head()
import requests

from bs4 import BeautifulSoup 



def preprocess_text(text):

    """ Apply any preprocessing methods"""

    text = BeautifulSoup(text).get_text()

    text = text.lower()

    text = text.replace('[^\w\s]','')

    return text



df["Text"] = df.Text.apply(preprocess_text)

df["Summary"] = df.Summary.apply(preprocess_text)
field_length = df.Text.astype(str).map(len)

print()

print("###This is the longest text###")

print (df.loc[field_length.idxmax(), 'Text'])

print()

print("###This is the shortest text###")

print()

print (df.loc[field_length.idxmin(), 'Text'])
from nltk import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

#not is in the stopword list. W/O adding whitelist, the summary would change from "do not recommend" to "recommend". This solution was borrowed from bertcarremans https://github.com/bertcarremans/TwitterUSAirlineSentiment

whitelist = ["n't", "not", "no"]



df['Text_after_removed_stopwords'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word in whitelist or word not in (stop)]))

print()

print('###Text after removed stopwords###'+'\n'+df['Text_after_removed_stopwords'][1])

print()

print('###Text before removed stopwords###'+'\n'+ df['Text'][1])

print()

df['Summary_after_removed_stopwords'] = df['Summary'].apply(lambda x: ' '.join([word for word in x.split() if word in whitelist or word not in (stop)]))

print('###Summary after removed stopwords###'+ '\n'+df['Summary_after_removed_stopwords'][1])

print()

print('###Summary before removed stopwords###'+'\n'+df['Summary'][1])
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

contractions = { 

"ain't": "am not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he's": "he is",

"how'd": "how did",

"how'll": "how will",

"how's": "how is",

"i'd": "i would",

"i'll": "i will",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it would",

"it'll": "it will",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"must've": "must have",

"mustn't": "must not",

"needn't": "need not",

"oughtn't": "ought not",

"shan't": "shall not",

"sha'n't": "shall not",

"she'd": "she would",

"she'll": "she will",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"that'd": "that would",

"that's": "that is",

"there'd": "there had",

"there's": "there is",

"they'd": "they would",

"they'll": "they will",

"they're": "they are",

"they've": "they have",

"wasn't": "was not",

"we'd": "we would",

"we'll": "we will",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"where'd": "where did",

"where's": "where is",

"who'll": "who will",

"who's": "who is",

"won't": "will not",

"wouldn't": "would not",

"you'd": "you would",

"you'll": "you will",

"you're": "you are"

}
#This two blocks of code was refered from https://www.kaggle.com/currie32/summarizing-text-with-amazon-reviews

def clean_text(text):



    # Replace contractions with longer forms in the above list

    if True:

        text = text.split()

        new_text = []

        for word in text:

            if word in contractions:

                new_text.append(contractions[word])

            else:

                new_text.append(word)

        text = " ".join(new_text)


clean_summaries = []

for summary in df.Summary:

    clean_summaries.append(clean_text(summary))

print("Summaries are complete.")



print(len(clean_summaries))





clean_texts = []

for text in df.Text:

    clean_texts.append(clean_text(text))

print("Texts are complete.")

print(len(clean_texts))

clean_text
df.head()
df.to_csv(r'/amazon_clean.csv',index=False)

len(df)
df1=df.sample(frac=0.001, replace=True, random_state=1)

len(df1)
df1 = df1.reset_index(drop=True)# we need to reindex the dataset after removed some rows
df1.tail()
df2=df1[['Summary','Text']]

df2.head()

docs=df2.apply(lambda x: " ".join(x), axis=1)

docs.head()
from sklearn.feature_extraction.text import CountVectorizer

#create a vocabulary of words, 

#ignore words that appear in 85% of documents, 

#eliminate stop words

docs=df2.apply(lambda x: " ".join(x), axis=1).tolist()



cv=CountVectorizer(max_df=0.85,max_features=10000)

word_count_vector=cv.fit_transform(docs)

 

list(cv.vocabulary_.keys())[:10]

 
from sklearn.feature_extraction.text import TfidfTransformer

 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(word_count_vector)
word_count_vector.shape
"""

https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XYtStuczbOQ

need the term frequency (term count) vectors for different tasks, use Tfidftransformer.

need to compute tf-idf scores on documents within“training” dataset, use Tfidfvectorizer

need to compute tf-idf scores on documents outside “training” dataset, use either one, both will work.



"""



# print idf values

df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

 

# sort ascending

df_idf.sort_values(by=['idf_weights'])
df3=df[['Summary','Text']]



df3.tail()
#https://github.com/kavgan/nlp-in-practice/blob/master/tf-idf/Keyword%20Extraction%20with%20TF-IDF%20and%20SKlearn.ipynb

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]



    score_vals = []

    feature_vals = []



    for idx, score in sorted_items:

        fname = feature_names[idx]

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])



    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results
# you only needs to do this once, this is a mapping of index to 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

feature_names=cv.get_feature_names()

 

# get the document that we want to extract keywords from

doc=df['Text'][239866]

 

#generate tf-idf for the given document

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

 

#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(tf_idf_vector.tocoo())

 

#extract only the top n; n here is 10

keywords=extract_topn_from_vector(feature_names,sorted_items,10)

 

# now print the results

print("\n=====Doc=====")

print(doc)

print("\n===Keywords===")

for k in keywords:

    print(k,keywords[k])

    

print("\n===original summary===")

print(df['Summary'][239866])

 
