!wget http://tinyurl.com/dec130-helperfunc

!mv dec130-helperfunc default-functions.py

%run default-functions.py
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

from matplotlib import rcParams

import scipy.stats as stats

import os

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading the datasets

## for comparisons

Reward_Satisfaction_Comparison_df = pd.read_csv("../input/deputy reward satisfaction comparison.csv", index_col=0)     

Mentor_Effort_Comparison_df = pd.read_csv("../input/mentor effort comparison.csv", index_col = 0)

Cooperation_Score_Comparison_df = pd.read_csv("../input/months cooperation score comparison.csv", index_col = 0)

Sustainability_Score_Comparison_df = pd.read_csv("../input/months sustainability score comparison.csv", index_col = 0)     



## for department evaluation

PMD_reg_df = pd.read_csv("../input/PMD final.csv", index_col = 0)



## for kamustahan

Kamustahan_df = pd.read_csv("../input/Kamustahan.csv", index_col = 0)



# this fills in the missing data

Reward_Satisfaction_Comparison_df = Reward_Satisfaction_Comparison_df.fillna(method='ffill')

Mentor_Effort_Comparison_df = Mentor_Effort_Comparison_df.fillna(method='ffill')

Cooperation_Score_Comparison_df = Cooperation_Score_Comparison_df.fillna(method='ffill')

Sustainability_Score_Comparison_df = Sustainability_Score_Comparison_df.fillna(method='ffill')

PMD_reg_df = PMD_reg_df.fillna(method='ffill')

Kamustahan_df = Kamustahan_df.fillna(method='ffill')
Reward_Satisfaction_Comparison_df.head()
Mentor_Effort_Comparison_df.head()
Cooperation_Score_Comparison_df.head()
Sustainability_Score_Comparison_df.head()
PMD_reg_df.head()
Kamustahan_df.head()
Reward_Satisfaction_Comparison_df.plot.line()
Mentor_Effort_Comparison_df.plot.line()
Cooperation_Score_Comparison_df.plot.line()
Sustainability_Score_Comparison_df.plot.line()
def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]
get_redundant_pairs(PMD_reg_df)

get_top_abs_correlations(PMD_reg_df, n=10)
import statsmodels.api as sm

PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Time Management Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  

PMD_TimeManagement_model.summary()
PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Problem-Solving Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  

PMD_TimeManagement_model.summary()
PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Planning Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  

PMD_TimeManagement_model.summary()
PMD_TimeManagement_model = sm.OLS(PMD_reg_df["D Management Improvement Score"], sm.add_constant(PMD_reg_df.drop(['D Time Management Improvement Score', 'D Problem-Solving Improvement Score','D Planning Improvement Score','D Management Improvement Score'], axis = 1))).fit()  

PMD_TimeManagement_model.summary()
Kamustahan_df.head()
# breaking down the text to single words

# example: "I love Spongebob" becomes "I" "love" "Spongebob"

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')



words_academic = [ tokenizer.tokenize(x.lower())  for x in Kamustahan_df.Academic_Goal]

words_PEERS = [ tokenizer.tokenize(x.lower())  for x in Kamustahan_df.PEERS_Goal]

words_other = [ tokenizer.tokenize(x.lower())  for x in Kamustahan_df.Others_Goal]



Kamustahan_df['Academic_Goal_words']= words_academic

Kamustahan_df['PEERS_Goal_words']= words_PEERS

Kamustahan_df['Others_Goal_words']= words_other

Kamustahan_df.head()
# importing data of stopwords AKA useless words like "the" "he" "she"

from nltk.corpus import stopwords

from IPython.display import clear_output

import nltk

nltk.download('stopwords')

stopwords.words('english')

clear_output()

stp = stopwords.words('english')
# adding my own stop words to the list

stp = stopwords.words('english') + ['ii','iii','read full','full article','read full article','silk','looking', 'information','read','full','article', 'glynn', 'wsj', 'com', 'jamesglynnwsj', 'james',

                              'james glynn wsj com', 'jamesglynnwsj', 'james glynn wsj com', 'james glynn wsj', 'james glynn', 'com jamesglynnwsj','glynn wsj com jamesglynnwsj', 'q3','3q','corresponding','graph', 'olga', 'cotaga','olgacotaga','olga', 'olga cotaga','jamesglynn','m2','2017','pm','daily shot','daily','shot']

                               
# Removing Stop Words

Kamustahan_df['Academic_Goal_words']=[[y for y in x if y not in stp] for x in Kamustahan_df['Academic_Goal']]

Kamustahan_df['PEERS_Goal_words']=[[y for y in x if y not in stp] for x in Kamustahan_df['PEERS_Goal']]

Kamustahan_df['Others_Goal_words']=[[y for y in x if y not in stp] for x in Kamustahan_df['Others_Goal']]
# stemming words to turn words like "Star" and "Wars" that have no meaning separately to "Star Wars"

from nltk.stem.porter import PorterStemmer

p_stemmer = PorterStemmer()

Kamustahan_df['Academic_Goal_words'] = [[p_stemmer.stem(y) for y in x] for x in Kamustahan_df['Academic_Goal']]

Kamustahan_df['PEERS_Goal_words'] = [[p_stemmer.stem(y) for y in x] for x in Kamustahan_df['PEERS_Goal']]

Kamustahan_df['Others_Goal_words'] = [[p_stemmer.stem(y) for y in x] for x in Kamustahan_df['Others_Goal']]
# this function makes columns that counts the number of times a word appears in each message

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def vectorize(dataset, nFeat, stp, how = 'CountVectorizer'):

    no_features = nFeat

    

    if how == 'CountVectorizer':

        tf_vectorizer = CountVectorizer(max_features=no_features, min_df = 3, max_df = 70, stop_words=stp, ngram_range = (1, 10))

    else:

        tf_vectorizer = TfidfVectorizer(max_features=no_features, min_df = 3, max_df = 70, stop_words=stp, ngram_range = (1, 10), norm=None)     

        

    tf = tf_vectorizer.fit_transform(dataset)

    tf_feature_names = tf_vectorizer.get_feature_names()

        

    return tf_vectorizer, tf, tf_feature_names
# doing inverse document frequency for academic goal messages

vect_idf1, corp_idf1, tf_names_idf1 = vectorize(Kamustahan_df['Academic_Goal'], 500, stp, how= TfidfVectorizer)   

tf_idf_doc1 = pd.DataFrame(corp_idf1.toarray())

tf_idf_doc1.columns = tf_names_idf1

tf_idf_doc1.head()
# doing inverse document frequency for PEERS goal messages

vect_idf2, corp_idf2, tf_names_idf2 = vectorize(Kamustahan_df['PEERS_Goal'], 500, stp, how= TfidfVectorizer)   

tf_idf_doc2 = pd.DataFrame(corp_idf2.toarray())

tf_idf_doc2.columns = tf_names_idf2

tf_idf_doc2.head()
# doing inverse document frequency for others goal messages

vect_idf3, corp_idf3, tf_names_idf3 = vectorize(Kamustahan_df['PEERS_Goal'], 500, stp, how= TfidfVectorizer)   

tf_idf_doc3 = pd.DataFrame(corp_idf2.toarray())

tf_idf_doc3.columns = tf_names_idf3

tf_idf_doc3.head()
from sklearn.decomposition import NMF, LatentDirichletAllocation

def LDArun(corpus, tf_feature_names, ntopics, nwords, print_yes):

    

    model = LatentDirichletAllocation(n_components=ntopics, max_iter=500,learning_method='online', random_state=10).fit(corpus)

    

    

    normprob = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    

    topics = {}

    

    

    for topic_idx, topic in enumerate(normprob):

        kw = {}

        k = 1

        

        for i in topic.argsort()[:-nwords -1:-1]:

                kw[k] = tf_feature_names[i]

                k+=1

                

        topics[topic_idx] = kw

        

        if print_yes:

            print('topic', topic_idx, "|",[tf_feature_names[i]  for i in topic.argsort()[:-nwords - 1:-1]])

    



    return (model, pd.DataFrame(topics).transpose())



def getTopic(x):

    if max(x) == 1/len(x):

        return ('N/A')

    else:

        return np.argmax(list(x))
# getting 5 topics of academic goals and the top 10 words in that topic to know what to name the topic

lda1, topics1 = LDArun(corp_idf1, tf_names_idf1,5, 10, True)
# naming the topics in the academic goals messages

topic_dict1 = {0:'Honors', 1:'Passing', 2:'Bawi', 3:'Latin Honors', 4:'Minor'}
# getting 5 topics of PEERS goals and the top 10 words in that topic to know what to name the topic

lda2, topics2 = LDArun(corp_idf2, tf_names_idf2,5, 10, True)
# naming the topics in the PEERS goals messages

topic_dict2 = {0:'Officer', 1:'Experiment', 2:'Mental Health', 3:'Crush', 4:'Help'}
lda3, topics3 = LDArun(corp_idf3, tf_names_idf3,5, 10, True)
# naming the topics in the others goals messages

topic_dict3 = {0:'Family', 1:'Friends', 2:'Church', 3:'Basketball', 4:'Spongebob'}
import numpy as np

topic_df1 = pd.DataFrame(lda1.transform(corp_idf1))

topic_df1["Partners_Student_Number"] = Kamustahan_df.Partners_Student_Number.values

topic_df1 = topic_df1.set_index("Partners_Student_Number")

topic_df1.columns = topic_dict1.values()

topic_df1['Final Topic'] = [topic_dict1[np.argmax([v, w, x, y, z])] for v, w, x, y, z in zip(topic_df1.iloc[:, 0], topic_df1.iloc[:, 1], topic_df1.iloc[:, 2], topic_df1.iloc[:, 3], topic_df1.iloc[:, 4])]
# Final Topic shows what they mostly talk about when talking about academic goals

topic_df1.head()
import numpy as np

topic_df2 = pd.DataFrame(lda2.transform(corp_idf2))

topic_df2["Partners_Student_Number"] = Kamustahan_df.Partners_Student_Number.values

topic_df2 = topic_df2.set_index("Partners_Student_Number")

topic_df2.columns = topic_dict2.values()

topic_df2['Final Topic'] = [topic_dict2[np.argmax([v, w, x, y, z])] for v, w, x, y, z in zip(topic_df2.iloc[:, 0], topic_df2.iloc[:, 2], topic_df2.iloc[:, 2], topic_df2.iloc[:, 3], topic_df2.iloc[:, 4])]

# Final Topic shows what they mostly talk about when talking about PEERS goals

topic_df2.head()
import numpy as np

topic_df3 = pd.DataFrame(lda3.transform(corp_idf3))

topic_df3["Partners_Student_Number"] = Kamustahan_df.Partners_Student_Number.values

topic_df3 = topic_df3.set_index("Partners_Student_Number")

topic_df3.columns = topic_dict3.values()

topic_df3['Final Topic'] = [topic_dict3[np.argmax([v, w, x, y, z])] for v, w, x, y, z in zip(topic_df3.iloc[:, 0], topic_df3.iloc[:, 3], topic_df3.iloc[:, 3], topic_df3.iloc[:, 3], topic_df3.iloc[:, 4])]

# Final Topic shows what they mostly talk about when talking about other goals

topic_df3.head()
cols = 'Final Topic'

specific_acads = topic_df1.loc[18234, cols]

specific_acads
specific_PEERS = topic_df2.loc[18234, cols]

specific_PEERS
specific_others = topic_df3.loc[18234, cols]

specific_others