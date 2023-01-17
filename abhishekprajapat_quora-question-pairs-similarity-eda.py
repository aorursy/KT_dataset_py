import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import os

import gc



import re
!unzip "../input/quora-question-pairs/train.csv.zip"
df = pd.read_csv("./train.csv")



print("Number of data points:",df.shape[0])
df.head()
df.info()
df.groupby("is_duplicate")['id'].count().plot.bar()
print('~> Total number of question pairs for training:\n   {}'.format(len(df)))
print('-> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(df['is_duplicate'].mean()*100, 2)))

print('\n-> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(df['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())

unique_qs = len(np.unique(qids))

qs_morethan_onetime = np.sum(qids.value_counts() > 1)

print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))

#print len(np.unique(qids))



print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime,qs_morethan_onetime/unique_qs*100))



print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 



q_vals=qids.value_counts()



q_vals=q_vals.values


x = ["unique_questions" , "Repeated Questions"]

y =  [unique_qs , qs_morethan_onetime]



plt.figure(figsize=(10, 6))

plt.title ("Plot representing unique and repeated questions  ")

sns.barplot(x,y)

plt.show()
#checking whether there are any repeated pair of questions



pair_duplicates = df[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()



print ("Number of duplicate questions",(pair_duplicates).shape[0] - df.shape[0])
plt.figure(figsize=(20, 10))



plt.hist(qids.value_counts(), bins=160)



plt.yscale('log', nonposy='clip')



plt.title('Log-Histogram of question appearance counts')



plt.xlabel('Number of occurences of question')



plt.ylabel('Number of questions')



print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 
#Checking whether there are any rows with null values

nan_rows = df[df.isnull().any(1)]

print (nan_rows)
# Filling the null values with ' '

df = df.fillna('')

nan_rows = df[df.isnull().any(1)]

print (nan_rows)
if os.path.isfile('df_fe_without_preprocessing_train.csv'):

    df = pd.read_csv("df_fe_without_preprocessing_train.csv",encoding='latin-1')

else:

    df['freq_qid1'] = df.groupby('qid1')['qid1'].transform('count') 

    df['freq_qid2'] = df.groupby('qid2')['qid2'].transform('count')

    df['q1len'] = df['question1'].str.len() 

    df['q2len'] = df['question2'].str.len()

    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))

    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))



    def normalized_word_Common(row):

        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

        return 1.0 * len(w1 & w2)

    df['word_Common'] = df.apply(normalized_word_Common, axis=1)



    def normalized_word_Total(row):

        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

        return 1.0 * (len(w1) + len(w2))

    df['word_Total'] = df.apply(normalized_word_Total, axis=1)



    def normalized_word_share(row):

        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

    df['word_share'] = df.apply(normalized_word_share, axis=1)



    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']

    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])



    df.to_csv("df_fe_without_preprocessing_train.csv", index=False)



df.head()
print ("Minimum length of the questions in question1 : " , min(df['q1_n_words']))



print ("Minimum length of the questions in question2 : " , min(df['q2_n_words']))



print ("Number of Questions with minimum length [question1] :", df[df['q1_n_words']== 1].shape[0])

print ("Number of Questions with minimum length [question2] :", df[df['q2_n_words']== 1].shape[0])
plt.figure(figsize=(12, 8))



plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:])



plt.subplot(1,2,2)

sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')

sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )

plt.show()
plt.figure(figsize=(12, 8))



plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = df[0:])



plt.subplot(1,2,2)

sns.distplot(df[df['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red')

sns.distplot(df[df['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue' )

plt.show()