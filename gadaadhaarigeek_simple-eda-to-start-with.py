import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install distance
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import zipfile
zf = zipfile.ZipFile("/kaggle/input/quora-question-pairs/train.csv.zip") 

data = pd.read_csv(zf.open('train.csv'))

print("Number of data points:", data.shape[0])
data.head()
data.info()
# data.groupby("is_duplicate")['id'].count().plot.bar()

sns.countplot(x="is_duplicate", data=data)
print('~> Total number of question pairs for training:\n   {}'.format(len(data)))
print('~> Question pairs are not Similar (is_duplicate = 0):\n   {}%'.format(100 - round(data['is_duplicate'].mean()*100, 2)))

print('\n~> Question pairs are Similar (is_duplicate = 1):\n   {}%'.format(round(data['is_duplicate'].mean()*100, 2)))
qids = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())

unique_qs = len(np.unique(qids))

qs_more_than_onetime = np.sum(qids.value_counts() > 1)

print ('Total number of  Unique Questions are: {}\n'.format(unique_qs))

#print len(np.unique(qids))



print ('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_more_than_onetime,qs_more_than_onetime/unique_qs*100))



print ('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 



q_vals=qids.value_counts()



q_vals=q_vals.values
x = ["unique_questions" , "Repeated Questions"]

y =  [unique_qs , qs_more_than_onetime]



plt.figure(figsize=(10, 6))

plt.title ("Plot representing unique and repeated questions  ")

sns.barplot(x,y)

plt.show()
#checking whether there are any repeated pair of questions



pair_duplicates = data[['qid1','qid2','is_duplicate']].groupby(['qid1','qid2']).count().reset_index()



print ("Number of duplicate questions",(pair_duplicates).shape[0] - data.shape[0])
plt.figure(figsize=(20, 10))



plt.hist(qids.value_counts(), bins=160)



plt.yscale('log', nonposy='clip')



plt.title('Log-Histogram of question appearance counts')



plt.xlabel('Number of times, a question appears')



plt.ylabel('Number of questions')



print ('Maximum number of times a single question is repeated: {}\n'.format(max(qids.value_counts()))) 
#Checking whether there are any rows with null values

nan_rows = data[data.isnull().any(1)]

print (nan_rows)
# Filling the null values with ' '

data = data.fillna('')

nan_rows = data[data.isnull().any(1)]

print (nan_rows)
data['freq_qid1'] = data.groupby('qid1')['qid1'].transform('count') 

data['freq_qid2'] = data.groupby('qid2')['qid2'].transform('count')

data['q1len'] = data['question1'].str.len() 

data['q2len'] = data['question2'].str.len()

data['q1_n_words'] = data['question1'].apply(lambda row: len(row.split(" ")))

data['q2_n_words'] = data['question2'].apply(lambda row: len(row.split(" ")))



def normalized_word_Common(row):

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

    return 1.0 * len(w1 & w2)

data['word_Common'] = data.apply(normalized_word_Common, axis=1)



def normalized_word_Total(row):

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

    return 1.0 * (len(w1) + len(w2))

data['word_Total'] = data.apply(normalized_word_Total, axis=1)



def normalized_word_share(row):

    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))

    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

data['word_share'] = data.apply(normalized_word_share, axis=1)



data['freq_q1+q2'] = data['freq_qid1'] + data['freq_qid2']

data['freq_q1-q2'] = abs(data['freq_qid1'] - data['freq_qid2'])



data.head()
print ("Minimum length of the questions in question1 : " , min(data['q1_n_words']))



print ("Minimum length of the questions in question2 : " , min(data['q2_n_words']))



print ("Number of Questions with minimum length [question1] :", data[data['q1_n_words']== 1].shape[0])

print ("Number of Questions with minimum length [question2] :", data[data['q2_n_words']== 1].shape[0])
data[data["q1_n_words"] == 1].head()
plt.figure(figsize=(15, 8))



plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'word_share', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['is_duplicate'] == 1.0]['word_share'][0:] , label = "1", color = 'red')

sns.distplot(data[data['is_duplicate'] == 0.0]['word_share'][0:] , label = "0" , color = 'blue' )

plt.show()
plt.figure(figsize=(15, 8))



plt.subplot(1,2,1)

sns.violinplot(x = 'is_duplicate', y = 'word_Common', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['is_duplicate'] == 1.0]['word_Common'][0:] , label = "1", color = 'red')

sns.distplot(data[data['is_duplicate'] == 0.0]['word_Common'][0:] , label = "0" , color = 'blue' )

plt.show()