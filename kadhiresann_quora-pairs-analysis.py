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
#Read the train data 

train_df = pd.read_csv('/kaggle/input/quoratrainandtestset/train.csv')

train_df.info()
#Display the train data

train_df.head()
#Read the test data

test_df = pd.read_csv('/kaggle/input/quoratrainandtestset/test.csv')

test_df.info()
#Display the test data

test_df.head()


import matplotlib.pyplot as plt



qids = pd.Series(train_df['qid1'].tolist() + train_df['qid2'].tolist())



plt.figure(figsize=(12, 8))

plt.hist(qids.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of occurance of question counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions');

train_df.isnull().sum()
test_df.isnull().sum()
#Retrieve the train rows which are null 

train_df[train_df.isnull().any(axis=1)]
#Retrieve the test rows which are null 

test_df[test_df.isnull().any(axis=1)]
#Filling the blanks which are null

train_df.fillna('',inplace=True)

test_df.fillna('',inplace=True)
print(f'No of missing entries in traninig  dataset is \n {train_df.isnull().sum()}, and test dataset  is \n {test_df.isnull().sum()}')
print(f'Percentage of repeated questions of qid1 in train data {round(100* np.sum(train_df.qid1.value_counts()> 1)/len(train_df.qid1),2)} %')

print(f'Percentage of repeated questions of qid2 in train data {round(100* np.sum(train_df.qid2.value_counts()> 1)/len(train_df.qid2),2)} %')
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



#Function to calculate the cosine similarity for a given dataframe

def get_cosine_sim(x):



    # Create the question Term Matrix



    count_vectorizer = CountVectorizer(stop_words='english') 

    count_vectorizer = CountVectorizer()

    

    questions=[x.question1,x.question2] # take two question columns

    

    sparse_matrix = count_vectorizer.fit_transform(questions)   

    similarity = cosine_similarity(sparse_matrix[0:1],sparse_matrix) # Create sparse matrix with cosine similarity value

    return similarity[0][1]
#Copy 2000 records of train dataset

trn_df=train_df.head(2000)
#Create a column called similarity to stroe the cosine value 

trn_df.loc[:,'similarity']=trn_df.apply(get_cosine_sim,axis=1)
# Make the similarity score to 1 if its score is greater than 0.5 else make it zero

trn_df.similarity=trn_df.similarity.apply(lambda x : 1 if x > 0.5 else 0)
#show the observations between predicted similarity and is_duplicate

trn_df.head(10) 
#True positive : i.e. question 1 and question 2 are similar and its identified correctly

accuracy_cosine_tp=100* len(trn_df.loc[((trn_df['is_duplicate'] == 1 ) & (trn_df['similarity']==1))])/2000



#True Negative : i.e question 1 and question 2 are different and its identified correctly

accuracy_cosine_tn=100* len(trn_df.loc[((trn_df['is_duplicate'] == 0 ) & (trn_df['similarity']==0))])/2000



print(f'Predicted score of True Positive cases for cosine similarity in train data {accuracy_cosine_tp}%')

print(f'Predicted score oof True Negative cases for cosine similarity in train data {accuracy_cosine_tn}%')

print(f'Total Predicted score {accuracy_cosine_tp + accuracy_cosine_tn}%')

#Copy only 2000 recrods of test data 

tst_df=test_df.head(2000)
#Create a columnn called is_duplicate to store the cosine value

tst_df.loc[:,'is_duplicate']=tst_df.apply(get_cosine_sim,axis=1)
#Display the dataframe after is_duplicate addition

tst_df.head()
# create a submission file from test data

submission_file=tst_df.loc[:,['test_id','is_duplicate']]

submission_file.head()
#Converting the submission file into CSV



submission_file.to_csv('Submisson_File',

                       sep=',',

                       header=True,

                       index=None

                      )

submission_file.to_csv(r'Submission_File.csv',index=False)
#Export the submission file

from IPython.display import FileLink

FileLink(r'Submission_File.csv')
from sklearn.model_selection import train_test_split



# train-test split

X_train, X_test, y_train, y_test = train_test_split(train_df.drop("is_duplicate",axis=1),train_df["is_duplicate"],test_size = 0.2, random_state = 7)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



y_test = y_test.tolist()

y_train = y_train.tolist()
# from the train dataset , merge q1& q2 and build a term-doc matrix

train_list=X_train['question1'].tolist()+ [""] + X_train['question2'].tolist()
# Create the question Term Matrix



count_vectorizer = CountVectorizer(stop_words='english') 

count_vectorizer = CountVectorizer()

    

# fitting a term-doc matrix

sparse_matrix_term_doc = count_vectorizer.fit_transform(train_list)
# transform training data, based on question1 and question2

train_sparse_matrix_term_doc_1 =  count_vectorizer.transform(X_train['question1'].tolist())

train_sparse_matrix_term_doc_2 =  count_vectorizer.transform(X_train['question2'].tolist())


# if the word exists in one question then it would be  1

# if the word exists in both questions then it would be 2 

# if the word doesn't exist it would be 0



x = train_sparse_matrix_term_doc_1  + train_sparse_matrix_term_doc_1  



y = y_train
#apply transform of count vectorizer to X_test



test_spare_matrix_term_doc1 = count_vectorizer.transform(X_test['question1'].tolist())

test_spare_matrix_term_doc2 = count_vectorizer.transform(X_test['question2'].tolist())
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.metrics import accuracy_score



lg= LogisticRegression(C=0.4, dual=True) # create logistic reg object  with penality 0.4



lg.fit(x, y) # fit the logsictic reg model on train data



#predicting target varible using train data

y_pred_class_train = lg.predict(x)



#predicting probabilities

y_pred_prob_train = lg.predict_proba(x)





print(f'Accuracy of training {accuracy_score(y_train,y_pred_class_train )}')

print(f'Log-loss of training {log_loss(y_train, y_pred_prob_train )}')



#predicting target varible using test data

y_pred_class_test = lg.predict(test_spare_matrix_term_doc1 + test_spare_matrix_term_doc2)



#predicting probabilities

y_pred_prob_test = lg.predict_proba(test_spare_matrix_term_doc2 + test_spare_matrix_term_doc2)



print(f'Accuracy of test {accuracy_score(y_test,y_pred_class_test)}')

print(f'Log-loss of test {log_loss(y_test, y_pred_prob_test)}')