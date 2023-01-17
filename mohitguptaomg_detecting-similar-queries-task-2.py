import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))
from sklearn.model_selection import train_test_split



def read_data():

    df = pd.read_csv("../input/train.csv")

    print ("Shape of base training File = ", df.shape)

    # Remove missing values and duplicates from training data

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)

    print("Shape of base training data after cleaning = ", df.shape)

    return df



df = read_data()

df_train, df_test = train_test_split(df, test_size = 0.02)

print ("\n\n", df_train.head(10))

print ("\nTrain Shape : ", df_train.shape)

print ("Test Shape : ", df_test.shape)
import matplotlib.pyplot as plt



def eda(data):

    dup_check = data['is_duplicate'].value_counts()

    plt.bar(dup_check.index, dup_check.values)

    plt.ylabel('Number of Queries')

    plt.xlabel('Is Duplicate')

    plt.title('Data Distribution', fontsize = 18)

    plt.show()

    

    print("\nAbove Graph Features :  [Is Not Duplicate | Is Duplicate]\n")

    print("Above Graph Indices  : ", dup_check.index)

    print("\nAbove Graph Values   : ", dup_check.values)



eda(df_train)

import re

import gensim

from gensim import corpora

from nltk.corpus import stopwords

from nltk.stem.porter import *
words = re.compile(r"\w+",re.I)

stopword = stopwords.words('english')

stemmer = PorterStemmer()



# Cleaning and tokenizing the queries.

def tokenize_questions(df):

    question_1_tokenized = []

    question_2_tokenized = []



    for q in df.question1.tolist():

        question_1_tokenized.append([stemmer.stem(i.lower()) for i in words.findall(q) 

                                     if i not in stopword])



    for q in df.question2.tolist():

        question_2_tokenized.append([stemmer.stem(i.lower()) for i in words.findall(q) 

                                     if i not in stopword])



    df["Question_1_tok"] = question_1_tokenized

    df["Question_2_tok"] = question_2_tokenized

    

    return df
df_train = tokenize_questions(df_train)

df_test = tokenize_questions(df_test)
df_train
df_test
def train_dictionary(df):

    

    questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()

    

    dictionary = corpora.Dictionary(questions_tokenized)

    dictionary.filter_extremes(no_below=5)

    dictionary.compactify()

    

    return dictionary
dictionary = train_dictionary(df_train)

print ("No of words in the dictionary = %s" %len(dictionary.token2id))
print(dictionary)
def get_vectors(df, dictionary):

    

    question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]

    question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]

    

    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))

    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))

    

    return question1_csc.transpose(),question2_csc.transpose()





q1_csc, q2_csc = get_vectors(df_train, dictionary)



print (q1_csc.shape)

print (q2_csc.shape)
q1_csc_test, q2_csc_test = get_vectors(df_test, dictionary)
'''

Similarity Measures:

    Cosine Similarity

    Manhattan Distance

    Euclidean Distance

'''



from sklearn.metrics.pairwise import cosine_similarity as cs

from sklearn.metrics.pairwise import manhattan_distances as md

from sklearn.metrics.pairwise import euclidean_distances as ed





def get_similarity_values(q1_csc, q2_csc):

    cosine_sim = []

    manhattan_dis = []

    eucledian_dis = []

        

    for i,j in zip(q1_csc, q2_csc):

        sim = cs(i,j)

        cosine_sim.append(sim[0][0])

        sim = md(i,j)

        manhattan_dis.append(sim[0][0])

        sim = ed(i,j)

        eucledian_dis.append(sim[0][0])

        

    return cosine_sim, manhattan_dis, eucledian_dis
cosine_sim, manhattan_dis, eucledian_dis = get_similarity_values(q1_csc, q2_csc)

y_pred_cos, y_pred_man, y_pred_euc = get_similarity_values(q1_csc_test, q2_csc_test)



print ("cosine_sim sample= \n", cosine_sim[0:5])

print ("\nmanhattan_dis sample = \n", manhattan_dis[0:5])

print ("\neucledian_dis sample = \n", eucledian_dis[0:5])
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



xtrain = pd.DataFrame({"cosine" : cosine_sim, "manhattan" : manhattan_dis,

                        "eucledian" : eucledian_dis})

ytrain = df_train.is_duplicate



xtest = pd.DataFrame({"cosine" : y_pred_cos, "manhattan" : y_pred_man,

                       "eucledian" : y_pred_euc})

ytest = df_test.is_duplicate

rf = RandomForestClassifier()

rf.fit(xtrain, ytrain)

rf_predicted = rf.predict(xtest)
logist = LogisticRegression(random_state=0)

logist.fit(xtrain, ytrain)

logist_predicted = logist.predict(xtest)
from sklearn.metrics import log_loss



def calculate_logloss(y_true, y_pred):

    loss_cal = log_loss(y_true, y_pred)

    return loss_cal
logloss_rf = calculate_logloss(ytest, rf_predicted)

log_loss_logist = calculate_logloss(ytest, logist_predicted)

print ("Log loss value using Random Forest is = %f" %logloss_rf)

print ("Log loss value using Logistic Regression is = %f" %log_loss_logist)
from sklearn.metrics import accuracy_score

test_acc_rf = accuracy_score(ytest, rf_predicted) * 100

test_acc_logist = accuracy_score(ytest, logist_predicted) * 100

print ("Accuracy of Random Forest Model : ", test_acc_rf)

print ("Accuracy of Logistic Regression Model : ", test_acc_logist)