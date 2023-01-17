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
#Import libraries here



import sqlite3 #For accessing sqlite3 data files.

from tqdm import tqdm #For visualizing the runtime of loop.



from sklearn.feature_extraction.text import CountVectorizer



from gensim.models import Word2Vec #For featurizing text into vectors using Word2Vec.

from gensim.models import KeyedVectors



from sklearn.neighbors import KNeighborsClassifier #For worling with KNN Classifiers.



from sklearn.model_selection import cross_val_score #For performing cross validation, and calculating Cross validation score

from sklearn.model_selection import train_test_split #To perform split of data.

from sklearn.metrics import accuracy_score #To calculate the accuracy of model.

print("cell executed")
#Load data file 

con = sqlite3.connect('../input/finals.sqlite')

final = pd.read_sql_query("""

SELECT * 

FROM Reviews

""", con)

print(final['Text'][0])

print(final['CleanedText'][0])
#We will balance the dataset.

df = final
df_positive = df[df['Score']=='positive']

df_negative = df[df['Score']=='negative']
df_negative = df_negative.sample(50000) #sample draws a random class

df_positive = df_positive.sample(50000)
from sklearn.utils import shuffle
final_df = shuffle(pd.concat([df_positive, df_negative]))
df_word2vec = final_df.sort_values(by='Time', inplace=False) #Store dataframe df into a new variable to work on.
#Featurize above sorted dataframe df using word2vec. I will use average-Word2Vec only right now. 

#This function trains the model.

def train_word2vec_model(df_word2vec):

    list_of_sentences = [] #A list consisting of each review, split into words.

    

    #This loop gives us list of sentences.

    for each_review in tqdm(df_word2vec['CleanedText'].values):

        list_of_sentences.append(each_review.split())

    print(df['CleanedText'][0])

    print(list_of_sentences[0])

    

    w2v_model = Word2Vec(list_of_sentences, min_count=5,size=50,workers=4) # Training of model, and this will take time .

    

    return w2v_model, list_of_sentences #Both of them will be used to featurize later using avg_word2vec or tfidf_word2vec.

print("cell executed")
#Train Word2Vec model on the basis of your text corpus.

w2v_model, list_of_sentences = train_word2vec_model(df_word2vec) #Calling function to train model.

print("cell executed")
#This function creates average word2vec .

def avg_word2vec(list_of_sentences, w2v_model):

    w2v_words = list(w2v_model.wv.vocab) #This is the list of all the words whose vector we have now

    avgw2v_each_review = [] #This list will store avg_w2v vector of each review.

    for each_sentence in tqdm(list_of_sentences):

        this_sentence_vector = np.zeros(50) #This vector will hold representation of this particular review ~ 'each sentence'.

        #This is intitalized zero since we will perform vector addition of the vector representation of each word in this review 

        #and then average their sum.

        cnt_words = 0 #It will keep on counting those words which we will consider for featurization. i.e present in w2v model

        for word in each_sentence:

            if word in w2v_words:

                vec = w2v_model.wv[word] #Returns the vector of word.

                this_sentence_vector += vec

                cnt_words += 1

        if cnt_words != 0:

            this_sentence_vector /= cnt_words #Took the average.

        avgw2v_each_review.append(this_sentence_vector)

    return avgw2v_each_review

print("cell executed")
avgw2v_each_review = avg_word2vec(list_of_sentences, w2v_model) #Calling function to perform avg word2vec model.

print(type(avgw2v_each_review))

print("cell executed")
#We will perform simple cross validation now to find best k for KNN



#Funtion to perform train test split.

def perform_train_test_split(X,y):

    

    X_train,y_train = X[0:70000], y[0:70000]

    X_test,y_test = X[70000:], y[70000:]

    '''

    For simple Cross Validation

    #X_1,X_test,y_1,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

    #X_train,X_cv,y_train,y_cv = train_test_split(X_1,y_1,test_size=0.3,random_state=0) 

    Do not use train_test_split while using time based splitting.

    '''

    return(X_train,y_train,X_test,y_test)
#Calling the function.

X = avgw2v_each_review #X is list

y = df_word2vec['Score'] #y is a Series

X_train,y_train,X_test,y_test = perform_train_test_split(X,y) 
#This function performs cross validation and help us finding the optimal value of 'k' to be used in KNN

'''def perform_cross_validation(X_train,y_train,X_cv,y_cv):

    

    for k in tqdm(range(1,30,2)): #We have considdering only odd values for k. The best k will be used in KNN.

        knn = KNeighborsClassifier(n_neighbors=k) #This will instantiate a knn model for each value of k.

        knn.fit(X_train,y_train) #Training of model.

        pred = knn.predict(X_cv) #Predicting class labels corresponding to X_cv using trained knn.

        accuracy = accuracy_score(y_cv,pred,normalize=True) * float(100) 

        #y_cv are original values, and 'pred' are predicted values, multiplying will simply gives us %age accuracy.

        

        print("accuracy for {} is {}".format(k,accuracy))

'''
# perform_cross_validation(X_train,y_train,X_cv,y_cv) #Calling function to perform simple cross validation.
#Function to perform 5-fold-cross-validation.



def k_fold_cv(X_train,y_train,max_cv):

    cv_scores = []

    for k in tqdm(range(1,50,2)): #For different values of k to be used in KNN

        knn = KNeighborsClassifier(n_neighbors=k) #Initialize classifier for each k.

        scores = cross_val_score(knn, X_train, y_train, cv=max_cv, scoring='accuracy') #Calculate cv scores for each k.

        cv_scores.append(tuple([k, scores.mean()*float(100)])) 

    return cv_scores



def get_optimal_k(cv_scores):

    cv_scores.sort(key = lambda x:x[1], reverse=True) #Sorting on the basis of scores,'reverse' performs descending.

    return (cv_scores[0][0],cv_scores[0][1])  #This will be the best k.
#Calling function to perform 5-fold-cv.

max_cv = 5 #Since we are performing 5-fold-cv

%time cv_scores = k_fold_cv(X_train,y_train, max_cv)

print(cv_scores)

k,cv_accuracy = get_optimal_k(cv_scores)

print("The best value of k is {} for which accuracy is {}".format(k,cv_accuracy))
#Function to perform knkn with optimal k.

def perform_knn_optimal_k(X_train,y_train,X_test,y_test,k):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    pred = knn.predict(X_test)

    test_accuracy = accuracy_score(y_test, pred,normalize=True) * float(100)

    #Test accuracy help us to test overfitting.

    print("The accuracy for k = {} is {}".format(k,test_accuracy))
optimal_k = k

perform_knn_optimal_k(X_train,y_train,X_test,y_test,optimal_k)