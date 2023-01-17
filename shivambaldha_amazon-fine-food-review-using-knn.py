#import all we needed module
import numpy as np
import pandas as pd 
import sqlite3
import matplotlib.pyplot as plt
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
#from sklearn import cross_validation******** this is not working
from collections import Counter
import os
print(os.listdir("../input/")) 
con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')

# we neglect the review having a score = 3

filtered_data = pd.read_sql_query('''select *from reviews where Score !=3''',con)

def partition(x):
    if x<3 :
        return 'negative'
    return 'positive'
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative
filtered_data.shape
filtered_data.head()
filtered_data['Score'].value_counts()
display = pd.read_sql_query('''select * from reviews where Score != 3 and userId = "AR5J8UI46CURR" order by ProductID''',con)
display.head()
# we sorting a data according to productid in ascending order

sorted_data = filtered_data.sort_values('ProductId',axis = 0,ascending=True,inplace = False,kind='quicksort',na_position='last')
final = sorted_data.drop_duplicates(subset={'UserId','ProfileName','Time','Text'})
final.shape
# After removing duplication , we will see how much %of data still remaing

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display = pd.read_sql_query('''select * from reviews where Score != 3 and Id = 44737 or Id= 64422 order by ProductId ''',con)
display.head()
#we remove duplication using HelpfulnessDenominator and HelpfulnessNumerator.

final = final[final.HelpfulnessNumerator<= final.HelpfulnessDenominator]
final.shape
final['Score'].value_counts()
import re 

i = 0
for sent in final['Text'].values:
    if(len(re.findall('<.>*?',sent))):
        print(i)
        print(sent)
        break;
    i+=1
import string
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') # initialising snowball stemmer

def cleanhtml(sentence): #function to clean word of any html tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr , ' ',sentence)
    return cleantext

def cleanpunc(sentence): #function to clean word of any punctuation or special character
    cleaned = re.sub(r'[?|!|\,|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r'',cleaned)
    return cleaned
print(stop)
print('***********************************************')
print(sno.stem('tasty'))
#Code for implementing step-by-step the checks mentioned in the pre-processing phase
# this code takes a while to run as it needs to run on 500k sentences.

i = 0
strl = ' '
final_string  = []
all_positive_words=[] # store words from +ve reviews here
all_negattive_words=[]#store words from -ve reviews here
s= ''

for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent) #remove html tag
    
    for w in sent.split():
        for cleaned_word in cleanpunc(w).split():
            if((cleaned_word.isalpha()) & (len(cleaned_word)>2)):
                if(cleaned_word.lower() not in stop):
                    s = (sno.stem(cleaned_word.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if(final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s) #list of all words use to store +ve list 
                    if (final['Score'].values)[i] == 'negative':
                        all_negattive_words.append(s) #list of all words use to store -Ve list
                else:
                    continue
            else:
                continue
                
    #print filtered sentens 
    strl = b" ".join(filtered_sentence) #final string of cleaned words
    final_string.append(strl)
    i+=1
final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
final.head(3) #below the processed review can be seen in the CleanedText Column 


# store final table into an SQlLite table for future.
conn = sqlite3.connect('finalassignment.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)
import sqlite3
con = sqlite3.connect('finalassignment.sqlite')
cleaned_data = pd.read_sql_query('select * from Reviews', con)
cleaned_data.shape
cleaned_data.head()
cleaned_data['Score'].value_counts()
# To randomly sample 5k points from both class

data_p = cleaned_data[cleaned_data['Score'] == 'positive'].sample(n = 5000)
data_n = cleaned_data[cleaned_data['Score'] == 'negative'].sample(n = 5000)
final_10k = pd.concat([data_p, data_n])
final_10k.shape
# Sorting data based on time
final_10k['Time'] = pd.to_datetime(final_10k['Time'], unit = 's')
final_10k = final_10k.sort_values(by = 'Time')
final_10k.head()
#function to compute the k value


def k_classifier_brute(X_train , y_train):
    
    #creatting odd list of k for knn
    myList = list(range(0,40))
    neighbors = list(filter(lambda x:x%2!=0 , myList))
    
    #empty list that will hold cv score
    cv_scores = []
    
    #perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors = k , algorithm = "brute")
        scores = cross_val_score(knn , X_train , y_train , cv = 10 , scoring = 'accuracy')
        cv_scores.append(scores.mean())
        
    #changing to misclassification error
    MSE = [1 - x for x in cv_scores]
    
    #determinning best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('\nThe optimal number of neighbors is %d' % optimal_k)
    
    #plot misclassification error vs k
    plt.plot(neighbors , MSE)
    
    
    for xy in zip(neighbors , np.round(MSE,3)):
        plt.annotate('(%s , %s)' %xy , xy=xy , textcoords = 'data')
    plt.title("Misclassification Error Vs K")
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()
    
    print('the misclassification error for each k value is : ' , np.round(MSE,3))
    return optimal_k
#7k data which will use to train model after vectorization

X = final_10k['CleanedText']
print('shape of X :' , X.shape)
y = final_10k['Score']
print('shape of y :' , y.shape)
# spliting data into 70% as a train and 30% as test data

from sklearn.model_selection import train_test_split
X_train , x_test , y_train , y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(X_train.shape,y_train.shape, x_test.shape, y_test.shape)
#train vectorizor

from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()
X_train = bow.fit_transform(X_train)
X_train
X_train.shape
#Test vectorizor

x_test = bow.transform(x_test)
x_test.shape
# to choss optimal k using brute force algorithm

from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection

#from sklearn import cross_validation not woeking

optimal_k_bow = k_classifier_brute(X_train, y_train)
optimal_k_bow
# instantiate learning model k  = optimal k
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_k_bow)

#fitting the model
knn_optimal.fit (X_train , y_train)

#predict the response
pred = knn_optimal.predict(x_test)
# accurary on train data 

train_acc_bow = knn_optimal.score(X_train , y_train)
print('Train Accuracy :' ,train_acc_bow)
print('Train Accuracy : %f%%' % (train_acc_bow)) # this is accuracy in %age
# error on train data 

train_err_bow = 1-train_acc_bow
print('Train Error: %f%%' % (train_err_bow) )
# evaluate accuracy on test data 
acc_bow = accuracy_score(y_test , pred)*100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k_bow,acc_bow))
# Confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , pred)
cm
# plot confusion matrix to discribe the performance of classifier.

import seaborn as sns

class_label = ['negative' , 'positive']
df_cm = pd.DataFrame(cm, index = class_label , columns = class_label)
sns.heatmap(df_cm , annot = True , fmt = 'd')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
# to show main classification report

from sklearn.metrics import classification_report
print(classification_report(y_test , pred))
# model for knn with bag of word
models = pd.DataFrame({'Model': ['KNN with Bow'], 'Hyper Parameter(K)': [optimal_k_bow], 'Train Error': [train_err_bow], 'Test Error': [100-acc_bow], 'Accuracy': [acc_bow ]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
# data

X = final_10k['CleanedText']
X.shape
# traget / class label

y = final_10k['Score']
y.shape

#split data

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(X_train.shape,y_train.shape, x_test.shape, y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
X_train = tf_idf_vect.fit_transform(X_train)
X_train
# convert test text data to its vectorizer

x_test = tf_idf_vect.transform(x_test)
x_test.shape
# to chossing optimal k

from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection

optimal_k_tfidf = k_classifier_brute(X_train, y_train)
optimal_k_tfidf
# instantiate learning model k = optimal k
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_k_tfidf)

#fitting the model
knn_optimal.fit(X_train , y_train)

#predict the response
pred = knn_optimal.predict(x_test)
#Accuracy of train data

train_acc_tfidf = knn_optimal.score(X_train , y_train)
print('train accuracy :' , train_acc_tfidf)
print('train accuracy : %f%%' % train_acc_tfidf)
# error in train data
train_err_tfidf = 1 - train_acc_tfidf 
print('train accuracy :' , train_err_tfidf)
print('train accuracy : %f%%' % train_err_tfidf)
# eveluate accuracy

acc_tfidf = accuracy_score(y_test , pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k_tfidf,acc_tfidf))
# Confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , pred)
cm
import seaborn as sns
class_label = ['negetive' , 'positive']
df_cm = pd.DataFrame(cm , index = class_label , columns = class_label)
sns.heatmap(df_cm , annot = True , fmt = 'd')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test , pred))
# model for knn with tfidf
models = pd.DataFrame({'Model': ['KNN with TfIdf'], 'Hyper Parameter(K)': [optimal_k_tfidf], 'Train Error': [train_err_tfidf], 'Test Error': [100-acc_tfidf], 'Accuracy': [acc_tfidf ]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
# data 
X = final_10k['Text']
X.shape
# target / class label 
y = final_10k['Score']
y.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_train, x_test, y_train, y_test =train_test_split(X, y, test_size = 0.3)
print(X_train.shape, x_test.shape, y_train.shape, y_test.shape)
import re

def cleanhtml(sentence): # function to clean the word of any html tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr , ' ' , sentence)
    return cleantext

def cleanpunc(sentence): # function to clean the word of any puntuation or special characters
    cleaned = re.sub(r'[?|!|\'|#]' , r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]' , r'' , sentence)
    return cleaned
# train  your own w2v model using your own train text corpus

import gensim
list_of_sent = []
for sent in X_train:
    filtered_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue
    list_of_sent.append(filtered_sentence)
w2v_model = gensim.models.Word2Vec(list_of_sent , min_count = 5 , size = 50 , workers = 4)
w2v_model.wv.most_similar('like')
w2v = w2v_model[w2v_model.wv.vocab]
w2v.shape
# Train your own Word2Vec model using your own test text corpus
import gensim
list_of_sent_test = []
for sent in x_test:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent_test.append(filtered_sentence)
w2v_model=gensim.models.Word2Vec(list_of_sent_test, min_count=5, size=50, workers=4)
w2v_model.wv.most_similar('like')
w2v = w2v_model[w2v_model.wv.vocab]
w2v.shape
# average Word2Vec for train data set.....
# compute average word2vec for each review.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sent: # for each review/sentence and this is for train data set...
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))
# average Word2Vec
# compute average word2vec for each review.
sent_vectors_test = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sent_test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors_test.append(sent_vec)
print(len(sent_vectors_test))
print(len(sent_vectors_test[0]))
X_train = sent_vectors

x_test = sent_vectors_test
optimal_k_avgw2v = k_classifier_brute(X_train , y_train)
optimal_k_avgw2v
# instantiate learning model k = optimal_k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k_avgw2v)

# fitting the model
knn_optimal.fit(X_train, y_train)
    
# predict the response
pred = knn_optimal.predict(x_test)
# Accuracy on train data
train_acc_avgw2v = knn_optimal.score(X_train, y_train)
print("Train accuracy", train_acc_avgw2v)
# Error on train data
train_err_avgw2v = 1-train_acc_avgw2v
print("Train Error %f%%" % (train_err_avgw2v))
# evaluate accuracy
acc_avg_w2v = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k_avgw2v, acc_avg_w2v))
print("Test Error %f%%" %(100-(acc_avg_w2v)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with tfidf
models = pd.DataFrame({'Model': ['KNN with average Word2Vec'], 'Hyper Parameter(K)': [optimal_k_avgw2v], 'Train Error': [train_err_avgw2v], 'Test Error': [100-acc_avg_w2v], 'Accuracy': [acc_avg_w2v ]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
# TF-IDF weighted Word2Vec train dataset
tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_of_sent: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1  
len(tfidf_sent_vectors)

X_train = tfidf_sent_vectors
# tf idf  weigthed word2vec test dataset

tfidf_feat = tf_idf_vect.get_feature_names() #tfidf words / col name

#final tf idf is the sparse matrix with row = sentence col=word and cell val = tfidf

tfidf_sent_vectors_test = [] #the tfdif w2v row = sentence review is stored in the list

row = 0

for sent in  list_of_sent_test: #for each review or sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length 
    weight_sum = 0
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            # obtain the tf idf of a word in sentence / review
            tfidf = final_tf_idf[row , tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    tfidf_sent_vectors_test.append(sent_vec)
    row += 1
len(tfidf_sent_vectors_test)
x_test = tfidf_sent_vectors_test
X_train = np.nan_to_num(X_train)
x_test = np.nan_to_num(x_test)
optimal_k_tfidf_w2v = k_classifier_brute(X_train , y_train)
optimal_k_tfidf_w2v
# instantiate Learning model k = optimal k 
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_k_tfidf_w2v)

#fitting the model
knn_optimal.fit(X_train , y_train)

#predict the response 
pred = knn_optimal.predict(x_test)
# accuracy on train data

train_acc_tfidf_w2v = knn_optimal.score(X_train, y_train)
print("Train accuracy", train_acc_tfidf_w2v)
# Error on train data
train_err_tfidf_w2v = 1-train_acc_tfidf_w2v
print("Train Error %f%%" % (train_err_tfidf_w2v))
# evaluate accuracy
acc_tfidf_w2v = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k_tfidf_w2v, acc_tfidf_w2v))
print("Test Error %f%%" %(100-(acc_tfidf_w2v)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with tfidf
models = pd.DataFrame({'Model': ['KNN with Tfidf Word2Vec'], 'Hyper Parameter(K)': [optimal_k_tfidf_w2v], 'Train Error': [train_err_tfidf_w2v], 'Test Error': [100-acc_tfidf_w2v], 'Accuracy': [acc_tfidf_w2v ]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
# model
models = pd.DataFrame({'Model': ['KNN with Bow', "KNN with TFIDF", "KNN with Average Word2Vec", "KNN with Tfidf Word2Vec "], 'Hyper Parameter(K)': [optimal_k_bow, optimal_k_tfidf, optimal_k_avgw2v, optimal_k_tfidf_w2v], 'Train Error': [train_err_bow, train_err_tfidf, train_err_avgw2v, train_err_tfidf_w2v], 'Test Error': [100-acc_bow, 100-acc_tfidf, 100-acc_avg_w2v, 100-acc_tfidf_w2v], 'Accuracy': [acc_bow, acc_tfidf, acc_avg_w2v, acc_tfidf_w2v]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy"])
models.sort_values(by='Accuracy', ascending=False)