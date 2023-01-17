# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data
df_true = pd.read_csv(r'/kaggle/input/fake-and-real-news-dataset/True.csv')
df_fake = pd.read_csv(r'/kaggle/input/fake-and-real-news-dataset/Fake.csv')
#Adding column
df_true['RealNews?'] = True
df_fake['RealNews?'] = False
df = df_true.append(df_fake)
#Data Aggregation + Transformation
df['document'] = df[['title', 'text']].agg(' '.join, axis=1)
df['document'] = df['document'].apply(lambda x: x.lower())
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

import re
import numpy as np


class WordCount:
    
    
    def __init__(self,df): 
        
        
        self.df = df
        self.__dict_real = {}
        self.__dict_fake = {}
        self.__total_vocab = set()
        self.__word_prob = {}
        self.p_real = 0
        self.p_fake = 0
    
    
    
    def predict(self,document):
        
        
        document = document.lower()
        words = re.split(r'\W+',document)
        
        log_of_real = np.log(self.p_real)
        log_of_fake = np.log(self.p_fake)
        
        for word in words:
            if(word in self.__word_prob):
                real_prob,fake_prob = self.__word_prob[word]
                log_of_real += np.log(real_prob)
                log_of_fake += np.log(fake_prob)
        
        
        if(log_of_real > log_of_fake):
            return 'Real News'
            
        return 'Fake News'
            
    
    def computeProbablities(self):
        
        total_nc = len(self.df)
        
        
        #Prior class prob
        self.p_real = len(self.df[self.df['RealNews?']]) / total_nc
        self.p_fake = 1 - self.p_real
        
    
        sum_wc_real = sum(self.__dict_real.values())
        sum_wc_fake = sum(self.__dict_fake.values())
        
        alpha = 1
        
        #vocab_count = len(self.__total_vocab)
        
        for word in self.__total_vocab:
            
            wc_r = self.__dict_real[word] if(word in self.__dict_real)  else 0
            wc_f = self.__dict_fake[word] if (word in self.__dict_fake) else 1
        
            p_wc_real = (wc_r + alpha) / (sum_wc_real + len(self.__dict_real))
            p_wc_fake = (wc_f + alpha) / (sum_wc_fake + len(self.__dict_fake))
        
            self.__word_prob[word] = (p_wc_real,p_wc_fake)
    
                 
    def wordCount(self):
        
        
        
        def processEachLine(line,type_):
            
            words = re.split(r'\W+',line)
            
            if(type_ == 'Real'):
                
                
                for word in words:
                    
                    self.__total_vocab.add(word)
                    
                    #Storing word count in real dict
                    if(word in self.__dict_real):
                        self.__dict_real[word] +=1
                        
                    else:
                        self.__dict_real[word] = 1
    
            
            else:
                for word in words:
                    
                    #Adding word to set
                    self.__total_vocab.add(word)
                    
                    #Storing word count in fake dict
                    if(word in self.__dict_fake):
                        self.__dict_fake[word] +=1
                        
                    else:
                        self.__dict_fake[word] = 1
            
            
            
        df_real = self.df[self.df['RealNews?']]
        df_fake = self.df[~self.df['RealNews?']]
        
        
        for i in range(len(df_real)):
            processEachLine(df_real['document'].iloc[i],'Real')
            
        for i in range(len(df_fake)):
            processEachLine(df_fake['document'].iloc[i],'Fake')
            
        
        #print(len(self.__dict_real))
        #print(len(self.__dict_fake))
        #print(len(self.__total_vocab))
        
wc = WordCount(df_train)
wc.wordCount()
wc.computeProbablities()
print('Completed')
#Prediction

count_true = 0
false_pos = 0
false_neg = 0
true_pos = 0
true_neg  =0 

for i in range(len(df_test)):
    
    line = df_test['document'].iloc[i]
    prediction = wc.predict(line) 
    
    ground_truth = 'Real News' if(df_test['RealNews?'].iloc[i]) else 'Fake News'
    
    if(ground_truth == 'Real News' and prediction == 'Fake News'):
        false_pos +=1
        
    elif(ground_truth == 'Fake News' and prediction == 'Real News'):
        false_neg +=1
        
    if(ground_truth == prediction):
        
        if(prediction == 'Real News'):
            true_pos +=1
            
        else:
            true_neg +=1
            

accuracy = (true_pos + true_neg) / len(df_test) * 100
precision = true_pos / (true_pos + false_pos) * 100
recall = true_pos / (true_pos + false_neg) * 100
f1 = (2 / (1/recall + 1/precision))
print('Accuracy ',accuracy)
print('Precision ',precision)
print('Recall ',recall)
print('F1 ',f1)
        
import re
from collections import Counter
from numba import jit, cuda,jitclass
from timeit import default_timer as timer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import pickle



class TfIdf:
    
    def __init__(self,df):
        self.df = df
        self.vocab = set()
        self.word_to_int = {}
        self.int_to_word = {}
    
    def preprocess(self,text):

        # Remove single character, digits, mutiple spaces, lowercase, whitespaces
        processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        processed_feature = re.sub('\d+',' ',processed_feature)
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        processed_feature = processed_feature.lower()
        processed_feature = processed_feature.rstrip().lstrip()
        
        return processed_feature

        
    def generateVectors(self,df):
        
        doc_dict = {}
        
        for i in range(len(df)):
            document = df['processed_doc'].iloc[i]
            
            sum_ = 0
            temp = {}
            for w in re.split(r'\W+',document):
                
                if(w in self.word_to_int):
                    if(w in temp):
                        temp[w] +=1
                    else:
                        temp[w] = 1
                        
                    sum_ +=1
                
            for vc in self.vocab:
                if not(vc in temp):
                    temp[vc] = 0
                    
            temp['sum'] = sum_
            doc_dict[i] = temp
            
        return doc_dict
    
    def computeTF(self,matrix,df): 
        
        
        tf_dict = {}
        
        for i in range(len(df)):
            
            total_words  = matrix[i]['sum']
            temp = {}
            for k,count in matrix[i].items():
                if(k == 'sum'):
                    continue
                    
                else:
                    temp[k] = 1 + np.log(count / total_words) if count != 0 else 0
            
            tf_dict[i] = temp
            
        
        return tf_dict
            
    def computeIDF(self,matrix):
    
        self.no_of_docs = {i:0 for i in self.vocab}
        tf_idf_matrix = {}

        for doc_id,d in matrix.items():
            for w,c in d.items():
                if(c != 0):
                    self.no_of_docs[w] +=1


        n_documents = len(self.df)

        #compute inverse frequency log values
        for w,dc in self.no_of_docs.items(): self.no_of_docs[w] = np.log(n_documents / dc)

        #compute tf-idf for each word and each document
        for doc_id,d in matrix.items():
            temp = {}
            for w,tf in d.items():
                temp[w] = tf * self.no_of_docs[w]

            tf_idf_matrix[doc_id] = temp

        return tf_idf_matrix
    
    
    
    def trainLogisticRegression(self,tf_idf):
  
      
        #Indexing the word for uniformity in training
        list_ = []

        for id_,d in tf_idf.items():
            
            temp = []
            for vc in self.vocab:
                temp.append(d[vc])
                
            list_.append(temp)
       
        X = pd.DataFrame(list_)

        #preparing label
        self.df['label'] = self.df['RealNews?'].apply(lambda x: 1 if x == True else 0)
        Y = self.df['label']

        #Splitting into train and test
        X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2, shuffle=True)

        #Train the model
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        #Printing metrics
        print('Metric In Train Follows Precision, Recall, Fscore and Support')
        print(precision_recall_fscore_support(y_test, y_pred,average='weighted'))
        
        #Saving a model after training
        pickle.dump(clf, open('lr_model', 'wb'))

    
    
    def tfidfCompute(self):
            
        self.df['processed_doc'] = self.df['document'].apply(lambda x: self.preprocess(x))
        
        text = ''
        for i in range(len(self.df)): text += self.df['processed_doc'].iloc[i] + ' '
            
        words = re.split(r'\W+',text)
        wc_dict = Counter(words)
        
        #Word count >= 3
        
        #Storing the corpus of words
        
        for k,v in wc_dict.items():
            if(v >= 4 and len(k) > 3):
                self.vocab.add(k)
        
        #No need for this part
        for idx,word in enumerate(self.vocab):
            self.word_to_int[word] = idx
            self.int_to_word[idx] = word
            
        
        print('Unique words in Train doc ',len(self.vocab))
        
        #generate vectors i.e word:count for each document
        doc2vec = self.generateVectors(self.df)
        
        
        #generate term freq for each word in each doc
        start = timer()
        df_tf = self.computeTF(doc2vec,self.df)
        print('TF computed in secs ',timer()-start)
        
        #generate inverse term freq and store in list of dict
        start = timer()
        tf_idf = self.computeIDF(df_tf)
        print('IDF computed in secs ',timer()-start)
        
        #train the model and test it's accuracy
        self.trainLogisticRegression(tf_idf)
        
        
train = TfIdf(df_train.iloc[0:1000,:])
train.tfidfCompute()


print('Completed') 
def predictTest(matrix,df):
    
    y_test = df['RealNews?'].apply(lambda x: 1 if x == True else 0)
    
    X_test = []

    for id_,d in matrix.items():
        temp = []
        for vc in train.vocab:
            temp.append(d[vc])
            
        X_test.append(temp)
    
    model = pickle.load(open('./lr_model', 'rb'))
    y_pred = model.predict(X_test)
    
    print('------Test Accuracy Metric for Tf-IDF-----------')
    print(precision_recall_fscore_support(y_test, y_pred,average='weighted'))
    
    
def computeTestTFIDF(matrix):
    
    tf_idf_matrix = {}
    
    #compute tf-idf for each word and each document
    
    for doc_id,d in matrix.items():
        temp = {}
        for w,tf in d.items():
            temp[w] = tf *  train.no_of_docs[w]

        tf_idf_matrix[doc_id] = temp

    return tf_idf_matrix
    
def checkIntoTrainVocab(dict_):
    
    vocab = train.vocab
    
    for did,v in dict_.items():
        temp = {}
        
        for w in vocab:

            if(w in v):
                temp[w] = v[w]
                
            else:
                temp[w] = 0
                
        v = temp
        
    return dict_

df_test['processed_doc'] = df_test['document'].apply(lambda x: train.preprocess(x))
test_set = df_test.iloc[0:1000]

test_vec = train.generateVectors(test_set)
test_vec = checkIntoTrainVocab(test_vec)
test_vec_tf = train.computeTF(test_vec,test_set)
test_tfidf = computeTestTFIDF(test_vec_tf)
predictTest(test_tfidf,test_set)
from sklearn.feature_extraction.text import TfidfVectorizer
#Define the train data corpus
corpus = df_train['document'].iloc[0:1000]
test_set = df_test.iloc[0:1000]
train_set = df_train.iloc[0:1000]


def nGram(n_gram=False):

    if(n_gram):
        tfIdfVectorizer=TfidfVectorizer(use_idf=True, ngram_range = (2,2),stop_words = {'english'})
        print('--------Training IDFs with n_gram--------')
        
    else:
        tfIdfVectorizer = TfidfVectorizer(use_idf=True)
        print('--------Training IDFs without n_gram--------')
        
    tfIdf = tfIdfVectorizer.fit_transform(corpus)
    
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)

    
    print(df.head(10))
    
    return tfIdfVectorizer,tfIdf

tfIdf_n,tf_n = nGram(n_gram=True)
tfIdf_wn,tf_wn = nGram(n_gram = False)

#Train the LR
def trainModel(tfIdf,n_gram):
    
    #Get the tfidf form of each doc

    X = tfIdf.todense()
    Y = train_set['RealNews?'].iloc[0:1000].apply(lambda x: 1 if x==True else 0)


    X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2, shuffle=True)

    lr = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    if(n_gram):
        
        pickle.dump(lr, open('lr_model_ngram_sklearn', 'wb'))
        print('Training Accuracy Metric with n_gram ------ Precision, Recall, Fscore and Support')
    else:
        pickle.dump(lr, open('lr_model_wngram_sklearn', 'wb'))
        print('Training Accuracy Metric without n_gram------ Precision, Recall, Fscore and Support')
        
    print(precision_recall_fscore_support(y_test, y_pred,average='weighted'))
    
trainModel(tf_n,True)
trainModel(tf_wn,False)

def testingModel(tf,n_gram):

    #Fit the test data 

    corpus_t = test_set['document'].iloc[0:1000]
    tfIdf_t = tf.transform(corpus_t)
    
    #Test the Prediction for Test Data

    X_test = tfIdf_t.todense()
    y_test = test_set['RealNews?'].apply(lambda x: 1 if x==True else 0)

    if(n_gram):
        model = pickle.load(open('./lr_model_ngram_sklearn', 'rb'))
        print('----Testing Accuracy for TF-IDF with ngram-------------')
    else:
        model = pickle.load(open('./lr_model_wngram_sklearn', 'rb'))
        print('----Testing Accuracy for TF-IDF without ngram-------------')
        
    y_pred = model.predict(X_test)
    
    print(precision_recall_fscore_support(y_test, y_pred,average='weighted'))
    
    
testingModel(tfIdf_n,True)
testingModel(tfIdf_wn,False)
from sklearn.naive_bayes import MultinomialNB
train_sub = df_train.iloc[0:10000]
test_sub = df_test.iloc[0:2000]

y = train_sub['RealNews?']
X = train_sub['document']

y_test = test_sub['RealNews?']
X_test = test_sub['document']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vec = vectorizer.fit_transform(X)
X = vec.toarray()

#fitting the test data
vec_test= vectorizer.transform(X_test)
X_test = vec_test.toarray()
nb = MultinomialNB()

def sklearn_NaiveBayes(X,y):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2, shuffle=True)
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)
    
    print('------------Training Accuracy ------------- ')
    print(precision_recall_fscore_support(y_pred,y_test,average='weighted'))
    

sklearn_NaiveBayes(X,y)


print('-------------Testing Accuracy---------------')
y_pred = nb.predict(X_test)
print(precision_recall_fscore_support(y_pred,y_test,average='weighted'))