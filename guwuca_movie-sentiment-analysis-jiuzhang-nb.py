import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk 
#nltk.download('wordnet')  -only need to do once
#nltk.download('stopwords')
dir="../input/kumarmanoj-bag-of-words-meets-bags-of-popcorn/"
trainData=pd.read_csv(dir+"labeledTrainData.tsv", delimiter='\t')
testData=pd.read_csv(dir+"testData.tsv", delimiter='\t')
trainData.head()

trainY=trainData['sentiment'].values   
trainX=trainData['review'].values     
testX=testData['review'].values
print(trainY.shape, trainX.shape, testX.shape)

from tqdm import tqdm

import re

#remove html <labels>, change n't to not, and remove 's, punctuation and nonwords
def clean_data(data_in):
    data=data_in.copy()
    for idx, line in tqdm(enumerate(data)):
        a=re.sub(r"n\'t"," not", line)
        data[idx]=re.sub(r"<.*?/>|\'s|[^A-Za-z]"," ",a).lower()
    return data
trainX_after_clean=clean_data(trainX)
testX_after_clean=clean_data(testX)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

#tokenize
def tokenize(data_in):
    res=[]
    for line in tqdm(data_in):
        line_tok=line.split()
        res.append(line_tok)
    return res

#lematize and remove stop word
def lemmatize_remove_stop(data_in):
    res=[]
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))

    for data in tqdm(data_in): 
#        res.append([lemmatizer.lemmatize(i,pos='v') for i in data if not i in stop_words and len(i)>=2])
         res.append([i for i in data if not i in stop_words and len(i)>=2])
    return res

#load human names corpus on Kaggle
#https://www.kaggle.com/nltkdata/names
name_copus1=pd.read_csv('../input/names/names/female.txt', header=None)
name_copus2=pd.read_csv('../input/names/names/male.txt',header=None)
human_names=set((name_copus1.append(name_copus2))[0].str.lower().values.ravel())

def class_mapping(data_in, name_copus):
    res=[]
    for data in tqdm(data_in):
        a=[(lambda x:("_humanname" if x in name_copus else x))(i) for i in data]
        res.append(a)
    return res

trainX_after_tok=tokenize(trainX_after_clean)
testX_after_tok=tokenize(testX_after_clean)
trainX_after_lem=lemmatize_remove_stop(trainX_after_tok)
testX_after_lem=lemmatize_remove_stop(testX_after_tok)
trainX_after_map=class_mapping(trainX_after_lem, human_names)
testX_after_map=class_mapping(testX_after_lem, human_names)
import copy
def add_ngram(data_in):
    data_copy=copy.deepcopy(data_in);
    for line in tqdm(data_copy):
        n=len(line)
        line.extend(["".join(line[i:i+2]) for i in range(n-1)])
        line.extend(["".join(line[i:i+3]) for i in range(n-2)])
    return data_copy

trainX_after_ngram=add_ngram(trainX_after_map)
testX_after_ngram=add_ngram(testX_after_map)
def gen_vocabulary(data_in):
    #data_in is a list of lists
    dictionary={}
    word_id=0
    for line in tqdm(data_in):
        for word in line:
            if word not in dictionary:
                dictionary[word]=word_id
                word_id+=1
    return dictionary

from scipy.sparse import csr_matrix
def td_idf_vectorize(data_in, dictionary):
    m=len(dictionary)
    n=len(data_in)
    col_idx=[]
    indptr=[0]
    data=[]
    word_doc_count=np.array([0]*(m+1))
    
    #generate idf matirx
    for doc in tqdm(data_in):
        term_set=set([])
        for term in doc:
            term_set.add(term)
        for term in term_set:
            word_doc_count[dictionary.get(term,m)]+=1
    idf=np.log((n+1)/(word_doc_count+1))+1  #+1 for smoothing
    
    #generate td matrix
    for doc in tqdm(data_in):
        word_count=len(doc)
        for term in doc:
            term_idx=dictionary.get(term,m)
            col_idx.append(term_idx)
            data.append(1/word_count*idf[term_idx])
        
        indptr.append(len(col_idx))
        
    td_matrix=csr_matrix((data,col_idx,indptr),shape=[n,m+1])    
    return td_matrix
    
dict_gen=gen_vocabulary(trainX_after_ngram)      
print(len(dict_gen))
trainX_vector=td_idf_vectorize(trainX_after_ngram,dict_gen)
testX_vector=td_idf_vectorize(testX_after_ngram,dict_gen)
print(trainX_vector.shape, testX_vector.shape)
class NBClassifier():
    """
    class variables:
        class_labels
        prior_matrix : num_classes x 1
        likely_hood_matrix : num_classes x feature_dim
    """
    
    def __init__(self):
        pass
    
    def fit(self, Xtrain, y):
        num_data,num_features=Xtrain.shape
        
        class_labels, count=np.unique(y,return_counts=True)
        self.class_labels=class_labels
        num_classes=class_labels.shape[0]
        
        #calculate priorMatrix
        self.log_prior=np.log(count/sum(count))
        
        #calculate likelyhood
        self.log_likely_hood=np.ones((num_classes,num_features))  # includes 1 for smoothing
        for idx, class_label in enumerate(class_labels):
            row_idx=np.nonzero(y==class_label)[0]
            self.log_likely_hood[idx]+=np.asarray(Xtrain[row_idx,:].sum(axis=0)).ravel()
            
        self.log_likely_hood=np.log(self.log_likely_hood/(self.log_likely_hood.sum(axis=1).reshape(2,1)-1))  
         
    def predict(self,Xtest):
        numData=Xtest.shape[0]
        num_classes=self.class_labels.shape[0]
        posterior=np.zeros((numData,num_classes))
        for idx, xtest in tqdm(enumerate(Xtest)):
            for idx2 in range(self.log_likely_hood.shape[0]):
                posterior[idx,idx2]=np.sum(xtest*self.log_likely_hood[idx2,:])+self.log_prior[idx2]

                              
        ytest=np.array([self.class_labels[i] for i in np.argmax(posterior, axis=1)])
        return ytest
#run model check

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix

trainX_split, valiX_split, trainY_split, valiY_split=train_test_split(trainX_vector, trainY, test_size=0.3, random_state=1)

nbc=NBClassifier()
nbc.fit(trainX_split,trainY_split)
ypred=nbc.predict(valiX_split)

print(accuracy_score(ypred,valiY_split))
print(f1_score(ypred,valiY_split))
print(confusion_matrix(ypred,valiY_split))

#train the model 
nbc2=NBClassifier()
nbc2.fit(trainX_vector,trainY)
ytestpred=nbc2.predict(testX_vector)
df=pd.DataFrame({'id':testData['id'],'sentiment':ytestpred})
df.to_csv('submission.csv',index=False,header=True)
#base line: test score = 0.81020, 
#using TF-IDF, lemmatizaing and stop word removal, test score=0.83812
#adding n-gram (1-3): test score=0.85168
#adding class mapping: test score=0.868
#remove lemmatization: test score=0.86992