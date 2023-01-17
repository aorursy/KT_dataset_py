import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库
from sklearn.model_selection import train_test_split
import pandas as pd


train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test = pd.read_csv('../input/testData.tsv', delimiter="\t")

data_train, data_vali, labels_train, labels_vali = train_test_split(
    train,
    train.sentiment, 
    test_size=0.2, 
    random_state=1)  
print(data_train.head())  
print (data_train.shape, data_vali.shape, labels_train.shape, labels_vali.shape)
labels_train=np.array(labels_train)
import re  #正则表达式

def review_to_wordlist(review):
#     print(review)

#   只保留英文单词
    review_text = re.sub("[^a-zA-Z]"," ", review)
#     print (review_text)
    
#   变成小写
    words = review_text.lower()
    
    return(words)

#y_train = train['sentiment']
train_data = []
for string in data_train['review']:
    train_data.append(review_to_wordlist(string))        
train_data = np.array(train_data)

vali_data = []
for string in data_vali['review']:
    vali_data.append(review_to_wordlist(string))        
vali_data = np.array(vali_data)

test_data = []
for string in test['review']:
    test_data.append(review_to_wordlist(string))    
test_data = np.array(test_data)
print(train_data.shape,vali_data.shape,test_data.shape)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def create_vocab(data):
    vocab=set([])
    for item in data:
        tmp=set(item.split())
        vocab.update(tmp)
    vocab.difference_update(stop_words)
    return {key:value for (key,value) in zip(vocab,range(len(vocab)))}

vocab=create_vocab(np.append(train_data,vali_data))

def vectorize(vocab,data):
    res_vector=np.zeros(len(vocab))
    tmp=data.split()
    for word in tmp:
            if word in vocab:res_vector[vocab[word]] +=1
    return res_vector

print("number of unique words is ",len(vocab))


def my_Bayes_Train(train_data,vocab):
    pos_vector,neg_vector=np.zeros([len(vocab)]),np.zeros([len(vocab)])
    pos_count,neg_count=0,0
    for i in range(len(train_data)):
        if i % 1000 == 0:
            print ('Train on the doc id:' + str(i))
        
        if  labels_train[i]==1:
            pos_vector += vectorize(vocab,train_data[i])
            pos_count +=1
        else:
            neg_vector += vectorize(vocab,train_data[i])
            neg_count +=1
    pos_word_count,neg_word_count=sum(pos_vector),sum(neg_vector)
#pos_unique_word_count,neg_unique_word_count=np.count_nonzero(pos_vector),np.count_nonzero(neg_vector)


    print(pos_count,neg_count,pos_word_count,neg_word_count)

#p_pos_vector=np.log((pos_vector+np.ones([len(vocab)]))/(pos_word_count+pos_unique_word_count))
#p_neg_vector=np.log((neg_vector+np.ones([len(vocab)]))/(neg_word_count+neg_unique_word_count))
    p_pos_vector=np.log((pos_vector+np.ones([len(vocab)]))/(pos_word_count+len(vocab)))
    p_neg_vector=np.log((neg_vector+np.ones([len(vocab)]))/(neg_word_count+len(vocab)))
    p_pos=np.log(pos_count/(pos_count+neg_count))
    p_neg=np.log(neg_count/(pos_count+neg_count))
    return p_pos_vector, p_pos, p_neg_vector, p_neg

p_pos_vector, p_pos, p_neg_vector, p_neg = my_Bayes_Train(train_data,vocab)                  
def Predict(test_word_vector,p_pos_vector, p_pos, p_neg_vector, p_neg):
    
    pos = sum(test_word_vector * p_pos_vector) + p_pos
    neg = sum(test_word_vector * p_neg_vector) + p_neg
   # print ("pos=",pos,"neg=",neg)
    if pos > neg:
        return 1
    else:
        return 0
predictions_baseline=[]

for review in vali_data:
    review_vector=vectorize(vocab,review)
    predictions_baseline.append(Predict(review_vector,p_pos_vector, p_pos, p_neg_vector, p_neg))
predictions_test=[]

for review in test_data:
    review_vector=vectorize(vocab,review)
    predictions_test.append(Predict(review_vector,p_pos_vector, p_pos, p_neg_vector, p_neg))
df = pd.DataFrame({"id": test['id'],"sentiment": predictions_test})
df.to_csv('dandan_movie_submission_stopwords.csv',index = False, header=True)