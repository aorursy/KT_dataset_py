import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt # 画图常用库

import pandas as pd


train = pd.read_csv('../input/labeledTrainData.tsv', delimiter="\t")
test = pd.read_csv('../input/testData.tsv', delimiter="\t")
train.head() 
print (train.shape)
print (test.shape)
test.head()["review"]
import re 

def review_to_wordlist(review):
    #只保留英文单词
    review_text = re.sub('[^a-zA-Z]',' ',review) #把非英文字母替换成空格
    words = review_text.lower()
    return words

y_train= train['sentiment']

train_data=[]
for review in train['review']:
    train_data.append(review_to_wordlist(review))

print(len(train_data))
train_data
    
train_data =np.array(train_data)
train_data.shape

#对test文本做提取
test_data =[]
for review in test['review']:
    test_data.append(review_to_wordlist(review))
test_data=np.array(test_data)
test_data.shape
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords

from nltk.corpus import stopwords
nltk.download('stopwords')
set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer() 

def GetVocabulary(data):
    vocab_dict={}
    wid=0
    for document in data:
        words= document.split()
        
        for word in words:
            word=lmtzr.lemmatize(word)
            if word not in stop_words and word not in vocab_dict:
                    
                    vocab_dict[word]=wid
                    
                    wid += 1
            else:
                continue
          
    return vocab_dict
        

vocab_dict = GetVocabulary(train_data)
print('Number of all the unique words : '+ str(len(vocab_dict.keys())))
def Document2Vector(vocab_dict,data):
    word_vector = np.zeros(len(vocab_dict.keys()))
    words = data.split()
    for word in words:
        word = word.lower()
        if word in vocab_dict:
            word_vector[vocab_dict[word]]+=1
    return word_vector

#把训练集的句子全部变成向量形式，这里面全是数字，每个词汇表里的单词 根据id排序的 出现在 该文章里的次数，即使没有出现 也是有 0
train_matrix =[]
for document in train_data:
#     words= document.split()
#     for word in words:
        word_vector = Document2Vector(vocab_dict,document)
        train_matrix.append(word_vector)

print(len(train_matrix))  # 有多少个文档
train_matrix[0:10]
print(len(train_matrix[0]))
def NaiveBayes_train(train_matrix,labels_train):
    num_docs = len(train_matrix)
    num_words = len(train_matrix[0])  # 对第一个样本取长度
    
     # 在每个分类下创建一个与词汇量大小相等的vector(即 numpy array) 用以计算每个单词在该类别下的频率
    good_word_counter =np.ones(num_words) 
    bad_word_counter = np.ones(num_words) #计算每个word出现的次数，初始化为1. 即使用拉普拉斯平滑
    
    good_total_count =0 
    bad_total_count = 0   #每一个类别 单词总的计数， 所有词出现在good里头的总数 good的总词数 （不去重 ）
    
    good_count =0  #good review的总数
    bad_count = 0
    
    for i in range(num_docs):
        if i%2000==0:
            print('Train on the doc id:'+ str(i))
        
        if y_train[i]==0:   #0 is bad review
            bad_word_counter += train_matrix[i]
            bad_total_count += sum(train_matrix[i])
            bad_count +=1
        else:
            good_word_counter += train_matrix[i]
            good_total_count += sum(train_matrix[i])
            good_count +=1
            
    #以下则是，每个单词 在各类别下出现的概率，并且取了log，为什么取log，就是怕太小变成0～这部分再看看了解下为什么
    #并且注意 在分母上也要加上平滑部分
    p_good_vector = np.log(good_word_counter/(good_total_count+num_words))
    p_bad_vector = np.log(bad_word_counter/(bad_total_count+num_words))

    return p_good_vector, np.log(good_count/num_docs), p_bad_vector,np.log(bad_count/num_docs)
    
p_good_vector, p_good, p_bad_vector, p_bad = NaiveBayes_train(train_matrix, y_train.values)


    
print(p_good_vector)
p_bad_vector
def predict(test_word_vector, p_good_vector,p_good,p_bad_vector,p_bad):
    
    good = sum(test_word_vector * p_good_vector)+ p_good
    bad =  sum(test_word_vector * p_bad_vector )+ p_bad
    
    if good > bad:
        return '1'
    else:
        return '0'
    
predictions =[]
i =0
for document in test_data:
    if i%2000 ==0:
        print('test on the doc id: '+ str(i))
    i+=1
    test_word_vector =Document2Vector(vocab_dict,document)
    ans= predict(test_word_vector,p_good_vector,p_good,p_bad_vector,p_bad)
    predictions.append(ans)
    
  
print(len(predictions))
df = pd.DataFrame({"id": test['id'],"sentiment": predictions})

df.to_csv('submission1.csv',index = False, header=True)


