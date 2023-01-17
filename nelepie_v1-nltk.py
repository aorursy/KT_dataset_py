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
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re
import nltk
import string
import operator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import WordPunctTokenizer
from bokeh.io import output_notebook
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from tqdm import tqdm_notebook
from copy import deepcopy
from array import *

import re
import nltk
import gensim
import string
import operator
import scipy.io
import itertools
import numpy as np
import pandas as pd
import bokeh.models as bm, bokeh.plotting as pl
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train  = train.dropna()
test = test.dropna()
def text_prepare(text):
    #Удалить стопслова.
    stopWords = set(stopwords.words('english'))
    for stopWord in stopWords:
        text = re.sub(r'\b{}\b'.format(stopWord), '', text)
    return text
def get_perm(original):
    result = []
    words = original.split(' ')
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n + 1):
            # Append tuple(original[i:j]) if that's what you are looking for
            result.append(words[i:j])
            result[-1] = ' '.join(result[-1])
    return result
train['permutations'] = train['text'].apply(get_perm)
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def choosing_selectedword(df_process,n):
    train_data = df_process['text']
    train_data_sentiment = df_process['sentiment']
    selected_text_processed = []
    analyser = SentimentIntensityAnalyzer()
    for j in range(0 , len(train_data)):
        text = re.sub(r'http\S+', '', str(train_data.iloc[j]))
        if(train_data_sentiment.iloc[j] == "neutral" or len(text.split()) < n):
            selected_text_processed.append(str(text))
        if(train_data_sentiment.iloc[j] == "positive" and len(text.split()) >=n):
            aa = re.split(' ', text)
            ss_arr = ""
            polar = 0
            for qa in range(0,len(aa)):
                score = analyser.polarity_scores(aa[qa])
                if score['compound'] >polar:
                    polar = score['compound']
                    ss_arr = aa[qa]
            if len(ss_arr) != 0:
                selected_text_processed.append(ss_arr)   
            if len(ss_arr) == 0:
                selected_text_processed.append(text)
        if(train_data_sentiment.iloc[j] == "negative"and len(text.split()) >= n):
            aa = re.split(' ', text)
        
            ss_arr = ""
            polar = 0
            for qa in range(0,len(aa)):
                score = analyser.polarity_scores(aa[qa])
                if score['compound'] < polar:
                    polar = score['compound']
                    ss_arr = aa[qa]
            if len(ss_arr) != 0:
                selected_text_processed.append(ss_arr)   
            if len(ss_arr) == 0:
                selected_text_processed.append(text)  
    return selected_text_processed
result_train = choosing_selectedword(train,4)
train_selected_data = train['selected_text']
average = 0;
for i in range(0,len(train_selected_data)):
    ja_s = jaccard(str(result_train[i]),str(train_selected_data.iloc[i]))
    average = ja_s+average
print('Training Data accuracy')
print(average/len(result_train))
train['len_w'] = t
np.where(train['len_w']>0)[0]
train.sentiment2 = train.sentiment.replace(to_replace = pd.unique(train.sentiment),value = [0,1,2])
train.sentiment2.values[np.where(train['len_w']>0)[0]]
train[(train.len_w>0)]
train[(train.len_w>0) &(train.sentiment == 'neutral')]
plt.hist(t)
plt.show()
import matplotlib.pyplot as plt
%matplotlib inline
t = np.array(list(map(len,result_train))) - np.array(list(map(len,train.selected_text)))
sizes = 100
tokenizer = WordPunctTokenizer()
#traint_tokenized = [tokenizer.tokenize(line.lower()) for line in new_test]
traint_tokenized = [tokenizer.tokenize(line.lower()) for line in train.text]
wv_embeddings = Word2Vec(traint_tokenized, # data for model to train on
                 size=sizes,         # embedding vector size     
                         # consider words that occured at least 5 times
                 window=5, min_count = 3).wv   
result_test = choosing_selectedword(test,4)
index = test.textID
submisstion = pd.DataFrame(columns = ['textID','selected_text'], data ={'textID':index,'selected_text':result_test})
def question_to_vec(question, embeddings, dim=300):
    """
        question: строка
        embeddings: наше векторное представление
        dim: размер любого вектора в нашем представлении
        
        return: векторное представление для вопроса
    """
    words = question.split(' ') #your code
    # убрать знак вопроса, если он есть
    n_known = 0
    result = np.array([0] * dim, dtype=float)
    
    for word in words:
        if word in embeddings:
            result += embeddings[word] #your code
            n_known += 1
            
    if n_known != 0:
        return result / n_known #your code
    else:
        return result
    
def text_prepare(text):
    """
        text: a string
        
        return: modified string
    """
    # Перевести символы в нижний регистр
    #text = text.lower() #your code
    
    
    # Заменить символы пунктуации на пробелы
    #text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
    
    #Удалить "плохие" символы
    #text = re.sub('[^A-Za-z0-9 ]', '', text)
    
    ##Удалить стопслова.
    #stopWords = set(stopwords.words('english'))
    #for stopWord in stopWords:
    #    text = re.sub(r'\b{}\b'.format(stopWord), '', text)
    return text
quora_vectors_emb = []
for num in train.text:
    q, *ex = num
    quora_vectors_emb.append(question_to_vec(q, wv_embeddings,100)) 
quora_tokenized = train.text
import operator
def find_closest_questions(question, k=5):
    """
    function that finds closest questions from dataset given question
    args:
        question: question, preprocessed using text_prepare 
        k: how many nearest questions to find
    """

    vec_question = question_to_vec(question,wv_embeddings,100).reshape(1,-1)
    dist_s = cosine_similarity(quora_vectors_emb, vec_question)[:,0]
    sort_dist_s = sorted(dist_s)[::-1][:k]
    sorted_questions = deepcopy(np.array(quora_tokenized)[dist_s.argsort()[::-1]])[:k]
    sort_dict = dict(zip(sorted_questions,sort_dist_s))
    sorted_d = sorted(sort_dict.items(), key=operator.itemgetter(1),reverse = True)
    return sorted_d
traintext = result_train
new_test = deepcopy(traintext)
for i in tqdm_notebook(range(len(traintext))):
    new_test[i] = text_prepare(traintext[i])
train_selected_data = train['selected_text']
average = 0;
for i in range(0,len(train_selected_data)):
    ja_s = jaccard(str(new_test[i]),str(train_selected_data.iloc[i]))
    average = ja_s+average
print('Training Data accuracy')
print(average/len(new_test))
find_closest_questions(result_train[0],k=10)
submisstion.to_csv('submission.csv', index = False)
print('done')