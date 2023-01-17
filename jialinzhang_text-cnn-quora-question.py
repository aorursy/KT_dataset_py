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
########################################################################################################################
# 用神经网络中Text-CNN构建文本分类模型,具体而言分为两个部分：
# ·数据处理
# ·CNN模型
########################################################################################################################
# question统一调整为 70 个 word,不足的填充<pad>，长的则截断
# vocabulary 大小为 <= 20万 , 且去除 词频小于1的词
# 本版本用130万条数据来训练,Threshold:0.5,相比version3,probability > Threshold,为 1 ；probability <= Threshold , 为 0
# 本版本做了如下改进：
# 原数据大约有130万条,但 insincere questions只有80442条,而sincere questions却有1219559条,insincere questions 占 sincere questions的比例只有 7%
# 将insincere questions and sincere questions 的比例设置为1:4,考虑到了样本不均衡给模型学习带来的影响
# 对 sincere questions随机采样时,是在全量sincere questions数据集上进行的
# 将Accuracy,Precision,Recall,F1_Score在训练过程中的变化以图表的形式展现出来了
# 卷积核大小 filters_size = [2,3,4,5,6,7] # 2 表示filter每次覆盖的单词数，其大小为 2*embedding_size
# Data_stage_1.py
# -*- coding: utf-8 -*-
# @Time    : 2019/1/7 14:56
# @Author  : Weiyang

'''
目标：读取train.csv和test.csv数据，去除标点符号等特殊符号，将question变为word列表，单词转为小写、
输入：train.csv,test.csv
输出1：questions = [[word1,word2,word3,..],...]
输出2：labels = [1,0,1,...]
输出3：test_questions = [[word1,word2,word3,..],...]
输出4：test_qids = [001256,05891,...]
输出5：empty_question_qids = [] # 如果去除特殊符号后,test.csv中question为空,则question类别为1,即是insincere question;其不再参与预测
注意:若去除特殊符号后,question为空,则将question直接删除,其不参与训练；而如果test.csv中出现此情况,则question类别确定为1,也无需参与预测;
'''

import pandas as pd
import re
import time
import numpy as np

start = time.time()
# read train.csv
train_data = pd.read_csv('../input/train.csv',encoding='utf-8-sig')

# 去除标点符号等特殊符号，将question变为词列表,将单词转为小写
questions = [] # [[word1,word2,..],..] question文本且分成一个一个词
labels = [] # labels
symbols = ['\''] # don't, I'm
randint = np.random.randint(0,len(train_data['question_text']),100) # 产生100个随机整数
count = 0
for question,label in zip(train_data['question_text'],train_data['target']):
    #if count > 30000:
        #break
    if count in randint:
        print('train.csv: ',question)
    question = re.sub("[\s+\.\!\/_,\\:;><{}\-$%^*()+\"\[\]]+|[+——！，。：；》《？?、~@#￥%……&*（）]+",' ',question)
    question = [str(word) for word in question.split() if len(word.strip()) !=0 and word not in symbols]
    question = [word.rstrip('\'').lstrip('\'') for word in question] # 去除左右两侧的单引号
    question = [word.rstrip('\\').lstrip('\\') for word in question]  # 去除左右两侧的\\
    question = [word.rstrip('’').lstrip('’') for word in question]  # 去除左右两侧的’
    question = [word.rstrip('‘').lstrip('‘') for word in question]  # 去除左右两侧的‘
    question = [word.rstrip('”').lstrip('”') for word in question]  # 去除左右两侧的”
    question = [word.rstrip('“').lstrip('“') for word in question]  # 去除左右两侧的“
    question = [word.rstrip('`').lstrip('`') for word in question]  # 去除左右两侧的`
    #question = [word.lower() for word in question]  # 将单词统一为小写
    
    # 将question首个单词变为小写，其余单词形式不变
    temp = []
    for i in range(len(question)):
        if i == 0:
            temp.append(question[i].lower())
        else:
            temp.append(question[i])
    question = temp[:]
    
    # 判断question是否为空
    if len(question) == 0:
        continue
    questions.append(question)
    labels.append(label)
    if count in randint:
        print('train.csv: ',question)
    count += 1

del train_data

# read test.csv
test_data = pd.read_csv('../input/test.csv',encoding='utf-8-sig')

# 去除标点符号等特殊符号，将question变为词列表,将单词转为小写
test_questions = [] # [[word1,word2,..],..] question文本且分成一个一个词
test_qids = [] #qids
empty_question_qids = [] # 如果去除特殊符号后,question为空,则question类别为1,即是insincere question;其不再参与预测
symbols = ['\''] # don't,I'm,...
randint = np.random.randint(0,len(test_data['question_text']),100) # 产生100个随机整数
count = 0
for question,qid in zip(test_data['question_text'],test_data['qid']):
    #if count > 3000:
        #break
    if count in randint:
        print('test.csv: ',question)
    question = re.sub("[\s+\.\!\/_,\\:;><{}\-$%^*()+\"\[\]]+|[+——！，。：；》《？?、~@#￥%……&*（）]+",' ',question)
    question = [str(word) for word in question.split() if len(word.strip()) !=0 and word not in symbols]
    question = [word.rstrip('\'').lstrip('\'') for word in question] # 去除左右两侧的单引号
    question = [word.rstrip('\\').lstrip('\\') for word in question]  # 去除左右两侧的\\
    question = [word.rstrip('’').lstrip('’') for word in question]  # 去除左右两侧的’
    question = [word.rstrip('‘').lstrip('‘') for word in question]  # 去除左右两侧的‘
    question = [word.rstrip('”').lstrip('”') for word in question]  # 去除左右两侧的”
    question = [word.rstrip('“').lstrip('“') for word in question]  # 去除左右两侧的“
    question = [word.rstrip('`').lstrip('`') for word in question]  # 去除左右两侧的`
    #question = [word.lower() for word in question]  # 将单词统一为小写
    
    # 将question首个单词变为小写，其余单词形式不变
    temp = []
    for i in range(len(question)):
        if i == 0:
            temp.append(question[i].lower())
        else:
            temp.append(question[i])
    question = temp[:]
    
    # 判断question是否为空
    if len(question) == 0:
        empty_question_qids.append(qid)
        continue
    test_questions.append(question)
    test_qids.append(qid)
    if count in randint:
        print('test.csv: ',question)
    count += 1

del test_data

print('The total time of Data_stage_1.py program is : %d s' %(time.time()-start))
# Data_stage_2.py
# -*- coding: utf-8 -*- 
# @Time    : 2019/1/13 10:23 
# @Author  : Weiyang

'''
目标：数据探索+预处理
输入：questions,labels,test_questions
输出1：
'''

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from collections import Counter
import tqdm
import time

start = time.time()

# ------------------------------------------------数据探索：基本的统计分析------------------------------------------------

'将train.csv分成insincere question 和 sincere question 两部分数据'
# 加载insincere question
# insincere question: 1 , sincere question: 0
insincere_questions = []
sincere_questions = [] 
for i in range(len(questions)):
    if labels[i] == 0:
        sincere_questions.append(questions[i])
    elif labels[i] == 1:
        insincere_questions.append(questions[i])

# 数据探索：描述性统计
# insincere questions and sincere questions 类别的比例
print('The Proportion of Insincere and sincere questions : %.2f'%(float(len(insincere_questions)/len(sincere_questions))))
# insincere questions 统计
print('-' * 30 + 'Insincere Questions' + '-' * 30)
print('Total number of insincere questions : {}'.format(len(insincere_questions)))
print('The average length of insincere questions: {}'.format(np.mean([len(question) for question in insincere_questions])))
print('The max length of insincere questions : {}'.format(np.max([len(question) for question in insincere_questions])))
print('The min length of insincere questions : {}'.format(np.min([len(question) for question in insincere_questions])))
# 统计高频词
c = Counter([word for question in insincere_questions for word in question]).most_common(100)
print('Most common words in insincere questions : \n{}'.format(c))

# sincere questions 统计
print('-' * 30 + 'Sincere Questions' + '-' * 30)
print('Total number fo sincere questions : {}'.format(len(sincere_questions)))
print('The average length of sincere questions: {}'.format(np.mean([len(question) for question in sincere_questions])))
print('The max length of sincere questions : {}'.format(np.max([len(question) for question in sincere_questions])))
print('The min length of sincere questions : {}'.format(np.min([len(question) for question in sincere_questions])))
# 统计高频词
c = Counter([word for question in sincere_questions for word in question]).most_common(100)
print('Most common words in sincere questions : \n{}'.format(c))

# --------------------------------------------------数据预处理----------------------------------------------------------
# ·构造词典Vocabulary
# ·构造映射表
# ·转换单词为tokens

# 句子最大长度
SENTENCE_LIMIT_SIZE = 70

# 构造词典
# 我们要基于整个语料来构造我们的词典，由于文本中包含许多干扰词汇，例如仅出现过1次的这类单词。对于这类极其低频词汇，我们可以对其
# 进行去除，一方面能加快模型执行效率，一方面也能减少特殊词带来的噪声。

# 将 questions+test_questions 扩展成单层列表:[word1,word2,....]
total_words = [word for question in (questions+test_questions) for word in question]
# 统计词汇
c = Counter(total_words)
# 倒序查看词频
print('The word frequency of train.csv and test.csv is : ')
print()
sorted(c.most_common(),key=lambda x : x[1],reverse=True)

# 初始化两个token: pad 和 unk
vocab = ['<pad>','<unk>']
vocab_max_length = 200000 # 词包最多能存储的单词数量

# 去除出现频次大于word_frequency的单词
word_frequency = 1
count = 1
for w,f in c.most_common():
    if count > vocab_max_length:
        break
    if f > word_frequency:
        vocab.append(w)
    count += 1
print('The size of vocabulary is : {}'.format(len(vocab)))

#构造映射
# 单词到编码的映射，例如：machine ---> 10256
word_to_token = {word: token for token,word in enumerate(vocab)}
# 编码到单词的映射，例如：10256---> machine
token_to_word = {token : word for token ,word in enumerate(vocab)}

# 转换文本：对文本进行编码
def convert_text_to_token(sentence,word_to_token_map=word_to_token,limit_size=SENTENCE_LIMIT_SIZE):
    '''
    根据单词--编码映射表 将单个句子转化为token
    :param sentence: 句子,类型是word list,[word1,word2,...]
    :param word_to_token_map: 单词到编码的映射
    :param limit_size: 句子最大长度。超过该长度的句子进行截断，不足的句子进行pad补全
    :return: 句子转换为token后的列表
    '''
    # 获取 unknow 单词和 pad的token
    unk_id = word_to_token_map['<unk>']
    pad_id = word_to_token_map['<pad>']

    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word,unk_id) for word in sentence]
    # 对句子长度进行规整，短的补全pad，长的截断 trunc
    # pad
    if len(tokens) < limit_size:
        tokens.extend([pad_id]*(limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]

    return tokens

# sincere questions随机采样,采样数量等于 insincere questions
print('Sampling sincere questions...')
sincere_insincere_proportion = 4/1 # The ratio of sincere/insincere 
index = np.random.randint(0,len(sincere_questions),int(len(insincere_questions)*sincere_insincere_proportion))
sincere_questions = np.array(sincere_questions)[index]
sincere_questions = sincere_questions.tolist()
questions = sincere_questions[:] + insincere_questions[:]
labels = [0]*len(sincere_questions) + [1]*len(insincere_questions)
del insincere_questions,sincere_questions
# 混洗数据
print('Shuffling questions....')
shuffled_index = np.random.permutation(range(len(labels)))
questions = np.array(questions)[shuffled_index]
questions = questions.tolist()
labels = np.array(labels)[shuffled_index]
labels = labels.tolist()

# 将 questions 转为 编码列表 : [[1,5,9,55,..],....]
print('Begining convert questions to tokens list...')
questions_tokens = []
for question in tqdm.tqdm(questions):
    tokens = convert_text_to_token(question)
    questions_tokens.append(tokens)
print('Del questions...')
del questions
# 将 test_questions 转为 编码列表 : [[1,5,9,55,..],....]
print('Begining convert test questions to tokens list...')
test_questions_tokens = []
for question in tqdm.tqdm(test_questions):
    tokens = convert_text_to_token(question)
    test_questions_tokens.append(tokens)
print('Del test questions...')
del test_questions

# 转换为numpy格式,方便处理
questions_tokens = np.array(questions_tokens)
test_questions_tokens = np.array(test_questions_tokens)
labels = np.array(labels).reshape(-1,1)

print('The shape of all question tokens in our corpus: ({},{})'.format(*questions_tokens.shape))
print('The shape of all targets in our corpus : ({},)'.format(*labels.shape))
print('The total time of Data_stage_2.py program is : %d s' %(time.time()-start))
# Data_stage_3.py
# -*- coding: utf-8 -*- 
# @Time    : 2019/1/13 12:32 
# @Author  : Weiyang

# ------------------------------------------------构造词向量-------------------------------------------------------------
# 这里使用 GoogleNews-vectors-negative300.bin预训练好的词向量来做embedding:
# ·如果当前词没有对应的词向量，则用随机数产生的向量替代
# ·如果当前词为 <PAD> ,则用 0向量替代

import warnings
warnings.filterwarnings(action="ignore")
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import time
import sys

start = time.time()

# loads 300x1 word vectors from file.
def load_bin_vec(fname):
    words_vector = KeyedVectors.load_word2vec_format(fname, binary=True) # 通过model来取对应词的词向量
    return words_vector
# 预训练的词向量的路径
vectors_file =  '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
print('Loading all_words ....')
print('Loading googleNew vector....')
words_vector = load_bin_vec(vectors_file)  # pre-trained vectors
google_words = set() # 存储 GoogleNews-vectors-negative300 中的 词
word_to_vec = {} # GoogleNews-vectors-negative300: {word:vector,...}
for word in tqdm.tqdm(words_vector.wv.vocab.keys()):
    google_words.add(word)
    word_to_vec[word] = np.array(words_vector[word],dtype=np.float32)     
print('The number of words which have pretrained-vectors in vocab is : {}'.format(len(set(vocab)&set(google_words))))
print()
print('The number of words which do not have pretrained-vectors in vocab is : {}'.format(len(set(vocab))-len(set(vocab)&set(google_words))))
del google_words,words_vector,vectors_file


# 构造词向量矩阵
VOCAB_SIZE = len(vocab) # 
EMBEDDING_SIZE = 300

# 初始化词向量矩阵(这里命名为 static是因为这个词向量矩阵用预训练好的填充，无需重新训练)
static_embeddings = np.zeros([VOCAB_SIZE,EMBEDDING_SIZE])
for word,token in tqdm.tqdm(word_to_token.items()):
    # 用google_vector词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word,0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token,:] = word_vector
del word_to_vec
    
# 重置PAD为0向量
pad_id = word_to_token['<pad>']
static_embeddings[pad_id,:] = np.zeros(EMBEDDING_SIZE)
static_embeddings = static_embeddings.astype(np.float32)

# --------------------------------------------分割训练集和测试集---------------------------------------------------------

def split_train_test(x,y,train_ratio=0.8,shuffle=True):
    '''
    分割train 和 test
    :param x: 输入特征序列
    :param y: 标签序列
    :param train_ratio: 训练样本比例
    :param shuffle: 是否shuffle
    :return:
    '''
    assert x.shape[0] == y.shape[0] , print('error shape!')

    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))
        x = x[shuffled_index]
        y = y[shuffled_index]
    # 分离 train 和 test
    train_size = int(x.shape[0] * train_ratio)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return x_train,x_test,y_train,y_test

# 划分train 和 test
x_train,x_test,y_train,y_test = split_train_test(questions_tokens,labels)
print('Del questions_token and labels ...')
del questions_tokens,labels

# ----------------------------------------------批量获取数据-------------------------------------------------------------

def get_batch(x,y,batch_size=300,shuffle=True):
    assert x.shape[0] == y.shape[0],print('error shape!')
    # shuffle
    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))
        x = x[shuffled_index]
        y = y[shuffled_index]
    # 统计共有几个完整的batch
    n_batches = int(x.shape[0] / batch_size)
    for i in range(n_batches - 1):
        x_batch = x[i * batch_size:(i+1)*batch_size]
        y_batch = y[i * batch_size:(i+1)*batch_size]
        yield x_batch,y_batch
        
print('The total time of Data_stage_3.py program is : %d s' %(time.time()-start))
# -*- coding: utf-8 -*- 
# @Time    : 2019/1/15 12:33 
# @Author  : Weiyang

########################################################################################################################
# CNN 模型实现文本分类
########################################################################################################################

import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import time

# 这里定义6种尺寸大小的filter，每种100个
filters_size = [2,3,4,5,6,7] # 2 表示filter每次覆盖的单词数，其大小为 2*embedding_size
num_filters = 100
# 超参数
BATCH_SIZE = 10000
EPOCHES = 50
LEARNING_RATE = 0.001
L2_LAMBDA = 10
KEEP_PROB = 0.8
embedding_size = 300
threshold = 0.5 # 阈值

start = time.time()
# 构建模型图
with tf.name_scope('CNN'):
    with tf.name_scope('placeholders'):
        inputs = tf.placeholder(dtype=tf.int32,shape=(None,SENTENCE_LIMIT_SIZE),name='inputs')
        targets = tf.placeholder(dtype=tf.float32,shape=(None,1),name='targets')

    # embeddings
    with tf.name_scope('embeddings'):
        #static_embeddings = tf.get_variable('embedding',[VOCAB_SIZE,EMBEDDING_SIZE]) # 不使用预训练的词向量
        embedding_matrix = tf.Variable(initial_value=static_embeddings,trainable=False,name='embedding_matrix')
        # embed = [None,sequence_limit_size,embedding_size]
        embed = tf.nn.embedding_lookup(embedding_matrix,inputs,name='embed')
        # 添加channel维度
        # embed_expanded = [None,sequence_limit_size,embedding_size,1] ，其中1 是新增加的维度,CNN需要输入channel信息
        embed_expanded = tf.expand_dims(embed,-1,name='embed_expand')

    # 用来存储max-pooling的结果
    pooled_outputs = []

    # 迭代多个filter
    for i,filter_size in enumerate(filters_size):
        with tf.name_scope('conv_maxpool_%s'%filter_size):
            filter_shape = [filter_size,embedding_size,1,num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape,mean=0.0,stddev=0.1),name='W')
            b = tf.Variable(tf.zeros(num_filters),name='b')
            # conv = [num_filters*len(filters_size),sequence_limit_size-filter_size+1,1,1]
            # 对于每个卷积核其卷积结果为[1, sequence_length - filter_size + 1, 1, 1]
            conv = tf.nn.conv2d(input=embed_expanded,filter=W,strides=[1,1,1,1],padding='VALID',name='conv')
            # 激活

            a = tf.nn.relu(tf.nn.bias_add(conv,b),name='activations')
            # 池化
            max_pooling = tf.nn.max_pool(value=a,
                                         ksize=[1,SENTENCE_LIMIT_SIZE - filter_size + 1,1,1],
                                         strides=[1,1,1,1],
                                         padding='VALID',
                                         name='max_pooling')
            pooled_outputs.append(max_pooling)

    # 统计所有的filter
    total_filters = num_filters * len(filters_size)
    total_pool = tf.concat(pooled_outputs,3)
    flattend_pool = tf.reshape(total_pool,(-1,total_filters))

    # dropout
    with tf.name_scope('dropout'):
        dropout = tf.nn.dropout(flattend_pool,KEEP_PROB)

    # output
    with tf.name_scope('output'):
        W = tf.Variable(tf.truncated_normal([total_filters,1],stddev=0.1),name='W_output')
        b = tf.Variable(tf.zeros(1),name='b_output')

        logits = tf.add(tf.matmul(dropout,W),b)
        predictions = tf.nn.sigmoid(logits,name='predictions')

    # loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets,logits=logits))
        loss = loss + L2_LAMBDA * tf.nn.l2_loss(W)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    # evaluation
    with tf.name_scope('evaluation'):
        correct_preds = tf.equal(tf.cast(tf.greater(predictions,threshold),tf.float32),targets)
        accuracy = tf.reduce_mean(tf.reduce_sum(tf.cast(correct_preds,tf.float32),axis=1))

# 训练模型
#存储训练损失
cnn_train_loss = []
# 存储准确率
cnn_train_accuracy = []
cnn_test_accuracy = []
# 存储精确率
cnn_train_precision = []
cnn_test_precision = []
# 存储召回率
cnn_train_recall = []
cnn_test_recall = []
# 存储F1值
cnn_train_F1 = []
cnn_test_F1 = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches = int(x_train.shape[0] / BATCH_SIZE)

    for epoch in range(1,EPOCHES+1):
        total_loss = 0
        for x_batch, y_batch in get_batch(x_train, y_train):
            _, batch_loss= sess.run([optimizer,loss], feed_dict={inputs: x_batch, targets: y_batch})
            total_loss += batch_loss
        #存储训练损失
        cnn_train_loss.append(total_loss/n_batches)
        
        # 在train 上的准确率: 随机抽取与测试数据集等量的数据进行测试
        index = np.random.randint(0,len(x_train),len(x_test))
        x_train_temp = x_train[index]
        y_train_temp = y_train[index]
        
        # 分批次预测
        batch_size = 20000
        label_pre = [] # predict label,[1,0,1,..]
        train_accuracy = []
        for start in range(0,len(x_train_temp),batch_size):
            train_questions = x_train_temp[start:start+batch_size]
            train_labels = y_train_temp[start:start+batch_size]
            accuracy_temp,label_pre_temp = sess.run([accuracy,predictions], feed_dict={inputs: train_questions,targets:train_labels})
            train_accuracy.append(accuracy_temp)
            label_pre_temp = [int(prob[0] > threshold) for prob in label_pre_temp]
            label_pre.extend(label_pre_temp)
        cnn_train_accuracy.append(np.mean(train_accuracy))
        
        y_train_temp = [label[0] for label in y_train_temp] # true label,[1,0,1,..]
        
        res_train_precision = precision_score(y_train_temp, label_pre).astype(np.float32) # train precision
        cnn_train_precision.append(res_train_precision)
        
        res_train_recall = recall_score(y_train_temp, label_pre).astype(np.float32) # train recall
        cnn_train_recall.append(res_train_recall)
        
        res_train_f1 = f1_score(y_train_temp, label_pre).astype(np.float32) # train F1 Score
        cnn_train_F1.append(res_train_f1)

        # 在 test上的准确率：用的是全量数据，而非批量数据
        # 分批次预测
        batch_size = 20000
        label_pre = [] # predict label,[1,0,1,..]
        label_pre_prob = [] # predict probability,[[0.5],[0.6],..]
        test_accuracy = []
        for start in range(0,len(x_test),batch_size):
            test_questions = x_test[start:start+batch_size]
            test_labels = y_test[start:start+batch_size]
            accuracy_temp,label_pre_temp = sess.run([accuracy,predictions], feed_dict={inputs: test_questions,targets:test_labels})
            # 最后一轮迭代,存储预测的概率以寻找最佳阈值
            if epoch == EPOCHES:
                label_pre_prob.extend(label_pre_temp)
            test_accuracy.append(accuracy_temp)
            label_pre_temp = [int(prob[0] > threshold) for prob in label_pre_temp]
            label_pre.extend(label_pre_temp)
        cnn_test_accuracy.append(np.mean(test_accuracy))
        
        y_test_temp = [label[0] for label in y_test] # true label,[1,0,1,..]
        
        res_test_precision = precision_score(y_test_temp, label_pre).astype(np.float32)
        cnn_test_precision.append(res_test_precision)
        
        res_test_recall = recall_score(y_test_temp, label_pre).astype(np.float32)
        cnn_test_recall.append(res_test_recall)
        
        res_test_f1 = f1_score(y_test_temp, label_pre).astype(np.float32)
        cnn_test_F1.append(res_test_f1)
        
        print('Threshold: {:.2f} , Epoch : {} , Train Loss: {:.4f} ,Train Accuracy : {:.4f} ,Test Accuracy : {:.4f} ,Train Precision : {:.4f} , Test Precision : {:.4f} , Train Recall : {:.4f} , Test Recall : {:.4f}'
              ' ,Train F1_Score : {:.4f} ,Test F1_Score : {:.4f}'.format(threshold, epoch, total_loss / n_batches, np.mean(train_accuracy), np.mean(test_accuracy),res_train_precision,res_test_precision,res_train_recall,res_test_recall,res_train_f1,res_test_f1))
        
        # 最后一轮迭代,寻找最佳阈值
        if epoch == EPOCHES:
             # 寻找最佳阈值
            thresholds_f1 = []
            thresholds_precision = []
            thresholds_recall = []
            y_test = [label[0] for label in y_test] # true label,[1,0,1,..]
            for thresh in np.arange(0.1, 0.501, 0.01):
                thresh = np.round(thresh, 2)
                label_predict = [int(prob[0] > thresh) for prob in label_pre_prob] # predict label,[1,0,1,..]
                res_accuracy = accuracy_score(y_test, label_predict).astype(np.float32)
                res_f1 = f1_score(y_test, label_predict).astype(np.float32)
                res_precision = precision_score(y_test, label_predict).astype(np.float32)
                res_recall = recall_score(y_test, label_predict).astype(np.float32)
                thresholds_f1.append([thresh, res_f1])
                thresholds_precision.append([thresh, res_precision])
                thresholds_recall.append([thresh, res_recall])
                print("Epoch: %d , Threshold:%.4f , Accuracy:%.4f , Precision:%.4f , Recall:%.4f ,F1:%.4f " % (epoch,
                    thresh, res_accuracy,res_precision, res_recall,res_f1))
            print('*' * 100)
            thresholds_f1.sort(key=lambda x: x[1], reverse=True)
            best_thresh = thresholds_f1[0][0]
            print('Best F1 Score at threshold {0} is {1}'.format(best_thresh, thresholds_f1[0][1]))

            thresholds_precision.sort(key=lambda x: x[1], reverse=True)
            best_thresh = thresholds_precision[0][0]
            print('Best Precision at threshold {0} is {1}'.format(best_thresh, thresholds_precision[0][1]))

            thresholds_recall.sort(key=lambda x: x[1], reverse=True)
            best_thresh = thresholds_recall[0][0]
            print('Best Recall at threshold {0} is {1}'.format(best_thresh, thresholds_recall[0][1]))

    figure = plt.figure(num='The Accuracy,Precision,Recall and F1_Score of CNN Model of Threshold : %.2f'%threshold)
    x = range(1, EPOCHES + 1)
    
    # 展现 train 上的 F1_Score 和 test上的 F1_Score
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('F1_Score')
    plt.plot(x, cnn_train_F1, label='train')
    plt.plot(x, cnn_test_F1, label='test')
    plt.xlim((0, EPOCHES + 1))
    plt.ylim((0.0, 1.01))
    plt.legend(loc='upper right')
    
    # 展现train上的准确率和test上的准确率
    ax2 = plt.subplot(2,4,5)
    ax2.set_title('Accuracy')
    plt.plot(x, cnn_train_accuracy, label='train')
    plt.plot(x, cnn_test_accuracy, label='test')
    plt.xlim((0, EPOCHES + 1))
    plt.ylim((0.0, 1.01))
    plt.legend(loc='upper right')
    
    # 展现train上的精确率和test上的精确率
    ax3 = plt.subplot(2,4,6)
    ax3.set_title('Precision')
    plt.plot(x, cnn_train_precision, label='train')
    plt.plot(x, cnn_test_precision, label='test')
    plt.legend(loc='upper right')
    plt.xlim((0, EPOCHES + 1))
    plt.ylim((0.0, 1.01))
    
    # 展现train上的召回率和test上的召回率
    ax4 = plt.subplot(2,4,7)
    ax4.set_title('Recall')
    plt.plot(x, cnn_train_recall, label='train')
    plt.plot(x, cnn_test_recall, label='test')
    plt.xlim((0, EPOCHES + 1))
    plt.ylim((0.0, 1.01))
    plt.legend(loc='upper right')
    
    # 展现train上的loss
    ax5 = plt.subplot(2,4,8)
    ax5.set_title('Train Loss')
    plt.plot(x, cnn_train_loss, label='train')
    plt.xlim((0, EPOCHES + 1))
    plt.ylim((0.0, 50.0))
    plt.legend(loc='upper right')
    
    figure.subplots_adjust(hspace=0.5) # 增加子图间隔
    figure.suptitle('The Train Loss , Accuracy , Precision, Recall and F1_Score of CNN Model of Threshold : %.2f' % threshold) # 大图标题
    plt.show()

    #print(cnn_train_loss)
    # 模型预测：在test上的准确率
    # 分批次预测
    batch_size = 20000
    label_pres = []
    for start in range(0,len(test_questions_tokens),batch_size):
        test_questions = test_questions_tokens[start:start+batch_size]
        label_pre = sess.run(predictions, feed_dict={inputs: test_questions})
        label_pre = [int(prob[0] > threshold) for prob in label_pre]
        label_pres.extend(label_pre)
    # 输出预测结果
    sub = pd.DataFrame(columns=['qid','prediction'])
    sub['qid'] = test_qids
    sub['prediction'] = label_pres
    # 将question为空的qid的类别添加进去并且设置为1
    if len(empty_question_qids) !=0:
        empty_questions = pd.DataFrame(columns=['qid','prediction'])
        empty_questions['qid'] = empty_question_qids
        empty_questions['prediction'] = [1]*len(empty_question_qids)
        sub = pd.concat([sub,empty_questions],axis=0)
    sub.to_csv('submission.csv',index=False)
    
print('The total time of CNN.py program is : %d min' % ((time.time() - start)/60))