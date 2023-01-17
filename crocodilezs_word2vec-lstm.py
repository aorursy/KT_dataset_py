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
import pandas as pd



filepath = '/kaggle/input/chinese-text-multi-classification/nCoV_100k_train.labled.csv'

file_data = pd.read_csv(filepath)
# 取前10000条数据

data = file_data.head(10000)
# 选择中文内容和情感倾向

data = data[['微博中文内容', '情感倾向']]
# 查看数据

data.head(10)
# 处理缺失值

data.isnull().sum()
data = data.dropna()
import re



def clean_zh_text(text):

    # keep English, digital and Chinese

    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')

    return comp.sub('', text)

 



data['微博中文内容'] = data.微博中文内容.apply(clean_zh_text)
# 分词

import jieba

def chinese_word_cut_w2v(mytext):

    return list(jieba.cut_for_search(mytext))



data['cut_comment_w2v'] = data.微博中文内容.apply(chinese_word_cut_w2v)
# 分词

def chinese_word_cut(mytext):

    return " ".join(jieba.cut(mytext))



data['cut_comment'] = data.微博中文内容.apply(chinese_word_cut)
x = data['cut_comment']

x_w2v = data['cut_comment_w2v']

y = data['情感倾向']
for i,v in y.items():

    if v=='-' or v=='4':

        print(i)

        y[i]='2'

    if  v=='-1':

        y[i]='2'
print(y)
y = y.astype(np.float32)
x.head(10)
x_w2v.head(10)
y.head(10)
# word2vec

from gensim.models import Word2Vec



model = Word2Vec(x_w2v, hs=1,min_count=1,window=10,size=400)
from gensim.test.utils import common_texts, get_tmpfile

path = get_tmpfile("word2vect.model") #创建临时文件

model.save("word2vect.model")

# model = Word2Vec.load("word2vec.model")
for key in model.wv.similar_by_word('开心', topn =100):

    print(key)
model.wv['开心']
from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument



documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(x_w2v)]

modeld = Doc2Vec(documents, dm=1, size=400, window=8, min_count=5, workers=4)
from gensim.test.utils import get_tmpfile



fname = get_tmpfile("doc2vec_model")



modeld.save(fname)

#modeld = Doc2Vec.load(fname)
corpus=modeld.docvecs
commentlst = []

for i in range(10000-30):

    commentlst.append(corpus[i].tolist())
yarray = np.array(y.tolist())
commentarray = np.array(commentlst)
# 分训练集和测试集

lentrain = 8000

lentest = 2000



x_train = commentarray[:lentrain]

y_train = yarray[:lentrain]

x_test = commentarray[(-1)*lentest-1:-1]

y_test = yarray[(-1)*lentest-1:-1]

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print(type(y_test[0]))
featuresTrain = torch.from_numpy(x_train)

targetsTrain = torch.from_numpy(y_train).type(torch.LongTensor)

featuresTest = torch.from_numpy(x_test)

targetsTest = torch.from_numpy(y_test).type(torch.LongTensor)
print(x_train.shape)
# Create RNN Model

class RNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):

        super(RNNModel, self).__init__()

        

        # Number of hidden dimensions

        self.hidden_dim = hidden_dim

        

        # Number of hidden layers

        self.layer_dim = layer_dim

        

        # RNN

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        

        # Readout layer

        self.fc = nn.Linear(hidden_dim, output_dim)

    

    def forward(self, x):

        

        # Initialize hidden state with zeros

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

            

        # One time step

        out, hn = self.rnn(x, h0)

        out = self.fc(out[:, -1, :]) 

        return out



# batch_size, epoch and iteration

batch_size = 100

n_iters = 20000

num_epochs = n_iters / (len(x_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch train and test sets

train = TensorDataset(featuresTrain,targetsTrain)

test = TensorDataset(featuresTest,targetsTest)



# data loader

train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

    

# Create RNN

input_dim = 20    # input dimension

hidden_dim = 200  # hidden layer dimension

layer_dim = 1     # number of hidden layers

output_dim = 3   # output dimension



model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)



# Cross Entropy Loss 

error = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.03

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
ftst0 = torch.zeros(100)

print(ftst0)
seq_dim = 20 

loss_list = []

iteration_list = []

accuracy_list = []

count = 0

for epoch in range(num_epochs):

    for i, (embeddings, labels) in enumerate(train_loader):



        train  = Variable(embeddings.view(-1, seq_dim, input_dim))

        labels = Variable(labels)

            

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        train = torch.tensor(train, dtype=torch.float32)

        outputs = model(train)

        

        # Calculate softmax and ross entropy loss

        loss = error(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

        

        count += 1

        

        if count % 250 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            rs = 0

            f1s = 0

            # Iterate through test dataset

            for embeddings, labels in test_loader:

                embeddings = Variable(embeddings.view(-1, seq_dim, input_dim))

                

                # Forward propagation

                embeddings = torch.tensor(embeddings, dtype=torch.float32)

                outputs = model(embeddings)

                

                # Get predictions from the maximum value

                predicted = torch.max(outputs.data, 1)[1]

                

                # Total number of labels

                total += labels.size(0)

                

                correct += (predicted == labels).sum()

                

                rs += recall_score(predicted, labels, average='weighted')

                

                f1s += f1_score(predicted, labels, average='weighted')

            

            accuracy = 100 * correct / float(total)

            r = rs / 20

            f1 = f1s / 20



            

            # store loss and iteration

            loss_list.append(loss.data)

            iteration_list.append(count)

            accuracy_list.append(accuracy)

            if count % 500 == 0:

                # Print Loss

                print('Iteration: {}  Loss: {}  Accuracy: {} %  Recall: {} %  f1: {} '.format(count, loss.item(), accuracy, r, f1))
import matplotlib.pyplot as plt



# visualization loss 

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("RNN: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("RNN: Accuracy vs Number of iteration")

plt.savefig('graph.png')

plt.show()