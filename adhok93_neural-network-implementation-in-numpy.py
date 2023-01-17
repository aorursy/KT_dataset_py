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
import pandas as pd

import numpy as np



data = pd.read_csv('../input/digit-recognizer/train.csv')

data = data.sample(frac=1)

print(data[['label']].groupby('label').size().reset_index())



one_hot = pd.get_dummies(data['label'].unique())

one_hot['label'] = one_hot.index



data = pd.merge(data,one_hot)

#data = data.drop('label',axis=1)

data = data.sample(frac=1)



data_train = data.iloc[0:40000]

data_test = data.iloc[40000:42000]

data_train.drop('label',axis=1,inplace=True)



data_test.drop('label',axis=1,inplace=True)



## Create the train and test set

X_train = np.array(data_train.drop([0,1,2,3,4,5,6,7,8,9],axis=1).values)/255

y_train = np.array(data_train[[0,1,2,3,4,5,6,7,8,9]].values)

X_test = np.array(data_test.drop([0,1,2,3,4,5,6,7,8,9],axis=1).values)/255

y_test = np.array(data_test[[0,1,2,3,4,5,6,7,8,9]].values)

one_hot
X_train = X_train.T

y_train = y_train.T

print(X_train.shape)

print(y_train.shape)

X_test = X_test.T

y_test = y_test.T



def sigmoid(x):

    return(1./(1+np.exp(-x)))



def softmax(x): 

    """Compute softmax values for each sets of scores in x.""" 



    e_x = np.exp(x - np.max(x)) 



    return (e_x / e_x.sum(axis=0)) 



import random

random.seed(42)

w1 = np.random.rand(100,784)/np.sqrt(784)

b0 = np.zeros((100,1))/np.sqrt(784)

w2 = np.random.rand(10,100)/np.sqrt(100)

b1 = np.zeros((10,1))/np.sqrt(100)

loss=[]

batches = 1000



lr = 0.5

batch_size = 200

beta = 0.9

count = 0

epochs = 500
loss_weight_dict = {

    

}

### Forward Pass

for i in range(epochs):

    if i%100==0:

        print('Epoch :',i)

    permutation = np.random.permutation(X_train.shape[1])

    X_train_shuffled = X_train[:, permutation]

    Y_train_shuffled = y_train[:, permutation]

    

    for j in range(batches):

        

        begin = j * batch_size

        end = min(begin + batch_size, X_train.shape[1] - 1)

        if begin>end:

            continue

        X = X_train_shuffled[:, begin:end]

        Y = Y_train_shuffled[:, begin:end]

        m_batch = end - begin

        x1 = sigmoid(w1@X+b0)

        x2 = softmax(w2@x1+b1)

        delta_2 = (x2-Y)

        delta_1 = np.multiply(w2.T@delta_2, np.multiply(x1,1-x1))

        if i==0 :

            dW1 = delta_1@X.T

            dW2 = delta_2@x1.T

            db0 = np.sum(delta_1,axis=1,keepdims=True)

            db1 = np.sum(delta_2,axis=1,keepdims=True)

        else:

            dW1_old = dW1

            dW2_old = dW2

            db0_old = db0

            db1_old = db1

            dW1 = delta_1@X.T

            dW2 = delta_2@x1.T

            db0 = np.sum(delta_1,axis=1,keepdims=True)

            db1 = np.sum(delta_2,axis=1,keepdims=True)

            dW1 = (beta * dW1_old + (1. - beta) * dW1)

            db0 = (beta * db0_old + (1. - beta) * db0)

            dW2 = (beta * dW2_old + (1. - beta) * dW2)

            db1 = (beta * db1_old + (1. - beta) * db1)





        w1 = w1 - (1./m_batch)*(dW1)*lr

        b0 = b0 - (1./m_batch)*(db0)*(lr)

        w2 = w2 - (1./m_batch)*(dW2)*lr

        b1 = b1 - (1./m_batch)*(db1)*(lr)

    

    x1 = sigmoid(w1@X_train+b0)

    x2_train = softmax(w2@x1+b1)

    

#     print('Training Loss...')

#     print(-np.mean(np.multiply(y_train,np.log(x2))))

    add_loss = {

        'loss' : -np.mean(np.multiply(y_train,np.log(x2_train))),

        'weight_1' : w1,

        'weight_2':w2,

        'b0' : b0,

        'b1': b1

    }

    

    

    

    

    

    x1 = sigmoid(w1@X_test+b0)

    x2_test = softmax(w2@x1+b1)

#     print('Testing Loss...')

#     print(-np.mean(np.multiply(y_test,np.log(x2))))

    

    add_loss['testing_loss'] = -np.mean(np.multiply(y_test,np.log(x2_test)))

    loss_weight_dict[count] = add_loss

    count = count + 1
test_loss = []



for i in range(len(loss_weight_dict)):

    test_loss.append(loss_weight_dict[i]['testing_loss'])

import matplotlib.pyplot as plt

plt.plot(test_loss)

plt.xlabel('Epochs')

plt.ylabel('Testing Loss')

plt.show()



print('Index where test loss is minimum :',test_loss.index(min(test_loss)))

print('Minimum Test Loss is :',min(test_loss))
train_loss = []



for i in range(len(loss_weight_dict)):

    train_loss.append(loss_weight_dict[i]['loss'])

import matplotlib.pyplot as plt

plt.plot(train_loss)

plt.xlabel('Epochs')

plt.ylabel('Training Loss')

plt.show()



print('Training Loss at index where test loss is minimum :', train_loss[test_loss.index(min(test_loss))])
testing_data = pd.read_csv('../input/digit-recognizer/test.csv')

test_data = np.array(testing_data.values)

print(test_data.shape)

test_data = test_data.T



test_data = test_data/255



x1_test = sigmoid(w1@test_data+b0)

y_test = softmax(w2@x1_test+b1)



y_test_df = pd.DataFrame(y_test)

y_test_df = (y_test_df == y_test_df.max()).astype(int)



y_test_df = y_test_df.transpose().merge(one_hot)[['label']] 



y_test_df = y_test_df.rename(columns = {'label':'Label'})
y_test_df['ImageId'] = y_test_df.index+1
y_test_df
y_test_df.groupby('Label').size().reset_index()
y_test_df.to_csv('sample_submission.csv',mode = 'w', index=False)