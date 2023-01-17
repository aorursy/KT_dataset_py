import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
print('Loading data...')

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(test.shape)
train.head()
test.head()
train_X = train.iloc[:, 1:].values.astype('float32') #pixel_0 to 778

train_y = train.iloc[:, 0].values.astype('int') #label

test_X = test.values.astype('float32') #pixel_0 to _778
import chainer

import chainer.links as L

import chainer.functions as F

from chainer.dataset.convert import concat_examples
class MLP(chainer.Chain):

    def __init__(self, n_hidden, n_out):

        super (MLP, self).__init__(

            l1 = L.Linear(None, n_hidden),

            l2 = L.Linear(n_hidden, n_hidden),

            l3 = L.Linear(n_hidden, n_out),)

            

    def __call__(self, x):

        h1 = F.relu(self.l1(x))

        h2 = F.relu(self.l2(h1))

        return self.l3(h2)





print('Constructing model...')

hidden_dim = 32  # Hidden dim for neural network

out_dim = 10     # Number of labels to classify, it is 10 for MNIST task.



model = MLP(hidden_dim, out_dim)
#aprint(model(train_X).data)
from chainer import optimizers

#https://docs.chainer.org/en/stable/reference/optimizers.html
optimizer = optimizers.Adam()

optimizer.setup(model)
from chainer import iterators

#https://docs.chainer.org/en/stable/reference/generated/chainer.iterators.MultiprocessIterator.html
#train data: 42000 

batch_size = 420
dataset_train = []

for X, y in zip(train_X, train_y):

    dataset_train.append((X, y))

train_iterator = iterators.SerialIterator(

    dataset_train, 

    batch_size

)
#print(train_iterator.epoch)

#print(len(train_iterator.next()))



#for i in range(1000):

#    train_iterator.next()

#print(train_iterator.epoch)
from chainer.datasets import TupleDataset

from sklearn.model_selection import train_test_split

from chainer import Variable
X_train, X_val, y_train, y_val= train_test_split(train_X, train_y, 

                                                 test_size=0.20, 

                                                 random_state=42)



train_dataset = TupleDataset(X_train, y_train)
print(train_dataset.__getitem__(10))
train_accuracy_log = []

val_accuracy_log = []



max_epoch = 10



while train_iterator.epoch < max_epoch:

    #Prepare batch data

    batch = train_iterator.next()

    X_batch, y_batch = chainer.dataset.concat_examples(batch)

    

    #Calculate the cost

    train_y_preds = model(X_batch)

    train_loss = F.softmax_cross_entropy(train_y_preds, y_batch)

    

    #Learning

    model.cleargrads()

    train_loss.backward()

    optimizer.update()

    

    #Check accuracy

    train_accuracy = F.accuracy(train_y_preds, y_batch)

    train_accuracy_log.append(float(train_accuracy.data))

    

    #Check val accuracy and generalization performance

    if train_iterator.is_new_epoch:

        print('******************'*5)

        val_y_preds = model(X_val)

        val_loss = F.softmax_cross_entropy(val_y_preds, y_val)

        val_accuracy = F.accuracy(val_y_preds, y_val)

        

        val_accuracy_log.append(float(val_accuracy.data))

        

        print('epoch{} train_accuracy:{}, val_accuracy:{}'.\

              format(train_iterator.epoch,

                     train_accuracy.data,

                     val_accuracy.data))

    

    

    
plt.plot(range(len(train_accuracy_log)), 

         train_accuracy_log)

plt.plot(range(100, len(val_accuracy_log)*101, 100),

         val_accuracy_log, color='red')

plt.show()
test_y_preds = model(test_X)

test_y_preds= test_y_preds.data

print(test_y_preds.shape)
print('Saving submission file …')



test_y_preds = np.argmax(test_y_preds,axis = 1)



test_y_preds = pd.Series(test_y_preds,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),test_y_preds],axis = 1)

submission.to_csv('submission_mlp.csv', index_label=False, index=False)


