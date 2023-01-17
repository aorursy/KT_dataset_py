from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,rmsprop

import pandas as pd

import numpy as np

dt=pd.read_csv('../input/diabetes.csv',header=None)

dt.shape

data=dt.values

data=data[1:,:]

len(data)
np.random.shuffle(data)
data


train_samples=data[1:600,:-1].astype(float)

test_samples=data[600:,:-1].astype(float)

train_labels=data[1:600,-1]

test_labels=data[600:,-1]



from keras.utils import to_categorical

test_labels.shape
mean=train_samples.mean(axis=0)

train_samples -= mean

std=train_samples.std(axis=0)

train_samples /=std

test_samples -=mean

test_samples /=std

train_samples[2]
train_labels.shape
# from keras.utils import to_categorical

# train_labels=to_categorical(train_labels)

# test_labels=to_categorical(test_labels)

model=Sequential()

model.add(Dense(16,activation='relu',input_shape=(train_samples.shape[1],)))

model.add(Dense(1,activation='sigmoid'))
model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

k=10 

num_val_samples=len(train_samples)//k

num_epochs=50

all_scores=[]

for i in range(k):

    print('processing fold #', i)

    val_data=train_samples[i * num_val_samples: (i+1) * num_val_samples]

    val_targets=train_labels[i * num_val_samples: (i+1)* num_val_samples]

    partial_train_data=np.concatenate([train_samples[:i * num_val_samples],

                                     train_samples[(i+1)*num_val_samples:]],

                                     axis=0)

    partial_train_targets=np.concatenate([train_labels[:i *num_val_samples],

                                         train_labels[(i+1)* num_val_samples:]],

                                        axis=0)

    model.fit(partial_train_data,partial_train_targets,

             epochs=num_epochs,batch_size=4)

    val_loss,val_acc=model.evaluate(val_data,val_targets,verbose=0)

  

    
print('model validation accuracy: ',val_acc)

print('model validation loss: ',val_loss)
test_loss,test_acc=model.evaluate(test_samples,test_labels,verbose=0)
test_acc
test_loss