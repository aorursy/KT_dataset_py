vocab = open('../input/taskset/latex_vocab.txt').read().split('\n')
vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
formulas = open('../input/taskset/formulas.norm.lst').read().split('\n')
idx_to_vocab = dict([ (i,vocab[i]) for i in range(len(vocab))])
vocab_to_idx #vocab dict
lab=idx_to_vocab.keys()
print(lab)
def formula_to_indices(formula):
    formula = formula.split(' ')
    res = [0]
    for token in formula:
      if token in vocab_to_idx:
        res.append( vocab_to_idx[token] + 4 )
      else:
        res.append(2)
    res.append(1)
    return res
formulas = list(map( formula_to_indices, formulas))
formulas
train = open('../input/taskset/train.lst').read().split('\n')[:-1]
val = open('../input/taskset/valid.lst').read().split('\n')[:-1]
test = open('../input/taskset/test.lst').read().split('\n')[:-1]    #reading files
import numpy as np
from PIL import Image
def import_images(datum):
    datum = datum.split(' ')
    img = np.array(Image.open('../input/processed-images/images_processed/'+datum[0]).convert('L'))
    return (img, formulas[ int(datum[1]) ])

train = list(map(import_images, train)) #reading the images and storing them
val = list(map(import_images, val))
test = list(map(import_images, test))
print(train[0])
shapes_list=[]
for i in range(0,len(train)):
    shapes_list.append(train[i][0].shape)
uni_shapes_list=list(set(shapes_list))
print(uni_shapes_list)
X_train=[]
y_train=[]                                        #arranging the data into X_train(images) and y_train(formula indices)
for i in range(0,len(train)):
    if(train[i][0].shape==(50, 320)):
        X_train.append(train[i][0])
        y_train.append(np.array(train[i][1]))
X_val=[]
y_val=[]
for i in range(0,len(val)):
    if(val[i][0].shape==(50, 320)):
        X_val.append(val[i][0])
        y_val.append(np.array(val[i][1]))
X_train=np.array(X_train)
y_train=np.array(y_train)
X_valid=np.array(X_val)
y_valid=np.array(y_val)
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
y_train[0].size
y_valid[0].size
size_y_train=[]
for i in range(0,len(y_train)):
    size_y_train.append(y_train[i].size)
size_y_valid=[]
for i in range(0,len(y_valid)):
    size_y_valid.append(y_valid[i].size)
max_size=158
#TODO :add for loop
for i in range(0,len(y_train)):                     # padding the sequences and making them into a constant length
    tmp=y_train[i]
    pad=np.pad(tmp,(0, max_size - tmp.size), 'constant', constant_values = 3)
    y_train[i]=np.expand_dims( pad, 0)
for i in range(0,len(y_valid)):
    tmp=y_valid[i]
    pad=np.pad(tmp,(0, max_size - tmp.size), 'constant', constant_values = 3)
    y_valid[i]=np.expand_dims( pad, 0)
print(y_train[0].size)
print(y_train[0])
print(y_valid[0].size)
print(y_valid[0])
y_train=list(y_train)
y_valid=list(y_valid)
y_train[0]
y_valid[0]
for i in range(0,len(y_train)):
     y_train[i]=y_train[i][0]
for i in range(0,len(y_valid)):
     y_valid[i]=y_valid[i][0]
y_train[0]
y_valid[0]
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
X_train.shape
X_valid.shape
y_train=np.array(y_train)
y_train.shape
y_valid=np.array(y_valid)
y_valid.shape
from keras.utils import to_categorical #one hot encoding 
y_train= to_categorical(y_train)
from keras.utils import to_categorical
y_valid= to_categorical(y_valid)
y_train.shape
y_valid.shape
y_train[0]
y_valid[0]
X_train = X_train.reshape(X_train.shape[0],320,50,1) #imp step to reshape the arrays
X_valid= X_valid.reshape(X_valid.shape[0],320,50,1)
X_train.shape
X_valid.shape
input_shape=(320,50,1)
X_train = np.array(X_train, dtype=np.float)
X_train = X_train - 128.
X_train = X_train / 128.
X_valid = np.array(X_valid, dtype=np.float)
X_valid = X_valid - 128.
X_valid = X_valid / 128.
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D ,Reshape,TimeDistributed,GRU
from keras.layers import AveragePooling2D,Concatenate, Add,MaxPooling2D, Dropout, GlobalMaxPooling2D,Flatten, GlobalAveragePooling2D,Bidirectional,LSTM
from keras.models import Model, Sequential
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=(320,50,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

conv_to_rnn_dims = (158,736)
model.add(Reshape(conv_to_rnn_dims))
model.add(Dense(128, activation='relu'))
model.add(Bidirectional(LSTM(60 , return_sequences = True )))
model.add(Bidirectional(LSTM(60 , return_sequences = True )))
model.add(Dense(503, activation='relu'))
model.summary()
#from keras.optimizers import Adam
#add = Adam(learning_rate=0.1,
 #   beta_1=0.99,
  #  beta_2=0.998,
   # epsilon=1e-06,
    #amsgrad=False,)
#model.compile(loss = 'categorical_crossentropy' , optimizer = add )
from keras.optimizers import SGD
opt = SGD(lr=0.01)
model.compile(loss = "categorical_crossentropy", optimizer = opt)
hist = model.fit(X_train,y_train,epochs = 5,validation_data = (X_valid,y_valid),shuffle = False)