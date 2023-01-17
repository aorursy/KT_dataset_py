import tensorflow
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, optimizers
import math

%matplotlib inline
tensorflow.__version__
# open the file as readonly
h5f = h5py.File('../input/street-view-house-nos-h5-file/SVHN_single_grey1.h5','r')
h5f.keys()
# load the already splited train, validation and test data
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]

X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]

print(f'Size of X_train is {X_train.shape}')
print(f'Size of y_train is {y_train.shape}\n')

print(f'Size of X_val is {X_val.shape}')
print(f'Size of y_val is {y_val.shape}\n')

print(f'Size of X_test is {X_test.shape}')
print(f'Size of y_test is {y_test.shape}')
plt.imshow(X_train[1],cmap='gray')
print(f'Label for the image is {y_train[1]}')
X_train = X_train.reshape(42000, 32*32)
X_val= X_val.reshape(X_val.shape[0], 32*32)
X_test = X_test.reshape(X_test.shape[0],32*32)

print(f'Shape of X_train is {X_train.shape}')
print(f'Shape of X_val is {X_val.shape}')
print(f'Shape of X_test is {X_test.shape}')
print(f'Min value is {X_train.min()}')
print(f'Max value is {X_train.max()}')
print('Before Normalization')
print(f'Min value is {X_train.min()}')
print(f'Max value is {X_train.max()}\n')

X_train = X_train/255.0
X_val= X_val/255.0
X_test = X_test/255.0

print('After Normalization')
print(f'Min value is {X_train.min()}')
print(f'Max value is {X_train.max()}')
print(f'Sample value before one hot encode {y_train[0]}\n')
y_train = tensorflow.keras.utils.to_categorical(y_train,num_classes=10)
y_val= tensorflow.keras.utils.to_categorical(y_val,num_classes=10)
y_test= tensorflow.keras.utils.to_categorical(y_test, num_classes=10)
print(f'Sample value after one hot encode {y_train[0]}')
plt.figure(figsize=(10,1))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_train[i].reshape(32,32),cmap='gray')
    print(f'Label for image at index {i+1} is {np.argmax(y_train[0:10][i])}')

def model_1(iterations):
    hidden_nodes=256
    output_nodes=10
    iterations=iterations
    
    model = Sequential()
    model.add(Reshape((1024,), input_shape=(32,32,), name='Input_layer'))
    model.add(BatchNormalization())
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(hidden_nodes,activation='relu'))
    
    model.add(Dense(output_nodes, activation='softmax'))
    
    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    #Fit the model
    model.fit(X_train,y_train,epochs=iterations, batch_size=1000, verbose=1)
    
    scores=[]
    score = model.evaluate(X_train,y_train, verbose=0)
    scores.append(score)
    score = model.evaluate(X_val,y_val, verbose=0)
    scores.append(score)
    score = model.evaluate(X_test,y_test, verbose=0)
    scores.append(score)
    return scores
model_1(1)
#running for more epochs
scores = model_1(100)
print(f'Training Dataset Loss is {scores[0][0]} Accuracy is {scores[0][1]}\n')
print(f'Validation Dataset Loss is {scores[1][0]} Accuracy is {scores[1][1]}\n')
print(f'Test Dataset Loss is {scores[2][0]} Accuracy is {scores[2][1]}\n')
def model_2(iterations, lr, Lambda, verb=0, eval_test=False):
    learning_rate=lr
    hidden_nodes=256
    output_nodes=10
    iterations=iterations
    # For early stopping of model.
    callbacks=tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    #model
    model = Sequential()
    model.add(Reshape((1024,), input_shape=(32,32,), name='Input_layer'))
    model.add(BatchNormalization())
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(hidden_nodes,activation='relu'))
    
    model.add(Dropout(0.3))
    
    model.add(Dense(hidden_nodes,activation='relu'))
    model.add(Dense(hidden_nodes,activation='relu'))
    
    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
    # adam optmizer with custom learning rate
    adam= optimizers.Adam(lr=learning_rate)
    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    #Fit the model
    model.fit(X_train,y_train, validation_data=(X_val,y_val),epochs=iterations, batch_size=1000, verbose=verb, callbacks=[callbacks])
    
    if eval_test == True:
        scores=[]
        score = model.evaluate(X_train,y_train, verbose=0)
        scores.append(score)
        score = model.evaluate(X_val,y_val, verbose=0)
        scores.append(score)
        score = model.evaluate(X_test,y_test, verbose=0)
        scores.append(score)
        return scores
    else:
        score = model.evaluate(X_val,y_val, verbose=(verb+1)%2)
        return score
iterations = 1
lr=0.0001
Lambda=0
score=model_2(iterations, lr, Lambda)
print(f'\nLoss is {score[0]} and Accuracy is {score[1]}')
iterations = 50
lr=1e-7
Lambda=1e-7
score=model_2(iterations, lr, Lambda)
print(f'Loss is {score[0]} and Accuracy is {score[1]}')
iterations = 50
lr=1e+7
Lambda=1e+7
score=model_2(iterations, lr, Lambda)
print(f'Loss is {score[0]} and Accuracy is {score[1]}')
iterations = 50
lr=1e-4
Lambda=1e-7
score=model_2(iterations, lr, Lambda)
print(f'Loss is {score[0]} and Accuracy is {score[1]}')
iterations = 50
lr=2
Lambda=1e-2
score=model_2(iterations, lr, Lambda)
print(f'Loss is {score[0]} and Accuracy is {score[1]}')
import math
results =[]
for i in range(10):
    lr=math.pow(10, np.random.uniform(-4.0,1.0))
    Lambda = math.pow(10, np.random.uniform(-7,-2))
    iterations = 20
    score=model_2(iterations, lr, Lambda,0)
    result=f'Loss is {score[0]} and Accuracy is {score[1]} with learing rate {lr} and Lambda {Lambda}\n'
    print(result)
    results.append(result)
import math
results =[]
for i in range(10):
    lr=math.pow(10, np.random.uniform(-4.0,-3.0))
    Lambda = math.pow(10, np.random.uniform(-7,-4))
    iterations = 100
    score=model_2(iterations, lr, Lambda,0)
    result=f'Loss is {score[0]} and Accuracy is {score[1]} with learing rate {lr} and Lambda {Lambda}\n'
    print(result)
    results.append([result,[score[0],score[1],lr,Lambda]])
lr= 0.0007
Lambda= 1.31e-05
iterations = 500
eval_test= True
scores = model_2(iterations, lr, Lambda, 1, eval_test)
print(f'Training Dataset Loss is {scores[0][0]} Accuracy is {scores[0][1]}\n')
print(f'Validation Dataset Loss is {scores[1][0]} Accuracy is {scores[1][1]}\n')
print(f'Test Dataset Loss is {scores[2][0]} Accuracy is {scores[2][1]}\n')
def model_3(iterations, lr, Lambda, verb=0,eval_test=False):
    learning_rate=lr
    hidden_nodes=500
    output_nodes=10
    iterations=iterations
    # for early stopping
    callbacks=tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    #model
    model = Sequential()
    model.add(Reshape((1024,), input_shape=(32,32,), name='Input_layer'))
    model.add(BatchNormalization())
    
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(0.5))   
    
    model.add(Dense(200,activation='relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(100,activation='relu'))
    
    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))
    # Custom adam optimizer
    adam= optimizers.Adam(lr=learning_rate)
    #Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    #Fit the model
    model.fit(X_train,y_train, epochs=iterations, batch_size=1000, verbose=verb, callbacks=[callbacks])
    if eval_test == True:
        scores=[]
        score = model.evaluate(X_train,y_train, verbose=0)
        scores.append(score)
        score = model.evaluate(X_val,y_val, verbose=0)
        scores.append(score)
        score = model.evaluate(X_test,y_test, verbose=0)
        scores.append(score)
        return scores
    else:
        score = model.evaluate(X_val,y_val, verbose=(verb+1)%2)
        return score

results =[]
print('Evaluating Validation data...\n')
for i in range(30):
    lr=math.pow(10, np.random.uniform(-4.0,1.0))
    Lambda = math.pow(10, np.random.uniform(-7,4))
    iterations = 20
    score=model_3(iterations, lr, Lambda,0)
    result=f'Loss is {score[0]} and Accuracy is {score[1]} with learing rate {lr} and Lambda {Lambda}\n'
    print(result)
    results.append([result,[score[0],score[1],lr,Lambda]])
for i in range(len(results)):
    if(results[i][1][1]>0.8):
        print(results[i][0])
import math

results =[]
print('Evaluating Validation data...\n')
for i in range(15):
    lr=math.pow(10, np.random.uniform(-4,-3))
    Lambda = math.pow(10, np.random.uniform(-5,-1))
    iterations = 100
    score=model_3(iterations, lr, Lambda,0)
    result=f'Loss is {score[0]} and Accuracy is {score[1]} with learing rate {lr} and Lambda {Lambda}\n'
    print(result)
    results.append([result,[score[0],score[1],lr,Lambda]])
for i in range(len(results)):
    if(results[i][1][1]>0.85):
        print(results[i][0])
lr=0.00055
iterations=500
Lambda = 0.00079
eval_test=True
scores = model_3(iterations, lr, Lambda, 1, eval_test)
print(f'Training Dataset Loss is {scores[0][0]} Accuracy is {scores[0][1]}\n')
print(f'Validation Dataset Loss is {scores[1][0]} Accuracy is {scores[1][1]}\n')
print(f'Test Dataset Loss is {scores[2][0]} Accuracy is {scores[2][1]}\n')