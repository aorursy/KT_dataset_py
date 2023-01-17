import numpy as np # linear algebra

import matplotlib.pyplot as plt # display images

import pandas as pd # data processing

from keras.models import Model # neural network generation, training and fitting 
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from keras.optimizers import SGD
def organize_data(X):    
    max_non_null_terms = 0
    num_rows = X.shape[0]

    # Sort each row in descending order.  
    for i in range(0,num_rows):
        temp = np.array(sorted(X[i,:],reverse=True))
        X[i,:] = temp
        # We count the number of non null values in each row and keep the largest.
        non_null_terms = np.sum(X[i,:] > 0)
        if non_null_terms > max_non_null_terms:
            max_non_null_terms = non_null_terms
    
    # After sorting we "cut" the last columns which contains only null terms.
    X = X[:,0:max_non_null_terms]
    
    return X
def normalize_data(X):   
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    X_normalized = np.zeros((num_rows,num_cols), dtype = np.float32)
    for i in range(0,num_rows):
        for j in range(0,num_cols):
            if X[i,j] > 0:
                X_normalized[i,j] = np.log10(X[i,j])
    
    return X_normalized
# load train dataset
train_data = pd.read_csv("../input/train.csv")        
train_val = train_data.values
Y = train_val[:,1].astype(np.float32)
X = train_val[:,2:].astype(np.float32)
# load test dataset
test_data = pd.read_csv("../input/test.csv")        
test_val = test_data.values
X_test = test_val[:,1:].astype(np.float32)
# stack vertically the train and test datasets
X_all = np.vstack((X, X_test))
# organize data
X_all = organize_data(X_all)
num_features = X_all.shape[1]
# normalize data
X_all = normalize_data(X_all)
Y = Y.reshape(Y.shape[0],1)
Y_normalized = normalize_data(Y)
# add new features
num_new_features = 5
X_add = np.zeros((X_all.shape[0],num_features + num_new_features), dtype = np.float32)
for i in range(0,X_all.shape[0]):
    X_add[i,0] = np.mean(X_all[i,X_all[i,:]!=0])
    X_add[i,1] = np.min(X_all[i,X_all[i,:]!=0])
    X_add[i,2] = np.max(X_all[i,:])
    X_add[i,3] = np.sum(X_all[i,:] > 0)
    X_add[i,4] = np.std(X_all[i,:])
X_add[:,5:] = X_all

# update number of features
num_features = num_features + num_new_features
# split this new dataset in train and test datasets to be used in training and prediction stages
X_train_final = X_add[0:X.shape[0],:]
X_test_final = X_add[X.shape[0]:,:]
# Check learning plots to see if the model is overfitting or not, if it is learning or not. 
# Now it is the time we choose the best parameters.
for cycles in [1,5,10]:
    for neurons in [10,50,100,150]:
        # input layer
        visible = Input(shape=(num_features,))
        # first feature extractor
        hid1 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hid2 = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hidpar2 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hid3 = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hidpar3 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        # second feature extractor
        hid1_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hid2_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hidpar2_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
        hid3_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
        hidpar3_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
        # interpretation layer
        feedback_hid = Dense(num_features, kernel_initializer='normal', activation = "relu")
    
        x = visible
        L = []
        LP = []
        for i in range(0,cycles):
            # first path (L = layer)
            L.append(hid1(x))
            L.append(hid2(L[0]))
            L.append(hidpar2(L[1]))
            L.append(hid3(L[2]))
            L.append(hidpar3(L[3]))
            L.append(concatenate([L[3],L[4]]))
            # second path (LP = layer in parallel)
            LP.append(hid1_(x))
            LP.append(hid2_(LP[0]))
            LP.append(hidpar2_(LP[1]))
            LP.append(hid3_(LP[2]))
            LP.append(hidpar3_(LP[3]))
            LP.append(concatenate([LP[3],LP[4]]))
            # merge both paths
            final_merge = concatenate([L[-1],LP[-1]])        
            x = feedback_hid(final_merge)
        
        # prediction output
        master = Dense(neurons, kernel_initializer='normal', activation='tanh')(x)
        output = Dense(1, kernel_initializer='normal', activation='softplus')(master)
        model = Model(inputs=visible, outputs=output)
        
        # compile the network
        sgd = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)

        # fit the model
        history = model.fit(X_train_final, Y_normalized,validation_split=0.33, epochs=150, batch_size=100, verbose=0)
        print('Cycles =',cycles)
        print('Neurons =', neurons)
        # show some information about the predictions
        print('\nLoss of predictions in train dataset:')
        predictions = model.predict(X_train_final)
        # Transform predictions to original format
        predictions = 10**predictions
        print( np.sqrt(1/Y.shape[0])*np.linalg.norm((np.log(predictions+1)-np.log(Y+1)),2) )

        # summarize history for loss
        plt.plot(history.history['loss'][2:])
        plt.plot(history.history['val_loss'][2:])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
# final model
cycles = 5
neurons = 150

# input layer
visible = Input(shape=(num_features,))
# first feature extractor
hid1 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hid2 = Dense(neurons, kernel_initializer='normal', activation = "relu")
hidpar2 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hid3 = Dense(neurons, kernel_initializer='normal', activation = "relu")
hidpar3 = Dense(neurons, kernel_initializer='normal', activation = "tanh")
# second feature extractor
hid1_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
hid2_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hidpar2_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
hid3_ = Dense(neurons, kernel_initializer='normal', activation = "tanh")
hidpar3_ = Dense(neurons, kernel_initializer='normal', activation = "relu")
# interpretation layer
feedback_hid = Dense(num_features, kernel_initializer='normal', activation = "relu")
# master layer
mast = Dense(neurons, kernel_initializer='normal', activation='tanh')
# output layer
out = Dense(1, kernel_initializer='normal', activation='softplus') 
    
x = visible
L = []
LP = []
for i in range(0,cycles):
    # first path (L = layer)
    L.append(hid1(x))
    L.append(hid2(L[0]))
    L.append(hidpar2(L[1]))
    L.append(hid3(L[2]))
    L.append(hidpar3(L[3]))
    L.append(concatenate([L[3],L[4]]))
    # second path (LP = layer in parallel)
    LP.append(hid1_(x))
    LP.append(hid2_(LP[0]))
    LP.append(hidpar2_(LP[1]))
    LP.append(hid3_(LP[2]))
    LP.append(hidpar3_(LP[3]))
    LP.append(concatenate([LP[3],LP[4]]))
    # merge both paths
    final_merge = concatenate([L[-1],LP[-1]])        
    x = feedback_hid(final_merge)
        
# prediction output
master = mast(x)
output = out(master)
model = Model(inputs=visible, outputs=output)
        
# compile the network
sgd = SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)
# fit the model to make predictions over the test dataset
history = model.fit(X_train_final, Y_normalized, epochs=150, batch_size=100, verbose=2)

# since X has more columns than X_test, we fill X_test with more null columns
predictions = model.predict(X_test_final)

# We plot some histogram to visualize the distribution of the predictions and make some comparisons. We expect
#that the histogram of the predictions are similar to the histogram of the outputs in the train dataset.
predictions = 10**predictions

print('Train dataset outputs.')
plt.hist(np.log10(Y),bins=100)
plt.show()

print('Predictions of the train dataset outputs.')
plt.hist(np.log10(predictions),bins=100)
plt.show()
# Save these predictions.
submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = predictions
submission.to_csv('submission.csv', index=False)
print(submission.head())
