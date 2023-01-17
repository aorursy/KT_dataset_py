%%time

# INSTALL RAPIDS FROM KAGGLE DATASET. TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.11.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
# LOAD LIBRARIES

import cudf, cuml

import pandas as pd, numpy as np

from sklearn.model_selection import train_test_split, KFold

from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

import matplotlib.pyplot as plt

print('cuML version',cuml.__version__)
# LOAD TRAINING DATA

train = cudf.read_csv('../input/digit-recognizer/train.csv')

print('train shape =', train.shape )

train.head()
# VISUALIZE DATA

samples = train.iloc[5000:5030,1:].to_pandas().values

plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(samples[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
%%time

# CREATE 20% VALIDATION SET

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:], train.iloc[:,0],\

        test_size=0.2, random_state=42)



# GRID SEARCH FOR OPTIMAL K

accs = []

for k in range(3,22):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # Better to use knn.predict() but cuML v0.11.0 has bug

    # y_hat = knn.predict(X_test)

    y_hat_p = knn.predict_proba(X_test)

    acc = (y_hat_p.to_pandas().values.argmax(axis=1)==y_test.to_array() ).sum()/y_test.shape[0]

    #print(k,acc)

    print(k,', ',end='')

    accs.append(acc)
%%time

# COMPUTE NEIGHBORS

row = 5; col = 7; sft = 10

knn = NearestNeighbors(n_neighbors=col)

knn.fit(X_train)

distances, indicies = knn.kneighbors(X_test)

# DISPLAY NEIGHBORS

displayV = X_test.to_pandas().iloc[sft:row+sft].values

displayT = X_train.to_pandas().iloc[indicies[sft:row+sft].to_pandas().values.flatten()].values

plt.figure(figsize=(15,row*1.5))

for i in range(row):

    plt.subplot(row,col+1,(col+1)*i+1)

    plt.imshow(displayV[i].reshape((28,28)),cmap=plt.cm.binary)

    if i==0: plt.title('Unknown\nDigit')

    for j in range(col):

        plt.subplot(row, col+1, (col+1)*i+j+2)

        plt.imshow(displayT[col*i+j].reshape((28,28)),cmap=plt.cm.binary)

        if i==0: plt.title('Known\nNeighbor '+str(j+1))

        plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
# PLOT GRID SEARCH RESULTS

plt.figure(figsize=(15,5))

plt.plot(range(3,22),accs)

plt.title('MNIST kNN k value versus validation acc')

plt.show()
%%time

# GRID SEARCH USING CROSS VALIDATION

for k in range(3,6):

    print('k =',k)

    oof = np.zeros(len(train))

    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (idxT, idxV) in enumerate( skf.split(train.iloc[:,1:], train.iloc[:,0]) ):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(train.iloc[idxT,1:], train.iloc[idxT,0])

        # Better to use knn.predict() but cuML v0.11.0 has bug

        # y_hat = knn.predict(train.iloc[idxV,1:])

        y_hat_p = knn.predict_proba(train.iloc[idxV,1:])

        oof[idxV] =  y_hat_p.to_pandas().values.argmax(axis=1)

        acc = ( oof[idxV]==train.iloc[idxV,0].to_array() ).sum()/len(idxV)

        print(' fold =',i,'acc =',acc)

    acc = ( oof==train.iloc[:,0].to_array() ).sum()/len(train)

    print(' OOF with k =',k,'ACC =',acc)
# LOAD TEST DATA

test = cudf.read_csv('../input/digit-recognizer/test.csv')

print('test shape =', test.shape )

test.head()
# FIT KNN MODEL

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(train.iloc[:,1:785], train.iloc[:,0])
%%time

# PREDICT TEST DATA

# Better to use knn.predict() but cuML v0.11.0 has bug

# y_hat = knn.predict(test)

y_hat_p = knn.predict_proba(test)

y_hat = y_hat_p.to_pandas().values.argmax(axis=1)
# SAVE PREDICTIONS TO CSV

sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sub.Label = y_hat

sub.to_csv('submission_cuML.csv',index=False)

sub.head()
# PLOT PREDICTION HISTOGRAM

plt.hist(sub.Label)

plt.title('Distribution of test predictions')

plt.show()
# TRAIN SKLEARN KNN MODEL

from sklearn.neighbors import KNeighborsClassifier as K2

knn = K2(n_neighbors=3,n_jobs=2)

knn.fit(train.iloc[:,1:].to_pandas(), train.iloc[:,0].to_pandas())
%%time

# PREDICT 1/28 OF ALL TEST IMAGES WITH CPU

y_hat = knn.predict(test.iloc[:1000,:].to_pandas())

print('Here we only infer 1000 out of 28,000 test images on CPU')