import csv

import numpy as np

import pandas as pd
# from google.colab import files

# uploaded = files.upload()
def csv_to_list(path):

    with open(path) as f:

        reader = csv.reader(f)

        data = list(reader)

        data = data[0]

        return data

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
data_1 = csv_to_list('/kaggle/input/promot/stress3a_promoters.csv')

data_0 = csv_to_list('/kaggle/input/promot/stress3b_promoters.csv')

data_n = csv_to_list('/kaggle/input/promot/promoters_neutral.csv')
df_up=pd.DataFrame(data_1)

df_down=pd.DataFrame(data_0)

df_net=pd.DataFrame(data_n)

len(df_net)
# print(df_down[0].nunique())

# print(df_up[0].nunique())

# print(pd.concat([df_up,df_down],ignore_index=True).nunique())

# print(df_net[0].nunique())

df_stre=pd.concat([df_up,df_down],ignore_index=True)

df_stre.info()

cond = (df_net[0].isin(df_stre[0]) == True) 

# print(df_net.drop(df_net[cond].index))

    
cond = ((df_net[0].isin(df_up[0]) == True) | (df_net[0].isin(df_down[0]) == True))

df_net.drop(df_net[cond].index, inplace = True)

print(len(df_net))

# cond1 = df_net[0].isin(df_down[0]) == True

# df_net.drop(df_down[cond1].index, inplace = True)

# print(len(df_net))
# from keras import backend as K

# K.tensorflow_backend._get_available_gpus()
print (len(data_0),len(data_1),len(data_n))
from copy import deepcopy

genes = []

labels = []

for i in data_0:#down

    genes.append(i.lower())

    labels.append(-1)

for i in data_1:#up

    genes.append(i.lower())

    labels.append(1)

for i in data_n:#up

    genes.append(i.lower())

    labels.append(0)

    





def encode(str):

    data = []

    for i in str:

        if (i=='a'):

            data.append([1,0,0,0])

        elif (i=='t'):

            data.append([0,1,0,0])

        elif (i=='g'):

            data.append([0,0,1,0])

        elif (i=='c'):

            data.append([0,0,0,1])

    return (data)
encoded_genes = []

for str in genes:

    encoded_genes.append(encode(str))
for lst in encoded_genes:

    length = len(lst)

    for i in range (length,1000):

        lst.append([0,0,0,0])
from sklearn.utils import shuffle



encoded_genes,labels = shuffle(encoded_genes,labels)

labels_single=deepcopy(labels)
labels=pd.get_dummies(labels)

labels=labels.values.tolist()
# len(X_train)

# len(encoded_genes)


from sklearn.model_selection import train_test_split



# X_train,X_test,y_train,y_test = train_test_split(encoded_genes,labels,test_size=0.15,random_state = 42)

X_train=encoded_genes

# Y_train=labels

y_train=labels

encoded_genes = []

len(X_train)
len(y_train)
batch_size  = 512
from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout,LSTM, Bidirectional

from keras.layers.embeddings import Embedding
def batch_generator(X, Y, batch_size = 32):

    indices = np.arange(len(X)) 

    batch=[]

    while True:

            # it might be a good idea to shuffle your data before each epoch

            np.random.shuffle(indices) 

            for i in indices:

                batch.append(i)

                if len(batch)==batch_size:

                    yield X[batch], Y[batch]

                    batch=[]
from keras.utils import Sequence



class MY_Generator(Sequence):



    def __init__(self, X_train, labels, batch_size):

        self.X_train, self.labels = X_train, labels

        self.batch_size = batch_size



    def __len__(self):

        return int(np.ceil(len(self.X_train) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.X_train[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        return np.array(batch_x), np.array(batch_y)
train_gen = MY_Generator(X_train, y_train, batch_size)
len(train_gen)
# train_generator = batch_generator(np.asarray(X_train), np.asarray(y_train), batch_size = 64)

# model = Sequential()

# # model.add(LSTM(100 ), input_shape=(1000,4))

# model.add(Bidirectional(LSTM(100),input_shape=(1000,4)))

# model.add(Dropout(0.2))

# model.add(Dense(3, activation='sigmoid'))

# # model.build((batch_size,1000,4))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# # model.fit_generator(train_generator,steps_per_epoch = len(X_train)//64,epochs=10)

# model.fit(X = X_train,y = y_train,epochs=1,verbose=True)

# # model.fit_generator(train_gen,steps_per_epoch = len(X_train)//batch_size,epochs=10)





from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import make_scorer,accuracy_score, precision_recall_fscore_support

import tensorflow as tf

def create_model( layers=100,dropout=0.2,activ='tanh'):

#     layers=100

#     dropout=0.2

#     activ='tanh'

    

    

    

    model = Sequential()

    # model.add(LSTM(100 ), input_shape=(1000,4))

    model.add(Bidirectional(LSTM(layers),input_shape=(1000,4)))

    model.add(Dropout(dropout))

    model.add(Dense(3, activation=activ))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.categorical_accuracy])



    

# model.build((batch_size,1000,4))

    

    return model

    

# model = KerasClassifier(build_fn=create_model,batch_size=512,epochs=5)   



    
# class_type = "4"

# kmerlen = "5"

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold

from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support





y_train  = np.array(y_train)

X_train = np.array(X_train)

# X_train=np.reshape(X_train, (37590, -1))

# Yy_train  = np.array(labels_single)



print(X_train.shape)

# classifier = "XGB"

seed = 1

# print("\n" + classifier + " BEGINS TRAINING with seed: "+ str(seed))

# print("\nTotal number of features = %d\nTotal number of training samples = %d\n" % (len(X_train[0]), len(X_train)))





# file_to_save_model = "model_xgb" + suffix_for_data    

# file_to_save_y = "y_of_model_xgb" + suffix_for_data



np.random.seed(seed)

indices = np.arange(X_train.shape[0])

np.random.shuffle(indices)

X_train = X_train[indices]

y_train = y_train[indices]



# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=seed)



# inner_cv = StratifiedKFold(n_splits=10,random_state=seed)

# outer_cv = StratifiedKFold(n_splits=10, random_state=seed)



inner_cv = KFold(n_splits=5,random_state=seed)

outer_cv = KFold(n_splits=5, random_state=seed)





# inner_cv.get_n_splits(X_train, y_train)

# outer_cv.get_n_splits(X_train, y_train)





fold_index=0

accuracy=[]

precision = []

recall = []

fscore = []

best_clf = []



# from google.colab import drive

# drive.mount('/content/drive')
# from keras import backend as K

# K.tensorflow_backend._get_available_gpus()
y_train.shape
model = Sequential()

model.add(Bidirectional(LSTM(25),input_shape=(1000,4)))

model.add(Dropout(0.3))

model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.model_selection import GridSearchCV,StratifiedKFold

from time import time



val_acc = 0



for train_index, test_index in outer_cv.split(X_train,y_train):

    #Working with cross validation

    print(train_index,test_index)

    startTrainTime = time()

    X_train_inner,X_test_inner = X_train[train_index], X_train[test_index]

    y_train_inner,y_test_inner = y_train[train_index], y_train[test_index]



    scorer = make_scorer(accuracy_score)





    train_generator = batch_generator(np.asarray(X_train_inner), np.asarray(y_train_inner), batch_size = 2048)

    

    



    model.fit_generator(train_generator,steps_per_epoch = len(X_train)//2048,epochs=3)

    # model.fit(X = X_train,y = y_train,epochs=1,verbose=True)

    # model.fit_generator(train_gen,steps_per_epoch = len(X_train)//batch_size,epochs=10)



    

    

    print("Train fold ",end='')

    print(fold_index,end='')

    print(" done in time %f" % (time()-startTrainTime))

    fold_index +=1



    y_predict = model.predict(X_test_inner)

    # y_score = best_clf.predict_proba(X_test_inner) #Used for generating ROC plot

    y_predict = (y_predict > 0.5)

    acc_score = accuracy_score(y_test_inner, y_predict)



    print("Accuracy score:{}\n".format(acc_score*100))



    evaluation_data = precision_recall_fscore_support(y_test_inner, y_predict, average='micro')



    print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))



    accuracy.append(acc_score)

    precision.append(evaluation_data[0])

    recall.append(evaluation_data[1])

    fscore.append(evaluation_data[2])



    y_val = model.predict(X_test_inner)

    val_acc += np.sum(y_val)

    print("val_acc:", val_acc)





    

    

    

    

    

    
print("Train fold ",end='')

print(fold_index,end='')

print(" done in time %f" % (time()-startTrainTime))

fold_index +=1



y_predict = model.predict(X_test_inner)

# y_score = best_clf.predict_proba(X_test_inner) #Used for generating ROC plot

y_predict = (y_predict > 0.5)

acc_score = accuracy_score(y_test_inner, y_predict)



print("Accuracy score:{}\n".format(acc_score*100))



evaluation_data = precision_recall_fscore_support(y_test_inner, y_predict, average='micro')



print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))



accuracy.append(acc_score)

precision.append(evaluation_data[0])

recall.append(evaluation_data[1])

fscore.append(evaluation_data[2])



y_val = model.predict(X_test_inner)

val_acc += np.sum(y_val)

print("val_acc:", val_acc)
print(outer_cv.split(X_train, y_train))
# from google.colab import drive

# drive.mount('/content/drive')
print(X_train.shape)

print(y_train.shape)

print(Yy_train.shape)
_,y = train_gen[0]

y
print(model.summary())



# Final evaluation of the modelz

# scores = model.evaluate(X_test, y_test, verbose=0)

# print("Accuracy: %.2f%%" % (scores[1]*100))
print(X_train[0])