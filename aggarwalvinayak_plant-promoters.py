import csv
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# # from google.colab import files
# uploaded = files.upload()
def csv_to_list(path):
    with open(path) as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[0]
        return data

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
data_1 = csv_to_list('/kaggle/input/promot/stress3a_promoters.csv')
data_0 = csv_to_list('/kaggle/input/promot/stress3b_promoters.csv')
data_n = csv_to_list('/kaggle/input/promot/red_neutral_95.csv')
# data_n = data_n[:6000]
data_0t=list()
print(len(data_n))
for i in data_n:
    if(len(i)>600):
        data_0t.append(i)
data_n=data_0t
print(len(data_n))


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
print(df_net.drop(df_net[cond].index))
    
cond = ((df_net[0].isin(df_up[0]) == True) | (df_net[0].isin(df_down[0]) == True))
df_net.drop(df_net[cond].index, inplace = True)
print(len(df_net))
# cond1 = df_net[0].isin(df_down[0]) == True
# df_net.drop(df_down[cond1].index, inplace = True)
# print(len(df_net))
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
print (len(data_0),len(data_1),len(data_n))
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE,ADASYN
print(imblearn.__version__)
import copy 


def stringToIarr(s):
    arr = []
    for i in s:
        if (i=='a'):
            arr.append(1.0)
        elif (i=='t'):
            arr.append(2.0)
        elif (i=='g'):
            arr.append(4.0)
        else :
            arr.append(8.0)
    return arr
X_smote = []
y_smote = []
for i in data_0:
    X_smote.append(stringToIarr(i.lower()))
    y_smote.append(2)
for i in data_1:
    X_smote.append(stringToIarr(i.lower()))
    y_smote.append(1)
    
for i in X_smote:
    while(len(i)<1000):
        i.append(0.0)    
X_cop=np.array(X_smote)
y_cop=copy.deepcopy(y_smote)

# X_cop = np.array(X_cop)
X_cop.reshape(1,-1)

oversample = SMOTE()
X_cop,y_cop = oversample.fit_resample(X_cop,y_cop)

for i in data_n:
    X_smote.append(stringToIarr(i.lower()))
    y_smote.append(0)
for i in X_smote:
    while(len(i)<1000):
        i.append(0.0)
    
        
X_smote = np.array(X_smote)
X_smote.reshape(1,-1)
X_smote.shape


oversample = SMOTE()
X_smote,y_smote = oversample.fit_resample(X_smote,y_smote)
counter = Counter(y_smote)
counter

# print(classification_report(y_cop,best_clf.predict(X_cop)))
# print(confusion_matrix(y_cop,best_clf.predict(X_cop), labels=[0,1,2]))
# def stringToIarr(s):
#     arr = []
#     for i in s:
#         if (i=='a'):
#             arr.append(1)
#         elif (i=='t'):
#             arr.append(-1)
#         elif (i=='g'):
#             arr.append(2)
#         else :
#             arr.append(-2)
#     return arr
# X_smote = []
# y_smote = []
# for i in data_0:
#     X_smote.append(stringToIarr(i.lower()))
#     y_smote.append(2)
# for i in data_1:
#     X_smote.append(stringToIarr(i.lower()))
#     y_smote.append(1)
# for i in data_n:
#     X_smote.append(stringToIarr(i.lower()))
#     y_smote.append(0)
# for i in X_smote:
#     while(len(i)<1000):
#         i.append(0.0)
    
        
# X_smote = np.array(X_smote)
# X_smote.reshape(1,-1)
# X_smote.shape
from copy import deepcopy
genes = []
labels = []
for i in data_0:#down
    genes.append(i.lower())
    labels.append(2)
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
for str1 in genes:
    encoded_genes.append(encode(str1))
for lst in encoded_genes:
    length = len(lst)
    for i in range (length,1000):
        lst.append([0,0,0,0])
for lst in encoded_genes:
    if(len(lst)<1000):
        print('.')
from sklearn.utils import shuffle

encoded_genes,labels = shuffle(encoded_genes,labels)
labels_single=deepcopy(labels)
labels=pd.get_dummies(labels)
labels=labels.values.tolist()
# labels

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
import tensorflow as tf
import keras.backend as K

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
# import numpy as np
# from sklearn.metrics import f1_score

# # Samples
# y_true = np.array([2,2,0,1,2])
# y_pred = np.array([2,2,0,2,2])

# print('Shape y_true:', y_true.shape)
# print('Shape y_pred:', y_pred.shape)

# # Results
# print('sklearn Macro-F1-Score:', f1_score(y_true, y_pred, average='macro'))
# print('Custom Macro-F1-Score:', K.eval(f1(y_true, y_pred)))
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer,accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from keras.layers import Activation
def create_model( layers,dropout,activ='sigmoid'):
#     layers=100
#     dropout=0.2
#     activ='tanh'    
    model = Sequential()
    # model.add(LSTM(100 ), input_shape=(1000,4))
    model.add(Bidirectional(LSTM(layers),input_shape=(1,1000)))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(3,activation='relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.build((batch_size,1000,4))
#     print(model.summary())
    return model
    
model = KerasClassifier(build_fn=create_model,batch_size=8192,epochs=100,verbose=0
                       )   
# model.summary()
# class_type = "4"
# kmerlen = "5"
from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support

y_train  = np.array(y_train)
X_train = np.array(X_train)
Yy_train  = np.array(labels_single)


X_smote = np.array(X_smote)
y_smote  = np.array(y_smote)


print(X_train.shape)
# classifier = "XGB"
seed = 1
# print("\n" + classifier + " BEGINS TRAINING with seed: "+ str(seed))
# print("\nTotal number of features = %d\nTotal number of training samples = %d\n" % (len(X_train[0]), len(X_train)))
# file_to_save_model = "model_xgb" + suffix_for_data    
# file_to_save_y = "y_of_model_xgb" + suffix_for_data

np.random.seed(seed)
# indices = np.arange(X_train.shape[0])
# np.random.shuffle(indices)
# X_train = X_train[indices]
# X_train = X_train.reshape(len(X_train),4,1000)
# y_train = y_train[indices]
# Yy_train = Yy_train[indices]


indices = np.arange(X_smote.shape[0])
np.random.shuffle(indices)
X_train = X_smote[indices]
Yy_train = y_smote[indices]
X_train = X_train.reshape(len(X_train),1,1000)

X_cop = X_cop.reshape(len(X_cop),1,1000)




# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=seed)

inner_cv = StratifiedKFold(n_splits=4,random_state=seed)
outer_cv = StratifiedKFold(n_splits=7, random_state=seed)

# inner_cv = KFold(n_splits=2,random_state=seed)
# outer_cv = KFold(n_splits=2, random_state=seed)
fold_index=0
accuracy=[]
precision = []
recall = []
fscore = []
best_clf = []
X_train.shape
X_cop.shape
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from time import time

val_acc = 0

for train_index, test_index in outer_cv.split(X_train,Yy_train):
    #Working with cross validation
#     print(train_index,test_index)
    startTrainTime = time()
    X_train_inner,X_test_inner = X_train[train_index], X_train[test_index]
    y_train_inner,y_test_inner = Yy_train[train_index], Yy_train[test_index]

    scorer = make_scorer(f1_score, average = 'macro')
    ############################################################
    # epochs=[10]
    layers=[50,75]
    dropout=[0.05]
#     dropout=[0.15]
    param_grid=dict(layers=layers,dropout=dropout)
    ###########################################################
    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring = scorer, cv=inner_cv)
    grid_fit = grid.fit(X_train_inner, y_train_inner,class_weight={0:0.3, 1:0.35,2:0.35})
#     print(start)
    
    
    
    
    best_clf = grid_fit.best_estimator_
    print('SCORE AVG:',grid_fit.best_score_)
    print("Train fold ",end='')
    print(fold_index,end='')
    print(" done in time %f" % (time()-startTrainTime))
    fold_index +=1
    from sklearn.metrics import classification_report
    print(classification_report(y_test_inner,best_clf.predict(X_test_inner)))
    # y_predict = best_clf.predict(X_test_inner)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test_inner,best_clf.predict(X_test_inner), labels=[0,1,2]))
    print("----------------------------------------------------------------")
    print(classification_report(y_cop,best_clf.predict(X_cop)))
    print(confusion_matrix(y_cop,best_clf.predict(X_cop), labels=[0,1,2]))
    print("________________________--------_____________________________-_")
    
    y_predict = best_clf.predict(X_test_inner)
    y_score = best_clf.predict_proba(X_test_inner) #Used for generating ROC plot

    acc_score = accuracy_score(y_test_inner, y_predict)

    print("Accuracy score:{}\n".format(acc_score*100))
    print(grid_fit.best_params_)
    evaluation_data = precision_recall_fscore_support(y_test_inner, y_predict, average='micro')
    
    print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))

    accuracy.append(acc_score)
    precision.append(evaluation_data[0])
    recall.append(evaluation_data[1])
    fscore.append(evaluation_data[2])
    
    y_val = best_clf.predict(X_test_inner)
    val_acc += np.sum(y_val)
grid_fit.best_params_
from sklearn.metrics import classification_report
print(classification_report(y_test_inner,best_clf.predict(X_test_inner)))
# y_predict = best_clf.predict(X_test_inner)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_inner,best_clf.predict(X_test_inner), labels=[0,1,2])
import statistics
acc_score = statistics.mean(accuracy)

print("Accuracy score:{}\n".format(acc_score*100))

evaluation_data = (statistics.mean(precision)*100,statistics.mean(recall)*100,statistics.mean(fscore)*100)

print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))
    
print(outer_cv.split(X_train, y_train))
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
