import csv

import numpy as np

import pandas as pd

from sklearn.metrics import f1_score

from sklearn.utils import shuffle

from collections import Counter

import imblearn

from imblearn.over_sampling import SMOTE,ADASYN

import copy 

from copy import deepcopy

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout,LSTM, Bidirectional,Activation

from keras.layers.embeddings import Embedding

from keras.utils import Sequence

from keras import backend as K

from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf

from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold

from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support,classification_report

from time import time

import statistics

import requests

import random


def csv_to_list(path):

    with open(path) as f:

        reader = csv.reader(f)

        data = list(reader)

        data = data[0]

        return data
path_1 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fstress1a_promoters.csv?alt=media&token=fad4db1b-a7cf-4231-9d27-138e99e29534"

path_2 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fstress2a_promoters.csv?alt=media&token=32bd8a8e-0bdd-4f03-8a9b-c14328f77155"

path_3 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fstress3a_promoters.csv?alt=media&token=12b2d734-3497-413a-a3e3-a9ca871519b0"

path_4 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fstress1b_promoters.csv?alt=media&token=7a8ac0ac-8ffa-4603-a31a-aa72579e7b23"

path_5 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fstress2b_promoters.csv?alt=media&token=ae3e1415-b4e2-47bc-a01b-494291e19914"

path_6 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fstress3b_promoters.csv?alt=media&token=e73575a3-4407-4b08-921c-743ff287724c"

path_7 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fred_neutral_95.csv?alt=media&token=dd117ed8-ef59-49f1-ad72-15657308c2f3"

path_8 = "https://firebasestorage.googleapis.com/v0/b/amazekart-bits.appspot.com/o/Promot%2Fpromoters_neutral.csv?alt=media&token=0556e29c-029c-40b9-9397-579af98a7188"



path1 = requests.get(path_1)

path2 = requests.get(path_2)

path3 = requests.get(path_3)

path4 = requests.get(path_4)

path5 = requests.get(path_5)

path6 = requests.get(path_6)

path7 = requests.get(path_7)

path8 = requests.get(path_8)



open('stress1a_promoters.csv', 'wb').write(path1.content)

open('stress2a_promoters.csv', 'wb').write(path2.content)

open('stress3a_promoters.csv', 'wb').write(path3.content)

open('stress1b_promoters.csv', 'wb').write(path4.content)

open('stress2b_promoters.csv', 'wb').write(path5.content)

open('stress3b_promoters.csv', 'wb').write(path6.content)

open('red_neutral_95.csv', 'wb').write(path7.content)

open('promoters_neutral.csv', 'wb').write(path8.content)
data_1_3a = csv_to_list('stress3a_promoters.csv')

data_0_3b = csv_to_list('stress3b_promoters.csv')

# data_1_2a = csv_to_list('stress1a_genes.csv')

# data_0_2b = csv_to_list('stress1b_genes.csv')

# data_1_1a = csv_to_list('stress2a_genes.csv')

# data_0_1b = csv_to_list('stress2b_genes.csv')

data_n = csv_to_list('promoters_neutral.csv')
# data_0t=list()

# print(len(data_n))

# for i in data_n:

#     if(len(i)>600):

#         data_0t.append(i)

# data_n=data_0t

# print(len(data_n))
df_up_3a=pd.DataFrame(data_1_3a)

df_down_3b=pd.DataFrame(data_0_3b)

# df_up_2a=pd.DataFrame(data_1_2a)

# df_down_2b=pd.DataFrame(data_0_2b)

# df_up_1a=pd.DataFrame(data_1_1a)

# df_down_1b=pd.DataFrame(data_0_1b)

df_net=pd.DataFrame(data_n)

len(df_net)
df_stre=pd.concat([df_up_3a,df_down_3b],ignore_index=True)

df_stre.info()

cond = (df_net[0].isin(df_stre[0]) == True) 

print(df_net.drop(df_net[cond].index))

    
# cond = ((df_net[0].isin(df_up[0]) == True) | (df_net[0].isin(df_down[0]) == True))

# df_net.drop(df_net[cond].index, inplace = True)

# print(len(df_net))
# def stringToIarr(s):

#     arr = []

#     for i in s:

#         if (i=='a'):

#             arr.append(1.0)

#         elif (i=='t'):

#             arr.append(2.0)

#         elif (i=='g'):

#             arr.append(4.0)

#         else :

#             arr.append(8.0)

#     return arr

def stringToIarr(str1):

    data = []

    for i in str1:

        if (i=='a'):

            data.append([1,0,0,0])

        elif (i=='t'):

            data.append([0,1,0,0])

        elif (i=='g'):

            data.append([0,0,1,0])

        elif (i=='c'):

            data.append([0,0,0,1])

    return (data)



def convert_data_to_smote(data_0,data_1,data_n):

    X_smote = []

    y_smote = []

    for i in data_0:

        X_smote.append(stringToIarr(i.lower()))

        y_smote.append(2)

    for i in data_1:

        X_smote.append(stringToIarr(i.lower()))

        y_smote.append(1)

    random.shuffle(data_n)

    for i in range(len(X_smote)//2):

        X_smote.append(stringToIarr(data_n[i].lower()))

        y_smote.append(0)

    

    maxx=0

#     for i in X_smote:

#         maxx=max(maxx,len(i))

    for i in X_smote:

        while(len(i)<1000):

            i.append([0,0,0,0])

            

    X_cop=np.array(X_smote)

    y_cop=copy.deepcopy(y_smote)



    # X_cop = np.array(X_cop)

#     X_cop.reshape(1,-1)



#     oversample = SMOTE()

#     X_cop,y_cop = oversample.fit_resample(X_cop,y_cop)

#     counter = Counter(y_cop)

#     print(counter)

    

#     for i in X_smote:

#         while(len(i)<1000):

#             i.append(0.0)



#     print("####",X_smote.shape)

    X_smote = np.array(X_smote)

#     X_smote.reshape(1,-1)

    print("####",X_smote.shape)







#     oversample = SMOTE()

#     X_smote,y_smote = oversample.fit_resample(X_smote,y_smote)

#     counter = Counter(y_smote)

#     counter

    return (X_smote,y_smote,X_cop,y_cop)



X_smote,y_smote,x,y = convert_data_to_smote(data_0_3b,data_1_3a,data_n)

# X_smote_3,y_smote_3,x_test3,y_test3 = convert_data_to_smote(data_0_3b,data_1_3a,data_n)

# X_smote_2,y_smote_2,x_test2,y_test2 = convert_data_to_smote(data_0_2b,data_1_2a,data_n)

# X_smote_1,y_smote_1,x_test1,y_test1 = convert_data_to_smote(data_0_1b,data_1_1a,data_n)
# counter = Counter(y_smote_3)

# print(counter)

# counter = Counter(y_smote_2)

# print(counter)

# counter = Counter(y_smote_1)

# print(counter)
genes = []

labels = []

for i in data_0_3b:#down

    genes.append(i.lower())

    labels.append(2)

for i in data_1_3a:#up

    genes.append(i.lower())

    labels.append(1)

for i in data_n:#neutral

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
encoded_genes,labels = shuffle(encoded_genes,labels)

labels_single=deepcopy(labels)

labels=pd.get_dummies(labels)

labels=labels.values.tolist()
from sklearn.model_selection import train_test_split



# X_train,X_test,y_train,y_test = train_test_split(encoded_genes,labels,test_size=0.15,random_state = 42)

X_train=encoded_genes

# Y_train=labels

y_train=labels

encoded_genes = []
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
batch_size = 256

train_gen = MY_Generator(X_train, y_train, batch_size)

len(train_gen)
def create_model( layers,dropout,activ='sigmoid'): 

    model = Sequential()

    # model.add(LSTM(100 ), input_shape=(1000,4))

    model.add(Bidirectional(LSTM(layers),input_shape=(1000,4)))

    model.add(Dropout(dropout))

    model.add(Dense(512, activation='sigmoid'))

    model.add(Dense(3,activation='relu'))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.build((batch_size,1000,4))

#     print(model.summary())

    return model

    

model = KerasClassifier(build_fn=create_model,batch_size=512,epochs=20,verbose=0)  
# y_train  = np.array(y_train)

# X_train = np.array(X_train)

# Yy_train  = np.array(labels_single)





# X_smote = np.array(X_smote)

# y_smote  = np.array(y_smote)



# print(X_train.shape)

# seed = 1



# np.random.seed(seed)



# indices = np.arange(X_smote.shape[0])

# np.random.shuffle(indices)

# X_train = X_smote[indices]

# Yy_train = y_smote[indices]

# X_train = X_train.reshape(len(X_train),1,1000)



# X_cop = X_cop.reshape(len(X_cop),1,1000)



# inner_cv = StratifiedKFold(n_splits=2,random_state=seed)

# outer_cv = StratifiedKFold(n_splits=2, random_state=seed)



# # inner_cv = KFold(n_splits=2,random_state=seed)

# # outer_cv = KFold(n_splits=2, random_state=seed)

# fold_index=0

# accuracy=[]

# precision = []

# recall = []

# fscore = []

# best_clf = []


# val_acc = 0



# for train_index, test_index in outer_cv.split(X_train,Yy_train):

#     #Working with cross validation

# #     print(train_index,test_index)

#     startTrainTime = time()

#     X_train_inner,X_test_inner = X_train[train_index], X_train[test_index]

#     y_train_inner,y_test_inner = Yy_train[train_index], Yy_train[test_index]



#     scorer = make_scorer(f1_score, average = 'macro')

#     ############################################################

#     # epochs=[10]

#     layers=[50,75]

#     dropout=[0.05]

# #     dropout=[0.15]

#     param_grid=dict(layers=layers,dropout=dropout)

#     ###########################################################

#     grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring = scorer, cv=inner_cv)

#     grid_fit = grid.fit(X_train_inner, y_train_inner,class_weight={0:0.3, 1:0.35,2:0.35})

# #     print(start)  

    

#     best_clf = grid_fit.best_estimator_

#     print('SCORE AVG:',grid_fit.best_score_)

#     print("Train fold ",end='')

#     print(fold_index,end='')

#     print(" done in time %f" % (time()-startTrainTime))

#     fold_index +=1

#     from sklearn.metrics import classification_report

#     print(classification_report(y_test_inner,best_clf.predict(X_test_inner)))

#     # y_predict = best_clf.predict(X_test_inner)

#     from sklearn.metrics import confusion_matrix

#     print(confusion_matrix(y_test_inner,best_clf.predict(X_test_inner), labels=[0,1,2]))

#     print("----------------------------------------------------------------")

#     print(classification_report(y_cop,best_clf.predict(X_cop)))

#     print(confusion_matrix(y_cop,best_clf.predict(X_cop), labels=[0,1,2]))

#     print("________________________--------_____________________________-_")

    

#     y_predict = best_clf.predict(X_test_inner)

#     y_score = best_clf.predict_proba(X_test_inner) #Used for generating ROC plot



#     acc_score = accuracy_score(y_test_inner, y_predict)



#     print("Accuracy score:{}\n".format(acc_score*100))

#     print(grid_fit.best_params_)

#     evaluation_data = precision_recall_fscore_support(y_test_inner, y_predict, average='micro')

    

#     print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))



#     accuracy.append(acc_score)

#     precision.append(evaluation_data[0])

#     recall.append(evaluation_data[1])

#     fscore.append(evaluation_data[2])

    

#     y_val = best_clf.predict(X_test_inner)

#     val_acc += np.sum(y_val)
# grid_fit.best_params_
# print(classification_report(y_test_inner,best_clf.predict(X_test_inner)))

# # y_predict = best_clf.predict(X_test_inner)

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_test_inner,best_clf.predict(X_test_inner), labels=[0,1,2])

# acc_score = statistics.mean(accuracy)



# print("Accuracy score:{}\n".format(acc_score*100))



# evaluation_data = (statistics.mean(precision)*100,statistics.mean(recall)*100,statistics.mean(fscore)*100)



# print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))

    
def modelSimple(trainX,trainY,testX,testY):

    X_smote = np.array(trainX)

    y_smote  = np.array(trainY)

    seed = 1

    np.random.seed(seed)

    indices = np.arange(X_smote.shape[0])

    np.random.shuffle(indices)

    X_train = X_smote[indices]

    Yy_train = y_smote[indices]

#     X_train = X_train.reshape(len(X_train),1,1000)

#     testX = testX.reshape(len(testX),1,1000)

    inner_cv = StratifiedKFold(n_splits=5,random_state=seed)



    fold_index=0

    accuracy=[]

    precision = []

    recall = []

    fscore = []

    best_clf = []   

    val_acc = 0



    startTrainTime = time()

    X_train_inner = X_train

    X_test_inner = testX

    y_train_inner = Yy_train 

    y_test_inner = testY



    scorer = make_scorer(f1_score, average = 'macro')

    layers=[27,53]

    # layers = [10]

#     dropout=[0.05,0.15]

    dropout = [0.15]

    param_grid=dict(layers=layers,dropout=dropout)

    grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring = scorer, cv=inner_cv)

    grid_fit = grid.fit(X_train_inner, y_train_inner,class_weight={0:0.3, 1:0.35,2:0.35})



    best_clf = grid_fit.best_estimator_

    print('SCORE AVG:',grid_fit.best_score_)

     

    print(classification_report(y_test_inner,best_clf.predict(X_test_inner)))

    # y_predict = best_clf.predict(X_test_inner)

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(y_test_inner,best_clf.predict(X_test_inner), labels=[0,1,2]))

    print("----------------------------------------------------------------")

    print(classification_report(testY,best_clf.predict(testX)))

    print(confusion_matrix(testY,best_clf.predict(testX), labels=[0,1,2]))

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

    return best_clf


def modelCrossNested(trainX,trainY):

    trainX = np.array(trainX)

    trainY  = np.array(trainY)

    print(trainX.shape)

    seed = 1

    np.random.seed(seed)

    indices = np.arange(X_smote.shape[0])

    np.random.shuffle(indices)

    X_train = trainX[indices]

    Yy_train = trainY[indices]

    X_train = X_train.reshape(len(X_train),1,1000)

#     testX = testX.reshape(len(testX),1,1000)



#     X_cop = X_cop.reshape(len(X_cop),1,1000)



    inner_cv = StratifiedKFold(n_splits=5,random_state=seed)

    outer_cv = StratifiedKFold(n_splits=4, random_state=seed)

    fold_index=0

    accuracy=[]

    precision = []

    recall = []

    fscore = []

    best_clf = []

    ############################################################################################################################

    

    val_acc = 0



    for train_index, test_index in outer_cv.split(X_train,Yy_train):



        startTrainTime = time()

        X_train_inner,X_test_inner = X_train[train_index], X_train[test_index]

        y_train_inner,y_test_inner = Yy_train[train_index], Yy_train[test_index]



        scorer = make_scorer(f1_score, average = 'macro')

        layers=[10,50]

        # layers = [10]

        dropout=[0.05,0.15]

#         dropout = [0.15]

        param_grid=dict(layers=layers,dropout=dropout)

        grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring = scorer, cv=inner_cv)

        grid_fit = grid.fit(X_train_inner, y_train_inner,class_weight={0:0.3, 1:0.35,2:0.35})



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

#         print("----------------------------------------------------------------")

#         print(classification_report(testY,best_clf.predict(testX)))

#         print(confusion_matrix(testY,best_clf.predict(testX), labels=[0,1,2]))

#         print("________________________--------_____________________________-_")



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



#     print(classification_report(y_test_inner,best_clf.predict(X_test_inner)))

#     # y_predict = best_clf.predict(X_test_inner)

#     confusion_matrix(y_test_inner,best_clf.predict(X_test_inner), labels=[0,1,2])

    acc_score = statistics.mean(accuracy)

    print("Accuracy score:{}\n".format(acc_score*100))

    evaluation_data = (statistics.mean(precision)*100,statistics.mean(recall)*100,statistics.mean(fscore)*100)

    print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))

    return evaluation_dat
def test(X,Y,best_clf):

#     X = X.reshape(len(X),1,1000)

    

    from sklearn.metrics import classification_report

    print(classification_report(Y,best_clf.predict(X)))

    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(Y,best_clf.predict(X), labels=[0,1,2]))



    y_predict = best_clf.predict(X)

    y_score = best_clf.predict_proba(X) #Used for generating ROC plot



    acc_score = accuracy_score(Y, y_predict)



    print("Accuracy score:{}\n".format(acc_score*100))

    evaluation_data = precision_recall_fscore_support(Y, y_predict, average='micro')



    print("Precision, Recall and Fscore are: {}\n".format(evaluation_data))
# output = modelCrossNested(X_smote_3,y_smote_3)
X_smote.shape
arr=[]

for i in range(5):

    X_smote,y_smote,x,y = convert_data_to_smote(data_0_3b,data_1_3a,data_n)

    model_train3 = modelSimple(X_smote,y_smote,x,y)    

    arr.append(test(x,y,model_train3))

arr
# test(x_test2,y_test2,model_train3)
# test(x_test1,y_test1,model_train3)
# model_train1 = modelSimple(X_smote_1,y_smote_1,x_test1,y_test1)
# test(x_test3,y_test3,model_train1)
# test(x_test2,y_test2,model_train1)
# test(x_test1,y_test1,model_train1)
# model_train2 = modelSimple(X_smote_2,y_smote_2,x_test2,y_test2)
# test(X_smote_3,y_smote_3,model_train2)
# test(X_smote_2,y_smote_2,model_train2)
# test(X_smote_1,y_smote_1,model_train2)
# output1 = modelCrossNested(X_smote_1,y_smote_1)
# output2 = modelCrossNested(X_smote_2,y_smote_2)
# output3 = modelCrossNested(X_smote_3,y_smote_3)