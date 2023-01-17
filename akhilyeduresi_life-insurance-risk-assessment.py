# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#loading libraries



import os

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_recall_curve

from sklearn.metrics import recall_score, classification_report, auc, roc_curve

from sklearn.metrics import precision_recall_fscore_support, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)



from keras import losses

from keras.utils import to_categorical

from keras.layers import Input, Dense, Dropout

from keras.models import Model, Sequential 

from keras.optimizers import Adam

from keras import optimizers

from keras import backend as K

from keras.callbacks import Callback

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers





import warnings

warnings.filterwarnings('ignore')



from scipy import stats

import tensorflow as tf

import pickle



from pylab import rcParams
#Loading train data

train=pd.read_csv("../input/train.csv")
#Data

train.head()
#Shape of data

train.shape
#Exploring missing values

train.isnull().sum()[train.isnull().sum() !=0]
#Exploring missing values

train_missing= train.isnull().sum()[train.isnull().sum() !=0]

train_missing=pd.DataFrame(train_missing.reset_index())

train_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)

train_missing['missing_count_percentage']=((train_missing['missing_count'])/59381)*100

plt.figure(figsize=(20,8))

sns.barplot(y=train_missing['features'],x=train_missing['missing_count_percentage'])

train_missing
#checking data types

train.dtypes.unique()
#Outliers detection

train.describe()
#Responce variable

aixs1 = plt.subplots(1,1,figsize=(10,5))

sns.countplot(x='Response',data=train)
#Categorical codes

train['Product_Info_2'] = train['Product_Info_2'].astype('category').cat.codes
# missing values

train_missing
#dropping columns containing missing values more than 80%

train = train.drop(['Medical_History_10','Medical_History_24','Medical_History_32'], axis=1)
#missing values AGAIN

train_missing= train.isnull().sum()[train.isnull().sum() !=0]

train_missing=pd.DataFrame(train_missing.reset_index())

train_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)

train_missing['missing_count_percentage']=((train_missing['missing_count'])/59381)*100

train_missing
#Mean Imputation fro continous variables

Continuos = ['Employment_Info_1','Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5',

                    'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']

train[Continuos] = train[Continuos].fillna(train[Continuos].mean())
#Mode Imputation fro continous variables

Categorical = ['Medical_History_1', 'Medical_History_15']

train[Categorical] = train[Categorical].apply(lambda x:x.fillna(x.value_counts().index[0]))
#Missing values again

train_missing= train.isnull().sum()[train.isnull().sum() !=0]

train_missing=pd.DataFrame(train_missing.reset_index())

train_missing.rename(columns={'index':'features',0:'missing_count'},inplace=True)

train_missing['missing_count_percentage']=((train_missing['missing_count'])/59381)*100

train_missing
#train data

train.head()
#Dataset split

train_data, test_data = train_test_split(train, test_size = 0.15)

print(train_data.shape)

print(test_data.shape)
#traindata

train_data.head()
#traindata

test_data.head()
#Predictor and responce variables

train_x = train_data.drop(['Id', 'Response'], axis=1)

train_y = train_data['Response']

test_x = test_data.drop(['Id', 'Response'], axis=1)

test_y = test_data['Response']

print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)



#train responce

train_y.head()
#test responce

test_y.head()
#converting to responce categorical class labels(0-7)

train_y = train_y-1

train_y = to_categorical(train_y, num_classes= 8)



print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
#Function for normalization

def normalization(data):

    return (data - data.min())/(data.max() - data.min())
#normalizing data

train_x = normalization(train_x)

test_x = normalization(test_x)
#traindata

train_x.head()
#testdata

test_x.head()
#Train and test data shapes

print(train_x.shape)

print(test_x.shape)
#assigning static parameter

nb_epoch = 20

batch_size = 512

input_dim = train_x.shape[1]

hidden_dim1 = 64 

hidden_dim2 = 32

hidden_dim3 = 16

learning_rate = 1e-7
#Function for auto encoder to get and fit model

def get_fit_encoder(xs_train,xs_cv,test_x):

    input_layer = Input(shape=(input_dim, ))

    encoder = Dense(input_dim, activation="relu",activity_regularizer=regularizers.l1(learning_rate))(input_layer)

    

    encoder = Dense(hidden_dim1, activation="relu")(encoder)

    encoder = Dense(hidden_dim2, activation="relu")(encoder)

    encoder = Dense(hidden_dim3, activation="relu", name="encoder")(encoder)

    

    decoder = Dense(hidden_dim3, activation="relu")(encoder)

    decoder = Dense(hidden_dim2, activation='relu')(decoder)

    decoder = Dense(hidden_dim1, activation='relu')(decoder)

    

    decoder = Dense(input_dim, activation='relu')(decoder)

    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    #autoencoder.summary()

    autoencoder.compile(optimizer='adam',

                        loss='binary_crossentropy')

    

    history = autoencoder.fit(x=xs_train, y=xs_train,

                          epochs=nb_epoch,

                          batch_size=batch_size,

                          shuffle=True,

                          validation_data=(xs_cv, xs_cv),

                          verbose=1)

    encoder = Model(autoencoder.input, autoencoder.get_layer('encoder').output)

    x_auto_train= encoder.predict(xs_train)

    x_auto_cv= encoder.predict(xs_cv)

    x_auto_test= encoder.predict(test_x)

    return x_auto_train,x_auto_cv,x_auto_test

    
#Function for Neural network to get and fit model

def get_fit_neuralnetwork(xs_encoder_train,xs_encoder_cv,xs_encoder_test,ys_train,ys_cv):

    classifier = Sequential()

    classifier.add(Dense(output_dim = input_dim , init = 'uniform', activation = 'relu', input_dim = 16))

    classifier.add(Dense(output_dim = 16 , init = 'uniform', activation = 'relu'))

    classifier.add(Dense(output_dim = 8 , init = 'uniform', activation = 'relu'))

    classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))

    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    cp = ModelCheckpoint(filepath="autoencoder_data.h5",

                         save_best_only=True,

                         verbose=0)

    tb = TensorBoard(log_dir='./logs',

                     histogram_freq=0,

                     write_graph=True,

                     write_images=True)

    history = classifier.fit(xs_encoder_train, ys_train,

                             batch_size=batch_size ,

                             epochs=nb_epoch ,

                             shuffle=True,

                             validation_data=(xs_encoder_cv,ys_cv),

                             verbose=1,

                            callbacks=[cp, tb]).history

    y_pred_NN = classifier.predict(xs_encoder_test, batch_size=batch_size, verbose=1)

    y_pred_NN = np.argmax(y_pred_NN,axis = 1) + 1

    return y_pred_NN
#Function for State of art model to get and fit model

def get_fit_SOA_Models(x_sampletrain,y_sampletrain,test_x):

    model1 = RandomForestClassifier()

    

    inside_train_y = np.argmax(y_sampletrain, axis = 1) + 1   

    

    model1.fit(x_sampletrain, inside_train_y)

    

    y_pred1 = model1.predict(test_x) 

    return y_pred1
#function for model evaluation

def model_evaluation (test_y,y_pred_NN,y_pred1):

   

    accuracy_NN = accuracy_score(test_y,y_pred_NN)

    F1_score_NN=f1_score(test_y, y_pred_NN,average='weighted')

    Precision_NN=precision_score(test_y, y_pred_NN,average='weighted')

    Recall_score_NN=recall_score(test_y, y_pred_NN,average='weighted')

    

    accuracy_SOAM1 = accuracy_score(test_y, y_pred1)

    F1_score_SOAM1=f1_score(test_y, y_pred1,average='weighted')

    Precision_SOAM1=precision_score(test_y, y_pred1,average='weighted')

    Recall_score_SOAM1=recall_score(test_y, y_pred1,average='weighted')

    

    

    print("Classification score for NN:", classification_report(test_y,y_pred_NN))

    print("Classification score for SOAM1:", classification_report(test_y, y_pred1))

       

    return accuracy_NN,F1_score_NN,Precision_NN,Recall_score_NN,accuracy_SOAM1,F1_score_SOAM1,Precision_SOAM1,Recall_score_SOAM1

    
#Function to pass sample data to autoencoder and neural network functions

def data_sampling(train_x, train_y, test_x, test_y):

    accuracy_list_NN= []

    F1_score_list_NN=[]

    Precision_list_NN=[]

    Recall_list_NN=[]

    

    accuracy_list_SOAM1= []

    F1_score_list_SOAM1=[]

    Precision_list_SOAM1=[]

    Recall_list_SOAM1=[]

    

    

    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 0.99]:

        print("data sample {}".format(i*100))

        x_sampletrain, _, y_sampletrain, _ = train_test_split(train_x, train_y, stratify= train_y, train_size=i)

        xs_train, xs_cv, ys_train, ys_cv = train_test_split(x_sampletrain, y_sampletrain, stratify=y_sampletrain, train_size=0.9)

        xs_train.shape, xs_cv.shape, ys_train.shape, ys_cv.shape

        xs_encoder_train,xs_encoder_cv,xs_encoder_test=get_fit_encoder(xs_train,xs_cv,test_x)

        y_pred_NN=get_fit_neuralnetwork(xs_encoder_train,xs_encoder_cv,xs_encoder_test,ys_train,ys_cv)

        

        y_pred1=get_fit_SOA_Models(x_sampletrain,y_sampletrain,test_x)

        

        accuracy_NN,F1_score_NN,Precision_NN,Recall_NN,accuracy_SOAM1,F1_score_SOAM1,Precision_SOAM1,Recall_SOAM1=model_evaluation(test_y,y_pred_NN,y_pred1)

        

        

        accuracy_list_NN.append(accuracy_NN)

        F1_score_list_NN.append(F1_score_NN)

        Precision_list_NN.append(Precision_NN)

        Recall_list_NN.append(Recall_NN)

        

        accuracy_list_SOAM1.append(accuracy_SOAM1)

        F1_score_list_SOAM1.append(F1_score_SOAM1)

        Precision_list_SOAM1.append(Precision_SOAM1)

        Recall_list_SOAM1.append(Recall_SOAM1)

        

        

    return accuracy_list_NN,F1_score_list_NN,Precision_list_NN,Recall_list_NN,accuracy_list_SOAM1,F1_score_list_SOAM1,Precision_list_SOAM1,Recall_list_SOAM1

#main code to run all functions to reach objective

accuracy_list_NN,F1_score_list_NN,Precision_list_NN,Recall_list_NN,accuracy_list_SOAM1,F1_score_list_SOAM1,Precision_list_SOAM1,Recall_list_SOAM1=data_sampling(train_x, train_y, test_x, test_y)
#Evalution output for Neural network

accuracy_list_NN
#Evalution output for SOAM network

accuracy_list_SOAM1
#Saving output to a file

with open('Accuracy_NN.txt', 'w') as f:

    print(accuracy_list_NN, file=f)

with open('Accuracy_SOAM.txt', 'w') as f:

    print(accuracy_list_SOAM1, file=f)



with open('F1_score_NN.txt', 'w') as f:

    print(F1_score_list_NN, file=f)

with open('F1_score_SOAM1.txt', 'w') as f:

    print(F1_score_list_SOAM1, file=f)



with open('Precision_Score_NN.txt', 'w') as f:

    print(Precision_list_NN, file=f)

with open('Precision_Score_SOAM1.txt', 'w') as f:

    print(Precision_list_SOAM1, file=f)

    

with open('Recall_Recall_NN.txt', 'w') as f:

    print(Recall_list_NN, file=f)

with open('Recall_Recall_SOAM1.txt', 'w') as f:

    print(Recall_list_SOAM1, file=f)
#output comparitive visualization

from matplotlib.pyplot import figure

plt.figure(figsize=(15, 5))

plt.plot(accuracy_list_NN,label='NN')

plt.plot(accuracy_list_SOAM1,label='SOAM1')

#plt.plot([10,20], accuracy_list_SOAM3,label='SOAM2')

plt.legend(loc='lower right')

plt.xlabel("data fraction")

plt.ylabel("Accuaracy Value")
