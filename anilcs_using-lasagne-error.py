from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_data():
    print('Loading data...')
    # load train data    
    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    
    remove = []
    for col in df_train.columns:
        if df_train[col].std() == 0:
            remove.append(col)

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)

    remove = []
    c = df_train.columns
    for i in range(len(c)-1):
        v = df_train[c[i]].values
        for j in range(i+1,len(c)):
            if np.array_equal(v,df_train[c[j]].values):
                remove.append(c[j])

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)

    target = df_train['TARGET'].values
    df_train =df_train.drop(['TARGET'],axis=1)
    id_test = df_test['ID'].values

    categoricalVariables = []
    for var in df_train.columns:
        vector=pd.concat([df_train[var],df_test[var]], axis=0)
        typ=str(df_train[var].dtype)
        if (typ=='object'):
            categoricalVariables.append(var)
    
    for col in categoricalVariables:
        df_train[col] = pd.factorize(df_train[col])[0] 
        df_test[col] = pd.factorize(df_test[col])[0]   
                                  
    list_train=df_train.columns.tolist()
    list_test =df_test.columns.tolist()
    
    #Remove sparse columns
    sparse_col=[]
    for col in df_train.columns:
        cls=df_train[col].values
        if sum(cls)<30:
            sparse_col.append(col) 
           
    df_train.drop(sparse_col, axis=1,inplace=True)
    df_test.drop(sparse_col, axis=1,inplace=True)

    df_train=df_train.fillna(-1)        
    df_test=df_test.fillna(-1)   
        
    feature_names=df_train.columns.values.tolist()

    X_train =df_train.values
    X_test = df_test.values

    scaler=StandardScaler()    
    X_train = scaler.fit_transform(X_train)    
    X_test=scaler.fit_transform(X_test)    
    X_fit, X_eval, y_fit, y_eval= train_test_split(X_train, target, test_size=0.3,random_state=123)   
    num_features=len(feature_names) 
    
    return X_fit, X_eval, y_fit, y_eval,X_test,id_test,num_features
X_fit, X_eval, y_fit, y_eval,X_test,id_test,num_features=load_data()
X_fit.shape
## create layer of nn
layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('output', DenseLayer)]
from lasagne.nonlinearities import sigmoid
#lasagne.updates.adagrad
from lasagne.updates import adagrad
net1 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=512,
                 dropout0_p=0.1,
                 dense1_num_units=256,
                 dropout1_p=0.1,
                 output_nonlinearity=sigmoid,
                 update=adagrad,
                 update_learning_rate=0.04,
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=15)
net1.fit(X_fit, y_fit)
