
import pandas as pd
import numpy as np
import zipfile
import random

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder  
#Used for category encoding. Will use embedding layers in Keras instead of doing one hot encoding

from sklearn import metrics #Using the AUC from sklear at the end of each epoch


from keras.layers import (Reshape, Lambda, Dropout, Input, Embedding, Dense, Conv1D, Conv2D, 
                          concatenate, Flatten)
                         
from keras.models import Model, load_model

from keras import regularizers, backend, optimizers

from keras.backend import expand_dims, relu

from keras import backend as K
import tensorflow as tf

#These two are needed to desplay the network
from keras.utils.vis_utils import model_to_dot, plot_model
from IPython.display import Image

import h5py

import pickle

#Memory cleanup after last use of large datasets 
import gc

import matplotlib.pyplot as plt


#For the first run should be set to true. The program will save the pre processed data in a few h5py and pickle files. 
#set it to False for all subsequent executions. The data will be loaded from the preprocessed files. 
#I am not sure the generated files persist between runs of the notebook in Kaggle. I just left it in case you download and run locally.
p_rebuild_datasets = True

#Setting to values other than zero allows to test with smaller subset of data. For testing purposes it can be set for 50k
p_subsample_lines = 0

k_datasets_prefix = "../input/"    

train_data_file = "application_train.csv"
predict_data_file = "application_test.csv"
bureau_data_file = "bureau.csv"
bureau_balance_data_file = "bureau_balance.csv"
pos_cash_balance_data_file = "POS_CASH_balance.csv"
credit_card_balance_data_file = "credit_card_balance.csv"
previous_application_data_file = "previous_application.csv"
installments_payments_data_file = "installments_payments.csv"

#This string will be used to identify auxiliary files.
p_data_model_version = "KerasNN_PAConv_SimpleV5"

p_objective = "TARGET"
p_index = "SK_ID_CURR"
def preproc_ds(df1, df2,combined=True,keys=None):
    if combined:
        df = df1.append(df2, ignore_index=True)[data.columns.tolist()]
        categoricals = df.columns[df.dtypes=="O"]
        for col in df.columns[df.dtypes!="O"]:
            if col != "TARGET" and not col in keys:
                mean = df[col].mean()
                std = df[col].std()
                df1[col] = (df1[col].fillna(mean) - mean) / std
                df2[col] = (df2[col].fillna(mean) - mean) / std
                
        print("combined")
        for col in df.columns[df.dtypes=="O"]:
            print(col)
            le = LabelEncoder()
            le.fit(df[col].fillna('PreProcEmpty'))
            print(le.classes_)
            df1[col] = le.transform(df1[col].fillna('PreProcEmpty'))
            df2[col] = le.transform(df2[col].fillna('PreProcEmpty'))

        return(df1, df2, categoricals)
    else:
        print("single")
        categoricals = df1.columns[df1.dtypes=="O"]
        for col in df1.columns[df1.dtypes!="O"]:
            if col != "TARGET" and not col in keys:
                mean = df1[col].mean()
                std = df1[col].std()
                df1[col] = (df1[col].fillna(mean) - mean) / std
                
        for col in df1.columns[df1.dtypes=="O"]:
            print(col)
            le = LabelEncoder()
            le.fit(df1[col].fillna('PreProcEmpty'))
            print(le.classes_)
            df1[col] = le.transform(df1[col].fillna('PreProcEmpty'))
        return(df1,categoricals)
    


def read_df(filename,subdir,index): 
    df = pd.read_csv(subdir + filename).sort_values(by=index)
    if index != None:
        df.set_index(index, inplace=False)
    return(df)
    

def generate_conv_tensor_simple(X,X_index, products,PA_cp):

    PA = PA_cp[PA_cp[p_index].isin(X[p_index])]
    st_products = 0
    samples = len(X)
    PA_length = len(PA)
    max_len_detected = 0
    max_prod_detected = 0
    nPA_in_vars = len(PA.iloc[0])
    PA_nX_tensor = np.zeros((samples,products,nPA_in_vars))
    X_sample=0                                                 
    x_progress_tic = samples/100
    x_progress = 0
    chunks_PA = (1,products,nPA_in_vars-2)
    PA_iloc = 0
    for X_iloc in range(samples):
        X_row = X.iloc[X_iloc,:]
        PA_row = PA.iloc[PA_iloc,:]
        product_n = 0        
        while PA_row[X_index] == X_row[X_index]:         
            if (product_n < products):
                PA_nX_tensor[X_iloc,product_n,:]= PA_row           
                st_products = st_products + 1 
            product_n = product_n + 1
            PA_iloc = PA_iloc + 1
            if PA_iloc >= PA_length: break
            PA_row = PA.iloc[PA_iloc,:]
        x_progress = x_progress + 1
        if x_progress > x_progress_tic:
            x_progress_tic = x_progress_tic + samples/100
            prog = "\r" + str(x_progress/samples*100) 
            print(prog, end="", flush=True)
            
    print("Products ",st_products)
    return(PA_nX_tensor[:,:,2:],chunks_PA)   

def pickledump(filename,dumpobj):    
    outfile = open(filename,'wb')
    pickle.dump(dumpobj,outfile, protocol=4)
    outfile.close()

def pickleload(filename):    
    print("Cargando pickle : ", filename)
    infile = open(filename,'rb')
    dumpobj = pickle.load(infile)
    infile.close()
    return(dumpobj)

def h5dump(dataset,array,hp5yf,chunks):
    return(hp5yf.create_dataset(dataset, data=array, compression="lzf",chunks=chunks))

def build_dump_tensor(X,p_index,p_products,h5py_dataset,PA,h5pyfilename):
    print("Saving :", h5py_dataset)
    ts_pa,chunks_pa = generate_conv_tensor_simple(X,p_index, p_products,PA)
    h5pyf = h5py.File(h5pyfilename, "a", driver = "core")
    ts_pa = h5dump(h5py_dataset, ts_pa,h5pyf,chunks_pa)
    h5pyf.flush()
    h5pyf.close()
    del h5pyf, ts_pa
    gc.collect()
    return()

#The number of previous applications/bureau reports by sample is limited to 24
p_products = 24

h5py_file_filename = "h5py_"+ p_data_model_version + ".hdf5" #to store train/test/unclassified main input
h5py_file_bd_filename = "h5py_"+ p_data_model_version + "_bd.hdf5" #to store train/test/unclassified bureau data
h5py_file_pa_filename = "h5py_"+ p_data_model_version + "_pa.hdf5"  #to store train/test/unclassified previous applications

if p_rebuild_datasets == True:  #Generate all pre-processed datasets
    

##### MAIN DATA
# Read train/unclassified main data
    data = read_df(train_data_file,k_datasets_prefix,p_index)
    data_unclassified = read_df(predict_data_file,k_datasets_prefix,p_index)

# Normalize + categoricals preprocess for main data
    data, data_unclassified, data_categs = preproc_ds(data, data_unclassified,True,[p_index,"TARGET"])

# This sorting is NOT mandatory since it will be resorted after train/test split
    data = data.sort_values(by=[p_index], ascending = True)

# This sorting is important for the tensor generation to work correctly      
    data_unclassified = data_unclassified.sort_values(by=[p_index], ascending = True)
    
# Save ID sorted list to match the predicted probabilities in the end
    data_unclassified["SK_ID_CURR"].to_csv("./" + p_data_model_version + "unclassified_ids.csv", header=True, index=None, sep=',', mode='w')
    
# Columns are re arranged so that categorical data is in the end. 
# Solving embeddings in the NN this will be simpler in this way
#    data = pd.concat([data.drop(data_categs,axis=1),data[data_categs]], axis=1)
#    data_unclassified = pd.concat([data_unclassified.drop(data_categs,axis=1),data_unclassified[data_categs]], axis=1)    
    data = data.reindex(columns=data.drop(data_categs,axis=1).columns.tolist()+data_categs.tolist())    
    data_unclassified = data_unclassified.reindex(columns=data_unclassified.drop(data_categs,axis=1).columns.tolist()+data_categs.tolist())  
#    Thanks GuangshuoChen @Kaggle
    
##### BUREAU DATA    
# Read + Normaulize + Pre-Process Bureau Data    
    X_bd, bd_categs = preproc_ds(read_df(bureau_data_file,k_datasets_prefix,[p_index,"SK_ID_BUREAU"]),None,False,[p_index,"SK_ID_BUREAU"])

# This sorting is important for the tensor generation to work correctly   
    X_bd = X_bd.sort_values(by=[p_index, 'SK_ID_BUREAU','DAYS_CREDIT' ], ascending=[1,1,1])
    
# Columns are re arranged so that categorical data is in the end. 
# Solving embeddings in the NN this will be simpler in this way
#    X_bd = pd.concat([X_bd.drop(bd_categs,axis=1),X_bd[bd_categs]], axis=1)
    X_bd = X_bd.reindex(columns=X_bd.drop(bd_categs,axis=1).columns.tolist()+bd_categs.tolist())    
#    Thanks GuangshuoChen @Kaggle
    
##### PREVIOUS APPLICATION DATA    
    X_pa, pa_categs = preproc_ds(read_df(previous_application_data_file,k_datasets_prefix,[p_index,"SK_ID_PREV"]),None,False,[p_index,"SK_ID_PREV"])

# This sorting is important for the tensor generation to work correctly   
    X_pa = X_pa.sort_values(by=[p_index, 'SK_ID_PREV' ], ascending=[1,1])

# Columns are re arranged so that categorical data is in the end. 
# Solving embeddings in the NN this will be simpler in this way    
#    X_pa = pd.concat([X_pa.drop(pa_categs,axis=1),X_pa[pa_categs]], axis=1)
    X_pa = X_pa.reindex(columns=X_pa.drop(pa_categs,axis=1).columns.tolist()+pa_categs.tolist())    
#    Thanks GuangshuoChen @Kaggle
    
# If need to do a quick test you can use the p_subsample_lines parameter to limit the dataset to a smaller portion. 
# Since the dataset has been sorted by ID it should only be used to check the logic is working properly
    if p_subsample_lines != 0:    
        data = data.iloc[:p_subsample_lines]

    
    X_predict = data_unclassified

# use the smallest of 10% or 10ksamples for test    
    if len(data)> 100000: test_size = 10000 
    else: test_size = 0.10

# split data. No need to use "the answer to life, the universe and everything" as random_state ;)    
    X_train, X_test = train_test_split(data, test_size=test_size, random_state=42, shuffle = True)

# This sorting is important for the tensor generation to work correctly  
    X_train = X_train.sort_values(by=[p_index], ascending = True)
    y_train = X_train[p_objective]
    X_train = X_train.drop(p_objective, axis=1)

# This sorting is important for the tensor generation to work correctly      
    X_test = X_test.sort_values(by=[p_index], ascending = True)
    y_test = X_test[p_objective]
    X_test = X_test.drop(p_objective, axis=1)

# Save preprocessed data for the main input data and category lists
    h5py_file = h5py.File(h5py_file_filename, "a", driver = "core")    
    X_train_h5 = h5dump("pd_"+ p_data_model_version +"_X_train",X_train,h5py_file,(5,X_train.shape[1]),)
    X_test_h5 = h5dump("pd_"+ p_data_model_version +"_X_test",X_test,h5py_file,(5,X_test.shape[1]))
    X_predict_h5 = h5dump("pd_"+ p_data_model_version +"_X_predict",X_predict,h5py_file,(5,X_predict.shape[1]))
    y_train_h5 = h5dump("pd_"+ p_data_model_version +"_y_train",y_train,h5py_file,None)
    y_test_h5 = h5dump("pd_"+ p_data_model_version +"_y_test",y_test,h5py_file,None)
    X_categs = pickledump("pk_"+ p_data_model_version +"_X_categs",data_categs)


    h5py_file.flush()
    h5py_file.close()

#Invoke the tensor build and save to H5PY for Bureau data in its three splits (train, Test and unclassified)    
    build_dump_tensor(X_test, p_index, p_products,"pd_"+ p_data_model_version +'_bd_test',X_bd, h5py_file_bd_filename)
    build_dump_tensor(X_predict, p_index, p_products,"pd_"+ p_data_model_version +'_bd_predict',X_bd, h5py_file_bd_filename)
    build_dump_tensor(X_train, p_index, p_products,"pd_"+ p_data_model_version +'_bd',X_bd, h5py_file_bd_filename)

#Invoke the tensor build and save to H5PY for Previous Application data in its three splits (train, Test and unclassified)    
    build_dump_tensor(X_test,p_index, p_products,"pd_"+ p_data_model_version +'_pa_test',X_pa,h5py_file_pa_filename)
    build_dump_tensor(X_predict, p_index, p_products,"pd_"+ p_data_model_version +'_pa_predict',X_pa,h5py_file_pa_filename)
    build_dump_tensor(X_train, p_index, p_products,"pd_"+ p_data_model_version +"_pa",X_pa,h5py_file_pa_filename)

#Save category list for embedding processing within NN
    pickledump("pk_"+ p_data_model_version +"_bd_categs",bd_categs)
    pickledump("pk_"+ p_data_model_version +"_pa_categs",pa_categs)       
        

# Reload from h5py all data. Makes sense for subsequent runs where data is not generated again
h5py_file = h5py.File("h5py_"+ p_data_model_version + ".hdf5", "r")
h5py_file_bd = h5py.File(h5py_file_bd_filename, "r")
h5py_file_pa = h5py.File(h5py_file_pa_filename, "r")

X_train = h5py_file["pd_"+ p_data_model_version +"_X_train"][:,1:]
X_train_labels = h5py_file["pd_"+ p_data_model_version +"_X_train"][:,:1]
X_test = h5py_file["pd_"+ p_data_model_version +"_X_test"][:,1:]
X_test_labels = h5py_file["pd_"+ p_data_model_version +"_X_test"][:,:1]
X_predict = h5py_file["pd_"+ p_data_model_version +"_X_predict"][:,1:]

y_train = h5py_file["pd_"+ p_data_model_version +"_y_train"]
y_test = h5py_file["pd_"+ p_data_model_version +"_y_test"]
X_categs = pickleload("pk_"+ p_data_model_version +"_X_categs")

X_bd_tensor = h5py_file_bd["pd_"+ p_data_model_version +"_bd"]
X_test_bd_tensor = h5py_file_bd["pd_"+ p_data_model_version +"_bd_test"]
X_predict_bd_tensor = h5py_file_bd["pd_"+ p_data_model_version +"_bd_predict"]
X_bd_categs = pickleload("pk_"+ p_data_model_version +"_bd_categs")

X_pa_tensor = h5py_file_pa["pd_"+ p_data_model_version +"_pa"]
X_test_pa_tensor = h5py_file_pa["pd_"+ p_data_model_version +"_pa_test"]
X_predict_pa_tensor = h5py_file_pa["pd_"+ p_data_model_version +"_pa_predict"]
X_pa_categs = pickleload("pk_"+ p_data_model_version +"_pa_categs")
#Determine features for all three input matrices/tensors
pa_in_vars = X_pa_tensor.shape[2]
bd_in_vars = X_bd_tensor.shape[2]
main_in_vars = X_train.shape[1]

total_samples = len(X_train)
from keras.callbacks import Callback


class roc_callback(Callback):
    def __init__(self,training_data,validation_data,early_stopping=False, patience=10, min_delta=0, checkpoint_file = None):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.auc_val = []
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.checkpoint_file = checkpoint_file


    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = 0
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_val, y_pred_val, pos_label=1)
        auc_val=metrics.auc(fpr, tpr)
        self.auc_val.append(auc_val)
        print('\rSKLearn roc-auc_val: %s' % str(round(auc_val,4)),end=100*' '+'\n')
        if self.early_stopping == True:
            current = auc_val

            if self.monitor_op(current-self.min_delta,self.best):
                self.best = current
                self.wait = 0
                if not self.checkpoint_file is None:
                    self.model.save_weights(self.checkpoint_file)
                    print("Saved checkpoint: ", self.checkpoint_file)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
            print("Early Stopping Best: ",str(round(self.best,5)), "Epochs w/o improvement: ", self.wait, "Must score ", str(round(self.best+self.min_delta,5)), "before  ",self.patience-self.wait," Epochs to extend",'\n' )
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
#This makes an aproximate AUC score calculation within the epoch
def auc_m(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P

#Custom function that allows cropping of tensors within keras
def crop(dimension, start, end=None):
    if not end is None:        
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
    else:
        def func(x):
            if dimension == 0:
                return x[start]
            if dimension == 1:
                return x[:, start]
            if dimension == 2:
                return x[:, :, start]
            if dimension == 3:
                return x[:, :, :, start]
            if dimension == 4:
                return x[:, :, :, :, start]            
    return Lambda(func)

def gen_conv_branch(name,  # name of layer
                    products, # number of prev applications or bureau reports as max
                    pa_filters, # number of filters in first convolution
                    pa_filters1, # number of filters in next convolution
                    pa_filters2, # number of filters in last convolution
                    lc, # Size of dense layer after convolutions
                    lc2, # Size of final dense layer before integrating with main input
                    features_pa, # number of features in prev applications or bureau reports
                    regu, # regularization parameter for convolutions and dense layers
                    pa_categs, # list of categorical fields
                    inputs, # list of input layers to this moment. function will append and return with the input for the new input
                    NN_data, # list with training data (h5py arrays). function will append and return with training data fot this branch
                    NN_val_data, # list with testing data (h5py arrays). function will append and return with testing data fot this branch
                    data_pa, # h5py object with train data
                    data_test_pa):   # h5py object with test data
    

        # Create input
        b_pa_input = Input(shape=(products,features_pa),name=name+"_PA_main", dtype='float32') 

        # Append new input to list of inputs already created
        inputs.append(b_pa_input)
        
        #Append train data to list of data already created
        NN_data.append(data_pa)

        #Append test data to list of data already created       
        NN_val_data.append(data_test_pa)

        #Number of non categorical features
        noncat_features = int(b_pa_input.shape[2]-len(pa_categs))

        #Splits from the input the part that has non categorical data. 
        # Data has (Sample, Product, Feature). This function will keep (Sample, Product, 0 to number of non-cat features)
        x_noncat_pa = crop(2,0,noncat_features)(b_pa_input)
#       x_noncat_pa = Reshape((products,noncat_features,))(x_noncat_pa)
        
        x_cat_pa = [x_noncat_pa] # initialize list of concatenation after dealing with embeddings

        
        #the following block creates the embedding layer for each categorical feature. 
        #It will crop each feature out of the input layer, determine the cardinality, create the embedding layer,
        #appends the resulting embedded layer to the list x_cat_pa (non-categoricals + categoricals)
        #and concatenate it all in a single result
        i=0 
        if len(pa_categs) > 0:
            for categ in pa_categs:
                cat_layer_pa = crop(2,noncat_features+i )(b_pa_input)
                cardinality = len(np.unique(data_pa[:,:,-len(pa_categs)+i]))
                target_cardinality = int(cardinality ** 0.25)
                cat_layer_pa = Embedding(cardinality, target_cardinality)(cat_layer_pa)
                cat_layer_pa = Reshape((products,target_cardinality,))(cat_layer_pa)         
                x_cat_pa.append(cat_layer_pa)
                i+=1
            b_pa_x= concatenate(x_cat_pa)
        else:
            b_pa_x= b_pa_input
        
        #The resulting tensor after embeddings still has the structure (sample, products, features)
        #it will pass through three 1D convolution with a kernel of size 1. 
        #The input features represent channels in the convolution. The idea behind using a convolution is to  
        #have a common "logic" for all instances of products
        
        b_pa_x = Conv1D(filters=pa_filters,
                      kernel_size=(1),
                      input_shape=(products,features_pa),
                      activation="relu",
                      kernel_regularizer=regularizers.l2(regu),
                      strides=1)(b_pa_x)
        b_pa_x = Conv1D(filters=pa_filters1,
              kernel_size=(1),
              input_shape=(products,pa_filters),
              activation="relu",
              kernel_regularizer=regularizers.l2(regu),
              strides=1)(b_pa_x)
        b_pa_x = Conv1D(filters=pa_filters2,
              kernel_size=(1),
              input_shape=(products,pa_filters1),
              activation="relu",
              kernel_regularizer=regularizers.l2(regu),
              strides=1)(b_pa_x)
        
        # The result id flattened and passed through two dense layers before connecting with the other inputs

        b_pa_x = Flatten()(b_pa_x)
        b_pa_x = Dropout(0.3)(b_pa_x)

        b_pa_x = Dense(lc, activation="relu",kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(regu))(b_pa_x)  
        b_pa_x = Dropout(0.3)(b_pa_x)
        
        b_pa_x = Dense(lc2, activation="relu",kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(regu))(b_pa_x)        

        return(b_pa_x, inputs, NN_data, NN_val_data)


    
def plot_roc(model,test_data,test_labels):

    prediction = model.predict(test_data)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, prediction, pos_label=1)
    auc=metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve '+ str(round(auc,6)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    plt.show()
    fig.savefig('./work/'+p_data_model_version+'_ROC.png')
    return(auc)    
    
#Set the weights of the classess based on the aproximate proportions of positive/negative samples
class_weight = {0: 1.,
                1: 16.}

#Regularization for convolution branches
regu=10**-7.56
#Redularization for main dense layers
regm=10**-4.2
#Learning rate
learn_rate = 10**-2.38
#Generally early stopping is activated before
epochs=50
#Set to lower size if out of GPU memory
batch_size = 2**12


#Input for main data file
main_input = Input(shape=(main_in_vars,),name='main_input', dtype='float32') #dtype='int32',

#Initialize inputs and data lists
model_inputs = [main_input]
NNtrain_data = [X_train]
NNtest_data = [X_test]



#Number of non categorical features
noncat_features = int(main_input.shape[1]-len(X_categs))

#Splits from the input the part that has non categorical data. 
#Data has (Sample, Feature). This function will keep (Sample, 0 to number of non-cat features)
x_noncat = crop(1,0,noncat_features)(main_input)
x_cat = [x_noncat]

#the following block creates the embedding layer for each categorical feature. 
#It will crop each feature out of the input layer, determine the cardinality, create the embedding layer,
#appends the resulting embedded layer to the list x_cat_pa (non-categoricals + categoricals)...
i=0
for categ in X_categs:
    cat_layer = crop(1,noncat_features+i )(main_input)
    cardinality = len(np.unique(X_train[:,-len(X_categs)+i]))
    target_cardinality = int(cardinality ** 0.25)   
    cat_layer = Embedding(cardinality, target_cardinality)(cat_layer)
    cat_layer = Reshape((target_cardinality,))(cat_layer)         
    x_cat.append(cat_layer)
    i+=1

#...and concatenate it all in a single result
x= concatenate(x_cat)


#First fully connected layer
x = Dense(90, activation="relu", kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(regm))(x)
x = Dropout(0.5)(x)


#Generate convolution branch for bureau data
bd_x, model_inputs, NNtrain_data, NNtest_data  = gen_conv_branch("bd_input",p_products, 
                                                  90, 60, 10,
                                                  60, 20, bd_in_vars,regu, 
                                                  X_bd_categs,
                                                  model_inputs, NNtrain_data, NNtest_data,
                                                  X_bd_tensor, X_test_bd_tensor)

#concatenate the result of the bureau data conv branch to the rest
x = concatenate([bd_x, x])

#Generate convolution branch for Previous Application
pcb_x, model_inputs, NNtrain_data, NNtest_data = gen_conv_branch("pa_input",p_products, 
                                                  90, 60, 10,
                                                  60,20,pa_in_vars,regu, 
                                                  X_pa_categs,
                                                  model_inputs, NNtrain_data, NNtest_data,
                                                  X_pa_tensor, X_test_pa_tensor)
x = concatenate([pcb_x, x])


#add two fully connected layers
x = Dense(150, activation="relu",kernel_initializer='he_normal',
    kernel_regularizer=regularizers.l2(regm))(x)
x = Dropout(0.5)(x)

x = Dense(30, activation="relu",kernel_initializer='he_normal',
    kernel_regularizer=regularizers.l2(regm))(x)     

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs=model_inputs, outputs=main_output)

#custom function to correctly measure AUC, deal with early stopping and save training progress
roc_cbk = roc_callback(training_data=None,validation_data=(NNtest_data, y_test),
                       early_stopping= True, patience=8, min_delta=0.0001, 
                       checkpoint_file = './'+p_data_model_version+'_weights.hdf5')   
callbacks_list = [roc_cbk]

#Compile model
model.compile(optimizer=optimizers.Adam(lr=learn_rate), loss='binary_crossentropy', metrics=[auc_m])

#Train and plot evolution of AUC
fitlog = model.fit(NNtrain_data, y_train, validation_data=(NNtest_data, y_test),
                   epochs=epochs, batch_size=batch_size, 
                   class_weight=class_weight, callbacks=callbacks_list, shuffle="batch")
#Plot training evolution
auc = fitlog.history["auc_m"]
plt.figure(figsize=(12,10))
plt.plot(auc)
plt.plot(roc_cbk.auc_val)
plt.ylim([0.73, 0.85])
plt.title('AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.legend('test', loc='upper right')
plt.plot(np.array(range(len(auc))),np.full(len(auc),np.mean(auc[-4:])))
plt.plot(np.array(range(len(roc_cbk.auc_val))),np.full(len(roc_cbk.auc_val),np.mean(roc_cbk.auc_val[-4:])))
plt.show()

print("Best AUC: ",max(roc_cbk.auc_val), " at epoch: ", np.argmax(roc_cbk.auc_val))
#Build prediction input data list
NNpredict_data = [X_predict, X_predict_bd_tensor, X_predict_pa_tensor ]

#load best weights from checkpoint file
model.load_weights('./'+p_data_model_version+'_weights.hdf5')

#Compute prediction
prediction = model.predict(NNpredict_data)
#create submission file
submission = pd.read_csv("./" + p_data_model_version + "unclassified_ids.csv", sep=',')
submission["TARGET"] = prediction
submission.to_csv('submission.csv', index=False)
