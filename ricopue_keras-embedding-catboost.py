import numpy as np

import pandas as pd

import datetime



import warnings

warnings.filterwarnings("ignore")
path_in = '../input/cat-in-the-dat-ii/'

X_train= pd.read_csv(path_in+'train.csv')

X_test= pd.read_csv(path_in+'test.csv')

samp_subm = pd.read_csv(path_in+'sample_submission.csv')
train_test =pd.concat([X_train, X_test],sort=False).reset_index()

y_train = X_train['target']

train_test = train_test.drop(['index','id','target'], axis = 1)
features_bin = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

features_low_nom = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

features_hi_nom = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

features_ord = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']

features_hi_ord = ['ord_5']

features_cyc = ['day', 'month']



numerics = ['float16', 'float32', 'float64']

categoricals=['int8','int16', 'int32', 'int64', ]
from sklearn.model_selection import StratifiedKFold

from category_encoders import WOEEncoder

from sklearn import preprocessing

import string 





def Convert_to_numeric(df):

      

    #Ordinal features

    map_ord = {'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4,

           'Freezing': 0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot': 4, 'Lava Hot':5}

    

    scii_letters_list=list(string.ascii_letters)

    map_ord_hex= dict(zip(scii_letters_list,range(0, len(scii_letters_list))))



    

    df['ord_0'] = df['ord_0']

    df['ord_1'] = df['ord_1'].replace(map_ord)

    df['ord_2'] = df['ord_2'].replace(map_ord)

    df['ord_3'] = df['ord_3'].replace(map_ord_hex)

    df['ord_4'] = df['ord_4'].replace(map_ord_hex)

        

    df[features_ord] = df[features_ord].fillna(df[features_ord].mean())

    

    StandardScaler_Encoder = preprocessing.StandardScaler()

    df[features_ord] = StandardScaler_Encoder.fit_transform(df[features_ord].astype(float))    

    



   #Binary, Low nominal and time features WOE encoder.

    n_splits=5

    WOE_features=features_bin+features_low_nom+features_cyc

    for col in WOE_features:

        df[f'{col}_Encode']=0

        for tr_idx, tst_idx in StratifiedKFold(n_splits=n_splits, random_state=2020, shuffle=True).split(df[:600000], y_train):

            WOE_encoder = WOEEncoder(cols=col)        

            WOE_encoder.fit(df[:600000].iloc[tr_idx, :], y_train.iloc[tr_idx])

            col_df=WOE_encoder.transform(df)[col]/n_splits

            df[f'{col}_Encode']= df[f'{col}_Encode']+col_df       

    df = df.drop(WOE_features, axis = 1)          

    

    

    #High Nominal Features Label encoder.

    Label_col=features_hi_nom+features_hi_ord

    for col in Label_col:

        Label_Encoder = preprocessing.LabelEncoder()

        df[col] = Label_Encoder.fit_transform(df[col].fillna("-1").astype(str).values)







        

    return df
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin



from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train_test=Convert_to_numeric(train_test)



train_test = reduce_mem_usage(train_test,use_float16=True)



X_train=train_test[:len(X_train)]

X_test=train_test[-len(X_test):]

#converting data to list format to match the network structure

def preproc(X_train, X_val, X_test):



    input_list_train = []

    input_list_val = []

    input_list_test = []    

    

    

    #the cols to be embedded: rescaling to range [0, # values)

    for c in X_train.select_dtypes(include=categoricals):

        raw_vals = np.unique(X_train[c])

        val_map = {}

        for i in range(len(raw_vals)):

            val_map[raw_vals[i]] = i       

        input_list_train.append(X_train[c].map(val_map).values)

        input_list_val.append(X_val[c].map(val_map).fillna(0).values)

        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

     

    #the rest of the columns

    other_cols = X_train.select_dtypes(include=numerics).columns.tolist()

    input_list_train.append(X_train[other_cols].values)

    input_list_val.append(X_val[other_cols].values)

    input_list_test.append(X_test[other_cols].values)

    

    return input_list_train, input_list_val, input_list_test 
#Keras embeddings based on

#https://github.com/mmortazavi/EntityEmbedding-Working_Example

    

import tensorflow as tf

import keras.backend as K

from keras.models import Sequential,Model  

from keras.models import Model as KerasModel

from keras.layers import Input, Dense, Activation, Reshape, BatchNormalization

from keras.layers import Concatenate, Dropout, SpatialDropout1D

from keras.layers.embeddings import Embedding

from tensorflow.keras import callbacks, utils 

from keras.optimizers import Adam

from tensorflow.python.keras.optimizers import TFOptimizer

from sklearn.metrics import roc_auc_score





def make_model():



    input_models=[]

    output_embeddings=[]

    numerics = ['float16', 'float32', 'float64']

    categoricals=['int8','int16', 'int32', 'int64']



    for categorical_var in X_train.select_dtypes(include=categoricals):

    

        #Name of the categorical variable that will be used in the Keras Embedding layer

        cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'

        # Define the embedding_size

        no_of_unique_cat  = X_train[categorical_var].nunique()

        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))

  

        #One Embedding Layer for each categorical variable

        input_model = Input(shape=(1,))

        output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)

        output_model = SpatialDropout1D(0.3)(output_model)

        output_model = Reshape(target_shape=(embedding_size,))(output_model)    

  

        #Appending all the categorical inputs

        input_models.append(input_model)

  

        #Appending all the embeddings

        output_embeddings.append(output_model)

        

    shape_numeric=len(X_train.select_dtypes(include=numerics).columns.tolist())

    #Other non-categorical data columns (numerical). 

    #I define single another network for the other columns and add them to our models list.

    input_numeric = Input(shape=(shape_numeric,))

    embedding_numeric = BatchNormalization()(input_numeric) 

    

    input_models.append(input_numeric)

    output_embeddings.append(embedding_numeric)



    #At the end we concatenate altogther and add other Dense layers

    output = Concatenate()(output_embeddings)

    output = BatchNormalization()(output)

    

    output = Dense(317,activation='relu',kernel_initializer="uniform")(output) 

    output= Dropout(0.3)(output) # To reduce ovwefiting

    output = BatchNormalization()(output)

    

    output = Dense(150,activation='relu',kernel_initializer="uniform")(output) 

    output= Dropout(0.2)(output) # To reduce ovwefiting

    output = BatchNormalization()(output)

    

    output = Dense(30,activation='relu')(output) 

    output= Dropout(0.1)(output) # To reduce ovwefiting

    output = BatchNormalization()(output)

    

       

    output = Dense(1, activation='sigmoid')(output)



    model = Model(inputs=input_models, outputs=output)

    



    return model
!pip install pydot_ng

import pydot_ng as pydot

from keras.utils import plot_model



model=make_model()

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

from IPython.display import Image

Image(retina=True, filename='model.png')
from sklearn.model_selection import train_test_split

from time import time

from sklearn.decomposition import PCA

from catboost import CatBoostClassifier







cls =CatBoostClassifier(eval_metric='AUC',

                        loss_function='CrossEntropy',

                        learning_rate=0.05,

                        l2_leaf_reg=3)



keras_oof = np.zeros(len(X_train))

cls_oof = np.zeros(len(X_train))

keras_preds = np.zeros(len(X_test))

cls_preds = np.zeros(len(X_test))



NFOLDS = 10



folds = StratifiedKFold(n_splits=NFOLDS, shuffle=False)





training_start_time = time()

for fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):

    

    train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]

    valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    

    X_train_list,X_val_list,X_test_list = preproc(train_x,valid_x, X_test)#Transfor to arrays to be accepted by 



    

    start_time = time()

    print(f'Training on fold {fold+1}')

    

    K.clear_session()

    model = make_model()

    adam = Adam(learning_rate=1e-4)

    model.compile(loss='binary_crossentropy',  optimizer=adam ,metrics=[tf.keras.metrics.AUC()])



    

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6,

                                 verbose=1, mode='min', baseline=None, restore_best_weights=True)



    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,min_delta=0.001,

                                      patience=2, min_lr=1e-6, mode='min', verbose=0)

     

   

    history  =  model.fit(X_train_list,train_y, validation_data=(X_val_list,valid_y) , epochs =  20,

                          batch_size = 1024, callbacks=[es,rl],verbose= 0)

        

    

    keras_fold_val= model.predict(X_val_list).ravel()

    keras_oof[valid_idx] += keras_fold_val/folds.n_splits

    keras_fold_auc=roc_auc_score(valid_y, keras_fold_val)

    keras_preds += model.predict(X_test_list).ravel()/folds.n_splits        

        

#######################CatBooost########################################



    layer_name = 'batch_normalization_2'

    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)    

    

    X_train_k = pd.DataFrame(intermediate_layer_model.predict(X_train_list))

    X_val_k = pd.DataFrame(intermediate_layer_model.predict(X_val_list))

    X_test_k = pd.DataFrame(intermediate_layer_model.predict(X_test_list))   

    

    cls.fit(X_train_k, train_y, eval_set=(X_val_k, valid_y),early_stopping_rounds=50,verbose=100,plot=False)



    cls_fold_val=cls.predict_proba(X_val_k)[:,1]

    cls_oof[valid_idx] += cls_fold_val/folds.n_splits

    cls_fold_AUC=roc_auc_score(valid_y, cls_fold_val)

    cls_preds +=cls.predict_proba(X_test_k)[:,1]/folds.n_splits

    

###########################Fold Results#########################

        

    print("\n")

    print('-' * 30)  

    print('Fold {} - Keras_OOF = {}'.format(fold + 1,keras_fold_auc ))

    print('Fold {} - CatBoost_OOF = {}'.format(fold + 1,cls_fold_AUC))

    print('-' * 30)

    print("\n")

    

    

#################Validation Results##############################

keras_val_auc=roc_auc_score(y_train, keras_oof)

cls_val_auc=roc_auc_score(y_train, cls_oof)

mix_val_auc=roc_auc_score(y_train, (keras_oof+cls_oof)/2)





print('-' * 30)  

print('# Final_Keras_OOF = {}'.format(keras_val_auc))

print('# Final_CatBoost_OOF = {}'.format(cls_val_auc))

print('# Final_Mix_OOF = {}'.format(mix_val_auc))

print('-' * 30)

print("\n")

predictions=(keras_preds+cls_preds)/2



num = samp_subm.id

output = pd.DataFrame({'id': num,

                       'target': predictions})

output.to_csv('keras-Embedding-catboost.csv', index=False)

output.head()