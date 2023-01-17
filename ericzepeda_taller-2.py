import tensorflow as tf

from keras.optimizers import adam

from keras.layers.normalization import BatchNormalization

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D,LeakyReLU,GRU,LSTM,SimpleRNN,Reshape

from keras.utils import to_categorical, plot_model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

from tensorflow.keras.initializers import he_uniform

from keras.constraints import max_norm
def txt_extract(names_data,path):

    data_list=[]

    for i in names_data:

        sentence = ""

        data_path = path + str(i)+".txt"

        text= open(data_path,'r')

        for x in text:

            sentence+=str(x)

        data_list.append(sentence)

        

        text.close()

    return data_list
def relation_extractor(data):

    full_extraction=[]

    for i in data:

        data_aux=""

        partial_extraction=[]

        data_aux=i.replace("'","")

        data_aux=data_aux.replace("[","")

        data_aux=data_aux.replace("]","")

        data_aux=data_aux.replace("->",",")

        data_aux=data_aux.replace(" ","")

        data_aux=data_aux.split("\n")

        for j in data_aux:

            if 'ROOT' in j:

                continue

            else:

                list_aux= j.split(",")

                partial_extraction.append(list_aux)

        full_extraction.append(partial_extraction)

    return full_extraction

def train_data_transform(data):

    for x in range(len(data)):

        for y in range(len(data[x])):

            for z in range(len(data[x][y])):

                if data[x][y][z]!='':

                    data[x][y][z] = float(data[x][y][z])

                else:

                    data[x][y][z]=0

    return data

def zeros_padding(data):

    largo_data = shape_calculator(data)

    len_max = max_array(data)

    new_data = np.zeros((largo_data,6))

    x=0

    for i in range(len(data)):

        for j in range(len(data[i])):

            y=0

            for k in range(len(data[i][j])):

                new_data[x][y]=data[i][j][k]

                

                y+=1

            x+=1

    return new_data

def max_array(data):

    largo_max=0

    for i in data:

        if len(i)>largo_max:

            largo_max=len(i)

        else:

            continue

    return largo_max

def shape_calculator(data):

    suma=0

    for i in data:

        suma+=len(i)

    return suma

def X_data_import(names_data,path):

    X = txt_extract(names_data,path)

    X = relation_extractor(X)

    X = train_data_transform(X)

    X = np.array(X)

    X=zeros_padding(X)

    return X
def txt_original_extract(name):

    original_list=[]

    data=open(name,'r')

    data_read=[line.rstrip('\n') for line in data]

    for x in data_read:

        aux_list=[]

        x_aux=x.split("\t",1)

        original_id=x_aux[0]

        x_aux.remove(original_id)

        x_aux = " ".join(str(w) for w in x_aux)

        aux_list.append(original_id)

        aux_list.append(x_aux)

        original_list.append(aux_list)

    data.close()

    return original_list
def cat_encode(data,cat):

    data=list(data)

    cat=list(cat)

    for i in range(len(data)):

        for j in range(len(cat)):

            if data[i]==cat[j]:

                data[i]=j

                break

            else:

                continue

    return keras.utils.to_categorical(data,len(cat))

    #return np.asarray(data)

def cat_decode(data,cat):

    #Y=data

    Y= [np.argmax(y, axis=None, out=None) for y in data]

    cat=list(cat)

    

    for i in range(len(Y)):

        Y[i]=cat[Y[i]]

    return Y
def auto_predict(data_x,data_set,cat,example_show):

    pred = model.predict(data_x)

    y_test = cat_decode(pred,cat)

    data_set['Expected']= y_test

    data_set.to_csv('sample_submission_1234.csv',columns=['Id','Expected'],index=False)

    if example_show==True:

        return print("Exito!\n",y_test[0:20])

    else:

        return print("Exito")
train_labels = pd.read_csv('/kaggle/input/taller/train_labels.csv')

test_labels = pd.read_csv('/kaggle/input/taller/sample_submission.csv')

train_path = '/kaggle/input/taller/train/'

test_path = '/kaggle/input/taller/test/'

train_message_path= "/kaggle/input/taller/train_source_tweets.txt"

test_message_path= "/kaggle/input/taller/test_source_tweets.txt"



###########        This creates original df with original message and label      ##################33

x_original = txt_original_extract(train_message_path)

df_original = pd.DataFrame(x_original,  columns =['id','original message'])

df_original['id']= df_original['id'].astype('int64')

df_train = pd.merge(df_original, train_labels, how='inner', left_on='id', right_on='id')

df_train.head(5)

##############################################################################33
train_message = df_train["original message"].values

train_relations = txt_extract(df_train["id"],train_path )

Y_train = df_train["label"]

cat = Y_train.unique()

Y_train=cat_encode(Y_train,cat)
train_message = df_train["original message"].values

train_relations = X_data_import(df_train["id"],train_path )

df_relations = pd.DataFrame(data=train_relations,columns=['P_uid', 'P_tweet_ID',"P_time", 'C_uid', 'C_tweet_ID',"C_time" ])

df_train_full = pd.merge(df_train, df_relations, how='inner', left_on='id', right_on='P_tweet_ID')

df_train_full.head()
X_train = df_train_full[["P_uid","P_tweet_ID","P_time","C_uid","C_tweet_ID","C_time"]].values

Y_train = df_train_full["label"]

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

cat = Y_train.unique()

Y_train=cat_encode(Y_train,cat)
X_train.shape
Y_train.shape

batch_size=128



model = Sequential()

model.add(keras.layers.BatchNormalization(input_shape = X_train.shape[1:]))

#model.add((Dense(518)))

#model.add(keras.layers.BatchNormalization())

model.add(LeakyReLU(alpha=0.1))

model.add(keras.layers.BatchNormalization())

#model.add(Reshape((128,6)))# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)

model.add(GRU(512,return_sequences=True))

model.add(keras.layers.BatchNormalization())

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)

model.add(GRU(512,return_sequences=True))

model.add(keras.layers.BatchNormalization())



model.add(GRU(512,return_sequences=False))

model.add(keras.layers.BatchNormalization())

#model.add(SimpleRNN(128))

#model.add(SimpleRNN(512, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros'))

model.add(Dense(512))

model.add(keras.layers.BatchNormalization())





model.add(Dense(4,activation='softmax'))



model.summary()
####################################        NUEVO                    #######################################

#Nota: Para el tercer modelo (test) con data augmentation

    

    

#########################################           CALLBACKS           #########################################

early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5)

batch_size=128

check_point = ModelCheckpoint(

    filepath='/tmp/checkpoint',

    save_weights_only=True,

    monitor='val_loss',

    mode='min',

    save_best_only=True,    

    verbose=1

)



reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.85, patience=4, verbose=0, mode='auto', min_lr=0.00006

)



#########################################           OPTIMIZER           #########################################

opt=tf.keras.optimizers.Adam()



########################################           LOSS FUCTION           #######################################



#########################################           Metrics           #########################################

model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["categorical_accuracy"])

train = model.fit(X_train,Y_train,epochs=20,validation_split=0.3,batch_size=batch_size,shuffle=True,callbacks=[early_stop,check_point,reduce_lr])

model.load_weights('/tmp/checkpoint')
X_train1 = df_train_full[["P_uid","P_tweet_ID","P_time"]].values

X_train1 = X_train1.reshape(X_train1.shape[0], 1, X_train1.shape[1])

batch_size=128



model = Sequential()

model.add(keras.layers.BatchNormalization(input_shape = X_train1.shape[1:]))

model.add((Dense(518)))

model.add(keras.layers.BatchNormalization())

model.add(LeakyReLU(alpha=0.1))

model.add(keras.layers.BatchNormalization())

#model.add(Reshape((128,6)))# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)

model.add(GRU(512,return_sequences=True))

model.add(keras.layers.BatchNormalization())

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)

model.add(GRU(512,return_sequences=False))

model.add(keras.layers.BatchNormalization())

#model.add(SimpleRNN(128))

#model.add(SimpleRNN(512, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros'))

model.add(Dense(512))

model.add(keras.layers.BatchNormalization())





model.add(Dense(4,activation='softmax'))



model.summary()

model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["categorical_accuracy"])

train = model.fit(X_train1,Y_train,epochs=20,validation_split=0.3,batch_size=batch_size,shuffle=True,callbacks=[early_stop,check_point,reduce_lr])