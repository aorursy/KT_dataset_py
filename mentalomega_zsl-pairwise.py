import scipy

from scipy.io import loadmat,savemat

import numpy as np

from scipy.spatial.distance import cdist

from scipy.linalg import hadamard
split_mat = loadmat("/kaggle/input/xlsa17/xlsa17/xlsa17/data/AWA1/att_splits.mat")

fea_mat = loadmat("/kaggle/input/xlsa17/xlsa17/xlsa17/data/AWA1/res101.mat")
def standardization(data):

    mu = np.mean(data, axis=0)

    sigma = np.std(data, axis=0)

    return (data - mu) / sigma



X = standardization(fea_mat["features"].T)

L = fea_mat["labels"] - 1

A = standardization(split_mat["att"].T)

index_test = np.squeeze(split_mat["test_unseen_loc"] - 1 )

index_train = np.squeeze(split_mat["trainval_loc"] - 1)

X_train = X[index_train]

X_test = X[index_test]

L_train = L[index_train]

L_test = L[index_test]

A_train = A[np.squeeze(L_train)]

A_test = A[np.squeeze(L_test)]

X_train.shape,X_test.shape,L_train.shape,L_test.shape,A.shape,A_train.shape,L.max(),L_train.min(),L_train.max(),L_test.min(),L_test.max(),A.max(),A.min()
L_unseen = np.unique(L_test)

A_unseen = A[L_unseen]

# H = hadamard(64)

# H_train = H[L_train]

# H_unseen = H[L_unseen]
import keras

from keras.models import Model

from keras.layers import *

from keras import backend as K

from keras.optimizers import Adam

from keras.losses import *

import tensorflow as tf
L = keras.utils.to_categorical(L)

L_train = L[index_train]

L_test = L[index_test]
def data_genarator():

    A_L = np.eye(A.shape[0])

    while True:

        index = np.random.permutation(X_train.shape[0])[:256]

        index_2 = np.random.choice(A.shape[0],256)

        yield (X_train[index],L_train[index],A_train[index],L_train[index]),None

#         yield (X_train[index],L_train[index],A[index_2],A_L[index_2]),None

data_gen = data_genarator()

next(data_gen)[0][0].shape,next(data_gen)[0][1].shape,next(data_gen)[0][2].shape,next(data_gen)[0][3].shape
input_x = Input([X.shape[-1]])

input_a = Input([A.shape[-1]])

input_xl = Input([L_train.shape[-1]])

input_al = Input([L.shape[-1]])

_ = input_x

_ = Dense(2048,activation="relu")(_)

_ = Dropout(0.5)(_)

_ = Dense(128,activation="tanh")(_)

output_x = _

model_x = Model(input_x,output_x)

model_x.summary()

_ = input_a

_ = Dense(512,activation="relu")(_)

_ = Dropout(0.5)(_)

_ = Dense(128,activation="tanh")(_)



output_a = _

model_a = Model(input_a,output_a)

model_a.summary()



def compute_Dij(inputs):

    x,y = inputs

    return K.reshape(K.dot(x,K.transpose(y)),[-1,1])

    return K.reshape((BIT- K.dot(x,K.transpose(y)))/2,[-1,1])



def compute_Sij(inputs):

    x,y = inputs

    return K.reshape(K.cast(K.dot(x, K.transpose(y)) > 0, "float32"),[-1,1])



def shuffling(x):

    idxs = K.arange(0, K.shape(x)[0])

    idxs = tf.random.shuffle(idxs)

    return K.gather(x, idxs)



output_Dij = Lambda(compute_Dij)([output_x,output_a])

output_Dij = Activation("sigmoid")(output_Dij)

output_Sij = Lambda(compute_Sij)([input_xl,input_al])

Sij_shuffle = Lambda(shuffling)(output_Sij)

D_S_1 = Concatenate()([output_Dij, output_Sij])

D_S_2 = Concatenate()([output_Dij, Sij_shuffle])



z_in = Input([1+1])

z = z_in

z = Dense(32, activation='relu')(z)

z = Dense(32, activation='relu')(z)

z = Dense(32, activation='relu')(z)

z = Dense(1, activation='sigmoid')(z)



GlobalDiscriminator = Model(z_in, z)

GlobalDiscriminator.summary()

D_S_1_scores = GlobalDiscriminator(D_S_1)

D_S_2_scores = GlobalDiscriminator(D_S_2)

global_info_loss = - K.mean(K.log(D_S_1_scores + 1e-6) + K.log(1 - D_S_2_scores + 1e-6))



model_train = Model([input_x,input_xl,input_a,input_al], [D_S_1_scores, D_S_2_scores])

model_train.add_loss(global_info_loss)





model_train.compile(optimizer=Adam(1e-4))





class Moniter(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        print("训练集精度",np.mean(

            np.argmin(cdist(model_x.predict(X_train),model_a.predict(A),metric="cosine"),-1) == np.argmax(L_train,-1)

            ))

        print("测试集精度",np.mean(

            L_unseen[np.argmin(cdist(model_x.predict(X_test),model_a.predict(A_unseen),metric="cosine"),-1)] == np.argmax(L_test,-1)

        ))

        

model_train.fit_generator(data_gen,epochs=20,steps_per_epoch=X_train.shape[0]//256,callbacks=[Moniter()])
input_x = Input([X.shape[-1]])

input_a = Input([A.shape[-1]])

input_xl = Input([L_train.shape[-1]])

input_al = Input([L.shape[-1]])

_ = input_x

_ = Dense(2048,activation="relu")(_)

_ = Dropout(0.5)(_)

_ = Dense(128,activation="tanh")(_)

output_x = _

model_x = Model(input_x,output_x)

model_x.summary()

_ = input_a

_ = Dense(512,activation="relu")(_)

_ = Dropout(0.5)(_)

_ = Dense(128,activation="tanh")(_)



output_a = _

model_a = Model(input_a,output_a)

model_a.summary()



def loss_struct(x, a, xl, al):

    s_bat = K.cast(K.dot(xl, K.transpose(al)) > 0, "float32")

    s_bat = s_bat * 2. - 1.

    theta = 0.5 * K.dot(x, K.transpose(a))

    return - K.sum(K.log(0.5 * (1. - s_bat) + s_bat * K.sigmoid(theta) + 1e-9))



loss = loss_struct(output_x,output_a,input_xl,input_al)

model_train = Model([input_x,input_xl,input_a,input_al],loss)

model_train.add_loss(loss)

model_train.compile(Adam(2e-4))
class Moniter(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        print("训练集精度",np.mean(

            np.argmin(cdist(model_x.predict(X_train),model_a.predict(A),metric="cosine"),-1) == np.argmax(L_train,-1)

            ))

        print("测试集精度",np.mean(

            L_unseen[np.argmin(cdist(model_x.predict(X_test),model_a.predict(A_unseen),metric="cosine"),-1)] == np.argmax(L_test,-1)

        ))

        

model_train.fit_generator(data_gen,epochs=20,steps_per_epoch=X_train.shape[0]//256,callbacks=[Moniter()])