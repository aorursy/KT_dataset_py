import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import pandas as pd

import numpy as np

import tensorflow as tf



from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
%config IPCompleter.greedy=True
df_mitbih_train = pd.read_csv('../input/heartbeat/mitbih_train.csv', header=None)

df_mitbih_test = pd.read_csv('../input/heartbeat/mitbih_test.csv', header=None)

df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0)



df_ptbdb_normal = pd.read_csv('../input/heartbeat/ptbdb_normal.csv', header=None)

df_ptbdb_abnormal = pd.read_csv('../input/heartbeat/ptbdb_abnormal.csv', header=None)

df_ptbdb = pd.concat([df_ptbdb_normal, df_ptbdb_abnormal], axis=0)



print(df_mitbih.info())

print(df_ptbdb.info())
# ptbdb

M_ptbdb = df_ptbdb.values

X_ptbdb = M_ptbdb[:,:-1]

y_ptbdb = M_ptbdb[:,-1]



# mitbih

M_mitbih = df_mitbih.values

X_mitbih = M_mitbih[:,:-1]

y_mitbih = M_mitbih[:,-1]
classes={0:"Normal",

         1:"Artial Premature",

         2:"Premature ventricular contraction",

         3:"Fusion of ventricular and normal",

         4:"Fusion of paced and normal"}

plt.figure(figsize=(15,4))

for i in range(0,5):

    plt.subplot(2,3,i + 1)

    all_samples_indexes = np.where(y_mitbih == i)[0]

    rand_samples_indexes = np.random.randint(0, len(all_samples_indexes), 3)

    rand_samples = X_mitbih[rand_samples_indexes]

    plt.plot(rand_samples.transpose())

    plt.title("Samples of class " + classes[i], loc='left', fontdict={'fontsize':8}, x=0.01, y=0.85)

classes={0:"Normal", 1:"Abnormal (MI)"}

plt.figure(figsize=(10,2))

for i in range(0,2):

    plt.subplot(1,2,i + 1)

    all_samples_indexes = np.where(y_ptbdb == i)[0]

    rand_samples_indexes = np.random.randint(0, len(all_samples_indexes), 3)

    rand_samples = X_ptbdb[rand_samples_indexes]

    plt.plot(rand_samples.transpose())

    plt.title("Samples of class " + classes[i], loc="left", fontdict={'fontsize':8})
repartition = df_mitbih[187].astype(int).value_counts()

print(repartition)
plt.figure(figsize=(5,5))

circle=plt.Circle( (0,0), 0.8, color='white')

plt.pie(repartition, labels=['n','q','v','s','f'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')

p=plt.gcf()

p.gca().add_artist(circle)

plt.show()
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Activation, Add, Dense, Flatten

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import LearningRateScheduler
input_shape = (187, 1)



def make_model(final_layer_size=5):

    I = Input(input_shape)

    C = Conv1D(filters=32, kernel_size=5)(I)



    C11 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(C)

    C12 = Conv1D(filters=32, kernel_size=5, padding='same')(C11)

    A11 = Add()([C, C12])

    R11 = Activation(activation='relu')(A11)

    M11 = MaxPool1D(pool_size=5, strides=2)(R11)



    C21 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(M11)

    C22 = Conv1D(filters=32, kernel_size=5, padding='same')(C21)

    A21 = Add()([M11, C22])

    R21 = Activation(activation='relu')(A21)

    M21 = MaxPool1D(pool_size=5, strides=2)(R21)



    C31 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(M21)

    C32 = Conv1D(filters=32, kernel_size=5, padding='same')(C31)

    A31 = Add()([M21, C32])

    R31 = Activation(activation='relu')(A31)

    M31 = MaxPool1D(pool_size=5, strides=2)(R31)



    C41 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(M31)

    C42 = Conv1D(filters=32, kernel_size=5, padding='same')(C41)

    A41 = Add()([M31, C42])

    R41 = Activation(activation='relu')(A41)

    M41 = MaxPool1D(pool_size=5, strides=2)(R41)



    C51 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(M41)

    C52 = Conv1D(filters=32, kernel_size=5, padding='same')(C51)

    A51 = Add()([M41, C52])

    R51 = Activation(activation='relu')(A51)

    M51 = MaxPool1D(pool_size=5, strides=2)(R51)



    F1 = Flatten()(M51)

    D1 = Dense(32)(F1)

    R1 = Activation(activation='relu')(D1)

    D2 = Dense(32)(R1)

    D3 = Dense(final_layer_size)(D2)



    O = Activation(activation='softmax')(D3)



    return Model(inputs=I, outputs=O)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=10000, decay_rate=0.75)

adam = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
n_classes = len(np.unique(y_ptbdb))

model_ptbdb = make_model(n_classes)

model_ptbdb.summary()
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(X_ptbdb, y_ptbdb, test_size=0.15)
model_ptbdb.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model_ptbdb.fit(np.expand_dims(X_train_ptbdb, axis=2), 

                          y_train_ptbdb, 

                          validation_split=0.15,

                          epochs=30,

                          batch_size=256,

                          verbose=0)
def plot_learning(history):

    plt.subplot(211)

    plt.plot(history.history['accuracy'])

    plt.legend(["accuracy"])

    plt.subplot(212)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'], label = "val_loss")

    plt.legend(["loss", "val_loss"])

    plt.show()

    
plot_learning(history)
unique, counts = np.unique(y_test_ptbdb, return_counts=True)

print(f"The testing set contains {counts[0]} normal recordings and {counts[1]} with myocardial infarction.\nLet's compute the confusion matrix.")
results = model_ptbdb.evaluate(np.expand_dims(X_test_ptbdb, axis=2), y_test_ptbdb, batch_size=128)

print(f"The accuracy on the testing set is {np.round(results[1]*100,1)}%")
y_pred_ptbdb = model_ptbdb.predict(np.expand_dims(X_test_ptbdb, axis=2))

y_pred_ptbdb_bool = np.argmax(y_pred_ptbdb, axis=1)

print(classification_report(y_test_ptbdb, y_pred_ptbdb_bool))



confusion_matrix = tf.math.confusion_matrix(y_test_ptbdb, y_pred_ptbdb_bool)

print(f"Confusion matrix :\n {confusion_matrix}")
print(f"{confusion_matrix[0][0]}/{counts[0]} MI were correctly classified")



print(f"{confusion_matrix[1][1]}/{counts[1]} normal beats were correctly classified")



print(f"{confusion_matrix[1][0]} beats were classified as MI")



print(f"{confusion_matrix[0][1]} MI were classified as normal")

X_train_mitbih, X_test_mitbih, y_train_mitbih, y_test_mitbih = train_test_split(X_mitbih, y_mitbih, test_size=0.15)
n_classes_mitbih = len(np.unique(y_mitbih))

model_mitbih = make_model(n_classes_mitbih)

model_mitbih.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model_mitbih.fit(np.expand_dims(X_train_mitbih, axis=2), 

                           y_train_mitbih, 

                           validation_split=0.15,

                           epochs=30,

                           batch_size=256,

                           verbose=0)
plot_learning(history)
results = model_mitbih.evaluate(np.expand_dims(X_test_mitbih, axis=2), y_test_mitbih, batch_size=128)

print(f"The accuracy on the testing set is {np.round(results[1]*100,1)}%")
predictions = model_mitbih.predict(np.expand_dims(X_test_mitbih, axis=2))

confusion_matrix = tf.math.confusion_matrix(y_test_mitbih, np.argmax(predictions[:], axis=1))

print(confusion_matrix)
y_pred_mitbih = model_mitbih.predict(np.expand_dims(X_test_mitbih, axis=2))

y_pred_mitbih_bool = np.argmax(y_pred_mitbih, axis=1)

print(classification_report(y_test_mitbih, y_pred_mitbih_bool))



confusion_matrix = tf.math.confusion_matrix(y_test_mitbih, y_pred_mitbih_bool)

print(f"Confusion matrix :\n {confusion_matrix}")
D1 = Dense(32)(model_mitbih.output)

D2 = Dense(32)(D1)

O = Dense(2, activation='softmax')(D2)

model = Model(inputs=model_mitbih.input, outputs=O)



for layer in model.layers[:-3]:

    layer.trainable = False



for layer in model.layers[-3:]:

    layer.trainable = True

    

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(np.expand_dims(X_train_ptbdb, axis=2), 

                    y_train_ptbdb, 

                    validation_split=0.15,

                    epochs=5,

                    batch_size=128,

                    verbose=0)
plot_learning(history)
print("Trainability of the layers \n")

for layer in model.layers:

    config = layer.get_config()

    print(f"{config['name']} : {config.get('trainable')}")
D1 = Dense(32)(model_mitbih.layers[-3].output)

D2 = Dense(32)(D1)

O = Dense(2, activation='softmax')(D2)

model = Model(inputs=model_mitbih.input, outputs=O)

model.summary()
for layer in model.layers[:-3]:

    layer.trainable = False



for layer in model.layers[-3:]:

    layer.trainable = True

    

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(np.expand_dims(X_train_ptbdb, axis=2), 

                    y_train_ptbdb, 

                    validation_split=0.15,

                    epochs=100,

                    batch_size=128,

                    verbose=0)

plot_learning(history)
D1 = Dense(32)(model_mitbih.layers[-3].output)

D2 = Dense(32)(D1)

O = Dense(2, activation='softmax')(D2)

model = Model(inputs=model_mitbih.input, outputs=O)

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(np.expand_dims(X_train_ptbdb, axis=2), 

                    y_train_ptbdb, 

                    validation_split=0.15,

                    epochs=100,

                    batch_size=128,

                    verbose=0)

plot_learning(history)