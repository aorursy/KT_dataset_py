#Bibliotecas necessárias



import numpy as np

import tensorflow as tf

from random import randint

import matplotlib.pyplot as plt
def generate_sequence(number_of_features, time_steps):

    return [randint(0,number_of_features-1) for i in range(time_steps)]
def one_hot_encode(number_of_features,x):

    max_ = number_of_features

    tmp = np.zeros(shape=(len(x),max_))

    

    for i in range(len(x)):

        tmp[i][x[i]] = 1

    

    return tmp



def one_hot_decode(x):

    tmp = []

    for encode in x:

        tmp.append(np.argmax(encode))

        

    return tmp



sequence = generate_sequence(10,5)

print(sequence)

encoded = one_hot_encode(10,sequence)

decoded = one_hot_decode(encoded)

print(decoded)
def generate_sample(size,time_steps, number_of_features, out_index):

    

    X = np.zeros(shape=(size,time_steps,number_of_features))

    Y = np.zeros(shape=(size,number_of_features))

    

    for i in range(size):

        

        #generate the sequence

        sequence = generate_sequence(number_of_features,time_steps)

        #one hot encode it

        encoded = one_hot_encode(number_of_features,sequence)

        #reshape it to be 3D (1 sample, length timesteps, nr_features features)

        x = encoded.reshape((1, time_steps, number_of_features))

        y = encoded[out_index].reshape(1, number_of_features)

        

        X[i] = x

        Y[i] = y

        



    return X, Y
def show_history(history):

    print(history.history.keys())



    # summarize history for accuracy

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy with 16 units')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()
number_of_features = 10

time_steps = 5

out_index = 2

size = 5000

X,y = generate_sample(size,time_steps,number_of_features,out_index)
def build_model(time_steps,nr_features,units):

    

    tf.keras.backend.clear_session()

    

    model = tf.keras.Sequential()

    

    model.add(tf.keras.layers.LSTM(units = units, input_shape= (time_steps, nr_features), return_sequences = False))

    

    model.add(tf.keras.layers.Dense(nr_features,activation='softmax'))



    model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics=["accuracy"])

    

    return model
def fit(epochs,model,X,y):

    

    from keras.callbacks.callbacks import EarlyStopping

    

    callbacks=[

            EarlyStopping(monitor='accuracy', patience=1, verbose=2, mode='max')

    ]

    

    history = model.fit(X,y,shuffle=False,verbose=2,epochs=epochs,callbacks=callbacks, validation_split = 0.2)

    

    return history
model  = build_model(time_steps, number_of_features,16)

model.summary()

history = fit(5000,model,X,y)
show_history(history)
nr_of_features = 10

steps = 5

out_index = 2

size = 5000



X,y = generate_sample(size,steps,nr_of_features,out_index)



model  = build_model(steps, nr_of_features, 128)

model.summary()



history = fit(5000,model,X,y)
show_history(history)
nr_of_features = 50

steps = 20

out_index = 2

size = 5000



X,y = generate_sample(size,steps,nr_of_features,out_index)



model  = build_model(steps, nr_of_features, 16)



model.summary()



from keras.callbacks.callbacks import EarlyStopping



#Vou mudar o campo patience, e monotorizar o validation data como o train data

callbacks=[

    EarlyStopping(monitor='accuracy', patience=4, verbose=2, mode='max')

]

    

history = model.fit(X,y,shuffle=False,verbose=2,epochs=5000, callbacks=callbacks ,validation_split = 0.2)

#history = fit(5000,model,X,y)
show_history(history)
nr_of_features = 50

steps = 20

out_index = 2

size = 5000



X,y = generate_sample(size,steps,nr_of_features,out_index)



model  = build_model(steps, nr_of_features, 128)

model.summary()



from keras.callbacks.callbacks import EarlyStopping



#Vou mudar o campo patience, e monotorizar o validation data como o train data

callbacks=[

    EarlyStopping(monitor='accuracy', patience=4, verbose=2, mode='max')

]

    

history = model.fit(X,y,shuffle=False,verbose=2,epochs=5000, callbacks=callbacks ,validation_split = 0.2)

#history = fit(5000,model,X,y)
show_history(history)
def build_model(time_steps,nr_features,units):

    

    tf.keras.backend.clear_session()

    

    model = tf.keras.Sequential()

    

    

    

    

    model.add(tf.keras.layers.LSTM(units = units,

                                   input_shape= (time_steps, nr_features),

                                   dropout = 0.25,

                                   recurrent_dropout = 0.25,

                                   return_sequences = True)

             )

    

    model.add(tf.keras.layers.LSTM(units = units,

                                   dropout = 0.25,

                                   recurrent_dropout = 0.25,

                                   return_sequences = False)

             )

                                 



    

    model.add(tf.keras.layers.Dense(nr_features,activation='softmax'))    





    model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics=["accuracy"])

    

    return model



def fit(epochs,model,X,y):

    

    from keras.callbacks.callbacks import EarlyStopping

    

    callbacks=[

            EarlyStopping(monitor='val_accuracy', patience=10, verbose=2, mode='max')

    ]

    

    history = model.fit(X,y,shuffle=False,verbose=2,epochs=epochs, validation_split = 0.2, callbacks=callbacks)

    

    return history
nr_of_features = 50

steps = 20

out_index = 2

size = 5000





X,y = generate_sample(size,steps,nr_of_features,out_index)



model  = build_model(steps, nr_of_features, 128)

model.summary()



history = fit(5000,model,X,y)
show_history(history)