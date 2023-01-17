import numpy as np

import keras as ke

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv("../input/train.csv").values



x_train = train[:, 1:].reshape(train.shape[0], 28, 28, 1)

x_train = x_train.astype(float)

x_train /= 255.0



#Use Entire Train-Set when trying this code on your own machine

x_train = x_train[:2500]
def build_model_enc():

    model = ke.models.Sequential()

    model.add(ke.layers.Conv2D(32, (5,5), padding="same", activation="relu", input_shape=(28, 28, 1)))

    model.add(ke.layers.Conv2D(64, (5,5), strides=(2,2), activation="relu", padding="same"))

    model.add(ke.layers.Conv2D(128, (5,5), strides=(2,2), activation="relu", padding="same"))

    model.add(ke.layers.Flatten())

    model.add(ke.layers.Dense(2, activation="linear"))



    return model



def build_model_dec():

    model = ke.models.Sequential()

    model.add(ke.layers.Dense(6272, input_shape=(2,)))

    model.add(ke.layers.Reshape((7, 7, 128)))

    model.add(ke.layers.Conv2D(64, (5,5), activation="relu", padding="same"))

    model.add(ke.layers.UpSampling2D())

    model.add(ke.layers.Conv2D(32, (5,5), activation="relu", padding="same"))

    model.add(ke.layers.UpSampling2D())

    model.add(ke.layers.Conv2D(1, (5,5), activation="sigmoid", padding="same"))



    return model



def build_model_disc():

    model = ke.models.Sequential()

    model.add(ke.layers.Dense(32, activation="relu", input_shape=(2,)))

    model.add(ke.layers.Dense(32, activation="relu"))

    model.add(ke.layers.Dense(1, activation="sigmoid"))

    return model
def build_model_aae():

    model_enc = build_model_enc()

    model_dec = build_model_dec()

    model_disc = build_model_disc()

    

    model_ae = ke.models.Sequential()

    model_ae.add(model_enc)

    model_ae.add(model_dec)

    

    model_enc_disc = ke.models.Sequential()

    model_enc_disc.add(model_enc)

    model_enc_disc.add(model_disc)

    

    return model_enc, model_dec, model_disc, model_ae, model_enc_disc



model_enc, model_dec, model_disc, model_ae, model_enc_disc = build_model_aae()



model_enc.summary()

model_dec.summary()

model_disc.summary()

model_ae.summary()

model_enc_disc.summary()



model_disc.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")

model_enc_disc.compile(optimizer=ke.optimizers.Adam(lr=1e-4), loss="binary_crossentropy")

model_ae.compile(optimizer=ke.optimizers.Adam(lr=1e-3), loss="binary_crossentropy")


def imagegrid(dec, epochnumber):        

        fig = plt.figure(figsize=[20, 20])

        

        for i in range(-5, 5):

            for j in range(-5,5):

                topred = np.array((i*0.5,j*0.5))

                topred = topred.reshape((1, 2))

                img = dec.predict(topred)

                img = img.reshape((28, 28))

                ax = fig.add_subplot(10, 10, (i+5)*10+j+5+1)

                ax.set_axis_off()

                ax.imshow(img, cmap="gray")

        

        fig.savefig(str(epochnumber)+".png")

        plt.show()

        plt.close(fig)

        

def settrainable(model, toset):

    for layer in model.layers:

        layer.trainable = toset

    model.trainable = toset
batchsize=50

#Set Number of Epochs to 10-20 or higher.

for epochnumber in range(1):

    np.random.shuffle(x_train)

    

    for i in range(int(len(x_train) / batchsize)):

        settrainable(model_ae, True)

        settrainable(model_enc, True)

        settrainable(model_dec, True)

        

        batch = x_train[i*batchsize:i*batchsize+batchsize]

        model_ae.train_on_batch(batch, batch)

        

        settrainable(model_disc, True)

        batchpred = model_enc.predict(batch)

        fakepred = np.random.standard_normal((batchsize,2))

        discbatch_x = np.concatenate([batchpred, fakepred])

        discbatch_y = np.concatenate([np.zeros(batchsize), np.ones(batchsize)])

        model_disc.train_on_batch(discbatch_x, discbatch_y)

        

        settrainable(model_enc_disc, True)

        settrainable(model_enc, True)

        settrainable(model_disc, False)

        model_enc_disc.train_on_batch(batch, np.ones(batchsize))

    

    print ("Reconstruction Loss:", model_ae.evaluate(x_train, x_train, verbose=0))

    print ("Adverserial Loss:", model_enc_disc.evaluate(x_train, np.ones(len(x_train)), verbose=0))

    

    

    imagegrid(model_dec, epochnumber)     

        