import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from IPython.display import clear_output

from matplotlib import pyplot as plt



from keras.callbacks import Callback

from keras.models import Model

import keras.layers as l
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []

        



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('Log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="acc")

        ax2.plot(self.x, self.val_acc, label="val_acc")

        ax2.legend()

        

        plt.show()

        

        

plot = PlotLearning()
data = pd.read_csv('../input/Iris.csv')

data = data.drop(['Id'] ,axis=1)

data.columns
encoder = LabelBinarizer()

unique_labels = np.unique(data.Species)

transfomed_label = encoder.fit_transform(unique_labels)

for label, encoding in zip(unique_labels, transfomed_label):

    print(label,'==>',encoding)
y = encoder.transform(data.Species.values)



data = data.drop(['Species'] ,axis=1)



X = data.values



x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, shuffle= True)
input_layer = l.Input((4,))



hidden = l.Dense(4, activation=None)(input_layer)

hidden = l.BatchNormalization()(hidden)

hidden = l.Activation('tanh')(hidden)



hidden = l.Dense(8, activation=None)(hidden)

hidden = l.BatchNormalization()(hidden)

hidden = l.Activation('tanh')(hidden)



hidden = l.Dense(8, activation=None)(hidden)

hidden = l.BatchNormalization()(hidden)

hidden = l.Activation('tanh')(hidden)



output = l.Dense(len(unique_labels), activation='softmax')(hidden)



model = Model(input_layer, output)

model.compile(

    optimizer='Adam',

    loss='categorical_crossentropy',

    metrics=['acc']

)



model.summary()

model.fit(x_train,y_train, epochs=100, validation_data=(x_valid, y_valid), shuffle=True, callbacks=[plot])
model.evaluate(x_valid, y_valid)
model.save('Iris Flower Classsifier.h5')