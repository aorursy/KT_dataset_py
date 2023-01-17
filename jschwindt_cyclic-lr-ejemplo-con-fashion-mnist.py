import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from plot_losses import PlotLosses

from cyclic_lr import CyclicLR
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint 

from keras.models import Sequential

from keras.layers.core import Dense, Flatten
!ls ../input/fashion-mnist-itba/fashion-mnist-itba-lab-ml-2018b/
datapath = '../input/fashion-mnist-itba/fashion-mnist-itba-lab-ml-2018b/'

X = np.load(datapath + 'train_images.npy')

y = np.loadtxt(datapath + 'train_labels.csv', delimiter=',', skiprows=1)

x_test = np.load(datapath + 'test_images.npy')
print(f'X.shape => {X.shape}')

print(f'x_test.shape => {x_test.shape}')

print(f'y.shape => {y.shape}')
X = X.astype('float32') / 255

x_test = x_test.astype('float32') / 255
y_categorical = to_categorical(y)

print(f'y_categorical.shape => {y_categorical.shape}')

print(y_categorical[0])
output_size = y_categorical.shape[1]



def create_model():

    model = Sequential()

    model.add(Flatten(input_shape=X.shape[1:]))

    model.add(Dense(40, activation='sigmoid'))

    model.add(Dense(output_size, name='Salida', activation='softmax'))

#     model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
batch_size = 64

epochs = 50

lr = 0.01
cyclic_lr = CyclicLR(

                base_lr=1e-9,

                max_lr=.1,

                step_size=X.shape[0]//batch_size,

                mode='triangular'

            )

model = create_model()

model.fit(X, y_categorical, epochs=1, batch_size=batch_size, verbose=1, callbacks=[cyclic_lr])
plt.semilogx(cyclic_lr.history['lr'], cyclic_lr.history['loss'])

plt.show()
x_train, x_valid, y_train, y_valid = train_test_split(X, y_categorical)
epochs = 1



# Callbacks

## Callback para graficar

plot_losses = PlotLosses(plot_interval=1, evaluate_interval=20, x_val=x_valid, y_val_categorical=y_valid)



## Callback para guardar pesos

checkpointer = ModelCheckpoint(filepath=f'checkpoint.hdf5', verbose=1,

                               save_best_only=True, monitor='val_acc', mode='max')

## Callback para Cyclic Learning Rate

cyclic_lr = CyclicLR(

                base_lr=1e-3,

                max_lr=1e-2,

                step_size=x_train.shape[0] // batch_size * 4,

                mode='triangular'

            )



model = create_model()



model.fit(x_train, 

    y_train,

    epochs=epochs, batch_size=batch_size, 

    verbose=2, 

    validation_data = (x_valid, y_valid),

    callbacks=[plot_losses, checkpointer, cyclic_lr],

)



loss, acc = model.evaluate(x_valid, y_valid)

print(f'ACC: {acc}')
model = create_model()

model.load_weights(f'checkpoint.hdf5')

final_prediction = model.predict(x_test)

    

test_labels = np.argmax(final_prediction, axis = 1)
import pandas as pd

df = pd.DataFrame(data={"Category": test_labels}).astype(int)

df.to_csv("./submission.csv", sep=',',index=True,  index_label='Id')