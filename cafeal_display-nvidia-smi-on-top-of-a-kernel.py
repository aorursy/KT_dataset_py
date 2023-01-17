import multiprocessing

import subprocess

from IPython import display

import time



def check_gpu_usage():

    while True:

        display.clear_output(wait=True)

        print(subprocess.check_output('nvidia-smi').decode().strip())

        time.sleep(1)

runner = multiprocessing.Process(target=check_gpu_usage)

runner.start()
import tensorflow.keras as K

(train_X, train_y), (test_X, test_y) = K.datasets.mnist.load_data()

train_X, test_X = train_X[:, :, :, None], test_X[:, :, :, None]

train_y, test_y = K.utils.to_categorical(train_y), K.utils.to_categorical(test_y)



model = K.Sequential([

    K.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),

    K.layers.MaxPool2D(2),

    K.layers.Conv2D(16, 3, activation='relu'),

    K.layers.MaxPool2D(2),

    K.layers.Conv2D(8, 3, activation='relu'),

    K.layers.Flatten(),

    K.layers.Dense(10, activation='softmax'),

])



sgd = K.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd)

result = model.fit(train_X, train_y, batch_size=64, epochs=10)