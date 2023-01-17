!pip install 'labml'
import tensorflow as tf

from labml import experiment
from labml.utils.keras import LabMLKerasCallback


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    YOUR_TOKEN = ''

    assert YOUR_TOKEN, 'please generate a token here https://web.lab-ml.com/'

    with experiment.record(name='MNIST Keras',token=YOUR_TOKEN):
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test),
                  callbacks=[LabMLKerasCallback()], verbose=None)


if __name__ == '__main__':
    main()