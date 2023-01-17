import tensorflow as tf


def train_mnist():

    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self,epoch,logs={}):

            if(logs.get('accuracy')>0.99):

                print('\nReached 99% accuracy so cancelling training!')

                self.model.stop_training=True

                

    callbacks=myCallback()



    mnist = tf.keras.datasets.mnist



    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train=x_train/256.0

    x_test=x_test/256.0

    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512,activation=tf.nn.relu),

        tf.keras.layers.Dense(1024,activation=tf.nn.relu),

        tf.keras.layers.Dense(10,activation=tf.nn.softmax)

    ])



    model.compile(optimizer='adam',

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    

    # model fitting

    history = model.fit(

        x_train,y_train,

        epochs=10,

        callbacks=[callbacks]

    )

    # model fitting

    return history.epoch, history.history['accuracy'][-1]
train_mnist()