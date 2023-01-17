import tensorflow as tf

import tensorflow.keras.datasets as datasets

import matplotlib.pyplot as plt
class MyLayer(tf.keras.layers.Layer):

    def __init__(self, units, input_dim, activation=tf.keras.activations.relu):

        super(MyLayer, self).__init__()

        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)

        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

        self.activation = activation

        self.times = 1000

        

        weight = self.w

        bias = self.b

        g_activation = self.activation

    

    def call(self, inputs):

#         return self.activation(

#             tf.matmul( self.activation(tf.matmul(inputs, self.w) + self.b ), self.w) + self.b)



#         d = self.activation( tf.matmul( self.activation(tf.matmul(inputs, self.w ) + self.b ), self.w) + self.b)

        d = self.activation(tf.matmul(inputs, self.w) + self.b)

        self.function2(d)

        print("d: ", d)

        return d

    

    def function2(self, a):

        self.result_tmp = a

        for i in range(self.times):

            self.result = self.function3(self.result_tmp)

            self.result_tmp = self.result  

        return self.result_tmp

    

    def function3(self, inputs_):

        return self.activation(tf.matmul(inputs_, self.w) + self.b)
def plotmodelhistory(history): 

    fig, axs = plt.subplots(1,2,figsize=(15,5)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy']) 

    axs[0].plot(history.history['val_accuracy']) 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss']) 

    axs[1].plot(history.history['val_loss']) 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper left')

    plt.show()
EPOCH = 10
def run_simple_mlp(x_train, y_train, x_test, y_test):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(3072, activation='relu'),

        tf.keras.layers.Dense(1000, activation='relu'),

        tf.keras.layers.Dense(100, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax'),

    ])

    

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    

    history = model.fit(x_train, y_train,

          batch_size=32,

          epochs=EPOCH,

          validation_data=(x_test, y_test),

          shuffle=True)

    print(model.summary())

#     x = model.fit(x_train, y_train, batch_size=32, epochs=EPOCH, verbose=0)



    

    # list all data in history

    print(history.history.keys())



    plotmodelhistory(history)



    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("simple_Train loss: %f, test loss: %f" % (train_loss, test_loss))

    print("simple_Train acc: %f, test acc: %f" % (train_acc, test_acc))

    return history
def run_shared_weight_mlp(x_train, y_train, x_test, y_test):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(3072, activation='relu'),

        tf.keras.layers.Dense(1000, activation='relu'),

        MyLayer(1000, 1000),

        tf.keras.layers.Dense(100, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax'),

    ])





    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

   

    history = model.fit(x_train, y_train,

          batch_size=32,

          epochs=EPOCH,

          validation_data=(x_test, y_test),

          shuffle=True)

    print(model.summary())

#     x = model.fit(x_train, y_train, batch_size=32, epochs=EPOCH, verbose=0)



    

    # list all data in history

    print(history.history.keys())



    plotmodelhistory(history)



    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("shared_weight_Train loss: %f, test loss: %f" % (train_loss, test_loss))

    print("shared_weight_Train acc: %f, test acc: %f" % (train_acc, test_acc))
history = None
def run_exp2():

    #here we use mnsit which can be downloaded online 

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()



    run_shared_weight_mlp(x_train, y_train, x_test, y_test)

    run_simple_mlp(x_train, y_train, x_test, y_test)
if __name__ == '__main__':

    run_exp2()