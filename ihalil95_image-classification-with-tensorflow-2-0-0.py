!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow.keras import Model
tf.__version__

dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = x_train/255.0

x_train = x_train[..., tf.newaxis]  # adding a channel dimension



x_test = x_test/255.0

x_test = x_test[..., tf.newaxis]  # adding a channel dimension
traindataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(128)

testdataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
class ConvNet(Model):

    def __init__(self):

        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')

        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.d1 = tf.keras.layers.Dense(256, activation='relu')

        self.d2 = tf.keras.layers.Dense(128, activation='relu')

        self.d3 = tf.keras.layers.Dense(10, activation='softmax')



    def call(self, x):

        x = self.conv1(x)

        x = self.pool(x)

        x = self.flatten(x)

        x = self.d1(x)

        x = self.d2(x)

        x = self.d3(x)

        return x
model = ConvNet()
epochs = 5

learningrate = 0.003
optimizer = tf.optimizers.Adam(learningrate)

lossf = tf.losses.SparseCategoricalCrossentropy()
train_loss = tf.metrics.Mean(name='train_loss')

train_accuracy = tf.metrics.SparseCategoricalAccuracy()



test_loss = tf.metrics.Mean(name='test_loss')

test_accuracy = tf.metrics.SparseCategoricalAccuracy()
@tf.function

def train(trainimages, trainlabels):

    with tf.GradientTape() as tape:

        trainpredictions = model(trainimages)

        trainloss = lossf(trainlabels, trainpredictions)

    gradients = tape.gradient(trainloss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



    train_loss(trainloss)

    train_accuracy(trainlabels, trainpredictions)
@tf.function

def test(testimages, testlabels):

    testpredictions = model(testimages)

    testloss = lossf(testlabels, testpredictions)



    test_loss(testloss)

    test_accuracy(testlabels, testpredictions)
for e in range(epochs):



    for images, labels in traindataset:

        train(images, labels)



    for timages, tlabels in testdataset:

        test(timages, tlabels)



    print(f"Epoch {e+1}", end='  ')

    print(f"Train loss: {train_loss.result()}, Test loss: {test_loss.result()}", end=',  ')

    print(f"Train accuracy: {100*train_accuracy.result()}, Test accuracy: {100*test_accuracy.result()}", end='\n')