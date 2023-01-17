# # Set your own project id here

# PROJECT_ID = 'your-google-cloud-project'

# from google.cloud import storage

# storage_client = storage.Client(project=PROJECT_ID)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

print (os.getcwd())
import sys

print(sys.executable)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time

import os

from IPython.display import clear_output

%matplotlib inline



import tensorflow as tf

# import tensorflow_addons as tfa



print('tensorflow version: {}'.format(tf.__version__))

print('GPU 사용 가능 여부: {}'.format(tf.test.is_gpu_available()))

print(tf.config.list_physical_devices('GPU'))
train_pandas = pd.read_csv("../input/digit-recognizer/train.csv")

test_pandas = pd.read_csv("../input/digit-recognizer/test.csv")
train_label = train_pandas["label"]

train_data = train_pandas.drop(["label"],axis=1)

test_data = test_pandas
print(train_data)
labels_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',

              5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

columns = 5

rows = 5

fig = plt.figure(figsize=(8, 8))



for i in range(1, columns*rows+1):

    data_idx = np.random.randint(len(train_data.values))

    img = train_data.values[data_idx].reshape([28, 28])

    label = labels_map[train_label[data_idx]]



    fig.add_subplot(rows, columns, i)

    plt.title(label)

    plt.imshow(img, cmap='gray')

    plt.axis('off')

plt.show()
from sklearn.model_selection import train_test_split



train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size = 0.1, random_state=1)
def normalize_image(input_image):

    input_image = (input_image / 255)



    return input_image
def reshape_image(input_image):

    return input_image.values.reshape(-1,28,28,1)
train_data = normalize_image(train_data)

train_data = reshape_image(train_data)

val_data = normalize_image(val_data)

val_data = reshape_image(val_data)

test_data = normalize_image(test_data)

test_data = reshape_image(test_data)



train_label = train_label.values

val_label = val_label.values
batch_size = 20

max_epochs = 20

learning_rate = 1e-4

num_classes = 10

batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((train_data,train_label))

train_dataset = train_dataset.shuffle(len(list(train_dataset))) # over or equal than dataset number

print(len(list(train_dataset)))

train_dataset = train_dataset.batch(batch_size) # batch_size = 128



val_dataset = tf.data.Dataset.from_tensor_slices((val_data,val_label))

val_dataset = val_dataset.batch(batch_size)



test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

test_dataset = test_dataset.batch(batch_size)



print(train_dataset)

print(val_dataset)

print(test_dataset)
class Conv(tf.keras.Model):

    def __init__(self, num_filters, kernel_size):

        super(Conv, self).__init__()

        ## 코드 시작 ##

        self.conv = tf.keras.layers.Conv2D(filters=num_filters,kernel_size=[kernel_size, kernel_size],padding='Same')

        self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))

        self.dropout = tf.keras.layers.Dropout(0.25)

        ## 코드 종료 ##



    def call(self, inputs, training=True):

        ## 코드 시작 ##

        x = self.conv(inputs)    # self.conv forward

        x = self.bn(x)    # self.bn   forward

        x = self.relu(x)    # self.relu forward

        x = self.pool(x)    # self.pool forward

        x = self.dropout(x)

        ## 코드 종료 ##



        return x
class SimpleCNN(tf.keras.Model):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        ## 코드 시작 ##

        self.conv1 = Conv(32,5)

        self.conv2 = Conv(64,3)



        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=256,activation=tf.nn.relu)

        self.dropout = tf.keras.layers.Dropout(0.25)

        self.dense2 = tf.keras.layers.Dense(units=num_classes,activation=tf.nn.softmax)

        ## 코드 종료 ##



    def call(self, inputs, training=True):

        ## 코드 시작 ##

        x = self.conv1(inputs)    # self.conv1 forward

        x = self.conv2(x)    # self.conv2 forward

        x = self.flatten(x)    # flatten 

        x = self.dense1(x)    # self.dense1 forward

        x = self.dropout(x)

        x = self.dense2(x)    # self.dense2 forward

        ## 코드 종료 ##



        return x
model = SimpleCNN()

for images, labels in train_dataset.take(1):

    print(images.shape)

    outputs = model(images, training=False)

model.summary()
model(train_data[0:3])
loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
mean_loss = tf.keras.metrics.Mean("loss")

mean_accuracy = tf.keras.metrics.Accuracy("accuracy")
def train_step(model, images, labels):

    ## 코드 시작 ##

    with tf.GradientTape() as tape:

        predictions = model(images)    # 위의 설명 1. 을 참고하여 None을 채우세요.

        # print(predictions)

        labels_onehot = tf.one_hot(labels,num_classes)

        loss_value = loss_object(labels_onehot,predictions)     # 위의 설명 2. 를 참고하여 None을 채우세요.



    gradients = tape.gradient(loss_value,model.trainable_variables)                # 위의 설명 3. 을 참고하여 None을 채우세요.

    optimizer.apply_gradients(zip(gradients,model.trainable_variables)) # 위의 설명 4. 를 참고하여 None을 채우세요.

    ## 코드 종료 ##

    

    mean_accuracy(labels, tf.argmax(predictions, axis=1))



    return loss_value
def validation(model, val_dataset, epoch):

    print('Start validation..')

    val_mean_loss = tf.keras.metrics.Mean("val_loss")

    val_mean_accuracy = tf.keras.metrics.Accuracy("val_acc")



    for step, (images, labels) in enumerate(val_dataset):

        predictions = model(images,training=False)       # 위의 설명 1. 을 참고하여 None을 채우세요.

        labels_onehot = tf.one_hot(labels,num_classes)

        val_loss_value = loss_object(labels_onehot,predictions)    # 위의 설명 2. 를 참고하여 None을 채우세요.

        val_mean_loss(val_loss_value)

        val_mean_accuracy(labels, tf.argmax(predictions, axis=1))

        

    print('Validation #{} epoch  Average Loss: {:.4g}  Accuracy: {:.4g}%\n'.format(

        epoch, val_mean_loss.result(), val_mean_accuracy.result() * 100))



    return val_mean_loss.result(), val_mean_accuracy.result()
def test(model, test_dataset):

    print('Start test..')

    test_mean_accuracy = tf.keras.metrics.Accuracy("test_acc")



    for step, (images, labels) in enumerate(test_dataset):

        predictions = model(images)

        test_mean_accuracy(labels, tf.argmax(predictions, axis=1))

        

    print('Test accuracy: {:.4g}%'.format(test_mean_accuracy.result() * 100))



    return test_mean_accuracy.result()
def save_model(model, epoch, train_dir):

    model_name = 'my_model_' + str(epoch)

    model.save_weights(os.path.join(train_dir, model_name))
train_dir = os.path.join('./train/exp2')

print_steps = 25

val_epoch = 1
def main(model, train_dataset, val_dataset, val_epoch, print_steps, save_dir):

    print('Start training..')

    num_batches_per_epoch = len(list(train_dataset))

    global_step = 0

    best_acc = 0.



    for epoch in range(max_epochs):

        for step, (images, labels) in enumerate(train_dataset):

            start_time = time.time()

            # train_step 함수 사용하여 loss 구하기

            ## 코드 시작 ##

            loss_value = train_step(model,images,labels)

            ## 코드 종료 ##

            mean_loss(loss_value)

            global_step += 1



            if global_step % print_steps == 0:

                duration = time.time() - start_time

                examples_per_sec = batch_size / float(duration)

                print("Epochs: [{}/{}] step: [{}/{}] loss: {:.4g} acc: {:.4g}%  ({:.2f} examples/sec; {:.3f} sec/batch)".format(

                    epoch+1, max_epochs, step+1, num_batches_per_epoch,

                    mean_loss.result(), mean_accuracy.result() * 100, examples_per_sec, duration))

        

        # clear the history            

        mean_loss.reset_states()

        mean_accuracy.reset_states()



        if (epoch + 1) % val_epoch == 0:

            # validation 함수 사용하여 검증하기

            # 여기서 epoch는 0부터 시작하기 때문에 + 1 을 해주시길 바랍니다.

            ## 코드 시작 ##

            val_mean_loss, val_mean_accuracy = validation(model,val_dataset,epoch+1)

            ## 코드 종료 ##

            if val_mean_accuracy > best_acc:

                print('Best performance at epoch: {}'.format(epoch + 1))

                print('Save in {}\n'.format(save_dir))

                best_acc = val_mean_accuracy

                save_model(model, epoch+1, save_dir)



    print('training done..')
main(model, train_dataset, val_dataset, val_epoch, print_steps, save_dir=train_dir)
test_batch_size = 25

batch_index = np.random.choice(

    len(val_data), size=test_batch_size, replace=False)



print(batch_index)



batch_xs = val_data[batch_index]

batch_ys = val_label[batch_index]

y_pred_ = model(batch_xs, training=False)



print(batch_xs.shape)

print(batch_ys.shape)

print(y_pred_)



fig = plt.figure(figsize=(10, 10))

for i, (px, py, y_pred) in enumerate(zip(batch_xs, batch_ys, y_pred_)):

    p = fig.add_subplot(5, 5, i+1)

    if np.argmax(y_pred) == py:

        p.set_title("{}".format(labels_map[py]), color='blue')

    else:

        p.set_title("{}/{}".format(labels_map[np.argmax(y_pred)],

                                   labels_map[py]), color='red')

    p.imshow(px.reshape(28, 28), cmap="gray")

    p.axis('off')
for step, (images) in enumerate(test_dataset):

        predictions = model(images)

        print(images.shape)
tmp_list = []

tmp_list = model(test_data[0:10])

tmp_list = np.argmax(tmp_list, axis=1)

print(tmp_list)

tmp_list = np.append(tmp_list,[1,2,3,4,5])

print(tmp_list)
def test(model, test_dataonly):

    print('Start test..')

    predictions = []

    

    tmp_list = []



    for step, (images) in enumerate(test_dataonly):

        predictions = np.append(predictions, np.argmax(model(images), axis=1)).astype(int)

        

    print("test done...")

    

    return predictions
predictions = test(model, test_dataset)
results = pd.Series(predictions,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)