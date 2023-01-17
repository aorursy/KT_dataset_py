from numpy.random import seed 
seed(888) 
import tensorflow as tf
tf.random.set_seed(404)
import os 
from IPython.display import clear_output
import numpy as np 
import tensorflow as tf 
import itertools

import keras 
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

from IPython.display import display 
from keras.preprocessing.image import array_to_img 
from keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix

from time import strftime

import matplotlib.pyplot as plt
%matplotlib inline
LOG_DIR = 'tensorboard_cifar_logs/'

LABEL_NAMES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

VALIDATION_SIZE = 10000
SMALL_TRAIN_SIZE = 1000

IMAGE_WIDTH = 32 
IMAGE_HEIGHT = 32 
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH 
COLOR_CHANNELS = 3 
TOTAL_INPUTS = IMAGE_PIXELS * COLOR_CHANNELS

NR_CLASSES = 10 
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
type(cifar10)
type(x_train_all)
x_train_all[0]
pic = array_to_img(x_train_all[0])
display(pic)
y_train_all[7][0]
LABEL_NAMES[y_train_all[7][0]]
plt.imshow(x_train_all[4])
plt.xlabel(LABEL_NAMES[y_train_all[4][0]])
plt.show()
plt.figure(figsize=(15,5))

for i in range(10): 
    plt.subplot(1,10,i+1)
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(LABEL_NAMES[y_train_all[i][0]])
    plt.imshow(x_train_all[i])

x_train_all[0].shape
nr_images, x, y, c = x_train_all.shape
print(f'images = {nr_images} \t| width = {x} \t| height = {y} \t| channels = {c}')
type(x_train_all[0][0][0][0])
x_train_all, x_test = x_train_all / 255.0, x_test / 255.0 
x_train_all.shape
x_train_all = x_train_all.reshape(len(x_train_all), 32*32*3)
x_train_all.shape
x_test = x_test.reshape(len(x_test), 32*32*3)
print(x_test.shape)
x_train = x_train_all[VALIDATION_SIZE:]
y_train = y_train_all[VALIDATION_SIZE:] 

x_val = x_train_all[:VALIDATION_SIZE]
y_val = y_train_all[:VALIDATION_SIZE]
x_train_xs = x_train[:SMALL_TRAIN_SIZE]
y_train_xs = y_train[:SMALL_TRAIN_SIZE]

model_1 = Sequential([
    Dense(units=128, input_dim=TOTAL_INPUTS, activation='relu'), 
    Dense(units=64, activation='relu'), 
    Dense(units=16, activation='relu'),
    Dense(units=10, activation='softmax')
])

model_1.compile(
    optimizer = 'adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model_2 = Sequential() 
model_2.add(Dropout(0.2, seed=42,input_shape=(TOTAL_INPUTS,)))
model_2.add(Dense(units=128, activation='relu'))
model_2.add(Dense(units=64, activation='relu'))
model_2.add(Dense(units=16, activation='relu'))
model_2.add(Dense(units=10, activation='softmax'))

model_2.compile(
    optimizer = 'adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model_3 = Sequential()
model_3.add(Dropout(0.2, seed=42,input_shape=(TOTAL_INPUTS,)))
model_3.add(Dense(units=128, activation='relu'))
model_3.add(Dropout(0.25, seed=42))
model_3.add(Dense(units=64, activation='relu'))
model_3.add(Dense(units=16, activation='relu'))
model_3.add(Dense(units=10, activation='softmax'))

model_3.compile(
    optimizer = 'adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
type(model_1)
model_1.summary() 
def get_tensorboard(model_name):

    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths = os.path.join(LOG_DIR, folder_name)

    try:
        os.makedirs(dir_paths)
    except OSError as err:
        print(err.strerror)
    else:
        print('Successfully created directory')

    return TensorBoard(log_dir=dir_paths)
class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

plot_losses = TrainingPlot()
samples_per_batch = 1000 
%%time 
nr_epochs = 150 
model_1.fit(x_train_xs, y_train_xs, epochs=nr_epochs, batch_size=samples_per_batch,verbose=0, 
           validation_data=(x_val,y_val))
%%time 
nr_epochs = 150 
model_2.fit(x_train_xs, y_train_xs, epochs=nr_epochs, batch_size=samples_per_batch,verbose=0, 
           validation_data=(x_val,y_val))
%%time 
nr_epochs = 150 
model_3.fit(x_train_xs, y_train_xs, epochs=nr_epochs, batch_size=samples_per_batch,verbose=0, 
           validation_data=(x_val,y_val))
x_val[0].shape
test = np.expand_dims(x_val[0], axis=0) 
test.shape
np.set_printoptions(precision=3)

model_2.predict(test)
model_2.predict(x_val) .shape
model_3.predict_classes(test)
y_val[0]
for number in range(10): 
    test_img = np.expand_dims(x_val[number], axis=0)
    predicted_val = model_3.predict_classes(test_img)[0] 
    print(f'Actual value: {y_val[number][0]} vs. predicted: {predicted_val}')
model_3.metrics_names
test_loss, test_accuracy = model_3.evaluate(x_test, y_test)
print(f'Test loss is {test_loss:0.3} and test accuracy is {test_accuracy:0.1%}')
predictions = model_3.predict_classes(x_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
conf_matrix.shape
nr_rows = conf_matrix.shape[0]
nr_cols = conf_matrix.shape[1]
conf_matrix.max() 
conf_matrix.min() 
conf_matrix[0]
plt.figure(figsize=(7,7), dpi= 200) 
plt.imshow(conf_matrix, cmap=plt.cm.Greens)
plt.title('Confusion Matrix', fontsize=16) 
plt.ylabel('Actual Labels', fontsize=12)
plt.xlabel('Predicted Labels', fontsize=12)

tick_marks = np.arange(NR_CLASSES)
plt.yticks(tick_marks, LABEL_NAMES)
plt.xticks(tick_marks, LABEL_NAMES)
plt.colorbar() 

for i, j in itertools.product(range(nr_rows), range(nr_cols)):
    plt.text(j, i, conf_matrix[i, j], horizontalalignment='center',
            color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
    
    
plt.show()
# True Posiives 
np.diag(conf_matrix)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
recall
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
precision
avg_recall = np.mean(recall)
print(f'Model 2 recall score is {avg_recall:.2%}')
avg_precision = np.mean(precision)
print(f'Model 2 precision score is {avg_precision:.2%}')

f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
print(f'Model 2 f score is {f1_score:.2%}')
