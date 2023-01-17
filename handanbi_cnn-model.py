import glob

import numpy as np

import os.path as path

from scipy import misc
IMAGE_PATH = '../input/planesnet/planesnet/planesnet'

file_paths = glob.glob(path.join(IMAGE_PATH, '*.png'))
#file_paths 

#잘 불러와졌는지 확인
images = [misc.imread(path) for path in file_paths]

images = np.asarray(images)
# Get image size

image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])

print(image_size)
#Sclae

images = images/255
n_images = images.shape[0]

labels=np.zeros(n_images)

for i in range(n_images):

    filename = path.basename(file_paths[i])[0]

    labels[i]=int(filename[0])

TRAIN_TEST_SPLIT=0.9
split_index=int(TRAIN_TEST_SPLIT*n_images)

shuffled_indices=np.random.permutation(n_images)

train_indices=shuffled_indices[0:split_index]

test_indices=shuffled_indices[split_index:]
x_train=images[train_indices,:,:]

y_train=labels[train_indices]

x_test=images[test_indices,:,:]

y_test=labels[test_indices]
import matplotlib.pyplot as plt
def visualize_data(positive_images,negative_images):

    figure=plt.figure()

    count=0

    

    for i in range(positive_images.shape[0]):

        count+=1

        figure.add_subplot(2,positive_images.shape[0],count)

        plt.imshow(positive_images[i,:,:])

        plt.axis('off')

        plt.title("1"),

        

        figure.add_subplot(1,positive_images.shape[0],count)

        plt.imshow(negative_images[i,:,:])

        plt.axis('off')

        plt.title("0")

    

    plt.show()
N_TO_VISUALIZE=10



positive_example_indices=(y_train == 1)

positive_examples = x_train[positive_example_indices,:,:]

positive_examples = positive_examples[0:N_TO_VISUALIZE,:,:]



negative_example_indices=(y_train == 0)

negative_examples = x_train[negative_example_indices,:,:]

negative_examples = negative_examples[0:N_TO_VISUALIZE,:,:]



visualize_data(positive_examples, negative_examples)
from keras.models import Sequential

from keras.layers import Activation,Dropout,Flatten,Dense,Conv2D,MaxPooling2D

from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.metrics import accuracy_score,f1_score

from datetime import datetime
N_LAYERS = 4
def cnn(size, n_layers):

    # INPUTS

    # size - size of the input images

    # n_layers - number of layers

    # OUTPUTS

    # model - compiled CNN

    

    # Define hyperparameters

    MIN_NEURONS = 20

    MAX_NEURONS = 120

    KERNEL = (3, 3)

    

    # Determine the # of neurons in each convolutional layer

    steps = np.floor(MAX_NEURONS / (n_layers + 1))

    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)

    neurons = neurons.astype(np.int32)

    

    # Define a model

    model = Sequential()

    

    # Add convolutional layers

    for i in range(0, n_layers):

        if i == 0:

            shape = (size[0], size[1], size[2])

            model.add(Conv2D(neurons[i], KERNEL, input_shape = shape))

        else:

            model.add(Conv2D(neurons[i], KERNEL))

            model.add(Activation('relu'))

    

    # Add max pooling layer

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())

    model.add(Dense(MAX_NEURONS))

    model.add(Activation('relu'))

    

    # Add output layer

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    

    # Compile the model

    model.compile(loss = 'binary_crossentropy',

                  optimizer = 'adam',

                  metrics = ['accuracy'])

    

    # Print a summary of the model

    model.summary()

    

    return model

            
model = cnn(size=image_size,n_layers=N_LAYERS)
EPOCHS=150

BATCH_SIZE=200
PATIENCE = 10

early_stopping = EarlyStopping(monitor='loss',min_delta=0,patience=PATIENCE,verbose=0,mode='auto')
LOG_DIRECTORY_ROOT = ''

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)

tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)
#그래프그리는건데 안돌아감ㅠ

#(tfKeras)founder@hilbert:~/tfKeras$tensorboard --logdir/home/founder/tfKeras/tensorlog

#TensorBoard 1.12.0 at http://hilbert:6006(Press CTRL +C to quit)
callbacks=[early_stopping,tensorboard]
model.fit(x_train, y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks,verbose=0)
test_predictions=model.predict(x_test)

test_predictions=np.round(test_predictions)
accuracy=accuracy_score(y_test,test_predictions)

print("Accuracy:"+str(accuracy))
import matplotlib.pyplot as plt

def visualize_incorrect_labels(x_data,y_real,y_predicted):

    count=0

    figure=plt.figure()

    incorrect_label_indices=(y_real !=y_predicted)

    y_real=y_real[incorrect_label_indices]

    y_predicted=y_predicted[incorrect_label_indices]

    x_data=x_data[incorrect_label_indices,:,:,:]

    

    maximum_square=np.ceil(np.sqrt(x_data.shape[0]))

    

    for i in range(x_data.shape[0]):

        count+=1

        figure.add_subplot(maximum_square,maximum_square,count)

        plt.imshow(x_data[i,:,:,:])

        plt.axis('off')

        plt.title("Predicted:"+str(int(y_predicted[i]))+"Real:"+str(int(y_real[i])),fontsize=10)

        

    plt.show()

visualize_incorrect_labels(x_test,y_test,np.asarray(test_predictions).ravel())
from keras.models import load_model

model.save('airplane_image_classification_model.h5')

print("Saved classifier to disk")
from keras.utils.vis_utils import plot_model



plot_model(model, show_shapes=True, show_layer_names=True)
from keras.models import load_model

model=load_model('airplane_image_classification_model.h5')