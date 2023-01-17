!pip install tensorflow keras numpy mnist matplotlib
import mnist #Dataset
import numpy as np
import matplotlib.pyplot as plt #Graph
from keras.models import Sequential #ANN
from keras.layers import Dense #Layers in ANN
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
#load dataset
train_images = mnist.train_images() #training data images
train_labels = mnist.train_labels() #training data labels
test_images = mnist.test_images() #testing data images
test_labels = mnist.test_labels() #testing data labels
#normalization of data
#normalize pixels from the range (0, 255) to (0, 1) to train our network easily.
train_images = train_images / 255
test_images = test_images / 255

#Flatten the image from 28 x 28 to 1-d array of size 784 to pass it to the neural network.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(test_images.shape) #(10000, 784)
print(train_images.shape) #(60000, 784)
#Building the model
model = Sequential()

#input layer
model.add(Dense(256, activation = 'tanh', input_dim = 784))

#hidden layer
model.add(Dense(7, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

#output layer
model.add(Dense(10, activation = 'softmax'))
#Compiling the model
#loss function measures how well the neural network worked
#and then tries to improve it using optimizer.
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#train the model
model.fit(train_images, to_categorical(train_labels), #to_categorical converts 2 into [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
      epochs = 25, #iterations
      batch_size = 64) #number of sample per gradient update during training

#evaluate the model
model.evaluate(test_images, to_categorical(test_labels))
model.save_weights('model.h5')
#predict on the first five test images
predictions = model.predict(test_images[:6])
print('Actual labels: ' , test_labels[:6])
print('predictions by our model: ' , np.argmax(predictions, axis = 1))
#ref https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
for i in range(0, 3):
    img = test_images[i]
    img = np.array(img, dtype = 'float')
    pixels = img.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()

predictions = model.predict(test_images)
print(confusion_matrix(test_labels, np.argmax(predictions, axis = 1)))

plot_confusion_matrix(confusion_matrix(test_labels, np.argmax(predictions, axis = 1)), 
                      normalize    = True,
                      target_names = ['0', '1', '2',  '3',  '4', '5',  '6', '7',  '8', '9'],
                      title        = "Confusion Matrix, Normalized")
import pandas as pd
sub = pd.DataFrame(np.argmax(predictions, axis = 1))
sub.index.name = 'ImageId'
sub.index += 1
sub.columns = ['Label']
sub.to_csv("submission_ffann.csv", header = True) 