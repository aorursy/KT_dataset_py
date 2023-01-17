import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.utils import plot_model
x_train = pd.read_csv("../input/fashion-mnist_train.csv")

x_test = pd.read_csv("../input/fashion-mnist_test.csv")
x_train.head()
x_test.head()
y_train = x_train['label']

y_test = x_test['label']



# Drop 'label' column

x_train = x_train.drop(labels = ['label'],axis = 1) 

x_test = x_test.drop(labels = ['label'], axis = 1)



y_train.value_counts()
def getImage(df, pos):

    pixels = df.loc[df.index == pos]

    pixels = np.array(pixels, dtype = 'uint8')

    pixels = pixels.reshape((28,28)) #Hay 784, su raiz cuadrada es 28 por tanto el tamaño de la dimension sera 28x28

    

    # Plot

    plt.title('Pice of clothing')

    plt.imshow(pixels, cmap='gray')

    #sns.heatmap(pixels)

    

    plt.show()
getImage(x_train, 2) #Label6

getImage(x_train, 3) #Label0
#Buscamos valores nulos en test y train
def getNull (df):

    return df.isnull().any().describe()
print ('(Train)-Null Data: ' + str(getNull(x_train)) + '\n' +

       '(Test)-Null Data: ' + str(getNull(x_test)))
original_test = x_test.copy()
x_train = x_train/255

x_test = x_test/255
x_train.shape
x_test.shape
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

x_train = x_train.values.reshape(60000,28,28,1)

x_test = x_test.values.reshape(10000,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])



y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)
# Set the random seed

random_seed = 12



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.15, random_state=random_seed)
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> [Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(3,3)))

model.add(Dropout(0.35))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(units = 256, activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(units = 10, activation = 'softmax'))
model.summary()
rmsprop = RMSprop()
model.compile(

    optimizer = rmsprop,

    loss = 'categorical_crossentropy',

    metrics = ['accuracy']

)
%%time



model_CNN = model.fit(X_train, Y_train, batch_size = 256, epochs = 50, 

          validation_data = (X_val, Y_val), verbose = 1)
score = model.evaluate(x_test, y_test)

print('Test accuracy:', score[1])
accuracy = model_CNN.history['acc']

val_accuracy = model_CNN.history['val_acc']

loss = model_CNN.history['loss']

val_loss = model_CNN.history['val_loss']

epochs = range(len(accuracy))



# Plot training & validation accuracy values

plt.plot(epochs, accuracy, 'bo')

plt.plot(epochs, val_accuracy, 'b')

#plt.ylabel('Accuracy')

#plt.xlabel('Epoch')

plt.title('Training and validation accuracy')

plt.legend(['Train', 'Test'], loc='upper left')



plt.figure()



# Plot training & validation loss values

plt.plot(epochs, loss, 'bo')

plt.plot(epochs, val_loss, 'b')

#plt.ylabel('Loss')

#plt.xlabel('Epoch')

plt.title('Training and validation loss')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
predictions = pd.Series(model.predict_classes(x_test))

y_test_label = pd.Series([y_test[i].argmax() for i in range(len(y_test))])

report = pd.concat([y_test_label, predictions], axis = 1)

report.head()
correct = np.nonzero(predictions==y_test_label)[0]

incorrect = np.nonzero(predictions!=y_test_label)[0]
print ("Our NN missclisified: " + str(len(incorrect)) + " clothes")
target_names = ["Class {}".format(i) for i in range(y_test_label.max()+1)]

print(classification_report(y_test_label, predictions, target_names=target_names))
getImage(original_test, correct[1])

getImage(original_test, correct[2])

getImage(original_test, correct[3])
getImage(original_test, incorrect[1])

getImage(original_test, incorrect[2])

getImage(original_test, incorrect[3])
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)





    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
confusion_mtx = confusion_matrix(y_test_label, predictions) 

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
category_index = {"T-shirt/top":0, "Trouser":1, "Pullover":2,

                 "Dress":3, "Coat":4, "Sandal":5,

                 "Shirt":6, "Sneaker":7, "Bag":8,

                 "Ankle boot":9}

category_reverse_index = dict((y,x) for (x,y) in category_index.items())
def probsPredictions(cloth):

    incorrect_label = model.predict(x_test) != y_test

    for i in cloth:

        print("-"*10)

        print("Predicted category: ", category_reverse_index[model.predict_classes(x_test, verbose=0)[i]])

        print("-"*10)

        category_reverse_index[y_test_label[i]]

        probabilities = model.predict(x_test, verbose=0)

        probabilities = probabilities[i]



        print("T-shirt/top Probability: ",probabilities[category_index["T-shirt/top"]] )

        print("Trouser Probability: ",probabilities[category_index["Trouser"]] )

        print("Pullover probability: ",probabilities[category_index["Pullover"]] )

        print("Dress Probability: ",probabilities[category_index["Dress"]] )

        print("Coat Probability: ",probabilities[category_index["Coat"]] )

        print("Sandal probability: ",probabilities[category_index["Sandal"]] )

        print("Shirt Probability: ",probabilities[category_index["Shirt"]] )

        print("Sneaker Probability: ",probabilities[category_index["Sneaker"]] )

        print("Bag probability: ",probabilities[category_index["Bag"]] )

        print("Ankle boot probability: ",probabilities[category_index["Ankle boot"]] )
probsPredictions(list(range(0, 5)))