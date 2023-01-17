# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import tensorflow as tf 

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,LeakyReLU

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")   

train = pd.read_csv("../input/train.csv")
train.head()
x_train = train.drop(["label"],axis=1)

y_train = train["label"]
x_train.isnull().any().describe()
test.isnull().any().describe()
# Normalization [-1,1]

m_train = np.mean(x_train)

X_train = (x_train - m_train) / 255.0

m_test = np.mean(test)

ttest = (test - m_test) / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

ttest = test.values.reshape(-1,28,28,1)
# one-hot encoding 

y_train = tf.keras.utils.to_categorical(y_train , num_classes = 10)
#fix a seed for the random number generator

random_seed = 7 

X_train , X_val , y_train , y_val = train_test_split (X_train,y_train, test_size = 0.2, random_state = random_seed)

# Some example

plt.imshow(X_train[147][:,:,0])
# I followed the 2X filter + pooling scheme but went a little experimental on kernel and filter sizes  

model = Sequential() 



# 1.filter & 2.filter + Maxpooling + dropout



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation = lambda x: tf.keras.activations.relu(x, alpha=0.1), input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation = lambda x: tf.keras.activations.relu(x, alpha=0.1)))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# 3.filter + 4.filer + Maxpooling + Dropout + strides



model.add(Conv2D(filters = 32, kernel_size = (2,2),padding = 'Same', 

                 activation = lambda x: tf.keras.activations.relu(x, alpha=0.1)))



model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', 

                 activation = lambda x: tf.keras.activations.relu(x, alpha=0.1)))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256))

model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.5))





model.add(Dense(10, activation = "softmax"))
model.summary()
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot',format='svg'))
# adam or RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
lr_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 35

batch_size = 64
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks=[lr_reduction])
def create_trace(x,y,ylabel,color):

        trace = go.Scatter(

            x = x,y = y,

            name=ylabel,

            marker=dict(color=color),

            mode = "markers+lines",

            text=x

        )

        return trace

    

def plot_accuracy_and_loss(history):

    hist = history.history

    acc = hist['acc']

    val_acc = hist['val_acc']

    loss = hist['loss']

    val_loss = hist['val_loss']

    epochs = list(range(1,len(acc)+1))

    

    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")

    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")

    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")

    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")

   

    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',

                                                             'Training and validation loss'))

    fig.append_trace(trace_ta,1,1)

    fig.append_trace(trace_va,1,1)

    fig.append_trace(trace_tl,1,2)

    fig.append_trace(trace_vl,1,2)

    fig['layout']['xaxis'].update(title = 'Epoch')

    fig['layout']['xaxis2'].update(title = 'Epoch')

    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])

    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])



    

    iplot(fig, filename='accuracy-loss')



plot_accuracy_and_loss(history)
# Plot the loss curve for training and validation 

plt.plot(history.history['val_loss'], color='r', label="validation loss")

plt.title("Validation Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curves for training and validation 

plt.plot(history.history['val_acc'], color='g', label="validation accuracy")

plt.title("Validation Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

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



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
print('-'*80)

print('train accuracy of the model: ', history.history['acc'][-1])

print('-'*80)
print('-'*80)

print('validation accuracy of the model: ', history.history['val_acc'][-1])

print('-'*80)
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# predict results

results = model.predict(ttest)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("Digit_Recognizer_Suad_Emre_Umar.csv",index=False)