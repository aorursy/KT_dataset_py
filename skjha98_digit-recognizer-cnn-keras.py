import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

X_train = train.drop(labels=["label"], axis = 1)
del train

sns.countplot(Y_train)
Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train/=255.0
test/=255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state = random_seed)
plt.imshow(X_train[0][:,:,0])
model = Sequential()

#model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
#model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
#model.add(MaxPool2D(pool_size = (2,2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
#model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
#model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(256, activation = 'relu'))
#model.add(Dropout(0.5))

#model.add(Dense(10, activation = 'softmax'))





model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(126, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# For model Visualization
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
lr_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
epochs = 30
batch_size = 86
# Without data augmentation i obtained an accuracy of 0.98114
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), verbose = 2)
# Plot the loss and accuracy curves for training and validation 
#fig, ax = plt.subplots(2,1)
#ax[0].plot(history.history['loss'], color='b', label="Training loss")
#ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
#legend = ax[0].legend(loc='best', shadow=True)

#ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
#ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
#legend = ax[1].legend(loc='best', shadow=True)
# Data Augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[lr_reduction])
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
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
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
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
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
model.save_weights("DigitRecogWeights.h5")
model.save("DigitRecogModel.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
test = pd.read_csv("../input/test.csv")
img1 = test.iloc[34].values.reshape(-1,28,28,1)
img1 = img1/255.0
plt.imshow(img1[0][:,:,0], cmap = 'gray')
pred = model.predict(img1)
pred_ans = np.argmax(pred)
print(pred_ans)

img1 = test.iloc[34].values
img1

img = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1,
         0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   1,   0,   1,   0,   0,   0,   0,   0,   0,
         0,   1,  14,  26,  23,   8,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   1,   0,   0,   1,   0,   0,   0,   0,
         0,   0,   5,  41, 115, 162, 145,  61,   8,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,
         0,   0,   0,   8,  68, 176, 240, 250, 235, 128,  19,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         1,   0,   0,   0,  13,  86, 206, 253, 253, 255, 236, 122,  17,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
         0,   0,   0,   1,   0,  13,  88, 211, 251, 254, 255, 233, 154,
        51,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
         0,   0,   1,   0,   0,   0,  10,  84, 214, 253, 255, 252, 215,
       111,  33,   4,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   4,  64, 198, 252, 254, 251,
       204,  84,  13,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,  30, 167, 248, 253,
       253, 208,  80,  10,   0,   0,   0,   0,   0,   0,   0,   1,   0,
         0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   8, 102, 235,
       253, 255, 217,  88,  12,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   1,  12,  22,  10,   0,   1,   0,   1,   0,  38,
       188, 252, 254, 239, 120,  16,   0,   0,   0,   0,   0,   0,   1,
         0,   0,   0,   0,   0,  13,  87, 143,  95,   0,   0,   1,   0,
         4,  90, 238, 254, 251, 181,  39,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,  41, 200, 248, 213,   0,   0,
         0,   0,  22, 156, 251, 255, 241, 103,   8,   0,   0,   1,   0,
         0,   0,   0,   0,   1,   0,   0,   0,   0,  68, 231, 254, 245,
         0,   0,   0,   0,  42, 208, 255, 253, 207,  51,   0,   1,   0,
         0,   0,   0,   0,   0,   1,   0,   0,   1,   4,  32, 136, 245,
       255, 247,   0,   0,   0,   0,  63, 235, 254, 254, 158,  22,   1,
         0,   0,   0,   0,   1,   0,   0,   0,   0,   7,  29,  79, 163,
       237, 254, 254, 236,   0,   1,   0,   0,  81, 247, 255, 250, 113,
         6,   0,   1,   0,   0,   0,   0,   0,   4,  18,  49,  97, 172,
       226, 251, 255, 254, 249, 172,   0,   0,   0,   3,  90, 249, 255,
       249,  98,   4,   0,   0,   1,   1,   9,  26,  52,  90, 148, 206,
       241, 252, 254, 255, 254, 245, 173,  58,   0,   0,   1,   1,  78,
       246, 255, 252, 161,  54,  35,  39,  53,  78, 117, 169, 212, 240,
       251, 253, 255, 254, 253, 245, 211, 129,  39,   4,   1,   1,   0,
         0,  53, 217, 254, 255, 242, 205, 186, 196, 216, 233, 246, 251,
       253, 254, 253, 254, 250, 236, 193, 126,  56,  15,   1,   0,   0,
         0,   1,   0,  17, 143, 247, 254, 254, 253, 252, 253, 253, 254,
       254, 254, 252, 249, 239, 209, 155,  90,  43,  13,   1,   0,   0,
         0,   1,   0,   0,   0,   2,  44, 170, 240, 250, 254, 253, 252,
       253, 249, 245, 230, 197, 144,  89,  49,  24,   5,   0,   1,   0,
         0,   1,   0,   0,   0,   0,   0,   0,   3,  39, 115, 175, 203,
       212, 207, 186, 151, 108,  70,  41,  18,   4,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   2,  14,
        30,  45,  49,  48,  35,  24,   8,   2,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
         0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0])
plt.imshow(img.reshape(28,28))
plt.show()
img = img.reshape(-1,28,28,1)
pred = model.predict(img)
print(pred, np.argmax(pred))












