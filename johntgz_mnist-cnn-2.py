import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import ReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import itertools
#1. extract training and val set with panda
root_path = "/kaggle/input/digit-recognizer"
X_train = pd.read_csv(root_path+"/train.csv")
test = pd.read_csv(root_path+"/test.csv")

#extract train_labels and train dataframe
Y_train = X_train["label"];
X_train = X_train.drop(['label'], axis=1)

#check for null and missing values, need to find a better way
# X_train.isnull().any().describe()
# test.isnull().any().describe()

#countplot
# g  = sns.countplot(Y_train)
#2. preprocess data

#2a. normalize
# X_train = X_train/255
# test = test/255

#2b. reshape and visualize
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1) 

#2c. Feature standardization
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x):
    return (x-mean_px)/std_px

num_training_examples = X_train.shape[0]
num_test_examples = test.shape[0]

#2d. One hot encoding
Y_train = to_categorical(Y_train, num_classes = 10)
# #2c. visualize
# rand_arr = np.random.randint(0, num_training_examples + 1, size = 4)

# plt.figure(figsize=(8,8))
# plt.subplot(221)
# plt.imshow(X_train[rand_arr[0]][:,:,0])
# plt.xlabel(Y_train[rand_arr[0]], bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# plt.subplot(222)
# plt.imshow(X_train[rand_arr[1]][:,:,0])
# plt.xlabel(Y_train[rand_arr[1]], bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# plt.subplot(223)
# plt.imshow(X_train[rand_arr[2]][:,:,0])
# plt.xlabel(Y_train[rand_arr[2]], bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# plt.subplot(224)
# plt.imshow(X_train[rand_arr[3]][:,:,0])
# plt.xlabel(Y_train[rand_arr[3]], bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

# plt.show()
X_train, X_val, y_train, y_val = train_test_split( X_train, Y_train, test_size = 0.1, random_state = 2)
#data augmentation
datagen = ImageDataGenerator( featurewise_center = False, #set input mean to 0 over the dataset
                            samplewise_center = False, #set each sample mean to 0
                             featurewise_std_normalization = False, #divide inputs by std of the dataset
                             samplewise_std_normalization = False, #divide each input by its std
                             zca_whitening = False, #apply ZCA whitening
                             rotation_range = 10, #randomly rotate images in the range (degrees, 0 to 180)
                             zoom_range = 0.1, #randomly zoom image
                             shear_range = 0.3,
                             width_shift_range = 0.1, #randomly shift images horizontally (fraction of total width)
                             height_shift_range = 0.1,#randomly shift images vertically (fraction of total height)
                             horizontal_flip = False, #randomly flip images
                             vertical_flip = False) #randomly flip
                             
datagen.fit(X_train)
# #CNN2
# model2 = Sequential()
# model2.add(Lambda(standardize, input_shape=(28,28,1)))

# model2.add(Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation='relu'))
# model2.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
# model2.add(Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation='relu'))
# model2.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding='valid'))

# model2.add(Dropout(0.25))
# model2.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))

# model2.add(Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation='relu'))
# model2.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
# model2.add(Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation='relu'))
# model2.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding='valid'))

# model2.add(Dropout(0.25))
# model2.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))

# model2.add(Flatten())
# model2.add(BatchNormalization())
# model2.add(Dense(512, activation='relu'))

# model2.add(Dropout(0.5))

# model2.add(BatchNormalization())
# model2.add(Dense(10,activation='softmax'))
# model2.summary()
#ensemble CNN
nets = 15
model = [0] * nets
for j in range(nets):
    model[j] = Sequential()
    
    model[j].add(Lambda(standardize, input_shape=(28,28,1)))
    
    model[j].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))
    
    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))
    
    model[j].add(Conv2D(128, kernel_size=4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))
    
    #compile with ADAM optimizer and cross entropy loss
    model[j].compile(optimizer='adam', loss ='categorical_crossentropy', metrics=['accuracy'])

model[0].summary()
# #optimizer
# opt_adam = Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
# opt_sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False, name='SGD')
# opt_rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False, name='RMSProp')

# #compile model
# model2.compile(optimizer=opt_adam, loss='categorical_crossentropy', metrics=['accuracy'])

# #learning rate annealer
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# #learning parameters
# epochs = 30
# batch_size = 64
# steps_per_epoch = X_train.shape[0] / batch_size #number of batch iterations before an epoch is complete
# #model1
# history1 = model1.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
#           epochs=epochs, 
#           validation_data = (X_val, y_val), 
#           verbose = 1, 
#           steps_per_epoch = steps_per_epoch,
#          callbacks = [learning_rate_reduction])

# #model2
# history2 = model2.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
#           epochs=epochs, 
#           validation_data = (X_val, y_val), 
#           verbose = 1, 
#           steps_per_epoch = steps_per_epoch,
#          callbacks = [learning_rate_reduction]
#                      )

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

#train the ensemble network
history = [0] * nets
epochs = 45
for j in range(nets):
    history[j] = model[j].fit_generator(datagen.flow(X_train, y_train, batch_size= 64), 
                                        epochs = epochs,
                                        steps_per_epoch = X_train.shape[0] // 64,
                                        validation_data = (X_val, y_val), 
                                        callbacks = [annealer],
                                        verbose = 1)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1, epochs, max(history[j].history['accuracy']), max(history[j].history['val_accuracy']) ))
history[0].history
#ensemble prediction
results = np.zeros( (test.shape[0], 10) )

for j in range(nets):
    results = results + model[j].predict(test)
results = np.argmax(results, axis=1)
results = pd.series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name="ImageId"), results], axis=1)
submission.to_csv("submission.csv", index=False)
#single model prediction
results = model2.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001), name="ImageId"),results], axis=1)
submission.to_csv("submission.csv", index=False)
! kaggle competitions submit -c digit-recognizer -f submission.csv -m "Message"
#Training and validation curves

#plot the loss and accuracy curves for training and validation
def get_key_values(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    
    return train_loss, val_loss, train_acc, val_acc

train_loss, val_loss, train_acc, val_acc = get_key_values(history2)

epoch_range = range(epochs)

plt.figure(figsize=(8,8))

plt.subplot(211)
plt.title('Loss')
plt.plot(epoch_range, train_loss, label='train_loss')
plt.plot(epoch_range, val_loss, label='val_loss')
plt.legend(loc='upper right')

plt.subplot(212)
plt.title('Accuracy')
plt.plot(epoch_range, train_acc, label='train_acc')
plt.plot(epoch_range, val_acc, label='val_acc')
plt.legend(loc='lower right')

plt.show()
#confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() /2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment = 'center',
                color = 'white' if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

Y_pred = model.predict(X_val)
# convert from OHE to class labels
Y_pred_classes = np.argmax(Y_pred, axis=1)
# convert from OHE to class labels
Y_true = np.argmax(y_val, axis=1)
#construct confusion matrix from sklearn.metrics
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
print(confusion_mtx)
plot_confusion_matrix(confusion_mtx, classes=range(10))
#display error results

#errors are differences between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

print('errors', errors)

Y_pred_errors = Y_pred[errors] #OHE labels
Y_pred_classes_errors = Y_pred_classes[errors]  #class labels

Y_true_classes_errors = Y_true[errors] #class labels
X_val_errors = X_val[errors] #misclassified input

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fix, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n+= 1
            

#probabilities of wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis = 1) 
#predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_classes_errors, axis=1))
#difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
#sorted list of the delta prob errors
sorted_delta_errors = np.argsort(delta_pred_true_errors)
#top 6 errors
most_important_errors = sorted_delta_errors[-6:]
print("most_important_errors",most_important_errors)
#show top 6 errors
display_errors(most_important_errors , X_val_errors, Y_pred_classes_errors, Y_true_classes_errors)
               
plt.show()
import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for file in filenames:
        print(os.path.join(dirname,file))