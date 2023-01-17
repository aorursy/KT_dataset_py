# Imports

import numpy as np
import pickle
import itertools

# System
import os, fnmatch

# Visualization
import seaborn #visualization library, must be imported before all other plotting libraries
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display, Image


# Machine Learning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import classification_report
from sklearn import preprocessing

# Deep Learning
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import History
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model

# Random Seed
#from tensorflow import set_random_seed
#from numpy.random import seed
#seed(1)
#set_random_seed(2)

# Audio
import librosa.display, librosa

# Configurations
path='../input/audio-instrument-classification/london_phill_dataset_multi'
# Get files
files = []
for root, dirnames, filenames in os.walk(path):
    for filename in fnmatch.filter(filenames, '*.mp3'):
        files.append(os.path.join(root, filename))

print("found %d audio files in %s"%(len(files),path))
filename="../input/pl-files/inst_labels.pl"
# Load labels
with open(filename, "rb") as f:
    labels = pickle.load( open( filename, "rb" ) )
# Encode Labels
labelencoder = LabelEncoder()
labelencoder.fit(labels)
print(len(labelencoder.classes_), "classes:", ", ".join(list(labelencoder.classes_)))
classes_num = labelencoder.transform(labels)

#print('Labels:', labels[:3])
#print('Encoded Classes: ', classes_num[0:3])
# Machine Learning Parameters
testset_size = 0.25 #Percentage of data for Testing

length_file = len(files)
sample_length=0
for i in range (length_file) :
    x_in,sr = librosa.load(files[i])
    if (np.size(x_in)>sample_length):
        sample_length= np.size(x_in)
print(sample_length)


x= np.zeros((length_file,sample_length))
for i in range (length_file) :
    x_in,sr = librosa.load(files[i])
    x_in_length = len(x_in)
    x[i, : x_in_length ] = x_in
    #print(x)
    #print(x_in)
     #print(x[i])
print(x.shape)
    
#zero_crossings
n0 = 1000
n1 = 7000
sum_zero_crossings_normalised = np.zeros((length_file,1))
for i in range (length_file) :
    zero_crossings = librosa.zero_crossings(x[i,n0:n1], pad=False)
    #zero_crossings_normalised = preprocessing.minmax_scale(zero_crossings,axis=0)
    sum_zero_crossings_normalised[i] = np.sum(zero_crossings)
    
sum_zero_crossings_normalised= preprocessing.minmax_scale(sum_zero_crossings_normalised)
#print(np.shape(zero_crossings))    
print(np.shape(sum_zero_crossings_normalised))
print(max(sum_zero_crossings_normalised))
#print(sum_zero_crossings_normalised)

#spectral_centroids
sum_spectral_centroids_normalised = np.zeros((length_file,1))
for i in range (length_file) :
    spectral_centroids = librosa.feature.spectral_centroid(x[i], sr=sr)[0]
    sum_spectral_centroids=np.sum(spectral_centroids)
    spectral_centroids_normalised = preprocessing.minmax_scale(spectral_centroids, axis=0)
    sum_spectral_centroids_normalised[i]=np.sum(spectral_centroids_normalised)
    
sum_spectral_centroids_normalised=preprocessing.minmax_scale(sum_spectral_centroids_normalised)   
#print(np.shape(spectral_centroids))
#print(spectral_centroids)
print(np.shape(sum_spectral_centroids_normalised))
#print(max(sum_spectral_centroids_normalised))
#print(sum_spectral_centroids_normalised)

#spectral_rolloff
sum_spectral_rolloff_normalised = np.zeros((length_file,1))
for i in range (length_file) :
    spectral_rolloff = librosa.feature.spectral_rolloff(x[i]+0.01, sr=sr)[0]
    spectral_rolloff_normalised = preprocessing.minmax_scale(spectral_rolloff,axis=0)
    sum_spectral_rolloff_normalised[i]=np.sum(spectral_rolloff_normalised)
sum_spectral_rolloff_normalised=preprocessing.minmax_scale(sum_spectral_rolloff_normalised)
#print(np.shape(spectral_rolloff))
#print(spectral_rolloff)
print(np.shape(sum_spectral_rolloff_normalised))
#print(max(sum_spectral_rolloff_normalised))
#print(sum_spectral_rolloff_normalised)

#mfcc
filename="../input/pl-files/mfcc_feature_vectors.pl"
with open(filename, "rb") as f:
    scaled_feature_vectors = pickle.load( open( filename, "rb" ) )
#print(np.amax(scaled_feature_vectors))

print(np.shape(scaled_feature_vectors))
#print(scaled_feature_vectors)
scaled_feature_vectors= np.concatenate((scaled_feature_vectors,sum_spectral_centroids_normalised),axis=1)
scaled_feature_vectors= np.concatenate((scaled_feature_vectors,sum_zero_crossings_normalised),axis=1)
scaled_feature_vectors= np.concatenate((scaled_feature_vectors,sum_spectral_rolloff_normalised),axis=1)                                 
print(np.shape(scaled_feature_vectors))
# Create Train and Test Set
splitter = StratifiedShuffleSplit(n_splits=1, test_size=testset_size, random_state=0)
splits = splitter.split(scaled_feature_vectors, classes_num)
for train_index, test_index in splits:
    train_set = scaled_feature_vectors[train_index]
    test_set = scaled_feature_vectors[test_index]
    train_classes = classes_num[train_index]
    test_classes = classes_num[test_index]
print("train_set shape:",train_set.shape)
print("test_set shape:",test_set.shape)
print("train_classes shape:",train_classes.shape)
print("test_classes shape:",test_classes.shape)
# DNN
# Use Keras Backend Type
train_set_d=train_set.astype(K.floatx())
test_set_d=test_set.astype(K.floatx())
# One Hot encode
onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
integer_encoded_train_classes =  train_classes.reshape(len( train_classes), 1)
onehot_encoded_train_classes = onehot_encoder.fit_transform(integer_encoded_train_classes,1)
integer_encoded_test_classes =  test_classes.reshape(len( test_classes),1)
onehot_encoded_test_classes = onehot_encoder.fit_transform(integer_encoded_test_classes,1)
#print('Integer Encoded:', integer_encoded_train_classes[:3] )
#print('One-hot Encoded: ', onehot_encoded_train_classes[:3] )
# Check Set Shapes
print("train_set shape:",train_set_d.shape)
print("test_set shape:",test_set_d.shape)
print("train_classes shape:",onehot_encoded_train_classes.shape)
print("test_classes shape:",onehot_encoded_test_classes.shape)
# Reshape Sets for Keras
train_set_d=train_set.reshape(train_set_d.shape[0],1,train_set_d.shape[1])
test_set_d=test_set.reshape(test_set_d.shape[0],1,test_set_d.shape[1])
train_classes_d_hot=onehot_encoded_train_classes.reshape(onehot_encoded_train_classes.shape[0],1,
                                                         onehot_encoded_train_classes.shape[1])
test_classes_d_hot=onehot_encoded_test_classes.reshape(onehot_encoded_test_classes.shape[0],1,
                                                       onehot_encoded_test_classes.shape[1])
# Creating Simple Model
model_input = Input(shape=(1,train_set.shape[1]))
fc1 = Dense(15, activation="relu")(model_input)
fc2 = Dense(10, activation="relu")(fc1)
fc3 = Dense(8, activation="relu")(fc2)
fc4 = Dense(6, activation="relu")(fc2)
n=onehot_encoded_train_classes.shape[1]
out = Dense(n, activation="softmax")(fc4)
model_d = Model(inputs=[model_input], outputs=[out])
model_d.summary()
plot_model(model_d, to_file='model_d.png', show_shapes=True)
Image('model_d.png')
# Compile Model
model_d.compile(loss      = 'categorical_crossentropy',
              optimizer = SGD(lr=0.05),
              metrics   =['accuracy'])
# Deep Learning Parameters
batch_size = 5 # Number of samples per gradient update.
epochs = 200    # An epoch is an iteration over the entire x and y data provided.

# Train Model
hist = model_d.fit(train_set_d, train_classes_d_hot, verbose=1, 
                    batch_size=batch_size, epochs=epochs, validation_data=(test_set_d,test_classes_d_hot))
# Plot Training Loss and Training Accuracy
plt.figure(figsize=(8,4))
plt.subplot(1, 2, 1)
plt.title("Training loss")
plt.plot(range(epochs),hist.history["loss"])

plt.subplot(1, 2, 2)
plt.title("Training Accuracy")
plt.plot(range(epochs),hist.history["accuracy"])

plt.tight_layout()
# Predict
predictions = model_d.predict(test_set_d)
predictions_round=predictions.round().astype('int')
predictions_int=np.argmax(predictions_round,axis=2)
predictions_labels=labelencoder.inverse_transform(np.ravel(predictions_int))
# Recall
print("Recall: ", recall_score(test_classes, predictions_int,average=None))

# Precision
print("Precision: ", precision_score(test_classes, predictions_int,average=None))

# F1-Score
print("F1-Score: ", f1_score(test_classes, predictions_int, average=None))

# Accuracy
print("Accuracy: ", accuracy_score(test_classes, predictions_int,normalize=False))
print("Number of samples:",test_classes.shape[0])

print(classification_report(test_classes, predictions_int))
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
plot_history(hist)
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_classes, predictions_int)
np.set_printoptions(precision=2)
# Function to Plot Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    """
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
plt.figure(figsize=(16,12))
plot_confusion_matrix(cnf_matrix, classes=labelencoder.classes_,
                      title='Confusion matrix, without normalization')
# Find wrong predicted samples indexes
wrong_predictions = [i for i, (e1, e2) in enumerate(zip(test_classes, predictions_int)) if e1 != e2]
# Find wrong predicted audio files
print(np.array(labels)[test_index[wrong_predictions]])
print(predictions_labels[wrong_predictions].T)
print(np.array(files)[test_index[wrong_predictions]])
