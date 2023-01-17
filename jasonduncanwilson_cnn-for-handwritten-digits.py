from IPython.display import Image
url = 'https://bplusmovieblog.files.wordpress.com/2016/08/pink-floyd-the-wall-18.png'
Image(url=url,width=800, height=600)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # one hot encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

%matplotlib inline
sns.set(style='white', context='notebook', palette='deep')

# Set random seed to get a consistent result
random_seed = 2
np.random.seed(random_seed)
# Load the training data and review
train = pd.read_csv("../input/train.csv")
train.head(5)
# Load the test data and review
test = pd.read_csv("../input/test.csv")
test.head(5)
# Compare columns between test and train
print("Columns in training data but not in testing data")
print([x for x in train.columns if x not in test.columns])
print("Columns in testing data but not in training data")
print([x for x in test.columns if x not in train.columns])
# Split into x and y for training
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_test = test
train_null_rows = sum(X_train.isnull().sum())
print("Number of train rows with null pixels:",train_null_rows)

# Drop rows with missing values as long as it doesn't exceed 2% of the data
if train_null_rows < len(X_train.index)*0.02:
    X_train = X_train.dropna()
    
test_null_rows = sum(X_test.isnull().sum())
print("Number of test rows with null pixels:",train_null_rows)

# Drop rows with missing values as long as it doesn't exceed 2% of the data
if test_null_rows < len(X_test.index)*0.02:
    X_test = X_test.dropna()
fig = plt.figure(figsize=(12, 5))
ax = sns.countplot(Y_train)
ax.set_title("Number of training examples per digit")
# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0
# Reshape into 3D matrices
X_train = X_train.values.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')
X_test = X_test.values.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')
Y_num_classes = Y_train.nunique()
print("the number of classes = %i" % Y_num_classes)
print("Dimension of images = {:d} x {:d}".format(X_train[1].shape[0],X_train[1].shape[1]))
images_and_labels = list(zip(X_train,  Y_train))
for index, (image, label) in enumerate(images_and_labels[:12]):
    plt.subplot(5, 4, index + 1)
    plt.axis('off')
    plt.imshow(image.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('label: %i' % label )
# Encode labels to one hot vectors
Y_train = to_categorical(Y_train, num_classes = Y_num_classes)
print("A few examples of the one hot encoding:")
print(Y_train[:5,:])
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, 
                                                  Y_train, 
                                                  test_size = 0.1, 
                                                  stratify=Y_train, #balance digits across data
                                                  random_state=random_seed)
model = Sequential()

model.add(Conv2D(filters = 32, 
                 kernel_size = (5,5),
                 #strides=2,
                 padding = 'Same', 
                 activation ='relu', 
                 input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, 
                 kernel_size = (5,5),
                 #strides=2,
                 padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))


model.add(Conv2D(filters = 64, 
                 kernel_size = (3,3),
                 padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, 
                 kernel_size = (3,3),
                 padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), 
                    strides=(2,2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(256, 
                activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, 
                activation = "softmax"))
# Define the optimizer
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer='adam', 
              #optimizer=optimizer,
              loss = "categorical_crossentropy", 
              metrics=["accuracy"])
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.2,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(X_train)
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# 30 epochs ran in roughly 130 minutes with CPU
# 40 epochs ran in roughly 9 minutes with GPU
epochs = 40 # set to something around 20 to 30 to increase accuracy
batch_size = 86

model.fit_generator(datagen.flow(X_train,
                                 Y_train,
                                 batch_size=batch_size),
                    epochs = epochs, 
                    validation_data = (X_val,Y_val),
                    verbose = 2, 
                    callbacks=[learning_rate_reduction],
                    steps_per_epoch=X_train.shape[0] // batch_size)
# Look at confusion matrix 

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(12, 5))
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
plot_confusion_matrix(confusion_mtx, classes = range(Y_num_classes))
# predict results
results = model.predict(X_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
results.head(5)
results_count = len(results)
submission = pd.concat([pd.Series(range(1,results_count+1),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)