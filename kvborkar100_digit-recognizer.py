import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = [8,6]

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# Load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
# Splitting training data into X and y
train_train = train_df.iloc[:,1:]
train_test = train_df.iloc[:,0]
train_train.shape,train_test.shape
# First lets look at our data
sns.countplot(train_test)
# Normalize the data
train_train = train_train / 255.0
test_df = test_df / 255.0
# We have data in the pandas dataframe format
# Convert it into 28 X 28 X 1 matrix, if we have color images we have to use 28 X 28 X 3
train_train = train_train.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)
# lets check for the first element
print(train_train[0])
print(train_test[0])
train_train[0][:,:,0]
grid_size = (1,3)
fig, axes = plt.subplots(1,3)
i =0
for ax in axes:
    ax.imshow(train_train[i][:,:,0])
    i+=1
# One hot encoding dependent varible
train_test = to_categorical(train_test,num_classes=10)
# Splitting data into training set and validation set
X_train, X_test, y_train, y_test = train_test_split(train_train,train_test,test_size = 0.2,random_state = 100)
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

classifier = Sequential()

# adding 2 convolution layer and 1 maxpooling layer
classifier.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
classifier.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# adding 2 convolution layer and 1 maxpooling layer
classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(256, activation = "relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation = "softmax"))
#Compile model
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
get_lr = ReduceLROnPlateau(monitor='val_acc', 
                            patience=3, 
                            verbose=1, 
                            factor=0.5, 
                            min_lr=0.00001)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1
    )
datagen.fit(X_train)
epochs = 40
batch_size = 100
classifier.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,y_test),
                              verbose = 2,
                              callbacks=[get_lr]
                                )
y_pred = classifier.predict(test_df)
y_pred = np.argmax(y_pred,axis=1)
output = pd.concat([pd.Series(range(1,28001),name="ImageId"),pd.Series(y_pred,name ="Label")],axis = 1)
output.head()
output.to_csv("1st_submission.csv",index=False)