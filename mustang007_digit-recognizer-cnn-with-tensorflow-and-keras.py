import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/digit-recognizer/train.csv')
df1 = pd.read_csv('../input/digit-recognizer/test.csv')
x_train = df.drop('label', axis = 1)
y_train = df['label']

test = df1
x_train.shape
test.shape
x_train = x_train.values.reshape(42000, 28,28)
test = test.values.reshape(28000, 28, 28)
x_train.shape
test.shape
single_image = x_train[0]
single_image
#matplotlib has a method to show these values in image format
plt.imshow(single_image)
#exploring labels
y_train
from tensorflow.keras.utils import to_categorical
#checking the shape of y_train
y_train.shape
y_example = to_categorical(y_train)
y_example.shape
y_example[0]
y_cat_train = to_categorical(y_train, num_classes=10)
#to_categorical takes num_classes on its own based on the label's unique values
#here it was from 0 to 9, hence, it took 10. You can specify them too using num_classes
y_cat_train[0]
#checking the maximum value of single_image
single_image.max()
#checking the minimum value of single_image
single_image.min()
x_train = x_train/255
test = test/255
#checking the scaled image
scaled_image = x_train[0]
scaled_image
scaled_image.max()
plt.imshow(scaled_image)
x_train.shape
#batch_size, width, height, color channels
x_train = x_train.reshape(42000, 28, 28, 1)
test.shape
test = test.reshape(28000, 28, 28, 3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(#rescale = 1./255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rotation_range = 20,
                               shear_range = 0.3,
                               zoom_range = 0.3,
                               horizontal_flip = True)
generator.fit(x_train)

random_seed = 2

x_train, x_val, y_cat_train, y_val = train_test_split(x_train, y_cat_train, test_size = 0.1, random_state=random_seed)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(4,4), strides = (1,1), input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten()) #flatten our layer, eg, our image is 28x28 so the flattened image will be 28*28=784 pixels
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))

#OUTPUT layer
model.add(Dense(10, activation='softmax')) #choosing softmax because of 'multiclass classification'


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
model.summary()
#Gonna import EarlyStopping in order to avoid overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
#Fitting the model
model.fit(x_train, y_cat_train, epochs = 10, validation_data = (x_val, y_val), callbacks = [early_stop])
metrics = pd.DataFrame(model.history.history)
metrics.head()
#Plotting loss and val_loss together
metrics[['loss', 'val_loss']].plot()
#Plotting accuracy and val_accuracy together
metrics[['accuracy', 'val_accuracy']].plot()
#Evaluating validation loss and accuracy
model.evaluate(x_val, y_val, verbose = 0)
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(x_val)

# Convert predictions classes to one hot vectors 
predictions_classes = np.argmax(predictions, axis = 1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val, axis = 1)
predictions
#we'll use y_true for predictions
print(classification_report(y_true, predictions_classes))
print(confusion_matrix(y_true, predictions_classes))
#visualizing confusion matrix
import seaborn as sns

plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_true, predictions_classes), annot=True)

# from keras.applications import MobileNetV2
# from keras.models import load_model
# model  = load_model("/kaggle/input/common-keras-pretrained-models/ResNet50.h5")
# for i in range(len(model.layers)-1):
#     model.layers[i].trainable = False
# model.summary()
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
# model.fit(x_train, y_cat_train, epochs = 10, validation_data = (x_val, y_val), callbacks = [early_stop])