# Base Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from bokeh.plotting import output_notebook, figure, show
output_notebook()

# Import models
from keras import models, layers, optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Other items
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
train.head(3)
validation = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
validation.head(3)
# Plot a few items
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(20,10))
i=1

for c in range(10):
    subset = train[train['label']==c]
    pictures = np.random.randint(low=0,high=subset.shape[0],size=20)
    for p in pictures:
        plt.subplot(10,20,i)
        picture_array = np.array(subset.iloc[p,1:]).reshape(28,28)
        plt.imshow(picture_array,cmap=plt.cm.binary)
        plt.title(class_names[c])
        plt.axis('off')
        i+=1
        
plt.tight_layout()
# Create variables for model

# Prepare training data
X = np.array(train.drop(columns=['label']))
y = np.array(train['label'])

# Prepare validation data
validation_x = validation.drop(columns=['label'])
validation_y = validation['label']

# Split training data into train and test
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert labels to categories
train_y_encoded = to_categorical(train_y)
test_y_encoded = to_categorical(test_y)
val_y_encoded = to_categorical(validation_y)

# Change to numpy arrays
train_x = np.array(train_x)/255
test_x = np.array(test_x)/255

train_y_encoded = np.array(train_y_encoded)
test_y_encoded = np.array(test_y_encoded)

validation_x = np.array(validation_x)
val_y_encoded = np.array(val_y_encoded)
# Model Definition
input_size = 784
output_size=10
m1 = models.Sequential()
m1.add(layers.Dense(input_size, activation='relu'))
m1.add(layers.Dense(output_size, activation='softmax'))

m1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
m1history = m1.fit(train_x, train_y_encoded, epochs=20, batch_size=128, validation_data=(test_x,test_y_encoded), verbose=0)
m1.save('m1.h5')
p = figure(plot_width=800, plot_height=400, title='Train & Test Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,21,1)
p.line(x,m1history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,m1history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "top_left"
show(p)
# Predict results
y_pred = m1.predict_classes(validation_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)
cm
# Classification report
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)
cr_df
plt.figure(figsize=(25,5))
sns.heatmap(cm, center=250, cmap='Blues')
print('Test Accuracy:', np.round(cr['accuracy']*100,2),'%')
# Model Definition
m2 = models.Sequential()
m2.add(layers.Dense(input_size, activation='relu'))
m2.add(layers.Dropout(0.5))
m2.add(layers.Dense(392, activation='relu'))
m2.add(layers.Dropout(0.5))
m2.add(layers.Dense(196, activation='relu'))
m2.add(layers.Dense(output_size, activation='softmax'))

m2.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
m2history = m2.fit(train_x, train_y_encoded, epochs=60, batch_size=128, validation_data=(test_x,test_y_encoded), verbose=0)
m2.save('m2.h5')
p = figure(plot_width=800, plot_height=400, title='Train & Test Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,61,1)
p.line(x,m2history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,m2history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "bottom_right"
show(p)
# Predict results
y_pred = m2.predict_classes(validation_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)

# Classification report
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)

plt.figure(figsize=(25,5))
sns.heatmap(cm, center=250, cmap='Blues')
print('Test Accuracy:', np.round(cr['accuracy']*100,2),'%')
# Create model inputs
image_shape=(28,28,1)

tr_x = np.array(train_x).reshape(train_x.shape[0],*image_shape)
te_x = np.array(test_x).reshape(test_x.shape[0],*image_shape)
va_x = np.array(validation_x).reshape(validation_x.shape[0],*image_shape)
m3 = models.Sequential()

m3.add(layers.Conv2D(28, kernel_size=(3,3),activation='linear',padding='same',input_shape=(28,28,1)))
m3.add(layers.LeakyReLU(alpha=0.1))
m3.add(layers.MaxPooling2D((2, 2),padding='same'))
m3.add(layers.Dropout(0.2))

m3.add(layers.Conv2D(56, (3,3), activation='linear',padding='same'))
m3.add(layers.LeakyReLU(alpha=0.1))
m3.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
m3.add(layers.Dropout(0.2))

m3.add(layers.Conv2D(56, (3,3), activation='linear',padding='same'))
m3.add(layers.LeakyReLU(alpha=0.1))                  
m3.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
m3.add(layers.Dropout(0.2))

m3.add(layers.Flatten())
m3.add(layers.Dense(112, activation='linear'))
m3.add(layers.LeakyReLU(alpha=0.1))           
m3.add(layers.Dropout(0.2))
m3.add(layers.Dense(10, activation='softmax'))
m3.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
m3history = m3.fit(tr_x, train_y_encoded, epochs=55, batch_size=128, validation_data=(te_x,test_y_encoded), verbose=0)
m3.save('m3.h5')
p = figure(plot_width=800, plot_height=400, title='Train & Test Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,56,1)
p.line(x,m3history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,m3history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "bottom_right"
show(p)
# Predict results
y_pred = m3.predict_classes(va_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)

# Classification report
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)

plt.figure(figsize=(25,5))
sns.heatmap(cm, center=250, cmap='Blues')
print('Test Accuracy:', np.round(cr['accuracy']*100,2),'%')
m4 = models.Sequential()

m4.add(layers.Conv2D(28, kernel_size=(3,3),activation='linear',padding='same',input_shape=(28,28,1)))
m4.add(layers.LeakyReLU(alpha=0.1))
m4.add(layers.MaxPooling2D((2, 2),padding='same'))
m4.add(layers.Dropout(0.3))

m4.add(layers.Conv2D(56, (3,3), activation='linear',padding='same'))
m4.add(layers.LeakyReLU(alpha=0.1))
m4.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
m4.add(layers.Dropout(0.3))

m4.add(layers.Conv2D(112, (3,3), activation='linear',padding='same'))
m4.add(layers.LeakyReLU(alpha=0.1))                  
m4.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
m4.add(layers.Dropout(0.3))

m4.add(layers.Flatten())
m4.add(layers.Dense(256, activation='linear'))
m4.add(layers.LeakyReLU(alpha=0.1))           
m4.add(layers.Dropout(0.3))
m4.add(layers.Dense(10, activation='softmax'))
datagen = ImageDataGenerator(rotation_range=5, 
                             width_shift_range=0.1,
                             height_shift_range=0.1, 
                             shear_range=0.1, 
                             zoom_range=0.1, 
                             horizontal_flip=False, 
                             fill_mode='nearest')
# Visualize augmented data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(20,10))

i=1
for c in range(10):
    subset = train[train['label']==c]
    pic = np.random.randint(low=0,high=subset.shape[0],size=1)
    for p in range(20):
        plt.subplot(10,20,i)
        picture_x = np.array(subset.iloc[pic,1:]).reshape(1,28,28,1)
        picture_y = np.array(subset.iloc[pic,1])
        picture_y = to_categorical(picture_y, num_classes=10).reshape(1,10)
        pic_x, pic_y = datagen.flow(picture_x, picture_y).next()
        plt.imshow(pic_x.reshape((28,28)),cmap=plt.cm.binary)
        plt.title(class_names[c])
        plt.axis('off')
        i+=1
plt.tight_layout()
m4.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
callbacks = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=150)
m4history = m4.fit_generator(datagen.flow(tr_x, train_y_encoded), 
                             steps_per_epoch=100, 
                             epochs=100, 
                             validation_data=(te_x,test_y_encoded),
                             validation_steps=50,
                             callbacks=callbacks,
                             verbose=2)
m4.save('m4.h5')
p = figure(plot_width=800, plot_height=400, title='Train & Test Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
x = np.arange(1,101,1)
p.line(x,m4history.history['accuracy'], legend_label='Train Accuracy', line_width=3)
p.line(x,m4history.history['val_accuracy'], legend_label='Validation Accuracy', color='green', line_width=3)
p.legend.location = "bottom_right"
show(p)
# Predict results
y_pred = m4.predict_classes(va_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(y_pred, validation_y), index=class_names, columns=class_names)

# Classification report
cr = classification_report(y_pred, validation_y,target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)

plt.figure(figsize=(25,5))
sns.heatmap(cm, center=250, cmap='Blues')
print('Test Accuracy:', np.round(cr['accuracy']*100,2),'%')
# Look through mis-classified items
validation['predicted']=y_pred
misclassified_shirts = validation[(validation['label']==6) & (validation['predicted']!=6)]
misclassified_shirts_pics = misclassified_shirts.drop(columns=['label', 'predicted']).reset_index(drop=True)
# Plot a few items
plt.figure(figsize=(20,5))
for i in range(30):
    pic_ind = np.random.randint(low=0,high=misclassified_shirts_pics.shape[0],size=1)
    pic = np.array(misclassified_shirts_pics.iloc[pic_ind]).reshape(28,28)
    plt.subplot(3,10,i+1)
    plt.imshow(pic,cmap=plt.cm.binary)
    plt.axis('off')
        
plt.tight_layout()
# Look through correctly classified items
validation['predicted']=y_pred
classified_shirts = validation[(validation['label']==6) & (validation['predicted']==6)]
classified_shirts_pics = classified_shirts.drop(columns=['label', 'predicted']).reset_index(drop=True)
# Plot a few items
plt.figure(figsize=(20,5))
for i in range(30):
    correct_ind = np.random.randint(low=0,high=classified_shirts_pics.shape[0],size=1)
    correct = np.array(classified_shirts_pics.iloc[correct_ind]).reshape(28,28)
    plt.subplot(3,10,i+1)
    plt.imshow(correct,cmap=plt.cm.binary)
    plt.axis('off')
        
plt.tight_layout()