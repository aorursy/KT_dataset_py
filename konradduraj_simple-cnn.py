import pandas as pd
import numpy as np

train_csv = pd.read_csv("../input/fashion-mnist_train.csv")
test_csv = pd.read_csv("../input/fashion-mnist_test.csv")

print(f'Number of rows: {len(train_csv.iloc[:,1])}')
print(f'Number of cols: {len(train_csv.iloc[1,:])}')
train_csv.head()

train_imgs = train_csv.iloc[:,1:]
train_labels = train_csv.iloc[:,0]

test_imgs = test_csv.iloc[:,1:]
test_labels = test_csv.iloc[:,0]

print(train_imgs.shape)
print(train_labels.shape)

train_imgs.head()
x_train = np.array(train_imgs, dtype = 'float32')
y_train = np.array(train_labels, dtype='float32')

x_test  = np.array(test_imgs, dtype = 'float32')
y_test = np.array(test_labels, dtype = 'float32')
x_train = x_train[:,:]/255
x_train

x_test = x_test[:,:]/255
x_test 
import matplotlib.pyplot as plt
%matplotlib inline

W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
axes = axes.ravel()
n_training = len(x_train) 

for i in np.arange(0, W_grid * L_grid): 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow(x_train[index].reshape((28,28)) )
    axes[i].set_title(y_train[index], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


from sklearn.model_selection import train_test_split
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2)
image_size = 28
image_shape = (image_size,image_size,1)
x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
simple_cnn = Sequential()

simple_cnn.add(Conv2D(32, (3,3), input_shape=(image_size, image_size, 1), activation = 'relu'))
simple_cnn.add(MaxPooling2D(pool_size = (2, 2)))

simple_cnn.add(Dropout(0.2))

simple_cnn.add(Conv2D(64, (3,3), activation = 'relu'))
simple_cnn.add(MaxPooling2D(pool_size = (2, 2)))

simple_cnn.add(Flatten())
simple_cnn.add(Dense(units = 32, activation = 'relu'))
simple_cnn.add(Dense(units = 10, activation = 'softmax'))
simple_cnn.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00001)
simple_cnn.compile(loss ='sparse_categorical_crossentropy', optimizer='adam',metrics =['accuracy'])
nb_epochs = 100
batch_size = 32

history = simple_cnn.fit(x_train, y_train,batch_size, epochs = nb_epochs, callbacks=[learning_rate_reduction],validation_data=(x_validate, y_validate))
evaluation = simple_cnn.evaluate(x_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))
predicted_classes = simple_cnn.predict_classes(x_test)
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(x_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)
def plot_accuracy_and_loss(history):
    hist = history.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, 'b', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'b', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()
plot_accuracy_and_loss(history)