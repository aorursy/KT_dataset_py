import pandas as pd
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input , BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras import Model
from sklearn.metrics import confusion_matrix
import seaborn as sn
# load data
df_train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
df_test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
df_train.head()
df_test.head()
df_train.columns
num_class = df_train.label.nunique()
num_class
df_test.columns
trainX = df_train.drop('label', axis=1).values
trainy = df_train['label'].values.reshape(-1,1)

testX = df_test.drop('label', axis=1).values
testy = df_test['label'].values.reshape(-1,1)
trainy[:5]
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape , testy.shape))
# plot first few images
for i in range(9):
    img = trainX[i].reshape(28,28)
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(img)
    
# show the figure
plt.show()
# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# one hot encode target values
trainy = to_categorical(trainy)
testy = to_categorical(testy)
# convert from integers to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')
# normalize to range 0-1
trainX = trainX / 255.0
testX = testX / 255.0
print(trainX.shape)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=7)
input_model = Input((trainX.shape[1],trainX.shape[2],trainX.shape[3]))


model1 = Conv2D(64,(5,5), activation='relu')(input_model)
model1 = Conv2D(64,(5,5), activation='relu', padding='same')(model1)
model1 = BatchNormalization()(model1)
model1 = MaxPooling2D((2, 2))(model1)
model1 = Conv2D(64,(3,3), activation='relu' ,padding='same')(model1)
model1 = Conv2D(64,(3,3), activation='relu' ,padding='valid')(model1)
model1 = BatchNormalization()(model1)
model1 = AveragePooling2D((2, 2))(model1)
model1 = Flatten()(model1)
#########################################################                          
model2 = Conv2D(128,(4,4), activation='relu')(input_model)  
model2 = Conv2D(64,(4,4), activation='relu', padding='same')(model2)
model2 = BatchNormalization()(model2)
model2 = MaxPooling2D((2, 2))(model2)
model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2) 
model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2) 
model2 = BatchNormalization()(model2)
model2 = AveragePooling2D((2, 2))(model2)
model2 = Conv2D(32,(2,2), activation='relu' ,padding='same')(model2)
model2 = Conv2D(32,(2,2), activation='relu' ,padding='valid')(model2)
model2 = BatchNormalization()(model2)
model2 = AveragePooling2D((2, 2))(model2)
model2 = Flatten()(model2)
########################################################
merged = Concatenate()([model1, model2])
merged = Dense(units = 512, activation = 'relu')(merged)
merged = Dropout(rate = 0.2)(merged)
merged = BatchNormalization()(merged)
merged = Dense(units = 20, activation = 'relu')(merged)
merged = Dense(units = 15, activation = 'relu')(merged)
output = Dense(units = num_class, activation = 'softmax')(merged)

model = Model(inputs= [input_model], outputs=[output])
plot_model(model, show_shapes=True)
model.summary()
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainy, epochs=100, batch_size=100, validation_split=0.25, verbose=1,callbacks=[es])
model.save_weights("FashionMNIST_weights.h5")
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(val_loss)
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Val error','Train error'], loc='upper right')
plt.savefig('plot_error.png')
plt.show()
val_accuracy = history.history['val_accuracy']
accuracy = history.history['accuracy']

plt.plot(val_accuracy)
plt.plot(accuracy)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['Val accuracy','Train accuracy'], loc='upper right')
plt.savefig( 'plot_accuracy.png')
plt.show()
pred = model.predict(testX)
pred = pd.DataFrame(pred)
pred['Perdiction'] = pred.idxmax(axis=1)
pred.head(5)
common = pred[["Perdiction"]].merge(df_test[['label']],left_on = pred.index , right_on = df_test.index)
common = common[["Perdiction",'label']]
common.head()
mat = confusion_matrix(common['label'],common['Perdiction'])
sn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
mask = (df_test.label == 2)
item2 = df_test[mask]
item2_pic = item2.drop('label', axis=1).values

# plot first few images
for i in range(9):
    img = item2_pic[i].reshape(28,28)
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(img)
    
# show the figure
plt.show()
mask = (df_test.label == 4)
item2 = df_test[mask]
item2_pic = item2.drop('label', axis=1).values

# plot first few images
for i in range(9):
    img = item2_pic[i].reshape(28,28)
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(img)
    
# show the figure
plt.show()
mask = (df_test.label == 0)
item2 = df_test[mask]
item2_pic = item2.drop('label', axis=1).values

# plot first few images
for i in range(9):
    img = item2_pic[i].reshape(28,28)
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(img)
    
# show the figure
plt.show()
mask = (df_test.label == 6)
item2 = df_test[mask]
item2_pic = item2.drop('label', axis=1).values

# plot first few images
for i in range(9):
    img = item2_pic[i].reshape(28,28)
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(img)
    
# show the figure
plt.show()
