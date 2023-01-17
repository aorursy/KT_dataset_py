import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau

train_X = pd.read_csv('../input/digit-recognition-dataset/Digit Recognition/trian_data1.csv')
test_X = pd.read_csv('../input/digit-recognition-dataset/Digit Recognition/test_data1.csv')
train_y = pd.read_csv('../input/digit-recognition-dataset/Digit Recognition/train.csv')
test= pd.read_csv('../input/digit-recognition-dataset/Digit Recognition/Test.csv')
sample_sub = pd.read_csv('../input/digit-recognition-dataset/Digit Recognition/Sample_submission.csv')
print('Train dataset has {} rows and {} columns'.format(train_X.shape[0],train_X.shape[1]))
print('test dataset has {} rows and {} columns'.format(test_X.shape[0],test_X.shape[1]))

train_X.head()
test_X.head()
train_y.head()
train_y = train_y.iloc[:,1]


train_y.head()

y = train_y.value_counts()
sns.barplot(y.index,y)

train_X = train_X /255
test_X =test_X /255
train_X= train_X.values.reshape(-1,28,28,1)
test_X = test_X.values.reshape(-1,28,28,1)

print('The shape of train set now is',train_X.shape)
print('The shape of test set now is',test_X.shape)
train_y = to_categorical(train_y)
X_train,X_test,y_train,y_test = train_test_split(train_X,train_y,random_state = 42 , test_size=0.20)
plt.imshow(X_train[0][:,:,0])
datagen = ImageDataGenerator(
            featurewise_center = False, # set input mean to 0 over the dataset
            samplewise_center = False,  # set each sample mean to 0
            featurewise_std_normalization = False, # divide inputs by std of the dataset
            samplewise_std_normalization = False,  # divide each input by its std
            zca_whitening = False,   # apply ZCA whitening
            rotation_range = 10,     # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1,       # Randomly zoom image 
            width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range = 0.1, # randomly shift images vertically (fraction of total height)
            horizontal_flip = False,  # randomly flip images
            vertical_flip = False     # randomly flip images
)

datagen.fit(X_train)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(BatchNormalization(momentum = .05))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization(momentum=0.05))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(BatchNormalization(momentum=.05))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation = "softmax"))
model.summary()
optimizer = Adam(learning_rate=0.001 , beta_1=0.9 ,beta_2 = 0.999)
model.compile(optimizer=optimizer , loss=['categorical_crossentropy'],metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',
                                            patience = 5 ,
                                            verbose = 1,
                                            factor = 0.5 , 
                                            min_lr = 0.00001)

epochs = 20
batch_size = 100
history = model.fit_generator(datagen.flow(X_train,y_train,batch_size = batch_size),
                              epochs = epochs ,
                              validation_data = (X_test,y_test),
                              verbose = 2,
                              steps_per_epoch = X_train.shape[0]//batch_size,
                              callbacks =[learning_rate_reduction])

fig,ax=plt.subplots(2,1)
fig.set
x=range(1,1+epochs)
ax[0].plot(x,history.history['loss'],color='red')
ax[0].plot(x,history.history['val_loss'],color='blue')

ax[1].plot(x,history.history['accuracy'],color='red')
ax[1].plot(x,history.history['val_accuracy'],color='blue')
ax[0].legend(['trainng loss','validation loss'])
ax[1].legend(['trainng acc','validation acc'])
plt.xlabel('Number of epochs')
plt.ylabel('accuracy')
y_pre_test=model.predict(X_test)
y_pre_test=np.argmax(y_pre_test,axis=1)
y_test=np.argmax(y_test,axis=1)
conf=confusion_matrix(y_test,y_pre_test)
conf=pd.DataFrame(conf,index=range(0,10),columns=range(0,10))


conf


plt.figure(figsize=(8,6))
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf, annot=True,annot_kws={"size": 16},cmap=plt.cm.Blues)# font size
x=(y_pre_test-y_test!=0).tolist()
x=[i for i,l in enumerate(x) if l!=False]
fig,ax=plt.subplots(1,4,sharey=False,figsize=(15,15))

for i in range(4):
    ax[i].imshow(X_test[x[i]][:,:,0])
    ax[i].set_xlabel('Real {}, Predicted {}'.format(y_test[x[i]],y_pre_test[x[i]]))
    
y_pre_test
test_y = model.predict(test_X)
test_y =np.argmax(test_y,axis=1)

test_y
test1 = test
test1 = test1.iloc[:,0:1]
test1
output = pd.DataFrame({'filename': test1.iloc[1:,0],
                     'label': test_y})
output.to_csv('submission1.csv', index=False)