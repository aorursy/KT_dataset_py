# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
np.random.seed(2)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from sklearn.model_selection import train_test_split
import itertools
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adagrad
from keras.optimizers import Nadam
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
subm = pd.read_csv("../input/predict/nas.csv") ## Upload Test Set
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) # Takes Data But Leaves Label Behind  
Y_train.value_counts()
X_train = X_train / 255.0
test = test / 255.0        ## Range of values was 0-255; now is 0-1.
subm = subm/ 255.0
X_train = X_train.values.reshape(-1,28,28,1)
subm = subm.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Labels= ["0","1", "2","3", "4","5", "6","7", "8","9"]  # Labels
Y_train = pd.get_dummies( Y_train, columns = Labels )  # Encoding of Labels
Y_train = Y_train.values   #Change into Numpy ndarray for model
#DrawMe = plt.imshow(X_train[0][:,:,0])  #[Index Of Number][Y,X,Channel]                     ## 
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Softmax, Add, Flatten, Activation , Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

inp = Input(shape=(28,28,1)) ## Model #1 - 4 Block ResNet 
C = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(inp)


C11 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(C)
C12 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(C11)
S11 = Add()([C12, C])

C21 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(S11)
C22 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(C21)
S21 = Add()([C22, S11])

C31 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(S21)
C32 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(C31)
S31 = Add()([C32, S21])

C41 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(S31)
C42 = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(C41)
S41 = Add()([C42, S31])

CXX = Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1))(S41)
M51 = MaxPool2D(pool_size=(2,2), strides=(2,2))(CXX)

F1 = Flatten()(M51)
D1 = Dense(32)(F1)
A6= Activation("relu")(D1)
DD1 = Dropout(0.2)(A6)
D2 = Dense(64)(DD1)
DD2 = Dropout(0.2)(D2)
D3 = Dense(10)(DD2)
A7 = Softmax()(D3)
model = Model(inputs=inp, outputs=A7)
model.summary()

optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3,  verbose=1, factor=0.5,  min_lr=0.00001)
epochs = 40 #10 # 1
batch_size = 86 #256

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
yxy=np.argmax(model.predict(subm),axis=1)
StringLabel = (['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine'])
pd.Series(yxy,name="Label")
for x in yxy:
    print(x,"-", StringLabel[x])