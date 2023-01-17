import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import gc   #Gabage collector for cleaning deleted data from memory

#glabl vars

COL_VALUES = ["x", "y", "z","xg","yg","zg"] #input file needs to always have obj as last column
MIN_VALUES = [-200, -200, -200, -10000, -10000, -10000]
MAX_VALUES = [200, 200, 200, 10000, 10000, 10000]

N_CLASSES = 5 #note that objects must be numbered from 0 to N_CLASSES
ROWS = 300 #how many lines per image snapshot
COLS = len(COL_VALUES)
DOWNSIZE_TO = 20000

data=pd.read_csv("../input/normal-position/out_normal_position.csv", names = COL_VALUES + ["obj"])
data2=pd.read_csv("../input/out-weird-position/out_weird_position.csv", names = COL_VALUES + ["obj"])
#data = data2.append(data)
#data2 = data2.drop(data2.index[0:2000])


def normalize(df):
    result = df.copy()
    i=0
    for feature_name in COL_VALUES:
        result[feature_name] = (df[feature_name] - MIN_VALUES[i]) / (MAX_VALUES[i] - MIN_VALUES[i])
        i=i+1
    return result

data=normalize(data)
data2=normalize(data2)
data.info()
import seaborn as sns

X = [] # images
y = [] # labels

def generate_images(imgs, labels, df, rows, cols, label):
    for i in range(0,min(df.shape[0]-rows, DOWNSIZE_TO*rows),rows):
        imgs.append((df.iloc[i:i+rows,0:cols]).values.tolist())
        labels.append(label)
    print("number of images after inserting label {} is {}".format(label, len(imgs)))
    return

#create the images for each class
for c in range(0, N_CLASSES):
    df=data[data['obj'] == c]
    generate_images(X, y, df, ROWS, COLS, c)
    
for c in range(0, N_CLASSES):
    df=data2[data['obj'] == c]
    generate_images(X, y, df, ROWS, COLS, c)
    
#plot labels to doublecheck
sns.countplot(np.array(y))
plt.title('labels for objects in processed dataset')

del data
del data2
gc.collect()
    
#shuffle and take a look at some random generated images
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def plot5imgs(arr_imgs, arr_labels, random_img):
    plt.figure(figsize=(ROWS,ROWS/2))
    columns = 5
    for i in range(columns):
        plt.subplot(5 / columns + 1, columns, i + 1)
        plt.title("obj #{}: {}".format(random_img, arr_labels[random_img]))
        if len(arr_imgs.shape) < 4:
            plt.imshow(arr_imgs[random_img])
        else:
            plt.imshow(arr_imgs[random_img][0])
        random_img=random_img+1


a = []
a = unison_shuffled_copies(np.array(X), np.array(y))
X = a[0]
y = a[1]

num=10
print(y[num:num+5])
#plot5imgs(X,y,num)
#X[3000]
#Convert list to numpy array
X = np.array(X)
y = np.array(y)
print("Shape of all images is:", X.shape)
print("Shape of all labels is:", y.shape)

#Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of train labels is:", y_train.shape)
print("Shape of validation labels is:", y_val.shape)

#clear memory
del X
del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (Activation, Dropout, Flatten, Dense, Convolution1D, MaxPooling1D, BatchNormalization, Conv1D, GlobalAveragePooling1D)
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 128
nb_epoch = 24

# Set random seed
np.random.seed(17)
 
# input image dimensions
img_rows, img_cols = ROWS, COLS
 
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=4),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
 
model = Sequential() 
 
#model.add(Convolution1D(32, 3, activation='relu', padding="valid", input_shape=(COLS, RAWS), data_format='channels_first'))

#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(N_CLASSES))
#model.add(Activation('softmax'))

#model.add(Convolution1D(nb_filter=32, kernel_size=3, activation='relu', padding="valid", input_shape=(ROWS, COLS)))
#model.add(BatchNormalization())
#model.add(Flatten())
#model.add(Dropout(0.4))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(1024, activation='relu'))
#model.add(Dense(N_CLASSES))
#model.add(Activation('softmax'))

#model.add(Conv1D(64, 3, activation='relu', input_shape=(ROWS, COLS)))
#model.add(Conv1D(64, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(2048, 3, activation='relu'))
#model.add(Conv1D(1024, 3, activation='relu'))
#model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.5))
#model.add(Dense(N_CLASSES, activation='softmax'))


model.add(Conv1D(32, 3, activation='relu', padding="valid", input_shape=(ROWS, COLS)))
model.add(BatchNormalization())
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(N_CLASSES, activation='softmax'))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.summary()
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, N_CLASSES)
Y_val = np_utils.to_categorical(y_val, N_CLASSES)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols)

model.fit(X_train, Y_train,
          batch_size=batch_size, 
          epochs=nb_epoch, verbose=1,
          callbacks=callbacks,
          validation_data=(X_val, Y_val))
#try predict from the validation set
#note that now Y contains a vector which represents the label. it's not a scalar 1,0 anymore

#plot5imgs(X_val, Y_val, 10)
#plt.imshow(X_val[num+i*20][0])

x = np.expand_dims(X_val[num], axis=0)
#print(x)
accuracy = model.evaluate(x=X_val,y=Y_val,batch_size=32)
print ("Accuracy: ", accuracy[1])

model.predict(x)
#try predict from external data
data=pd.read_csv("../input/out-normal-position-others/out_normal_position_others.csv", names = COL_VALUES + ["obj"])
data=normalize(data)
random_chair=pd.read_csv("../input/prova-chair/prova_randomchair.csv", names = COL_VALUES + ["obj"])
random_chair=normalize(random_chair)
random_door=pd.read_csv("../input/weird-toilet-out/weird_door_toilet_out.csv", names = COL_VALUES + ["obj"])
random_door=normalize(random_door)






print ("\n\n========================= Others =====================")

X = []
y = []
TEST_CLASSES=5
#create the images for each class
for c in range(0, TEST_CLASSES):
    df=data[data['obj'] == c]
    generate_images(X, y, df, ROWS, COLS, c)
a = []
a = unison_shuffled_copies(np.array(X), np.array(y))
X = a[0]
y = a[1]  
y = np.array(y)
X = np.array(X)
num = 0
#plot5imgs(X,y,num)
#for i in range(0,9):
 #   x = np.expand_dims(X[num+i], axis=0)
  #  x = np.expand_dims(x, axis=0)
   # print ("{}: {}".format(i, model.predict(x)))
    
y = np_utils.to_categorical(y, N_CLASSES)
X = X.reshape(X.shape[0], img_rows, img_cols)

accuracy = model.evaluate(x=X,y=y,batch_size=16)
print ("Accuracy Normal Others: ", accuracy[1])


print ("\n\n========================= Door =====================")
X = []
y = []
#create the images for each class
df=random_door[random_door['obj'] == 0]
generate_images(X, y, df, ROWS, COLS, 0)
a = []
a = unison_shuffled_copies(np.array(X), np.array(y))
X = a[0]
y = a[1]  
y = np.array(y)
X = np.array(X)
num = 0
#plot5imgs(X,y,num)
#for i in range(0,9):
 #   x = np.expand_dims(X[num+i], axis=0)
  #  x = np.expand_dims(x, axis=0)
   # print ("{}: {}".format(i, model.predict(x)))
    
y = np_utils.to_categorical(y, N_CLASSES)
X = X.reshape(X.shape[0], img_rows, img_cols)

accuracy = model.evaluate(x=X,y=y,batch_size=16)
print ("Accuracy Random Door: ", accuracy[1])
X = np.expand_dims(X[2], axis=0)
print (model.predict(X))
print (y[2])


print ("\n\n========================= Chair =====================")
X = []
y = []
#create the images for each class
df=random_chair[random_chair['obj'] == 1]
generate_images(X, y, df, ROWS, COLS, 1)
a = []
a = unison_shuffled_copies(np.array(X), np.array(y))
X = a[0]
y = a[1]  
y = np.array(y)
X = np.array(X)
y = np_utils.to_categorical(y, N_CLASSES)
X = X.reshape(X.shape[0], img_rows, img_cols)
accuracy = model.evaluate(x=X,y=y,batch_size=16)
print ("Accuracy Random Chair: ", accuracy[1])
X = np.expand_dims(X[0], axis=0)
print (model.predict(X))
print (y[0])

del X
del y
del a
del data
gc.collect()
