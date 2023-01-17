from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file

df1 = pd.read_csv('/kaggle/input/jaffe_pixels.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'jaffe_pixels.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
import os,cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10



from sklearn.utils import shuffle

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

import keras



from keras.utils import np_utils



from keras import backend as K



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam

from keras.preprocessing.image import ImageDataGenerator


data_path = '../input/jaffe/jaffe'

data_dir_list = os.listdir(data_path)



img_rows=256

img_cols=256

num_channel=1



num_epoch=10



img_data_list=[]





for dataset in data_dir_list:

    img_list=os.listdir(data_path+'/'+ dataset)

    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    for img in img_list:

        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        input_img_resize=cv2.resize(input_img,(128,128))

        img_data_list.append(input_img_resize)

        

img_data = np.array(img_data_list)

img_data = img_data.astype('float32')

img_data = img_data/255

img_data.shape
num_classes = 7



num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,),dtype='int64')



labels[0:29]=0 #30

labels[30:58]=1 #29

labels[59:90]=2 #32

labels[91:121]=3 #31

labels[122:151]=4 #30

labels[152:182]=5 #31

labels[183:]=6 #30



names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']



def getLabel(id):

    return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]
Y = np_utils.to_categorical(labels, num_classes)



#Shuffle the dataset

x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

x_test=X_test

#X_train=X_train.reshape(X_train.shape[0],128,128,1)

#X_test=X_test.reshape(X_test.shape[0],128,128,1)
from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization
input_shape=(128,128,3)



model = Sequential()



model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation = 'softmax'))





# Classification

# model.add(Flatten())

# model.add(Dense(64))

# model.add(Activation('relu'))

# model.add(Dropout(0.5))

# model.add(Dense(num_classes))

# model.add(Activation('softmax'))



#Compile Model

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
model.summary()

model.get_config()

model.layers[0].get_config()

model.layers[0].input_shape

model.layers[0].output_shape

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable
from keras import callbacks

filename='model_train_new.csv'

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"



csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]

callbacks_list = [csv_log]
hist = model.fit(X_train, y_train, batch_size=7, epochs=50, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
results = model.predict_classes(X_train)

print(results)
from keras import backend as K

get_3rd_layer_output = K.function([model.layers[0].input],[model.layers[7].output])

layer_output = get_3rd_layer_output([X_train])[0] 

print(layer_output)
# create array with hstack 

from numpy import array 

from numpy import hstack 

import numpy as np 

a1 = layer_output 

x1 = results  

#x2=y_train 

#a2=x2.reshape(-1,1) 

a3=x1.reshape(-1,1)  #print(a2) #a2.shape 

a4 = hstack((a1,a3)) 

print(a4)
import csv

d=a4  

csv.register_dialect('myDialect', quoting=csv.QUOTE_ALL, skipinitialspace=True)

with open('jjj.csv', 'w') as f:

    writer = csv.writer(f, dialect='myDialect')

    for row in d:

        writer.writerow(row)

    f.close()