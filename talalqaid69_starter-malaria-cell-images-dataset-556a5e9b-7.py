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

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#(os.listdir("../input"))

import glob

import cv2

import tensorflow as tf

from keras import layers

from keras.layers import Dropout , Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Convolution2D

from keras.models import Model, load_model,Sequential

from keras.initializers import glorot_uniform

from sklearn.model_selection import train_test_split

import keras.backend as K

from sklearn.utils import shuffle

# Any results you write to the current directory are saved as output.

print("yes")
X_para = []

Y_para = []

myfiles = glob.glob("../input/cell_images/cell_images/Parasitized/*.png")

for file in myfiles:

    #kernel = np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])

    #img = cv2.filter2D( cv2.resize(cv2.imread(file) , (120,120)) , -1 , kernel)

    #image_yuv = cv2.cvtColor(img ,cv2.COLOR_BGR2YUV )

    #image_yuv[: ,: , 0] = cv2.equalizeHist(image_yuv[:,:,0])

    #image = cv2.cvtColor(image_yuv , cv2.COLOR_YUV2RGB)

    image=cv2.resize(cv2.imread(file) , (128,128))

    X_para.append(image)

    Y_para.append(1)

    
X_un , Y_un = [],[]

unfiles = glob.glob("../input/cell_images/cell_images/Uninfected/*.png")

for file in unfiles:

    #kernel = np.array([[0,-1,0],[-1,6,-1],[0,-1,0]])

    #img = cv2.filter2D( cv2.resize(cv2.imread(file) , (120,120)) , -1 , kernel)

    #image_yuv = cv2.cvtColor(img ,cv2.COLOR_BGR2YUV )

    #image_yuv[: ,: , 0] = cv2.equalizeHist(image_yuv[:,:,0])

    #image = cv2.cvtColor(image_yuv , cv2.COLOR_YUV2RGB)

    image=cv2.resize(cv2.imread(file) , (128,128))

    X_un.append(image)

    Y_un.append(0)
X = X_para + X_un

Y = Y_para + Y_un

X,Y = shuffle = (X,Y)

X,Y = shuffle = (X,Y)

X,Y = shuffle = (X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3 , random_state =42)

X = np.array(X)
inp = Input(shape = (120 , 120 , 3))

x = Conv2D(filters = 16 , kernel_size = (3,3) , strides = (1,1) , padding = "valid" , kernel_initializer=glorot_uniform(seed = 2))(inp)

x = Activation("relu")(x)

x = Dropout(0.2)(x)

x = Conv2D(filters = 32 , kernel_size = (4,4) , strides = (2,2) , padding = "valid" , kernel_initializer=glorot_uniform(seed = 2))(x)

x = Activation("relu")(x)

x = MaxPooling2D(pool_size = (2,2) , strides = (2,2) , padding = "valid")(x)

x = Dropout(0.2)(x)

x = Conv2D(filters = 64 , kernel_size = (3,3) , strides = (2,2) , padding = "valid" , kernel_initializer = glorot_uniform(seed = 2))(x)

x = Activation("relu")(x)

x = Dropout(0.2)(x)

x = Conv2D(filters = 128 , kernel_size = (3,3) , strides = (1,1) , padding = "valid" , kernel_initializer = glorot_uniform())(x)

x = Activation("relu")(x)

x = MaxPooling2D(pool_size = (2,2) , strides = (2,2) , padding = "valid")(x)

x = Dropout(0.2)(x)

x = Conv2D(filters = 256 , kernel_size = (2,2) , strides = (2,2) , padding = "valid" , kernel_initializer = glorot_uniform())(x)

x = Activation("relu")(x)

x = AveragePooling2D(pool_size = (3,3) , strides = (1,1) , padding = "valid")(x)

x = Dropout(0.2)(x)

x = Flatten()(x)

x = Dense(120)(x)

x = Activation("relu")(x)

x = Dropout(0.2)(x)

x = Dense(60)(x)

x = Activation("relu")(x)

x = Dropout(0.2)(x)

x = Dense(10)(x)

x = Activation("relu")(x)

x = Dropout(0.)(x)

x = Dense(1)(x)

output = Activation("sigmoid")(x)

model  = Model(inputs =inp , outputs = output )

print("yes")
model.compile(loss = "binary_crossentropy" , optimizer = "adam" , metrics = ["accuracy"])

history = model.fit(np.array(X_train) ,np.array(Y_train) , epochs = 11 ,validation_split = 0.2 )

print("tes")
# Initialising the CNN

model = Sequential()



# Step 1 - Convolution

model.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))



# Step 2 - Pooling

model.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

model.add(Convolution2D(32, 3, 3, activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

model.add(Flatten())



# Step 4 - Full connection

model.add(Dense(output_dim = 128, activation = 'relu'))

model.add(Dense(output_dim = 1, activation = 'sigmoid'))

print("yes")

model.compile(loss = "binary_crossentropy" , optimizer = "adam" , metrics = ["accuracy"])

history = model.fit(np.array(X_train) ,np.array(Y_train) , epochs = 11 ,validation_split = 0.2 )

print("yes")

y_pre = model.predict(np.array(X_test))

y_pre = np.reshape(y_pre ,(len(y_pre),) )

Y_test = np.array(Y_test)

fil = y_pre >= 0.5

y_pre[fil] = 1

fil = y_pre < 0.5

y_pre[fil] = 0

print(np.sum(Y_test == y_pre)/len(y_pre))