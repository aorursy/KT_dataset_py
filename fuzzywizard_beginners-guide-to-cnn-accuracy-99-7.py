# Ignore warnings :

import warnings

warnings.filterwarnings('ignore')





# Handle table-like data and matrices :

import numpy as np

import pandas as pd

import math 

import itertools







# Modelling Algorithms :



# Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis , QuadraticDiscriminantAnalysis



# Regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV, ElasticNet

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor 

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor









# Modelling Helpers :

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score







#preprocessing :

from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder







#evaluation metrics :



# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 



# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  





# Deep Learning Libraries

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.utils import to_categorical







# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno







# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plt.rcParams.update(params)
# Center all plots

from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""");
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

df = train.copy()

df_test = test.copy()
# How the Data looks

df.head()
print("Train: ", df.shape)

print("Test: ", df_test.shape)
df.describe()
# Train

df.isnull().any().sum()
# Test

df_test.isnull().any().sum()
# Generate Random Numbers

rdm = np.random.randint(0,42000,size=4)

print(rdm)
fig, ax = plt.subplots(2,2, figsize=(12,6))

fig.set_size_inches(10,10)

ax[0,0].imshow(df.drop('label',axis=1).values[rdm[0]].reshape(28,28),cmap='gray')

ax[0,1].imshow(df.drop('label',axis=1).values[rdm[1]].reshape(28,28),cmap='gray')

ax[1,0].imshow(df.drop('label',axis=1).values[rdm[2]].reshape(28,28),cmap='gray')

ax[1,1].imshow(df.drop('label',axis=1).values[rdm[3]].reshape(28,28),cmap='gray')
df['label'].value_counts()
plt.figure(figsize=(13,7))

plt.hist( x=df['label'] , bins=19 ,color='c')

plt.xlabel('Digits')

plt.ylabel('Frequency')

plt.title('Distribution of Label\'s')
## Setting the seeds for Reproducibility.

seed = 66

np.random.seed(seed)
X = train.iloc[:,1:]

Y = train.iloc[:,0]

x_train , x_test , y_train , y_test = train_test_split(X, Y , test_size=0.1, random_state=seed)
# The first parameter in reshape indicates the number of examples.

# We pass it as -1, which means that it is an unknown dimension and we want numpy to figure it out.



# reshape(examples, height, width, channels)

x_train = x_train.values.reshape(-1, 28, 28, 1)

x_test = x_test.values.reshape(-1, 28, 28, 1)

df_test=df_test.values.reshape(-1,28,28,1)


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
df.dtypes.head()
# You need to make sure that your Image is cast into double/float from int before you do this scaling 

# as you will most likely generate floating point numbers.

# And had it been int, the values will be truncated to zero.



x_train = x_train.astype("float32")/255

x_test = x_test.astype("float32")/255

df_test = df_test.astype("float32")/255
datagen.fit(x_train)
y_train = to_categorical(y_train, num_classes=10)

y_test = to_categorical(y_test, num_classes=10)



print(y_train[0])
# Building ConvNet



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',

                 input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))

model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
# Optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999 )
# Compiling the model

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
batch_size = 64

epochs = 20
# Fit the Model

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, 

                              validation_data = (x_test, y_test), verbose=2, 

                              steps_per_epoch=x_train.shape[0] // batch_size,

                              callbacks = [reduce_lr])
model.evaluate(x_test, y_test)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Test'])

plt.show()
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Test'])

plt.show()
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(x_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
pred_digits_test=np.argmax(model.predict(df_test),axis=1)

image_id_test=[]

for i in range (len(pred_digits_test)):

    image_id_test.append(i+1)

d={'ImageId':image_id_test,'Label':pred_digits_test}

answer=pd.DataFrame(d)

answer.to_csv('answer.csv',index=False)