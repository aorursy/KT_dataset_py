# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import missingno as msno

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 

import tensorflow as tf
import random as rn
train=pd.read_csv(r'../input/train.csv')
test=pd.read_csv(r'../input/test.csv')
df=train.copy()
df_test=test.copy()
df.head()
df.shape
df_test.shape 
df['label'].value_counts()
sns.factorplot(data=df,kind='count',x='label',size=5,aspect=1.5)
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
count=0
for i in range(5):
    for j in range (2):
        ax[i,j].imshow(df.drop('label',axis=1).values[count].reshape(28,28),cmap='gray')
        count+=1
X=df.drop('label',axis=1).values
Y=df['label'].values
X
Y=to_categorical(Y,10)  
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
x_train=x_train/255  
x_test=x_test/255
num_test=df_test.values
num_test=num_test/255
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
num_test=num_test.reshape(num_test.shape[0],28,28,1)
x_train.shape
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
# modelling starts using a CNN.

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',data_format="channels_last",activation='relu',
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',data_format="channels_last",activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format="channels_last"))
model.add(Dropout(0.20))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',data_format="channels_last",activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',data_format="channels_last",activation='relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format="channels_last"))
model.add(Dropout(0.20))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',data_format="channels_last",activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',data_format="channels_last",activation='relu'))
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format="channels_last"))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.50))
model.add(Dense(10, activation='softmax'))

batch_size=64
epochs=20
from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1,min_lr=0.0001)
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


datagen.fit(x_train)
model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])  
model.summary()
History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size,callbacks=[red_lr])

pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)
image_id=[]
for i in range (len(pred_digits)):
    image_id.append(i+1)
len(image_id)
pred_digits
model.evaluate(x_test, y_test)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==6):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==6):
        break

count=0
fig,ax=plt.subplots(3,2)
fig.set_size_inches(10,10)
for i in range (3):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]].reshape(28,28),cmap='gray')
        ax[i,j].set_title("Predicted Label : "+str(pred_digits[prop_class[count]])+"\n"+"Actual Label : "+str(np.argmax(y_test[prop_class[count]])))
        plt.tight_layout()
        count+=1
        
count=0
fig,ax=plt.subplots(3,2)
fig.set_size_inches(10,10)
for i in range (3):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]].reshape(28,28),cmap='gray')
        ax[i,j].set_title("Predicted Label : "+str(pred_digits[mis_class[count]])+"\n"+"Actual Label : "+str(np.argmax(y_test[mis_class[count]])))
        plt.tight_layout()
        count+=1
        
pred_digits_test=np.argmax(model.predict(num_test),axis=1)
image_id_test=[]
for i in range (len(pred_digits_test)):
    image_id_test.append(i+1)
d={'ImageId':image_id_test,'Label':pred_digits_test}
answer=pd.DataFrame(d)
answer.to_csv('answer.csv',index=False)

