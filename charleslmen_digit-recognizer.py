# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Import train datastet
train = pd.read_csv('../input/train.csv')
train.head()
test=pd.read_csv('../input/test.csv')
test.head()
#Checking missing values
def find_missing(train,test):
    count_missing_train=train.isnull().sum().values
    count_missing_test=train.isnull().sum().values
    total_train=train.shape[0]
    total_test=test.shape[0]
    ratio_missing_train=count_missing_train/total_train*100
    ratio_missing_test=count_missing_test/total_test*100
    return pd.DataFrame({'Missing_train':count_missing_train,'Missing_Ratio_Train':ratio_missing_train,
                        'Missing_test':count_missing_test,'Missing_Ratio_Test':ratio_missing_test},
                       index=train.columns)
df_missing=find_missing(train.drop(columns='label',axis=1),test)
df_missing=df_missing[df_missing['Missing_Ratio_Train']>0].sort_values(by='Missing_Ratio_Train',ascending=False)
df_missing.head()
# There is no missing values in train and test sets
# Splitting to target
train_target=train.iloc[:,0].values
# Normalization
train=train.drop(columns='label')/255.0
test=test/255.0
# Reshape
train=train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
# Encoding the target
train_target=pd.get_dummies(train_target)
# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,train_target,test_size=0.2,random_state=0)
# Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier=Sequential()
# Convolution
classifier.add(Convolution2D(32,5,5,input_shape=(28,28,1),activation='relu'))
classifier.add(Convolution2D(32,5,5,input_shape=(28,28,1),activation='relu'))
# Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# Another Convolution
classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(Convolution2D(64,3,3,activation='relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

# Flatten
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=10,activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_datagen.fit(train)
classifier.fit_generator(
train_datagen.flow(train,train_target,batch_size=84),nb_epoch=2,
    validation_data=(x_train,y_train),verbose=2,samples_per_epoch=train.shape[0])
predictions=classifier.predict(test)
predictions=np.argmax(predictions,axis=1)
predictions=pd.Series(predictions)
submission=pd.DataFrame({'ImageId':range(1,28001)})
submission['label']=predictions
submission.to_csv('submission.csv',index=False)
