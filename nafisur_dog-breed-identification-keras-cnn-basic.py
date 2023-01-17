from keras.layers import Dense,Dropout,Input,MaxPooling2D,ZeroPadding2D,Conv2D,Flatten
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_colwidth=150
df1=pd.read_csv('../input/labels.csv')
df1.head()
# path of the dogs images
img_file='../input/train/'
df=df1.assign(img_path=lambda x: img_file + x['id'] +'.jpg')
df.head()
#Number of different breed type
df.breed.value_counts()
# Take a look at the class/breed distribution
ax=pd.value_counts(df['breed'],ascending=True).plot(kind='barh',fontsize="40",title="Class Distribution",figsize=(50,100))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)
plt.show()
#Top 20 breed
top_20=list(df.breed.value_counts()[0:20].index)
top_20
df2=df[df.breed.isin(top_20)]
df2.shape
img_pixel=np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in df2['img_path'].values.tolist()])
img_pixel.shape
img_label=df2.breed
img_label=pd.get_dummies(df2.breed)
img_label.head()
X=img_pixel
y=img_label.values
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow(X_train,y=y_train,batch_size=32)
testing_set=test_datagen.flow(X_test,y=y_test,batch_size=32)
model=Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(299,299,3)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(20,activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()
# history=model.fit_generator(training_set,
#                       steps_per_epoch = 16,
#                       validation_data = testing_set,
#                       validation_steps = 4,
#                       epochs = 2,
#                       verbose = 1)
