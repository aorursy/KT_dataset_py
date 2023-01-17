import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
t = pd.read_csv('../input/training/training.csv')
ts = pd.read_csv('../input/test/test.csv')
t.shape[0]
#get image

def get_img(df):
    imgs = []
    img = df.iloc[:,-1].str.split(' ')
    for j in range(0,df.shape[0]):
        im = [0 if i=='' else i for i in img[j]]
        imgs.append(im)
    imgs = np.array(imgs, dtype=float).reshape(-1,96,96,1)
    return imgs
X = get_img(t)/255
Y = np.array(t.drop('Image', axis=1).fillna(method='ffill'),dtype=float)
X_ts = get_img(ts)/255
#X[...,0][0]
def image(i):
    img = X[...,0][i]
    plt.imshow(img)
    pt = np.vstack(np.split(Y[i],15)).T
    plt.scatter(pt[0],pt[1],c='red',marker = '*')
image(5)
#import packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D,Flatten, LeakyReLU
#from keras.layers import LeakyReLU(alpha=0.3) as activation
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='linear', input_shape = (96,96,1)))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "linear"))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(128, activation = "linear"))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(30))
# Compile the model
model.compile(optimizer = 'adam' , loss = 'mse', metrics=['mae','accuracy'])
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.fit(X, Y, epochs=100, batch_size=128,validation_split = 0.2)
Y_ts = model.predict(X_ts)
def image_(i):
    img = X_ts[...,0][i]
    plt.imshow(img)
    pt = np.vstack(np.split(Y_ts[i],15)).T
    plt.scatter(pt[0],pt[1],c='red',marker = '*')
#image_(0)
image_(10)
look_id = pd.read_csv('../input/IdLookupTable.csv')
look_id.drop('Location',axis=1,inplace=True)
ind = np.array(t.columns[:-1])
value = np.array(range(0,30))
maps = pd.Series(value,ind)
look_id['location_id'] = look_id.FeatureName.map(maps)
df = look_id.copy()

location = pd.DataFrame({'Location':[]})
for i in range(1,1784):
    ind = df[df.ImageId==i].location_id
    location = location.append(pd.DataFrame(Y_ts[i-1][list(ind)],columns=['Location']), ignore_index=True)

look_id['Location']=location
look_id[['RowId','Location']].to_csv('Sub1.csv',index=False)