# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/training/training.csv")

test_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/test/test.csv")

sample_submit_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/SampleSubmission.csv")

lookup_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")
train_df.columns.tolist()
test_df.head()




# imag = []

# for i in range(0,7049):

#     img = train_data['Image'][i].split(' ')

#     img = ['0' if x == '' else x for x in img]

#     imag.append(img)

    

    





img_list = []

for i in range(len(train_df.Image)):

    



    img  = np.array(train_df.Image[i].split(" "),dtype=float)

    img = img.reshape(-1,96,96,1)

    img_list.append(img)



for im in img_list[:5]:

    

    plt.imshow(im.reshape(96,96),cmap='gray')

    plt.show()
#Test data



test_img_list =[]

for i in range(len(test_df.Image)):

    img  = np.array(test_df.Image[i].split(" "),dtype=float)

    img = img.reshape(96,96,1)

    test_img_list.append(img)
test_df.count()

len(test_img_list)
#showing images of test data set

for im in test_img_list[:5]:

    plt.imshow(im.reshape(96,96),cmap='gray')

    plt.show()
# plotting points in image

# For Train Data



cols= train_df.columns.tolist()



# for i in range(len(img_list)):

#     plt.imshow(img_list[i].reshape(96,96),cmap='gray')



#     for j in range(0,len(cols)-1,2):

#         plt.scatter(train_df[cols[j]][i],train_df[cols[j+1]][i],color='red')



#     plt.show()


plt.imshow(img_list[1].reshape(96,96),cmap='gray')



for j in range(0,len(cols)-1,2):

    plt.scatter(train_df[cols[j]][1],train_df[cols[j+1]][1],color='green')



plt.show()
# train_df[train_df.isnull()]



train_df.isnull().sum()/len(train_df)

train_df.isnull().sum()
drop_df = train_df.dropna(axis=0).reset_index()

# len(drop_df)

len(drop_df)/len(train_df)

# we have only 30 percent data remain for now if we removed all the rows conataining nan values
# extra index was came so i removed the previous index and reset 

drop_df.drop(['index'],axis=1)

# drop df has images data to be converted into list

drop_img_list =[]

for i in range(len(drop_df.Image)):

    dimg  = np.array(drop_df.Image[i].split(" "),dtype=float)

    dimg = dimg.reshape(96,96,1)

    drop_img_list.append(dimg)

#     print(img)

# print(len(drop_df.Image))



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Activation

from keras.layers import Flatten, Dense

from keras.layers.normalization import BatchNormalization

from keras import optimizers

from keras.callbacks import ModelCheckpoint, History





def the_model():

    model = Sequential()

    

    model.add(Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(96, 96, 1))) # Input shape: (96, 96, 1)

    

    model.add(MaxPooling2D(pool_size=2))

    

    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    

    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    

    # Convert all values to 1D array

    model.add(Flatten())

    

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))



    model.add(Dense(30))

    

    return model
# for data in train_df[:1]:

#     print(data)

    

# x_train =drop_df.T[-1:].T







x_train = np.array(drop_img_list)





y_train = drop_df.T[:-1].T



# y train has extra index columns so we have to remove this

y_train = y_train.drop(['index'],axis=1)



print(len(x_train))

print(len(y_train))



print('x shape ',x_train.shape)

print('y shape ',y_train.shape)

y_train
# x_train.reshape(96,96,1)

# y_train.shape
# print("Training datapoint shape: X_train.shape:{}".format(x_train.shape))

# print("Training labels shape: y_train.shape:{}".format(y_train.shape))



epochs = 60

batch_size = 64



model = the_model()

hist = History()



checkpointer = ModelCheckpoint(filepath='checkpoint1.hdf5', 

                               verbose=1, save_best_only=True)



# training the model

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])



# new_x_train = x_train[1:]

model_fit = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, hist], verbose=1)



model.save('model1.h5')
# %reload_ext tensorboard.notebook

# %tensorboard --logdir logs
# test_df.shape

len(test_img_list)
test_imgs = np.array(test_img_list)

test_imgs.shape

res= model.predict(test_imgs)

len(res)
# res[0]



for x in range(100):

    res_img =x



    # print(res[2])

    plt.imshow(test_imgs[res_img].reshape(96,96),cmap='gray')

    for i in range(0,len(res[res_img]),2):



        plt.scatter(res[res_img][i],res[res_img][i+1],color='green')



    plt.show()

# except image column

main_cols = cols[:-1]

new_lookup_df=pd.DataFrame(columns=['RowId', 'ImageId', 'FeatureName','Location'])

new_lookup_df=lookup_df







# lookup_df['ImageId'].unique()



for RowId,ImageId,FeatureName,Location in lookup_df.values:



#     print(res[ImageId][main_cols.index(str(FeatureName))])

    new_lookup_df['Location'][RowId]=res[int(ImageId)-1][main_cols.index(str(FeatureName))]

    

#     new_lookup_df = new_lookup_df.append({'RowId':RowId,'ImageId':ImageId,'FeatureName':FeatureName,'Location':res[RowId][main_cols.index(str(FeatureName))]},ignore_index=True)

#     new_lookup_df[RowId]= res[ImageId][main_cols.index(str(FeatureName))]

    



print(new_lookup_df)



#     print( )

    



#         print(res[ImageId][j])

#         if FeatureName == colName:

# #             print('yes')

#             print(FeatureName)



new_lookup_df = new_lookup_df.drop(['ImageId','FeatureName'],axis=1)

new_lookup_df.to_csv('output.csv')



from IPython.display import FileLink

FileLink(r'output.csv')