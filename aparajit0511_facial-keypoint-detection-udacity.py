# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pandas.io.parsers import read_csv

from sklearn.utils import shuffle



FTRAIN = '../input/training/training.csv'

FTEST = '../input/test/test.csv'





def load(test=False, cols=None):

    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.

    Pass a list of *cols* if you're only interested in a subset of the

    target columns.

    """

    fname = FTEST if test else FTRAIN

    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe



    # The Image column has pixel values separated by space; convert

    # the values to numpy arrays:

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))



    if cols:  # get a subset of columns

        df = df[list(cols) + ['Image']]



    print(df.count())  # prints the number of values for each column

    df = df.dropna()  # drop all rows that have missing values in them



    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]

    X = X.astype(np.float32)



    if not test:  # only FTRAIN has any target columns

        y = df[df.columns[:-1]].values

        y = (y - 48) / 48  # scale target coordinates to [-1, 1]

        X, y = shuffle(X, y, random_state=42)  # shuffle train data

        y = y.astype(np.float32)

    else:

        y = None



    return X, y





X, y = load()

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(

    X.shape, X.min(), X.max()))

print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(

    y.shape, y.min(), y.max()))
#  I changed the X dimension structure to have (Nsample, Nrows in frame, N columns in frame, 1) in load2d.

def load2d(test=False,cols=None):



    re = load(test, cols)

    

    X = re[0].reshape(-1,96,96,1)

    y = re[1]



    return X, y
%%time

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.optimizers import SGD

from keras.layers import Dropout



model = Sequential()

model.add(Dense(128,input_dim=X.shape[1]))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dense(30))



model.summary()
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd)



hist = model.fit(X, y, nb_epoch=50,batch_size=128, validation_split=0.2,verbose=False)


import matplotlib.pyplot as plt

%matplotlib inline



def plot_loss(hist,name,plt,RMSE_TF=False):

    '''

    RMSE_TF: if True, then RMSE is plotted with original scale 

    '''

    loss = hist['loss']

    val_loss = hist['val_loss']

    if RMSE_TF:

        loss = np.sqrt(np.array(loss))*48 

        val_loss = np.sqrt(np.array(val_loss))*48 

        

    plt.plot(loss,"--",linewidth=3,label="train:"+name)

    plt.plot(val_loss,linewidth=3,label="val:"+name)



plot_loss(hist.history,"model 1",plt)

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("log loss")

plt.show()
X_test, _ = load(test=True)

y_test = model.predict(X_test)
#converting the images back to 96*96 pixels so i can check the performance of my model on the image dataset



def plot_sample(x, y, axis):

    img = x.reshape(96, 96)

    axis.imshow(img, cmap='gray')

    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)





fig = plt.figure(figsize=(10, 7))

fig.subplots_adjust(

    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)



for i in range(16):

    axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])

    plot_sample(X_test[i], y_test[i], axis)



plt.show()
from keras.models import load_model

# import h5py

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# del model  # deletes the existing model



# # returns a compiled model

# # identical to the previous one

model = load_model('my_model.h5')
X.shape
# #Deleting these as i will use a new structure



# del X, y, X_test, y_test
X,y = load2d()
print(X.shape)

print(y.shape)
from keras.layers import MaxPooling2D, Conv2D , Flatten, Dropout

from keras.layers.normalization import BatchNormalization
def CNN():

    model2 = Sequential()



    model2.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(96,96,1)))

    model2.add(Dropout(0.1))

    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

    model2.add(BatchNormalization())



    model2.add(Conv2D(32, 5, 5,activation="relu"))

    # model.add(Activation("relu"))

    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

    model2.add(Dropout(0.2))

    model2.add(BatchNormalization())



    model2.add(Conv2D(64, 5, 5,activation="relu"))

    # model.add(Activation("relu"))

    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

    model2.add(BatchNormalization())



    model2.add(Conv2D(128, 3, 3,activation="relu"))

    # model.add(Activation("relu"))

    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid"))

    model2.add(Dropout(0.4))

    model2.add(BatchNormalization())



    model2.add(Flatten())



    model2.add(Dense(500, activation="relu"))

    model2.add(Dropout(0.1))



    model2.add(Dense(128, activation="relu"))

    model2.add(Dropout(0.1))



    model2.add(Dense(30))





    model2.summary()

    model2.compile(optimizer='adam', 

              loss='mse',

              metrics=['mae','accuracy'])

    return(model2)
model2 = CNN()

hist2 = model2.fit(X, y, nb_epoch=500,batch_size=128, validation_split=0.2,verbose=False)
# print(hist2.history)
# hist2.history
# Comparing model1 and model2

plt.figure(figsize=(4,4))

plot_loss(hist.history,"model 1",plt)

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()



plot_loss(hist2.history,"model 2",plt)

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()
sample1,_ = load(test=True)

sample2,_ = load2d(test=True)

y_pred1 = model.predict(sample1)

y_pred2 = model2.predict(sample2)
#Comparing model1 and model2 on images



fig = plt.figure(figsize=(10,10))

fig.subplots_adjust(hspace=0.001,wspace=0.001,

                    left=0,right=1,bottom=0, top=1)

Npicture = 5

count = 1

for irow in range(Npicture):

    ipic = np.random.choice(sample2.shape[0])

    ax = fig.add_subplot(Npicture, 2, count,xticks=[],yticks=[])        

    plot_sample(sample1[ipic],y_pred1[ipic],ax)

    if count < 3:

        ax.set_title("model 1")

        

    count += 1

    ax = fig.add_subplot(Npicture, 2, count,xticks=[],yticks=[])  

    plot_sample(sample2[ipic],y_pred2[ipic],ax)

    if count < 3:

        ax.set_title("model 2")

    count += 1

plt.show()
model2.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'

# del model  # deletes the existing model



# # returns a compiled model

# # identical to the previous one

model2 = load_model('my_model2.h5')
# from keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,

#     samplewise_center=False,

#     featurewise_std_normalization=False,

#     samplewise_std_normalization=False,

#     zca_whitening=False,

#     rotation_range=0.,

#     width_shift_range=0.,

#     height_shift_range=0.,

#     shear_range=0.,

#     zoom_range=0.,

#     channel_shift_range=0.,

#     fill_mode='nearest',

#     cval=0.,

#     horizontal_flip=False,

#     vertical_flip=False,

#     dim_ordering='th')
## Using ImageDataGenerator to flip the images and flip indices will be used to manually flipping

## the keypoints on the face



from keras.preprocessing.image import ImageDataGenerator

class FlippedImageDataGenerator(ImageDataGenerator):

    flip_indices = [

        (0, 2), (1, 3),

        (4, 8), (5, 9), (6, 10), (7, 11),

        (12, 16), (13, 17), (14, 18), (15, 19),

        (22, 24), (23, 25),

        ]



    def next(self):

        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()

        batch_size = X_batch.shape[0]

        indices = np.random.choice(batch_size, batch_size/2, replace=False)

        X_batch[indices] = X_batch[indices, :, :, ::-1]



        if y_batch is not None:

            

            y_batch[indices, ::2] = y_batch[indices, ::2] * -1



            # left_eye_center_x -> right_eye_center_x のようにフリップ

            for a, b in self.flip_indices:

                y_batch[indices, a], y_batch[indices, b] = (

                    y_batch[indices, b], y_batch[indices, a]

                )



        return X_batch, y_batch
## splitting the data

from sklearn.model_selection import train_test_split



X, y = load2d()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model3 = CNN()

flipgen = FlippedImageDataGenerator()

hist3 = model3.fit_generator(flipgen.flow(X_train, y_train),

                             samples_per_epoch=X_train.shape[0],

                             nb_epoch=300,

                             validation_data=(X_val, y_val))



## Comparing mode1, model2 and model3 using pyplot

plt.figure(figsize=(8,8))

plot_loss(hist.history,"model 1",plt)

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()



plot_loss(hist2.history,"model 2",plt)

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()



plot_loss(hist3.history,"model 3",plt)

plt.legend()

plt.grid()

plt.yscale("log")

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()
X_train.shape
model3.save('my_model3.h5')  # creates a HDF5 file 'my_model.h5'

# del model  # deletes the existing model



# # returns a compiled model

# # identical to the previous one

model3 = load_model('my_model3.h5')
SPECIALIST_SETTINGS = [

    dict(

        columns=(

            'left_eye_center_x', 'left_eye_center_y',

            'right_eye_center_x', 'right_eye_center_y',

            ),

        flip_indices=((0, 2), (1, 3)),

        ),



    dict(

        columns=(

            'nose_tip_x', 'nose_tip_y',

            ),

        flip_indices=(),

        ),



    dict(

        columns=(

            'mouth_left_corner_x', 'mouth_left_corner_y',

            'mouth_right_corner_x', 'mouth_right_corner_y',

            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',

            ),

        flip_indices=((0, 2), (1, 3)),

        ),



    dict(

        columns=(

            'mouth_center_bottom_lip_x',

            'mouth_center_bottom_lip_y',

            ),

        flip_indices=(),

        ),



    dict(

        columns=(

            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',

            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',

            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',

            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',

            ),

        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),

        ),



    dict(

        columns=(

            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',

            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',

            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',

            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',

            ),

        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),

        ),

    ]

from collections import OrderedDict



def fit_specialists(freeze=True,

                    print_every=10,

                    epochs=100,

                    prop=0.1,

                    name_transfer_model="my_model3.h5"):

    specialists = OrderedDict()

 



    for setting in SPECIALIST_SETTINGS:

        

        cols = setting['columns']

        flip_indices = setting['flip_indices']

        

        X, y = load2d(cols=cols)

        X_train, X_val, y_train, y_val = train_test_split(X, y, 

                                                          test_size=0.2, 

                                                          random_state=42)

        model4 = load_model(name_transfer_model) 

        if freeze:

            for layer in model.layers:

                layer.trainable = False

            

        model4.layers.pop() # get rid of output layer

        model4.outputs = [model4.layers[-1].output]

        model4.layers[-1].outbound_nodes = []

        model4.add(Dense(len(cols))) # add new output layer



        model4.compile(loss='mean_squared_error', optimizer="adam")

        

        flipgen = FlippedImageDataGenerator()

        flipgen.flip_indices = setting['flip_indices']

        print(X_train.shape)

        print(y_train.shape)

        print(X_val.shape)

        print(y_val.shape)

        hist_final = model4.fit_generator(flipgen.flow(X_train, y_train),

                                     samples_per_epoch=X_train.shape[0],

                                     nb_epoch=epochs,

                                     validation_data=(X_val, y_val))

        

        ## print(model.summary()) 

        

       

        specialists[cols] = model4

    return(specialists)

%%time

specialists1 = fit_specialists(freeze=True,

                    print_every=10,

                    epochs=100,

                    name_transfer_model="my_model3.h5")
X_train.shape
type(specialists1)
from pandas import DataFrame, concat



X_test,_ = load2d(test=True)



## prediction with model 3

y_pred3 = model3.predict(X_test)

landmark_nm = read_csv(os.path.expanduser(FTRAIN)).columns[:-1].values

df_y_pred3 = DataFrame(y_pred3,columns = landmark_nm)



## prediction with specialist model

def predict_specialist(specialists1,X_test):

    y_pred_s = []

    for columns, value in specialists1.items():

        smodel = value



        y_pred = smodel.predict(X_test)

        y_pred = DataFrame(y_pred,columns=columns)

        y_pred_s.append(y_pred)



    df_y_pred_s = concat(y_pred_s,axis=1)

    return(df_y_pred_s)

df_y_pred_s = predict_specialist(specialists1,X_test)

y_pred_s = df_y_pred_s.values
FIdLookup = '../input/IdLookupTable.csv'



IdLookup = read_csv(os.path.expanduser(FIdLookup))



def prepare_submission(y_pred4,filename):

    '''

    save a .csv file that can be submitted to kaggle

    '''

    ImageId = IdLookup["ImageId"]

    FeatureName = IdLookup["FeatureName"]

    RowId = IdLookup["RowId"]

    

    submit = []

    for rowId,irow,landmark in zip(RowId,ImageId,FeatureName):

        submit.append([rowId,y_pred4[landmark].iloc[irow-1]])

    

    submit = DataFrame(submit,columns=["RowId","Location"])

    ## adjust the scale 

    submit["Location"] = submit["Location"]*48 + 48

    print(submit.shape)

#     loc = "result/" + filename + ".csv"

    if filename == "model3":

       submit.to_csv("model3.csv",index=False) 

    else:

        submit.to_csv("special.csv",index=False)

    

#     print("File is saved at:" +  loc)



prepare_submission(df_y_pred_s,"special")    

prepare_submission(df_y_pred3,"model3")



# sample3,_ = load2d(test=True)

# sample_special,_ = load2d(test=True)



# y_pred3 = model3.predict(X_test)



# def predict_specialist_case(specialists1,X_test):

#     for columns, value in specialists1.items():

#         smodel = value



#         y_pred_sample = smodel.predict(X_test)

#         return y_pred_sample



# y_pred_sample = predict_specialist_case(specialists1,X_test)







# fig = plt.figure(figsize=(12, 20))

# fig.subplots_adjust(hspace=0.001,wspace=0.001,

#                     left=0,right=1,bottom=0, top=1)

# Npicture = 7

# count = 1

# for irow in range(Npicture):

#     ipic = np.random.choice(X_test.shape[0])

#     ax = fig.add_subplot(Npicture, 2, count,xticks=[],yticks=[])        

#     plot_sample(X_test[ipic],y_pred3[ipic],ax)

#     if count < 3:

#         ax.set_title("model 3")

        

#     count += 1

#     ax = fig.add_subplot(Npicture, 2, count,xticks=[],yticks=[])  

#     plot_sample(X_test[ipic],y_pred_sample[ipic],ax)

#     if count < 3:

#         ax.set_title("special model")

#     count += 1

# plt.show()
df_y_pred_s = df_y_pred_s[df_y_pred3.columns]

df_compare = {}

df_compare["difference"] = ((df_y_pred_s - df_y_pred3)**2).mean(axis=1)

df_compare["RowId"] = range(df_y_pred_s.shape[0])

df_compare = DataFrame(df_compare)

df_compare = df_compare.sort_values("difference",ascending=False)
fig = plt.figure(figsize=(12,35))



Nsample = 13

pic_index = df_compare["RowId"].iloc[:Nsample].values

pic_index_good = df_compare["RowId"].iloc[-Nsample:].values

count = 1





for ipic_g,ipic in zip(pic_index_good,pic_index):

    ## good model 3

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic_g],y_pred3[ipic_g],ax)

    ax.set_title("Good:model3:pic"+str(ipic_g))

    

    ## good special

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic_g],y_pred_s[ipic_g],ax)

    ax.set_title("Good:special:pic"+str(ipic_g))

    

    ## bad model 3

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic],y_pred3[ipic],ax)

    ax.set_title("Bad:model3:pic"+str(ipic))

    

    ## bad special

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic],y_pred_s[ipic],ax)

    ax.set_title("Bad:special:pic"+str(ipic))



plt.show()
def fit_specialists(freeze=True,

                    print_every=10,

                    epochs=100,

                    prop=0.1,

                    name_transfer_model="my_model2.h5"):

    specialists = OrderedDict()

 



    for setting in SPECIALIST_SETTINGS:

        

        cols = setting['columns']

        flip_indices = setting['flip_indices']

        

        X, y = load2d(cols=cols)

        X_train, X_val, y_train, y_val = train_test_split(X, y, 

                                                          test_size=0.2, 

                                                          random_state=42)

        model4 = load_model(name_transfer_model) 

        if freeze:

            for layer in model.layers:

                layer.trainable = False

            

        model4.layers.pop() # get rid of output layer

        model4.outputs = [model4.layers[-1].output]

        model4.layers[-1].outbound_nodes = []

        model4.add(Dense(len(cols))) # add new output layer



        model4.compile(loss='mean_squared_error', optimizer="adam")

        

        flipgen = FlippedImageDataGenerator()

        flipgen.flip_indices = setting['flip_indices']

        

        print(X_train.shape)

        print(y_train.shape)

        print(X_val.shape)

        print(y_val.shape)

        

        hist_final = model4.fit_generator(flipgen.flow(X_train, y_train),

                                     samples_per_epoch=X_train.shape[0],

                                     nb_epoch=epochs,

                                     validation_data=(X_val, y_val))

        

        

       

        specialists[cols] = model4

    return(specialists)



%%time

specialists2 = fit_specialists(freeze=True,

                    print_every=10,

                    epochs=100,

                    name_transfer_model="my_model2.h5")
X_test,_ = load2d(test=True)



def predict_specialist(specialists2,X_test):

    y_pred_s = []

    for columns, value in specialists2.items():

        smodel = value



        y_pred = smodel.predict(X_test)

        y_pred = DataFrame(y_pred,columns=columns)

        y_pred_s.append(y_pred)



    df_y_pred_s = concat(y_pred_s,axis=1)

    return(df_y_pred_s)

df_y_pred_s = predict_specialist(specialists2,X_test)

y_pred_s = df_y_pred_s.values
## prediction with model 2

y_pred2 = model2.predict(X_test)

landmark_nm = read_csv(os.path.expanduser(FTRAIN)).columns[:-1].values

df_y_pred2 = DataFrame(y_pred2,columns = landmark_nm)
FIdLookup = '../input/IdLookupTable.csv'



IdLookup = read_csv(os.path.expanduser(FIdLookup))



def prepare_submission(y_pred2,filename):

    '''

    save a .csv file that can be submitted to kaggle

    '''

    ImageId = IdLookup["ImageId"]

    FeatureName = IdLookup["FeatureName"]

    RowId = IdLookup["RowId"]

    

    submit = []

    for rowId,irow,landmark in zip(RowId,ImageId,FeatureName):

        submit.append([rowId,y_pred2[landmark].iloc[irow-1]])

    

    submit = DataFrame(submit,columns=["RowId","Location"])

    ## adjust the scale 

    submit["Location"] = submit["Location"]*48 + 48

    print(submit.shape)



    if filename == "model2":

        submit.to_csv("model2.csv",index=False) 

    else:

        submit.to_csv("special_model2.csv",index=False)

    



prepare_submission(df_y_pred_s,"special_model2")    

prepare_submission(df_y_pred2,"model2")
df_y_pred_s = df_y_pred_s[df_y_pred3.columns]

df_compare = {}

df_compare["difference"] = ((df_y_pred_s - df_y_pred2)**2).mean(axis=1)

df_compare["RowId"] = range(df_y_pred_s.shape[0])

df_compare = DataFrame(df_compare)

df_compare = df_compare.sort_values("difference",ascending=False)
fig = plt.figure(figsize=(12,35))



Nsample = 13

pic_index = df_compare["RowId"].iloc[:Nsample].values

pic_index_good = df_compare["RowId"].iloc[-Nsample:].values

count = 1





for ipic_g,ipic in zip(pic_index_good,pic_index):

    ## good model 2

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic_g],y_pred2[ipic_g],ax)

    ax.set_title("Good:model2:pic"+str(ipic_g))

    

    ## good special

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic_g],y_pred_s[ipic_g],ax)

    ax.set_title("Good:special:pic"+str(ipic_g))

    

    ## bad model 2

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic],y_pred2[ipic],ax)

    ax.set_title("Bad:model2:pic"+str(ipic))

    

    ## bad special

    ax = fig.add_subplot(Nsample,4,count,xticks=[],yticks=[])

    count += 1

    plot_sample(X_test[ipic],y_pred_s[ipic],ax)

    ax.set_title("Bad:special:pic"+str(ipic))



plt.show()
# def fit_specialists(freeze=True,

#                     print_every=10,

#                     epochs=100,

#                     prop=0.1,

#                     name_transfer_model="my_model.h5"):

#     specialists = OrderedDict()

 



#     for setting in SPECIALIST_SETTINGS:

        

#         cols = setting['columns']

#         flip_indices = setting['flip_indices']

        

#         X, y = load2d(cols=cols)

# #         X.reshape(7049, 96, 96, 1)

#         X_train, X_val, y_train, y_val = train_test_split(X, y, 

#                                                           test_size=0.2, 

#                                                           random_state=42)

# #         X_val = np.expand_dims(X_val, axis=0)

#         model4 = load_model(name_transfer_model) 

#         if freeze:

#             for layer in model.layers:

#                 layer.trainable = False

            

#         model4.layers.pop() # get rid of output layer

#         model4.outputs = [model4.layers[-1].output]

#         model4.layers[-1].outbound_nodes = []

#         model4.add(Dense(len(cols))) # add new output layer



#         model4.compile(loss='mean_squared_error', optimizer="adam")

        

#         flipgen = FlippedImageDataGenerator()

#         flipgen.flip_indices = setting['flip_indices']

# #         print(y_val.shape)

# # #         X_val.reshape(1407,9216)

# # #         y_val.reshape()

# #         print(X_train.shape)

#         print(X_train.shape)

#         print(y_train.shape)

#         print(X_val.shape)

#         print(y_val.shape)

#         X_train.reshape(5626,9216)

#         X_val.reshape(1407,9216)

#         hist_final = model4.fit_generator(flipgen.flow(X_train, y_train),

#                                      samples_per_epoch=X_train.shape[0],

#                                      nb_epoch=epochs,

#                                      validation_data=(X_val, y_val))

    

       

#         specialists[cols] = model4

#     return(specialists)

# %%time

# specialists3 = fit_specialists(freeze=True,

#                     print_every=10,

#                     epochs=100,

#                     name_transfer_model="my_model.h5")