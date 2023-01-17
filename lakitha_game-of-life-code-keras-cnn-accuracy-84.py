# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import Conv1D,Conv2D

from sklearn.preprocessing import MinMaxScaler

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.layers import Flatten

from keras.layers import Dropout,Activation

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

import pickle

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')

print(df_train.iloc[:,1:627].columns)

print(df_train.iloc[:,627:].columns)



for del_c in range(1,6):



    df_train_grouped=df_train.groupby('delta')

    selected_df_train=df_train_grouped.get_group(del_c)



    output_column_name_list=selected_df_train.iloc[:,1:627].columns



    input_data_df=pd.DataFrame()

    input_data_df=selected_df_train.iloc[:,627:]

    # input_data_df['delta']=selected_df_train['delta'].tolist()

    np_input_data=input_data_df.values



    # np_input_data=np_input_data.reshape(-1,1)



    print(f'Input Data {np_input_data}')

    print(f'Input shape {np_input_data.shape}')



    output_data_df=pd.DataFrame()

    output_data_df=selected_df_train.iloc[:,2:627]

    np_output_data=output_data_df.values

    # np_output_data=np_output_data.reshape(-1,1)



    print(f'Output Data {np_output_data}')

    print(f'Output shape {np_output_data.shape}')





    def reshape_input(X):

        return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)



    complete_input_array=[]





    for i in range(np_input_data.shape[0]):



        row_array=[]

        np_input = np.empty((0,25), int)

        column_count=1

        for j in range(np_input_data.shape[1]):



            if column_count < 26:

                row_array.append(np_input_data[i][j])



                if column_count == 25:

                    np_input=np.append(np_input,np.array([np.asarray(row_array)]), axis=0)

                    row_array=[]

                    column_count=0



            column_count=column_count+1







        complete_input_array.append(np_input)



    complete_input_array_=reshape_input(np.asarray(complete_input_array))





    complete_output_array=[]



    for i in range(np_output_data.shape[0]):



        row_array=[]

        np_input = np.empty((0,25), int)

        column_count=1

        for j in range(np_output_data.shape[1]):



            if column_count < 26:

                row_array.append(np_output_data[i][j])



                if column_count == 25:

                    np_input=np.append(np_input,np.array([np.asarray(row_array)]), axis=0)

                    row_array=[]

                    column_count=0



            column_count=column_count+1



        complete_output_array.append(np_input)

    complete_output_array_=reshape_input(np.asarray(complete_output_array))





    X_train, X_test, y_train, y_test = train_test_split(complete_input_array_, complete_output_array_, test_size=0.33, random_state=42)





    filters = 25

    kernel_size = (3, 3) # look at all 8 neighboring cells, plus itself

    strides = 1

    hidden_dims = 128



    model = Sequential()

    model.add(Conv2D(

        filters, 

        kernel_size,

        padding='same',

        activation='relu',

        strides=strides,

        input_shape=(25, 25, 1)

    ))

    model.add(Conv2D(

        filters, 

        kernel_size,

        padding='same',

        activation='relu',

        strides=strides,

        input_shape=(25, 25, 1)

    ))

    model.add(Dense(256))

    model.add(Activation('relu'))

    model.add(Dense(128))

    model.add(Activation('relu'))

    model.add(Dense(64))

    model.add(Activation('relu'))

    model.add(Dense(32))

    model.add(Activation('relu'))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size=256

    epochs=100

    model.fit(X_train, y_train, 

            batch_size=batch_size, 

            epochs=epochs,

            validation_data=(X_test, y_test)

        )

    

    model_name='delta_'+str(del_c)+'.h5'

    model.save(model_name)
def reshape_input(X):

    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)



df_test=pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/test.csv')

print(df_test)

delta_list=df_test['delta'].tolist()

np_input_data=df_test.iloc[:,2:].values

print(np_input_data)

complete_input_array=[]





for i in range(np_input_data.shape[0]):

    

    row_array=[]

    np_input = np.empty((0,25), int)

    column_count=1

    for j in range(np_input_data.shape[1]):

       

        if column_count < 26:

            row_array.append(np_input_data[i][j])

            

            if column_count == 25:

                np_input=np.append(np_input,np.array([np.asarray(row_array)]), axis=0)

                row_array=[]

                column_count=0

        

        column_count=column_count+1

    

    

    

    complete_input_array.append(np_input)



complete_input_array_=reshape_input(np.asarray(complete_input_array))



# result = (result > 0.2).astype(int)

# print(result.flatten().reshape((25, 25)))



# print(result.flatten().reshape(-1,625))

#complete_input_array_.shape[0]



from tensorflow import keras



list_=[]



for j in range(len(delta_list)):



    for i in range(complete_input_array_.shape[0]):

        

        if delta_list[j]==1:

            model= keras.models.load_model('delta1_model.h5')

        elif delta_list[j]==2:

            model= keras.models.load_model('delta2_model.h5')

        elif delta_list[j]==3:

            model= keras.models.load_model('delta3_model.h5')

        elif delta_list[j]==4:

            model= keras.models.load_model('delta4_model.h5')

        elif delta_list[j]==5:

            model= keras.models.load_model('delta5_model.h5')

    

        result=model.predict_on_batch(complete_input_array_[i:i+1])

        result = (result > 0.4).astype(int)

        list_.append(result.flatten().reshape(-1,625)[0])

    

# print(np.asarray(list_))

submission_df = pd.DataFrame(np.asarray(list_),columns=list(output_column_name_list[1:]))

print(submission_df)

# print(submission_df)





df_sample=pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/sample_submission.csv')



submission_df.insert(0,'id', df_sample['id'].tolist())

print(submission_df)



submission_df.to_csv('submission.csv',index=False)


# import numpy as np

# import matplotlib.pyplot as plt



# H = np.array([[1, 2, 3, 4],

#               [5, 6, 7, 8],

#               [9, 10, 11, 12],

#               [13, 14, 15, 16]])  # added some commas and array creation code



# fig = plt.figure(figsize=(6, 3.2))



# ax = fig.add_subplot(111)

# ax.set_title('colorMap')

# plt.imshow(complete_input_array_[0][1])

# ax.set_aspect('equal')



# # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])

# cax.get_xaxis().set_visible(False)

# cax.get_yaxis().set_visible(False)

# cax.patch.set_alpha(0)

# cax.set_frame_on(False)

# plt.colorbar(orientation='vertical')

# plt.show()