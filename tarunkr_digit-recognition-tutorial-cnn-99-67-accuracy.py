import pandas as pd

import numpy as np



import matplotlib.pyplot as plt 

import cv2 as cv



from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D

from keras import models

from keras.optimizers import Adam,RMSprop 

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import pickle



%matplotlib inline
np.random.seed(1) # seed

df_train = pd.read_csv("../input/digit-recognizer/train.csv") # Loading Dataset

df_train = df_train.iloc[np.random.permutation(len(df_train))] # Random permutaion for dataset (seed is used to resample the same permutation every time)
df_train.head(5)
df_train.shape
sample_size = df_train.shape[0] # Training set size

validation_size = int(df_train.shape[0]*0.1) # Validation set size 



# train_x and train_y

train_x = np.asarray(df_train.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0

train_y = np.asarray(df_train.iloc[:sample_size-validation_size,0]).reshape([sample_size-validation_size,1]) # taking column 0



# val_x and val_y

val_x = np.asarray(df_train.iloc[sample_size-validation_size:,1:]).reshape([validation_size,28,28,1])

val_y = np.asarray(df_train.iloc[sample_size-validation_size:,0]).reshape([validation_size,1])
train_x.shape,train_y.shape
df_test = pd.read_csv("../input/digit-recognizer/test.csv")

test_x = np.asarray(df_test.iloc[:,:]).reshape([-1,28,28,1])
# convirting pixel values in range [0,1]

train_x = train_x/255

val_x = val_x/255

test_x = test_x/255
# Cheacking frequency of digits in training and validation set

counts = df_train.iloc[:sample_size-validation_size,:].groupby('label')['label'].count()

# df_train.head(2)

# counts

f = plt.figure(figsize=(10,6))

f.add_subplot(111)



plt.bar(counts.index,counts.values,width = 0.8,color="orange")

for i in counts.index:

    plt.text(i,counts.values[i]+50,str(counts.values[i]),horizontalalignment='center',fontsize=14)



plt.tick_params(labelsize = 14)

plt.xticks(counts.index)

plt.xlabel("Digits",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Frequency Graph training set",fontsize=20)

plt.savefig('digit_frequency_train.png')  

plt.show()
# df_train.iloc[sample_size-validation_index:,1:]

# Cheacking frequency of digits in training and validation set

counts = df_train.iloc[sample_size-validation_size:,:].groupby('label')['label'].count()

# df_train.head(2)

# counts

f = plt.figure(figsize=(10,6))

f.add_subplot(111)



plt.bar(counts.index,counts.values,width = 0.8,color="orange")

for i in counts.index:

    plt.text(i,counts.values[i]+5,str(counts.values[i]),horizontalalignment='center',fontsize=14)



plt.tick_params(labelsize = 14)

plt.xticks(counts.index)

plt.xlabel("Digits",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Frequency Graph Validation set",fontsize=20)

plt.savefig('digit_frequency_val.png')

plt.show()
rows = 5 # defining no. of rows in figure

cols = 6 # defining no. of colums in figure



f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 



for i in range(rows*cols): 

    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration

    plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 

    plt.axis("off")

    plt.title(str(train_y[i]), y=-0.15,color="green")

plt.savefig("digits.png")
# # Loading pickled resources

# !wget https://www.kaggleusercontent.com/kf/31703703/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..f6l0FhjgIxCd7bfZA3MF_A.oHtR56AzfPZhslGD-R2Uca6Kce1yVCG805lWUk25H5T-MQciLjAjmoTd9RPGh1zdUlwtbMZPuAi9_1BrHNZfFlJ5duoYajHON-Sk_mMy7OIePjqNRqo8vkEnTHmbV6Oj1Z6MR9dGzT2Gch2soeaLaZnIjxJt5e8DsFaia6dTjxRzzzrKaQDWLikdsjo2xwbQp9yo4-8htw6adclSbtnXsMV4kJNBs25d-qqRLuUSuhqxxCbJMkuuZgPjnEzqO7aLU0zqcGYUXDDdx1O-oU2ncMAYpXYqssqzQgD6-t4Fl83XWQnNqRv5wec5cdD-7IF9cbjyD_CE-Ib863pPJ9RJc-IYypbUvvKfMQuhahe9NiuRGNSNodVlSiuSzk0nudl5uHqf7V7_1h_juPPVj8mUUOqleLye9_ZtJ2S8pD6hUXT9p7kPy6v6RdoaE_LgkrijyvmJhmS-yMETpazrlQlKp96A3W0EVdhtVxmW7QUwbjIlzdEs7whAe4EcqQIzd4H69TR6hLCqVlaZkMBBPvBWr_dCTxu6htDP8qE2QCH08H1VPXyZLERTMH1SRENnwa_BxMTVkc_pP70tkvGA2xtgoJHzAlcOZZfqsa5fmCa8tqIOkME1hn88Xgm5eh58JXT2ZXyA7hxfpzKP3UQXegeBnPaEPOLa5eEoND9E9ypyi5I1QWa7QTMe9ruEbXr6DUbJ.GvQemyhjKibaOnfNrnRUXQ/model.h5

# !wget https://www.kaggleusercontent.com/kf/31703703/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..yUikFJ_W78H0Zfo7LsQ-2g.1-XOTVZwAUnN9uzbSyaPEuXbdS_5mlLvYRUOaFjMZewxYpcCFASz2fROWEIHjC-KHCOOO-DYjLY7pWHvN81cOW25n28C2mT6aFnWkObrkUYXlbZW0sS7iaKwxUw4a_XwYYUmMfeNuOpKf5OsBdv-L2y0GOLb-fT8hDqteihh6qP97VOT1fmvdv2gYcG0WqKw4mcWwtWAYyqQAScE_knALnyTvWZTF73LR99gaxy_w41t3PHo0rfHy037Cll0lWznkf6ppfAIv_CKr_v2HfpB9go0DJbXKqD87LkcnNF94e_b95i0UYfS2pMugXS4ob1WnSblE34Df_n9rB3pWMXn9DPnqlwDqa1snJOc3CLeK9MTsPQ9NuzEFIjxMnWXQhNGQ7sXYX8qAPqanQ3-OQSZHHD8lufpcJEguVqOuOWY5qZlKe4ZTzn3g8fhbssulphnajM4ZThC5pceWnVT2rowvERPtlvijy6AdAGEy57BxN7FeweBVZAR1Zv5RA_wE2qo4DFmyK36j8p_bVZHTnVvqFFWz0SAimJqzMmzxNrZY7_NGuqDy5rjUmoa2y0e0-qPFBcfFWvT2Pe_RKW21dBOKH9HbI1j0WS_Ua72FI1-MToX8DUHRdN9UNLJhnugwWY_lWvppU-fRCJFhkSPIYyfzbx1cH2ykLLZRz4-sjCegE78I03ue3096Q7kUMw9-ZGd.a3Ge-GJWsUMNPuyQ0gpMfg/model_img_augmentation.h5

# !wget https://www.kaggleusercontent.com/kf/31703703/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Hq9sgF7KDfRSLRlO3zlQSg.dQKW1YCUjhwAZbX-54g1ygj0qTWSH6egywwOdIqm-hwUxTWnj6L9EkNK8qncdH4FeNbEtAjYrQ2oStYGIVVJzsqFDQwaXBsnzvKQx2icckW3ww-aNtDxATdkVY2ZiqMLAMIMyy8_nFoFvt2tQm48XeIUecn3OTL6Te3VfRr3OXQ1QWU7lbY-8BetDVy0tbrLV_vIkV-fUy_FLGsa7QgH4Xerxi7xXoItAmJCbEIXBt5pR-_frNz0rdDoj30e-xGdUkLRiBNe1Nk9_1EFacGwTwhM3KqE8SDF5CtVqP7XsFhFIxVal-lmUy1hzsPi0xZt_ikRIbmfTb6K5HmQ97Jzm4nd7YdQ_u2ScbhLLFy2pj0n4XapGZgMYE5o_2_Cv2c_3uquTGDTpjiOAA25ylrFypwZGmenAeCSrJZIto0ta_onqVPz_euNQY4PHJ5P3aJi2KWHJzNl3LxBU2u-LfkfOzwYXu1DIOOqkzSSLWraxZoleean-9oawdqq0doQhogfx2wBr844ApCpVYDxEEn8CxAwfK-RoiE7MdEasDYtWujlmcOnMQsQjUohnJcOWwQSIOIEpqevcXBU4PPoiw-ex-vEUtflupnNRtlwwLRzjyrV_6VJbdQkus1F0HfToHEBjB6tkZyydreIK3Zi8u0l1YvT3VRtWMw3amc3s7f86ilZZ_uoGhW8fc7k14JmubAX.FiOfKdbaYf0TStGFpmhDSw/history_1.hs

# !wget https://www.kaggleusercontent.com/kf/31703703/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..IWrH9XKSk0oDOCNg7wTlbg.A49rZubI479_e2da4l7JhZBGm7YdJ87UviDnyAP6924Kez0dRIPjzw-67wCI_iMtJGQQJrnn74pVwzt0cPa8CVTJIApHs2t4qZd-7dlWDcOqcwlC8zaLECXT836Jey6auUY1JE1C8YApSHBfDr-WB_KO75JQRzDdXV6LFEdPfzTyoXWf0zmuyg4bi41rPhH2UkhEfPowmze9G_nOWg64WHfJTBCcWzMDf56GbyNmp46RtdKuqriMH0sAHHFrz94DQkHnnz0U149yYC6oUkqlvty-l7jvEGM8Nc1_-LZbptB_8cZxu9gOFYprjdUyAerb5Izz-J1nkextV-0OUe8SUYj8XwG5pF6pNDbZrkoDTurkkORlMIEAKNhl6SZvPjmb1Mv-owxnrpSdHnHejp61kFp-FOi_oFFzAJiW5NVm8pmMln7RP3JgcuCfWKQUEHlrsC5NaMvsFG9CNlXeul9E4PXS6ycuRzflz5uWxvoe1Dt-Tm4Pnoaa-CvJ2hsLKsX-oH66LE6sWQm8UrDTJ0eBxaWG8QAbJ6WeOICv9zykRx8ro5j18RWgnOLK2x9PLJmYA7rL6Zm7kxavLQMTovSg-IZ9EyuTuSmGv-0Rd_0Gmb5-2qvBsYKGcn8lbJMmOXfeDWEqTT0XBK-VfVcDNxX8EZd2iYD1c3nJa6R324v4JGTOLHZJbL7UNtGXtQn-3WVc.ShxYOf_zson_2CAlaDOT8w/history_2.hs

!wget https://www.kaggleusercontent.com/kf/32045042/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..LcCpstRicdTOqMGGjh1wmg.LJi-FKBMbLQL5cc2XHtVyNpT--iVgBeKlbby9C_YjBjm7dSLbbqY4F27EpcgOWZYo22EOWWwyVMAd7VsYOzQkUGK-sj6gvWswv9CTc8I8ZDMkZvGdRRy3COevzGk21yXTlbRG1D--BVHlwCZVaYSMBKt9gIeCELgl-ZcFKoOjZSUpcZXBrmfuBA_OK63fc_olldErb5p5S_qWVecdJ1r49anGVs-x682Q0y5cCs9lr7q1n2B2saWkiC4d8tMGRMyUTCt1xXl6nENaYNh4C4u0DGOBfT14jEuWfyzFH3obgs_TtD4Bqg2zzxjjUwtJd7ymBNCVq7td-_chLlds-lps5uq4BLckniKl9QgbJ7ZcL6qoMonLzPn5VUE-vlIfUCfBieYOHKkWb3Buz557IrG4mk9GqAvOPp_Vzd5n9h9SGEvOFCsiG-_IinSIA7a2NxSPvtNEKmYcG37CS0xLfEV8copUh3pIHdVVLn9SXd2r5Wg2Q5q8zUnEvnHSxcPNg7LheviS4NMX18bsL9Wqm2PP_WDd7QkO_wJ0AsGyDveLWKebHB26wS_iz3w7X-EAheRmBrCyiUfJWHSM-IJPAJk9EIhRH1zr8WVtwsc19PRevFcY4RDHx7fyvPFVrJ2fFsicZMaDfJchFR_lP_uYDs6NvPu5AXXEkXnFMsrJ0IAICiEDwadnj0eV5w5b-DozhrP.8pmfOFY56dYt_Z2UVg-UWA/model_img_augmentation.h5
model = models.Sequential()
# Block 1

model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model.add(LeakyReLU())

model.add(Conv2D(32,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Block 2

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation="sigmoid"))
initial_lr = 0.001

loss = "sparse_categorical_crossentropy"

model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

model.summary()
epochs = 20

batch_size = 256

history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])
model.save("model.h5")

with open('history_1.hs', 'wb') as history:

    pickle.dump(history_1,history)
# model = models.load_model("model.h5")

# with open('history_1.hs', 'rb') as history:

#     history_1 = pickle.load(history)
# Diffining Figure

f = plt.figure(figsize=(20,7))



#Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p = np.argmax(model.predict(val_x),axis =1)



error = 0

confusion_matrix = np.zeros([10,10])

for i in range(val_x.shape[0]):

    confusion_matrix[val_y[i],val_p[i]] += 1

    if val_y[i]!=val_p[i]:

        error +=1

        

print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])

print("\nValidation set Shape :",val_p.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

plt.xticks(np.arange(0,10),np.arange(0,10))

plt.yticks(np.arange(0,10),np.arange(0,10))



threshold = confusion_matrix.max()/2 



for i in range(10):

    for j in range(10):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix1.png")

plt.show()
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



datagen.fit(train_x)
lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.00001)
epochs = 20

history_2 = model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=[val_x,val_y],callbacks=[lrr])
model.save("model_img_augmentation.h5")

with open('history_2.hs', 'wb') as history:

    pickle.dump(history_2,history)
model = models.load_model("model_img_augmentation.h5")

# with open('history_2.hs', 'rb') as history:

#     history_2 = pickle.load(history)
# Diffining Figure

f = plt.figure(figsize=(20,7))

f.add_subplot(121)



#Adding Subplot 1 (For Accuracy)

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['accuracy']+history_2.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_accuracy']+history_2.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()





#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['loss']+history_2.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_loss']+history_2.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p = np.argmax(model.predict(val_x),axis =1)



error = 0

confusion_matrix = np.zeros([10,10])

for i in range(val_x.shape[0]):

    confusion_matrix[val_y[i],val_p[i]] += 1

    if val_y[i]!=val_p[i]:

        error +=1

        

confusion_matrix,error,(error*100)/val_p.shape[0],100-(error*100)/val_p.shape[0],val_p.shape[0]



print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])

print("\nValidation set Shape :",val_p.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

plt.xticks(np.arange(0,10),np.arange(0,10))

plt.yticks(np.arange(0,10),np.arange(0,10))



threshold = confusion_matrix.max()/2 



for i in range(10):

    for j in range(10):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix2.png")

plt.show()
rows = 4

cols = 9



f = plt.figure(figsize=(2*cols,2*rows))

sub_plot = 1

for i in range(val_x.shape[0]):

    if val_y[i]!=val_p[i]:

        f.add_subplot(rows,cols,sub_plot) 

        sub_plot+=1

        plt.imshow(val_x[i].reshape([28,28]),cmap="Blues")

        plt.axis("off")

        plt.title("T: "+str(val_y[i])+" P:"+str(val_p[i]), y=-0.15,color="Red")

plt.savefig("error_plots.png")

plt.show()
test_y = np.argmax(model.predict(test_x),axis =1)
rows = 5

cols = 10



f = plt.figure(figsize=(2*cols,2*rows))



for i in range(rows*cols):

    f.add_subplot(rows,cols,i+1)

    plt.imshow(test_x[i].reshape([28,28]),cmap="Blues")

    plt.axis("off")

    plt.title(str(test_y[i]))
# Extracts the outputs of all layers except Flatten and Dense layers

output_layers = [layer.output for layer in model.layers[:-4]]

# Creates a model that will return these outputs, given the model input (This is multi output model)

activation_model = models.Model(inputs=model.input, outputs=output_layers)
# predicting the output of each layers

activations_2  = activation_model.predict(val_x[2].reshape([1,28,28,1]))

activations_6  = activation_model.predict(val_x[7].reshape([1,28,28,1]))

first_activation_layer  = activations_2[0]

first_activation_layer.shape
rows = 4

cols = 2



f = plt.figure(figsize=(2*cols,2*rows))



for i in range(4):

    f.add_subplot(rows,cols,2*i+1)

    plt.imshow(activations_2[0][0,:,:,i].reshape([28,28]),cmap="Blues")

    plt.axis("off") 



    f.add_subplot(rows,cols,2*i+2)

    plt.imshow(activations_6[0][0,:,:,i].reshape([28,28]),cmap="Blues")

    plt.savefig("layer_output_comparision"+str(i)+".png")

    plt.axis("off")
def plot_layer(layer,i,layer_name = None):

    rows = layer.shape[-1]/16

    cols = 16



    f = plt.figure(figsize=(1*cols,1*rows))

    # plt.imshow(first_activation_layer[0,:,:,:].reshape([14*4,14*16]),cmap="Blues")

    for i in range(layer.shape[-1]):

        f.add_subplot(rows,cols,i+1)

        plt.imshow(layer[0,:,:,i].reshape([layer.shape[2],layer.shape[2]]),cmap="Blues")

        plt.axis("off")

    f.suptitle(layer_name,fontsize=14)

    plt.savefig("intermidiate_layers"+str(i)+".png")

    plt.show()
# Visualising each layers

for i,layer in enumerate(activation_model.predict(val_x[6].reshape([1,28,28,1]))):

    plot_layer(layer,i,output_layers[i].name)
df_submission = pd.DataFrame([df_test.index+1,test_y],["ImageId","Label"]).transpose()

df_submission.to_csv("submission.csv",index=False)