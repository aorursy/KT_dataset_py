import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

import seaborn as sns

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from  IPython.display import display

from tensorflow.random import set_seed

np.random.seed(11)

set_seed(11)

random.seed(11)

!PYTHONHASHSEED=0
df = pd.read_csv("../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv")
df.shape
df.head()
def img_arr(x):

    '''

    Function to convert pixel data (string) into array of pixels

    '''

    x=x.reset_index(drop=True)

    n = len(x) #number of rows

    for i in range(n):

        if i==0:

            arr = np.array(x[i].split()).astype(np.int16) #Initializing the array

        else:

            arr = np.append(arr,np.array(x[i].split()).astype(np.int16),axis=0) #Appending data to the array

    return arr.reshape(n,48,48,1) #reshaping the array to 4-dim image pixel array
#Splitting dataset into X and y

X = df.iloc[:,4].copy()

y = df.iloc[:,2].copy()
# As seen below the class is fairly balanced

y.value_counts()
y.value_counts().plot(kind="bar")

plt.title("Label Distribution")

plt.xlabel("Labels")

plt.ylabel("Count");
#splitting the data into train and te sets. 'te' set will be further split into validation and test sets 

X_train,X_te,y_train,y_te = train_test_split(X,y,test_size=0.3,random_state=11)
#splitting 'te' set into validation and test set

X_val,X_test,y_val,y_test = train_test_split(X_te,y_te,test_size=0.15,random_state=11)
#Converting the string of pixels into image array for each of train, val and test set

X_train = img_arr(X_train)

X_test = img_arr(X_test)

X_val = img_arr(X_val)
y_train = y_train.values

y_test = y_test.values

y_val = y_val.values
rows=20 #rows in subplots

cols=5 #columns in subplots

samp = random.sample(range(X_train.shape[0]),rows*cols) #selecting 100 random samples

x_samp = X_train[samp,:,:,:]

y_samp = y_train[samp]



fig,ax = plt.subplots(rows,cols,figsize=(12,45))

r = 0

c = 0

for i in range(rows*cols):

    aa = x_samp[i,:,:,:].reshape(48,48)

    ax[r,c].axis("off")

    ax[r,c].imshow(aa,cmap="gray")

    ax[r,c].set_title(f"Gender: {'Female' if y_samp[i]==1 else 'Male'}")

    c+=1

    if c == cols:

        c=0

        r+=1

plt.show()
set_seed(11)

random.seed(11)

np.random.seed(11)
train_data_gen = ImageDataGenerator(rotation_range=30,

                                   width_shift_range=1,

                                    brightness_range=[0.8,1.2],

                                    zoom_range=[0.8,1.2],

                                    rescale=1/255

                                   )





val_data_gen = ImageDataGenerator(rescale=1/255)



test_data_gen = ImageDataGenerator(rescale=1/255)
fig,ax = plt.subplots(10,5,figsize=(15,25))

for n in range(10):    

    r = random.sample(range(X_train.shape[0]),1)[0]

    ax[n,0].imshow(X_train[r].reshape(48,48),cmap="gray")

    ax[n,0].set_title("Original")

    ax[n,0].axis("off")

    for i in range(1,5):

        ax[n,i].imshow(train_data_gen.random_transform(X_train[r]).reshape(48,48),cmap="gray")

        ax[n,i].set_title("Augmented")

        ax[n,i].axis("off")

plt.show()
set_seed(11)

random.seed(11)

np.random.seed(11)

training_data = train_data_gen.flow(X_train,y_train,

                                   seed=11)



val_data = val_data_gen.flow(X_val,y_val,

                                   seed=11,shuffle=False)



test_data = test_data_gen.flow(X_test,y_test,

                                   seed=11,shuffle=False)
INPUT_SHAPE = (48,48,1)
random.seed(11)

set_seed(11)

np.random.seed(11)

model = Sequential()



model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation="relu",input_shape=INPUT_SHAPE))

model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))



model.add(Conv2D(filters=128,kernel_size=3,strides=1,activation="relu"))

model.add(Conv2D(filters=128,kernel_size=3,strides=1,activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))



model.add(Flatten())



model.add(Dense(units=512,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(units=1024,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(units=1,activation="sigmoid"))



model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["binary_accuracy"])
early_stop = EarlyStopping(monitor="val_loss",patience=5,mode="min") #Ensure the model doesn't overfit
random.seed(11)

set_seed(11)

np.random.seed(11)

history = model.fit(training_data,batch_size=32,epochs=500,callbacks=early_stop,validation_data=val_data)
#Dataframe capturing the accuracy and loss per epoch

loss_df = pd.DataFrame(history.history)

loss_df
loss_df.plot();
Final_train = np.append(X_train,X_val,axis=0)

Final_val = np.append(y_train,y_val,axis=0)
final_training_data = train_data_gen.flow(Final_train,Final_val,

                                   seed=11)
random.seed(11)

set_seed(11)

np.random.seed(11)

final_model_history = model.fit(final_training_data,batch_size=32,epochs=20)
model.evaluate(test_data)
prediction = model.predict(test_data).flatten()
print(prediction)
prediction = np.round(prediction) #rounding so that the prediction >0.5 becones 1 and everything else becomes 0
prediction
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,cbar=False,fmt="d")

plt.xlabel("Prediction")

plt.ylabel("Actual");
print(classification_report(y_test,prediction))
error_index = (y_test != prediction)#finding error indices

y_test_error = y_test[error_index]

X_test_error = X_test[error_index]

prediction_error = prediction[error_index]
rows=int(np.floor(sum(error_index)/3)) #rows in subplots

cols=3 #columns in subplots

x_samp = X_test_error

y_samp = y_test_error



fig,ax = plt.subplots(rows,cols,figsize=(15,200))

r = 0

c = 0

for i in range((rows*cols)-1):

    aa = x_samp[i].reshape(48,48)

    ax[r,c].axis("off")

    ax[r,c].imshow(aa,cmap="gray")

    actual_lab = "Female" if y_samp[i]==1 else "Male"

    pred_lab = "Female" if int(prediction_error[i])==1 else "Male"

    ax[r,c].set_title(f'Actual: {actual_lab}\nPred: {pred_lab}')

    c+=1

    if c == cols:

        c=0

        r+=1

plt.show()