# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras

import numpy as np

import pandas as pd

from PIL import Image as img

import glob

from tqdm import tqdm

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing import image

keras.backend.image_data_format()

keras.backend.set_image_data_format("channels_first")

keras.backend.image_data_format()

%matplotlib inline
y_train = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")

blind_train_list = []

for i in list(y_train['id_code']):

    path = "/kaggle/input/aptos2019-blindness-detection/train_images/"+i+".png"

    blind_train_list.append(path)
x_train = []

for i in tqdm(blind_train_list):

    temp = img.open(i).resize((64, 64))

    temp = temp.convert("L") 

    x_train.append((np.array(temp) - np.mean(temp)) / np.std(temp))

print("aptos blindness detection images loading is done")



a = np.asarray(x_train)

x_train = a.reshape(a.shape[0], 1, a.shape[1], a.shape[2])
y_train_ = y_train['diagnosis']

y_train_df = pd.DataFrame()

zero = []

one = []

two = []

three = []

four = []

for i in list(y_train_):

    if i == 0:

        zero.append(1)

        one.append(0)

        two.append(0)

        three.append(0)

        four.append(0)

    elif i == 1:

        zero.append(0)

        one.append(1)

        two.append(0)

        three.append(0)

        four.append(0)

    elif i == 2:

        zero.append(0)

        one.append(0)

        two.append(1)

        three.append(0)

        four.append(0)

    elif i == 3:

        zero.append(0)

        one.append(0)

        two.append(1)

        three.append(0)

        four.append(0)

    elif i == 4:

        zero.append(0)

        one.append(0)

        two.append(0)

        three.append(0)

        four.append(1)



y_train_df = pd.DataFrame({'zero':zero,'one':one,'two':two,'three':three,'four':four})  

y_train_val = np.array(y_train_df)
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(1,64,64)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
X_train_, X_val, y_train_, y_val = train_test_split(x_train, y_train_val, random_state=42, test_size=0.1)

X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, random_state=42, test_size=0.1)

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=100)
plt.figure(figsize=(20, 7))

plt.subplot(1, 2, 1)

plt.plot(model.history.history["acc"])

plt.plot(model.history.history["val_acc"])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train", "Val"], loc="upper left")



plt.xticks(np.arange(len(model.history.history["acc"])), np.arange(1, len(model.history.history["val_acc"])+1, 1))



plt.subplot(1, 2, 2)

plt.plot(model.history.history["loss"])

plt.plot(model.history.history["val_loss"])

plt.title("Model Loss")

plt.ylabel("Loss")

plt.xlabel("Epoch")

plt.legend(["Train", "Val"], loc="upper right")

plt.xticks(np.arange(len(model.history.history["loss"])), np.arange(1, len(model.history.history["loss"])+1, 1))

plt.show()
result_val = model.predict(x=X_val)

df_val = pd.DataFrame(result_val)

pred_val_list = []

for j in range(367):

    zero_ = float(df_val.iloc[j,0:1])

    one_ = float(df_val.iloc[j,1:2])

    two_ = float(df_val.iloc[j,2:3])

    thre_ = float(df_val.iloc[j,3:4])

    four_ = float(df_val.iloc[j,4:])

    

    if zero_ > one_ and zero_ > two_ and zero_ > thre_ and zero_ > four_:

        pred_val_list.append(0)

    elif one_ > zero_ and one_ > two_ and one_ > thre_ and one_ > four_:

        pred_val_list.append(1)

    elif two_ > zero_ and two_ > one_ and two_ > thre_ and two_ > four_:

        pred_val_list.append(2)

    elif thre_ > zero_ and thre_ > one_ and thre_ > two_ and thre_ > four_:

        pred_val_list.append(3)

    elif four_ > zero_ and four_ > one_ and four_ > two_ and four_ > thre_:

        pred_val_list.append(4)

    

df_val_ = pd.DataFrame(y_val)

val_list = []

for j in range(367):

    zero_ = float(df_val_.iloc[j,0:1])

    one_ = float(df_val_.iloc[j,1:2])

    two_ = float(df_val_.iloc[j,2:3])

    thre_ = float(df_val_.iloc[j,3:4])

    four_ = float(df_val_.iloc[j,4:])

    

    if zero_ > one_ and zero_ > two_ and zero_ > thre_ and zero_ > four_:

        val_list.append(0)

    elif one_ > zero_ and one_ > two_ and one_ > thre_ and one_ > four_:

        val_list.append(1)

    elif two_ > zero_ and two_ > one_ and two_ > thre_ and two_ > four_:

        val_list.append(2)

    elif thre_ > zero_ and thre_ > one_ and thre_ > two_ and thre_ > four_:

        val_list.append(3)

    elif four_ > zero_ and four_ > one_ and four_ > two_ and four_ > thre_:

        val_list.append(4)
df_comb = pd.DataFrame({'actual':val_list,'predicted':pred_val_list})

sm = 0

for i in range(367):

    fst = int(df_comb.iloc[i,0:1])

    snd =  int(df_comb.iloc[i,1:])

    if fst == snd:

        sm = sm + 1  

sm/df_comb.shape[0]
y_test = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")

blind_test_list = []

for i in list(y_test['id_code']):

    path = "/kaggle/input/aptos2019-blindness-detection/test_images/"+i+".png"

    blind_test_list.append(path)

    

x_test = []

for i in tqdm(blind_test_list):

    temp_test = img.open(i).resize((64, 64))

    temp_test = temp_test.convert("L")

    x_test.append((np.array(temp_test) - np.mean(temp_test)) / np.std(temp_test))

print("aptos blindness test images loading is done")

b = np.asarray(x_test)

x_test = b.reshape(b.shape[0], 1, b.shape[1], b.shape[2])
result_test = model.predict(x=x_test)

df_test = pd.DataFrame(result_test)

pred_test_list = []

for j in range(1928):

    zero_ = float(df_test.iloc[j,0:1])

    one_ = float(df_test.iloc[j,1:2])

    two_ = float(df_test.iloc[j,2:3])

    thre_ = float(df_test.iloc[j,3:4])

    four_ = float(df_test.iloc[j,4:])

    

    if zero_ > one_ and zero_ > two_ and zero_ > thre_ and zero_ > four_:

        pred_test_list.append(0)

    elif one_ > zero_ and one_ > two_ and one_ > thre_ and one_ > four_:

        pred_test_list.append(1)

    elif two_ > zero_ and two_ > one_ and two_ > thre_ and two_ > four_:

        pred_test_list.append(2)

    elif thre_ > zero_ and thre_ > one_ and thre_ > two_ and thre_ > four_:

        pred_test_list.append(3)

    elif four_ > zero_ and four_ > one_ and four_ > two_ and four_ > thre_:

        pred_test_list.append(4)

        

y_test_ = y_test['id_code']

sub = pd.DataFrame({'id_code':y_test_,'diagnosis':pred_test_list})

sub.to_csv("../input/sampleSubmission.csv",index=False)
pd.read_csv("../input/sampleSubmission.csv").to_csv('../sample_Submission.csv')

#print(sub_df.head(5))