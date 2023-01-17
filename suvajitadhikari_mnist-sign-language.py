# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd



train_df=pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")

test_df=pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
train_df.head(10)
train_df.info()
X_train_full=train_df

X_train_full.head(10)
y_train_full=X_train_full.iloc[:,0]

X_train_full=X_train_full.iloc[:,1:]

y_train_full.head(10)
y_train_full.value_counts().sort_index()
X_train_full.shape
X_valid,X_train=X_train_full.iloc[:5491,:]/255,X_train_full.iloc[5491:,:]/255

y_valid,y_train=y_train_full.iloc[:5491],y_train_full.iloc[5491:]

(y_valid.value_counts()/len(y_valid)).sort_index()
(y_train.value_counts()/len(y_train)).sort_index()
y_valid=y_valid.to_numpy()

y_train=y_train.to_numpy()



X_train=X_train.to_numpy()

X_valid=X_valid.to_numpy()
import matplotlib.pyplot as plt



plt.imshow(X_train[0].reshape(28,28),interpolation="gaussian")

plt.axis("off")

plt.show()
y_train[0] # The above figure is a 7 in sign language.
def index_sign_labels(target):

    labels_index=[None]*25   # to store index of individual labels

    for i in range(len(target)):

        labels_index[target[i]]=i

    return labels_index
idx=index_sign_labels(y_train)



n_rows=5

n_columns=5



i=0

plt.figure(figsize=(15,12))

for rows in range(n_rows):

    for columns in range(n_columns):

        if i!=9:

            index=n_columns*rows+columns

            plt.subplot(n_rows,n_columns,index+1)

            plt.imshow(X_train[idx[i]].reshape(28,28),interpolation="gaussian")

            plt.axis("off")

            plt.title(i)

        i+=1
import tensorflow as tf

from tensorflow import keras
early_stopping_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)



model=keras.models.Sequential([keras.layers.Dense(128,activation='relu',input_shape=[784]),

                               keras.layers.Dense(100,activation='relu'),

                               keras.layers.Dense(100,activation='relu'),

                               keras.layers.Dense(100,activation='relu'),

                               keras.layers.Dense(25,activation='softmax')])
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=150,validation_data=(X_valid,y_valid),callbacks=[early_stopping_cb])
pd.DataFrame(history.history).plot(figsize=(12,8))

plt.grid(True)

plt.ylim(0,1.2)

plt.xlabel("Epochs",fontsize=14)

plt.ylabel("Loss / Accuracy",fontsize=14)

plt.show()
X_test_full=test_df



X_test=X_test_full.iloc[:,1:]/255

y_test=X_test_full.iloc[:,0]



X_test=X_test.to_numpy()

y_test=y_test.to_numpy()



model.evaluate(X_test,y_test)
size=len(y_test)

n_class=24



X_new=X_test[-10:,:]

output_probs=model.predict(X_new)

y_pred=[]



for i in range(10):

    m=output_probs[i][0]

    x=0

    for j in range(1,n_class):

        if output_probs[i][j]>m:

            m=output_probs[i][j]

            x=j

    y_pred.append(x)



y_pred
y_test[-10:]
y_pred=np.array(y_pred)
acc=sum(y_pred==y_test)/len(y_test)

acc
idx=index_sign_labels(y_test)



n_rows=5

n_columns=5



i=0

plt.figure(figsize=(15,12))

for rows in range(n_rows):

    for columns in range(n_columns):

        if i!=9:

            index=n_columns*rows+columns

            plt.subplot(n_rows,n_columns,index+1)

            plt.imshow(X_train[idx[i]].reshape(28,28),interpolation="gaussian")

            plt.axis("off")

            plt.title(i)

        i+=1