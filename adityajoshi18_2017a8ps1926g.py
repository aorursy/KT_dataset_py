import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.utils import to_categorical

# Set callback functions to early stop training and save the best model so far

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam











from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
data=pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv",encoding="utf-8",index_col=0)



data.head()

data["Class"].value_counts()
for columns in data.columns:

  data.drop( data[ data[columns] == "?" ].index , inplace=True)
data["Size"].unique()
data.shape

data['Size'] = data['Size'].map({

    'Small': 0, 

    'Medium': 1,

    'Big':2

    })

data.head()
len(data.columns)
x_train = data.iloc[:,:11]




y_train = data.iloc[:,11]
x_train.shape




y_train.shape

x_train["Size"].unique()


y_train.unique()















y_train_enc = to_categorical(y_train)
y_train_enc

y_train
x_train.head()
x_train.info()


def create_model():

  model = Sequential()

  model.add(Dense(32,input_dim=11, activation='relu'))

  model.add(Dense(16, activation='relu'))

  model.add(Dropout(rate=0.2))

  model.add(Dense(8, activation='relu'))

  model.add(Dropout(rate=0.2))

  model.add(Dense(6,activation='softmax'))

  



  # Compile 

  adam = Adam(lr=10**-3)

  model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

  print(model.summary())

  return model





def model_train(X_train, Y_train, n_folds, iterr, bsize):

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=7)

    hist = []

    val_hist=[]

    acc=[]

    val_acc=[]

    for train, test in kfold.split(X_train, Y_train):

        #----------------Build NN model--------

        new_model = create_model()

        #----------------Fit the model-----------------

        xtr = X_train.iloc[train]

        ytr = Y_train[train]

        xval = X_train.iloc[test]

        yval = Y_train[test]

    

        

        history = new_model.fit(xtr, ytr,validation_data=(xval, yval), epochs=iterr, batch_size=bsize, verbose=1)

        hist.append(history.history['loss'])

        val_hist.append(history.history['val_loss'])

        acc.append(history.history['accuracy'])

        val_acc.append(history.history['val_accuracy'])

        plt.figure()

        plt.plot(history.history['loss'], label = "Training Loss")

        plt.plot(history.history['val_loss'], label = "Validation loss")

        plt.xlabel('Number of epochs')

        plt.legend()

        plt.show()

    mean_val_hist = np.mean(val_hist,axis=0)

    main_iterr = np.argmin(mean_val_hist)

    print('main iteration is:',main_iterr)

    return main_iterr

        

        
main_iterr = model_train(x_train, y_train_enc, 5, 170, 20)
new_model = create_model()

new_model.fit(x_train, y_train_enc, epochs=main_iterr, batch_size=20, verbose=1)
scores = new_model.evaluate(x_train, y_train_enc, verbose=0)

print(scores)
df=pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",encoding="utf-8",index_col=0)


df.head()





for columns in df.columns:

  df[columns].unique()

  print(columns,df[columns].unique())
df.shape

df['Size'] = df['Size'].map({

    'Small': 0, 

    'Medium': 1,

    'Big':2

    })

df.head()
y_pred = new_model.predict(df)
y_pred
np.shape(y_pred)












df.shape

list1 = []

for i in range(len(y_pred)):

  temp = y_pred[i,:]

  ind = np.argmax(temp)

  list1.append(ind)

y_test = np.array(list1)



y_test


df
df_2 = df



col_name = df.columns
df_2["Class"] = 0
df_2
for i in range(len(y_test)):

  df_2.iloc[i,11] = y_test[i]





df_2
df_2 = df_2.drop(columns=col_name)
df_2.to_csv("submission.csv")




df_2
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)

create_download_link(df_2)