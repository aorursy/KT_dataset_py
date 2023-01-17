import pandas as pd

import os

import numpy as np

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')

print(df_train.head(2),df_test.head(2))
def getAccuracy(a,b):return sum(getInt(a)==getInt(b))*100/b.shape[0]



def oneVsAll(y):

    size = y.size

    y_new = np.zeros((size,10))

    for i in range(0,size):

        y_new[i][int(y[i])] = 1

    return y_new



def getInt(y):

    size = y.shape[0]

    y_new = np.zeros((size,1))

    for i in range(0,size):

        result = np.where(y[i] == np.amax(y[i]))

        y_new[i] = result

    return y_new
y_temp = df_train['label'].values

y = oneVsAll(y_temp)

df_train=df_train.drop('label',axis=1)

X = df_train.values/255

X_test = df_test.values
# Importing requirements for dnn

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping
def DNN(num_layers,train_X,train_y):

    #create model

    model = Sequential()

    #get number of columns in training data

    n_cols = train_X.shape[1]

    #num_layers = [200,30,15,10]

    # Input Layer #200

    model.add(Dense(num_layers[0], activation='relu', input_shape=(n_cols,)))

    

    # Hidden Layers and Output Layer #index 1,2  1->len(num_layers)-2 

    for i in range(1,len(num_layers)-1):

        model.add(Dense(num_layers[i], activation='relu'))

    model.add(Dense(num_layers[-1]))

    model.compile(optimizer='adamax', loss='mean_squared_error')

    

    #set early stopping monitor so the model stops training when it won't improve anymore

    early_stopping_monitor = EarlyStopping(patience=2)

    #train model

    model.fit(train_X, train_y, 

              validation_split=0.1, epochs=25, callbacks=[early_stopping_monitor])

    return model
m1 = DNN([1500,600,400,100,40,10],X,y)
preds = m1.predict(X_test/255)
fin_pred = getInt(preds)
fin_pred=fin_pred.astype('int')

test_id = np.arange(1,fin_pred.size+1)
my_submission = pd.DataFrame({'ImageId': test_id, 'Label': fin_pred.reshape(fin_pred.size,)})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
x = pd.read_csv('submission.csv')