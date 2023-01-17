import pandas as pd

import numpy as np

 

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



# https://www.kaggle.com/uciml/iris/downloads/iris-species.zip

data = pd.read_csv('../input/Iris.csv')

data = np.array(data)



class iris:

    """ Simple way to hold data"""

    dictionary={}

    

    

def getV(n):

    mint = int(np.round(n)) 

    if mint in dictionary:

        return iris.dictionary[mint]

    else:

        return -1



        

# label encode the categorical variables

for i in range(data.shape[1]):

    if i in [5]:  # colums to convert. Just the last one in this case

        lbl = preprocessing.LabelEncoder()

        lbl.fit(data[:,i])

        

        values   = np.unique(data[:,5])

        keys = lbl.transform(np.unique(data[:,5]))

        iris.dictionary = dict(zip(keys, values))



        data[:,i] = lbl.transform(data[:,i])

        

# See class above

iris.target = data[:,-1]

iris.data   = data[:,1:-1]



# Training on 10% of the data

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.10, random_state=5)



X_train.shape,y_train.shape,X_train.shape[1]
# Make sure you get kind of an even balance.

# You want at least one of each.



np.unique(y_train, return_counts=True)
import pandas as pd

import numpy as np





from sklearn import preprocessing

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout



import matplotlib

import matplotlib.pyplot as plt









# Keras model

def findMin(dense=64, drop=[],epoch=33):

    np.random.seed(seed=23) 

    PYTHONHASHSEED=0

    model = Sequential()

    model.add(Dense(dense, input_dim=X_train.shape[1]))

    model.add(Activation('relu'))

    model.add(Dropout(drop[0]))

    model.add(Dense(dense))

    model.add(Activation('relu'))

    model.add(Dropout(drop[1]))

    model.add(Dense(1))



    #model.compile(loss='mse', optimizer='rmsprop')

    model.compile(loss='mse', optimizer='adam')



    history = model.fit(X_train, y_train, nb_epoch=epoch, batch_size=32, verbose=0)

    myMin=min(history.history['loss'])

    return history.history['loss'].index(myMin),myMin,history,model



p=[]

for dense in [20,32,64,128]:

    a,b,c,model = findMin(dense,[0,0])

    print("Min loss: {:^9.3f}   dense: {:^9.3f}".format(b,dense))

    plt.plot(c.history['loss'])

    p.append('Dense %s' % dense)

    

    

plt.title("Loss")

plt.legend(p);
p=[]

for dense in [64,128]:

    a,b,c,model = findMin(dense,[0,0],100)

    print("Min loss: {:^9.3f}   dense: {:^9.2f}".format(b,dense))

    plt.plot(c.history['loss'])

    p.append('Dense %s' % dense)

    

    



plt.legend(p);
p=[]

g=[]

for dense in [32]:

    index,b,c,model = findMin(dense,[0,0],epoch=1500)

    print("Min loss: {:^9.3f}   dense: {:^9.2f}  index: {:d}".format(b,dense,index))

    g.append(c.history['loss'])

    plt.plot(c.history['loss'])

    p.append('Dense %s' % dense)

    

    



plt.legend(p);
a,b,c,model=findMin(32,[0,0],1500)

model.summary()
# calculate predictions

predictions = model.predict(X_test)



status = [ int(np.round(i)) == int(np.round(j)) for i,j in zip(predictions.flatten(),y_test)]

print("correct: ",status.count(True),"/",len(status))
from keras.models import load_model



model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

del model  # deletes the existing model



# returns a compiled model

# identical to the previous one

model = load_model('my_model.h5')





# calculate predictions

predictions = model.predict(X_test)

status = [ int(np.round(i)) == int(np.round(j)) for i,j in zip(predictions.flatten(),y_test)]

print("correct: ",status.count(True),"/",len(status))