# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
coloum=['ID','symbol','last_price','change','change_percentage','bid_size','bid','offer','offer_size','turnOver','high','low','open','last_volume','total_trades','last_trade','time_stamp']

df=pd.read_csv("../input/Stock_Data_06-11.csv", names=coloum,header=None)

df.head()

df.drop("ID",axis=1,inplace=True)
len(df)
df.describe(include="all")
print("unique Comapnies: ",str(len(df.symbol.unique())))

company=df.symbol.unique()

sizes=df.symbol

x=pd.value_counts(sizes)

print("And Every Company have Same number of Records\n")

print(x)

#taking out dates from time stamps

date=[]

for i in range (0,len(df)):

    a=df.time_stamp.iloc[i]

    b=int(a[8:10])

    date.append(b)

np.unique(date)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")



# Load an example dataset with long-form data





# Plot the responses for different events and regions

plt.subplots(figsize=(50,40))



b=sns.lineplot(x=date, y="last_price",

             hue="symbol",palette="Set2",

             data=df)

b.tick_params(labelsize=50)

sns.heatmap(df.isnull())
sns.heatmap(df.corr())


df_test=df.iloc[1800000:2442515]

df=df.iloc[0:1800000]

print(len(df))

print(len(df_test))
hbl_df=df[df["symbol"]=="HBL"]

hbl_df.to_csv("hbl_df")

len(hbl_df)
import matplotlib.pyplot as plt

#taking Last price coloum

Stock_price_HBL = hbl_df.iloc[:,2:3].values

plt.plot(Stock_price_HBL[1:100])

plt.show()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HBL)
X=hbl_df.drop("last_price",axis=1)

y=hbl_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)
x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)
#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

# Some functions to help out with

def return_rmse(test,predicted):

    rmse = math.sqrt(mean_squared_error(test, predicted))

    print("The root mean squared error is {}.".format(rmse))
def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_Hbl=model()

regressor_Hbl.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_Hbl.to_json()

with open("model_hbl.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_Hbl.save_weights("model_hbl.h5")

print("Saved model to disk")
# make predictions

trainPredict = regressor_Hbl.predict(x_train)

testPredict = regressor_Hbl.predict(x_test)


from sklearn.metrics import r2_score

rnn_score = r2_score(y_test,testPredict)

rnn_score

# plot baseline and predictions



plt.plot(testPredict)

plt.plot(y_test)

plt.show()
plt.plot(y_train)

plt.plot(trainPredict)

plt.show()
FABL_df=df[df["symbol"]=="FABL"]



len(FABL_df)
import matplotlib.pyplot as plt

#taking Last price coloum

Stock_price_FABL = FABL_df.iloc[:,2:3].values

plt.plot(Stock_price_FABL[1:100])

plt.show()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FABL)
X=FABL_df.drop("last_price",axis=1)

y=FABL_df["last_price"]

x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)

#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_Fabl=model()

regressor_Fabl.fit(x_train,y_train,epochs =20, batch_size = 5000)

model_json = regressor_Fabl.to_json()

with open("model_FABL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_Fabl.save_weights("model_FABL.h5")

print("Saved model to disk")
AKBL_df=df[df["symbol"]=="AKBL"]

print(len(AKBL_df))





#taking Last price coloum

Stock_price_AKBL = AKBL_df.iloc[:,2:3].values

plt.plot(Stock_price_AKBL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_AKBL)



X=AKBL_df.drop("last_price",axis=1)

y=AKBL_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_AKBL=model()

regressor_AKBL.fit(x_train,y_train,epochs =20, batch_size = 5000)

model_json = regressor_AKBL.to_json()

with open("model_AKBL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_AKBL.save_weights("model_AKBL.h5")

print("Saved model to disk")
SNGP_df=df[df["symbol"]=="SNGP"]

print(len(SNGP_df))





#taking Last price coloum

Stock_price_SNGP =SNGP_df.iloc[:,2:3].values

plt.plot(Stock_price_SNGP[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_SNGP)



X=SNGP_df.drop("last_price",axis=1)

y=SNGP_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_SNGP=model()

regressor_SNGP.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_AKBL.to_json()

with open("model_SNGP.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_SNGP.save_weights("model_SNGP.h5")

print("Saved model to disk")
KEL_df=df[df["symbol"]=="KEL"]

print(len(KEL_df))





#taking Last price coloum

Stock_price_KEL =KEL_df.iloc[:,2:3].values

plt.plot(Stock_price_KEL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_KEL)



X=KEL_df.drop("last_price",axis=1)

y=KEL_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_KEL=model()

regressor_KEL.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_KEL.to_json()

with open("model_KEL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_KEL.save_weights("model_KEL.h5")

print("Saved model to disk")
BYCO_df=df[df["symbol"]=="BYCO"]

print(len(BYCO_df))





#taking Last price coloum

Stock_price_BYCO =BYCO_df.iloc[:,2:3].values

plt.plot(Stock_price_BYCO[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_BYCO)



X=BYCO_df.drop("last_price",axis=1)

y=BYCO_df["last_price"]





x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_BYCO=model()

regressor_BYCO.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_BYCO.to_json()

with open("model_BYCO.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_BYCO.save_weights("model_BYCO.h5")

print("Saved model to disk")

    
DGKC_df=df[df["symbol"]=="DGKC"]

print(len(DGKC_df))





#taking Last price coloum

Stock_price_DGKC =DGKC_df.iloc[:,2:3].values

plt.plot(Stock_price_DGKC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DGKC)



X=DGKC_df.drop("last_price",axis=1)

y=DGKC_df["last_price"]









x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_DGKC=model()

regressor_DGKC.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_DGKC.to_json()

with open("model_DGKC.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_DGKC.save_weights("model_DGKC.h5")

print("Saved model to disk")

    

ENGRO_df=df[df["symbol"]=="ENGRO"]

print(len(ENGRO_df))





#taking Last price coloum

Stock_price_ENGRO =ENGRO_df.iloc[:,2:3].values

plt.plot(Stock_price_ENGRO[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ENGRO)



X=ENGRO_df.drop("last_price",axis=1)

y=ENGRO_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_ENGRO=model()

regressor_ENGRO.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_ENGRO.to_json()

with open("model_ENGRO.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_ENGRO.save_weights("model_ENGRO.h5")

print("Saved model to disk")

    

INDU_df=df[df["symbol"]=="INDU"]

print(len(INDU_df))





#taking Last price coloum

Stock_price_INDU =INDU_df.iloc[:,2:3].values

plt.plot(Stock_price_INDU[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_INDU)



X=INDU_df.drop("last_price",axis=1)

y=INDU_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_INDU=model()

regressor_INDU.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_INDU.to_json()

with open("model_INDU.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_INDU.save_weights("model_INDU.h5")

print("Saved model to disk")

    

FEROZ_df=df[df["symbol"]=="FEROZ"]

print(len(FEROZ_df))





#taking Last price coloum

Stock_price_FEROZ =FEROZ_df.iloc[:,2:3].values

plt.plot(Stock_price_FEROZ[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FEROZ)



X=FEROZ_df.drop("last_price",axis=1)

y=FEROZ_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))









#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_FEROZ=model()

regressor_FEROZ.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_FEROZ.to_json()

with open("model_FEROZ.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_FEROZ.save_weights("model_FEROZ.h5")

print("Saved model to disk")

    

DCL_df=df[df["symbol"]=="DCL"]

print(len(DCL_df))





#taking Last price coloum

Stock_price_DCL =DCL_df.iloc[:,2:3].values

plt.plot(Stock_price_DCL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DCL)



X=DCL_df.drop("last_price",axis=1)

y=DCL_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





    

    

x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_DCL=model()

regressor_DCL.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_DCL.to_json()

with open("model_DCL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_DCL.save_weights("model_DCL.h5")

print("Saved model to disk")

    

FFBL_df=df[df["symbol"]=="FFBL"]

print(len(FFBL_df))





#taking Last price coloum

Stock_price_FFBL =FFBL_df.iloc[:,2:3].values

plt.plot(Stock_price_FFBL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FFBL)



X=FFBL_df.drop("last_price",axis=1)

y=FFBL_df["last_price"]





x_train=[]

y_train=[]



for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_FFBL=model()

regressor_FFBL.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_FFBL.to_json()

with open("model_FFBL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_FFBL.save_weights("model_FFBL.h5")

print("Saved model to disk")

    

GHNL_df=df[df["symbol"]=="GHNL"]

print(len(GHNL_df))





#taking Last price coloum

Stock_price_GHNL =GHNL_df.iloc[:,2:3].values

plt.plot(Stock_price_GHNL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_GHNL)



X=GHNL_df.drop("last_price",axis=1)

y=GHNL_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_GHNL=model()

regressor_GHNL.fit(x_train,y_train,epochs =20, batch_size = 5000)

model_json = regressor_GHNL.to_json()

with open("model_GHNL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_GHNL.save_weights("model_GHNL.h5")

print("Saved model to disk")

    

HCAR_df=df[df["symbol"]=="HCAR"]

print(len(HCAR_df))





#taking Last price coloum

Stock_price_HCAR =HCAR_df.iloc[:,2:3].values

plt.plot(Stock_price_HCAR[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HCAR)



X=HCAR_df.drop("last_price",axis=1)

y=HCAR_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_HCAR=model()

regressor_HCAR.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_HCAR.to_json()

with open("model_HCAR.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_HCAR.save_weights("model_HCAR.h5")

print("Saved model to disk")

    

FDIBL_df=df[df["symbol"]=="FDIBL"]

print(len(FDIBL_df))





#taking Last price coloum

Stock_price_FDIBL =FDIBL_df.iloc[:,2:3].values

plt.plot(Stock_price_FDIBL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_FDIBL)



X=FDIBL_df.drop("last_price",axis=1)

y=FDIBL_df["last_price"]





x_train=[]

y_train=[]



for i in range(60,X[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_FDIBL=model()

regressor_FDIBL.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_FDIBL.to_json()

with open("model_FDIBL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_FDIBL.save_weights("model_FDIBL.h5")

print("Saved model to disk")

    

KOHC_df=df[df["symbol"]=="KOHC"]

print(len(KOHC_df))





#taking Last price coloum

Stock_price_KOHC =KOHC_df.iloc[:,2:3].values

plt.plot(Stock_price_KOHC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_KOHC)



X=KOHC_df.drop("last_price",axis=1)

y=KOHC_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_KOHC=model()

regressor_KOHC.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_KOHC.to_json()

with open("model_KOHC.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_KOHC.save_weights("model_KOHC.h5")

print("Saved model to disk")

    

HUBC_df=df[df["symbol"]=="HUBC"]

print(len(HUBC_df))





#taking Last price coloum

Stock_price_HUBC =HUBC_df.iloc[:,2:3].values

plt.plot(Stock_price_HUBC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HUBC)



X=HUBC_df.drop("last_price",axis=1)

y=HUBC_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_HUBC=model()

regressor_HUBC.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_HUBC.to_json()

with open("model_HUBC.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_HUBC.save_weights("model_HUBC.h5")

print("Saved model to disk")

    

BAFL_df=df[df["symbol"]=="BAFL"]

print(len(BAFL_df))





#taking Last price coloum

Stock_price_BAFL =BAFL_df.iloc[:,2:3].values

plt.plot(Stock_price_BAFL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_BAFL)



X=BAFL_df.drop("last_price",axis=1)

y=BAFL_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_BAFL=model()

regressor_BAFL.fit(x_train,y_train,epochs =20, batch_size = 5000)

model_json = regressor_BAFL.to_json()

with open("model_BAFL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_BAFL.save_weights("model_BAFL.h5")

print("Saved model to disk")

    

ISL_df=df[df["symbol"]=="ISL"]

print(len(ISL_df))





#taking Last price coloum

Stock_price_ISL =ISL_df.iloc[:,2:3].values

plt.plot(Stock_price_ISL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ISL)



X=ISL_df.drop("last_price",axis=1)

y=ISL_df["last_price"]





x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))









#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_ISL=model()

regressor_ISL.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_ISL.to_json()

with open("model_ISL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_ISL.save_weights("model_ISL.h5")

print("Saved model to disk")

    

KAPCO_df=df[df["symbol"]=="KAPCO"]

print(len(KAPCO_df))





#taking Last price coloum

Stock_price_KAPCO =KAPCO_df.iloc[:,2:3].values

plt.plot(Stock_price_KAPCO[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_KAPCO)



X=KAPCO_df.drop("last_price",axis=1)

y=KAPCO_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_KAPCO=model()

regressor_KAPCO.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_KAPCO.to_json()

with open("model_KAPCO.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_KAPCO.save_weights("model_KAPCO.h5")

print("Saved model to disk")

    

ANL_df=df[df["symbol"]=="ANL"]

print(len(ANL_df))





#taking Last price coloum

Stock_price_ANL =ANL_df.iloc[:,2:3].values

plt.plot(Stock_price_ANL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_ANL)



X=ANL_df.drop("last_price",axis=1)

y=ANL_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_ANL=model()

regressor_ANL.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_ANL.to_json()

with open("model_ANL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_ANL.save_weights("model_ANL.h5")

print("Saved model to disk")

    

DFML_df=df[df["symbol"]=="DFML"]

print(len(DFML_df))





#taking Last price coloum

Stock_price_DFML =DFML_df.iloc[:,2:3].values

plt.plot(Stock_price_DFML[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_DFML)



X=DFML_df.drop("last_price",axis=1)

y=DFML_df["last_price"]







x_train=[]

y_train=[]



for i in range(60,X.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    







x_train,y_train = np.array(x_train),np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print(x_train.shape)







#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_DFML=model()

regressor_DFML.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_DFML.to_json()

with open("model_DFML.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_DFML.save_weights("model_DFML.h5")

print("Saved model to disk")

    

BOP_df=df[df["symbol"]=="BOP"]

print(len(BOP_df))





#taking Last price coloum

Stock_price_BOP =BOP_df.iloc[:,2:3].values

plt.plot(Stock_price_BOP[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_BOP)



X=BOP_df.drop("last_price",axis=1)

y=BOP_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_BOP=model()

regressor_BOP.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_BOP.to_json()

with open("model_BOP.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_BOP.save_weights("model_BOP.h5")

print("Saved model to disk")

    

HASCOL_df=df[df["symbol"]=="HASCOL"]

print(len(HASCOL_df))





#taking Last price coloum

Stock_price_HASCOL =HASCOL_df.iloc[:,2:3].values

plt.plot(Stock_price_HASCOL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_HASCOL)



X=HASCOL_df.drop("last_price",axis=1)

y=HASCOL_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_HASCOL=model()

regressor_HASCOL.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_HASCOL.to_json()

with open("model_HASCOL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_HASCOL.save_weights("model_HASCOL.h5")

print("Saved model to disk")

    

MARI_df=df[df["symbol"]=="MARI"]

print(len(MARI_df))





#taking Last price coloum

Stock_price_MARI =MARI_df.iloc[:,2:3].values

plt.plot(Stock_price_MARI[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_MARI)



X=MARI_df.drop("last_price",axis=1)

y=MARI_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_MARI=model()

regressor_MARI.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_MARI.to_json()

with open("model_MARI.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_MARI.save_weights("model_MARI.h5")

print("Saved model to disk")

    

LUCK_df=df[df["symbol"]=="LUCK"]

print(len(LUCK_df))





#taking Last price coloum

Stock_price_LUCK =LUCK_df.iloc[:,2:3].values

plt.plot(Stock_price_LUCK[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_LUCK)



X=LUCK_df.drop("last_price",axis=1)

y=LUCK_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_LUCK=model()

regressor_LUCK.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_LUCK.to_json()

with open("model_LUCK.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_LUCK.save_weights("model_LUCK.h5")

print("Saved model to disk")

    

PIAA_df=df[df["symbol"]=="PIAA"]

print(len(PIAA_df))





#taking Last price coloum

Stock_price_PIAA =PIAA_df.iloc[:,2:3].values

plt.plot(Stock_price_PIAA[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PIAA)



X=PIAA_df.drop("last_price",axis=1)

y=PIAA_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_PIAA=model()

regressor_PIAA.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_PIAA.to_json()

with open("model_PIAA.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_PIAA.save_weights("model_PIAA.h5")

print("Saved model to disk")

    

PASL_df=df[df["symbol"]=="PASL"]

print(len(PASL_df))





#taking Last price coloum

Stock_price_PASL =PASL_df.iloc[:,2:3].values

plt.plot(Stock_price_PASL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_PASL)



X=PASL_df.drop("last_price",axis=1)

y=PASL_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_PASL=model()

regressor_PASL.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_PASL.to_json()

with open("model_PASL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_PASL.save_weights("model_PASL.h5")

print("Saved model to disk")

    

POL_df=df[df["symbol"]=="POL"]

print(len(POL_df))





#taking Last price coloum

Stock_price_POL =POL_df.iloc[:,2:3].values

plt.plot(Stock_price_POL[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_POL)



X=POL_df.drop("last_price",axis=1)

y=POL_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_POL=model()

regressor_POL.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_POL.to_json()

with open("model_POL.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_POL.save_weights("model_POL.h5")

print("Saved model to disk")

    

OGDC_df=df[df["symbol"]=="OGDC"]

print(len(OGDC_df))





#taking Last price coloum

Stock_price_OGDC =OGDC_df.iloc[:,2:3].values

plt.plot(Stock_price_OGDC[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_OGDC)



X=OGDC_df.drop("last_price",axis=1)

y=OGDC_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_OGDC=model()

regressor_OGDC.fit(x_train,y_train,epochs = 3, batch_size = 5000)

model_json = regressor_OGDC.to_json()

with open("model_OGDC.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_OGDC.save_weights("model_OGDC.h5")

print("Saved model to disk")

    

NBP_df=df[df["symbol"]=="NBP"]

print(len(NBP_df))





#taking Last price coloum

Stock_price_NBP =NBP_df.iloc[:,2:3].values

plt.plot(Stock_price_NBP[1:100])

plt.show()



import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(Stock_price_NBP)



X=NBP_df.drop("last_price",axis=1)

y=NBP_df["last_price"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.10, random_state=42)

print(X_train.shape)

print(X_test.shape)





x_train=[]

y_train=[]

x_test=[]

y_test=[]

for i in range(60,X_train.shape[0]):

    x_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i,0])

    





for i in range(60,X_test.shape[0]):

    

    x_test.append(training_set_scaled[i-60:i, 0])

    y_test.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_train.shape)

print(x_test.shape)





#Building the RNN LSTM model

# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers.core import Dense, Activation, Dropout

#Using TensorFlow backend.

def model():

    regressor = Sequential()

    regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50,return_sequences = True))

    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))

    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')

    return regressor

regressor_NBP=model()

regressor_NBP.fit(x_train,y_train,epochs = 20, batch_size = 5000)

model_json = regressor_NBP.to_json()

with open("model_NBP.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

regressor_NBP.save_weights("model_NBP.h5")

print("Saved model to disk")

    
