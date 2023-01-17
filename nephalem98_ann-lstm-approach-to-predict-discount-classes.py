import pandas as pd #Data Analysis

import numpy as np #Linear Algebra

import seaborn as sns #Data Visualization

import matplotlib.pyplot as plt #Data Visualization
import os

print(os.listdir("../input"))
#This is the Product_sales_train_and_test dataset but without the "[]" in the Customer Basket.

df1=pd.read_csv("../input/remove/data.csv")
df2=pd.read_csv("../input/discount-prediction/Train.csv")

df3=pd.read_csv("../input/discount-prediction/test.csv")
df1.fillna(float(0.0),inplace=True)

df2.fillna(float(0.0),inplace=True)
from sklearn.feature_extraction.text import CountVectorizer

cv1 = CountVectorizer(max_features=500)

y = cv1.fit_transform(df1["Customer_Basket"]).toarray()
thirty= list(y)

thirty1=pd.DataFrame(thirty)
final=pd.concat([df1,thirty1],axis=1)
df2=df2[df2["BillNo"]!=float(0.0)]
finaltrain=pd.merge(final,df2,on="BillNo",how="inner")

finaltest=pd.merge(final,df3,on="BillNo",how="inner")
finaltrain.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)

finaltest.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)
X=finaltrain.drop(["Discount 5%","Discount 12%","Discount 18%","Discount 28%"],axis=1)

y=finaltrain[["Discount 5%","Discount 12%","Discount 18%","Discount 28%"]]
X1, y2 = np.array(X), np.array(y)
var = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 500))

classifier.add(Dropout(0.2))



# Adding the second hidden layer

classifier.add(Dense(units =32 , kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))



classifier.add(Dense(units =16 , kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))



# Adding the output layer

classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X, y, batch_size = 10, epochs = 50)
annpredictions=classifier.predict(finaltest)
discountann=list(annpredictions)
abbasann=pd.DataFrame(discountann)
abbasann=(abbasann> 0.4)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (var.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a fourth LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units=4, activation='softmax'))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'categorical_crossentropy')



# Fitting the RNN to the Training set

regressor.fit(var, y2, epochs = 1, batch_size = 32)

finaltest1=np.array(finaltest)

baas=np.reshape(finaltest1, (finaltest1.shape[0], finaltest1.shape[1], 1))
discountclass=regressor.predict(baas)
discountbaas=list(discountclass)
abbas=pd.DataFrame(discountbaas)
abbas= (abbas > 0.3)