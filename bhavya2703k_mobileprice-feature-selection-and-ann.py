import pandas as pd

import numpy as np
data= pd.read_csv("../input/mobile-price-classification/train.csv")
data.head()
data.info()

data.count()
import matplotlib.pyplot as plt

import seaborn as sns
data['price_range'].describe()

corrmat = data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
data.drop(columns= ["three_g",'four_g',"m_dep","n_cores","sc_h","pc","wifi","blue"],axis=1, inplace=True)
print("No. of touch screen phones is %f",sum(data['touch_screen']==1))

data.drop(columns=['touch_screen','mobile_wt','clock_speed'],axis=1, inplace=True)
data
y= data.iloc[:,-1]

x= data.iloc[:,0:9]
x
import seaborn as sns

x.boxplot(figsize=(14,8))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)
type(x)
y= y.to_numpy()

type(y)
y= y.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder

ohot = OneHotEncoder()

y = ohot.fit_transform(y)
y= y.toarray()

y
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

import tensorflow as tf

from tensorflow import keras

from keras.layers import Flatten,Dense

from keras.models import Sequential
model=Sequential()

model.add(Dense(10, activation='relu', input_dim=9 ))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='softmax'))

model.summary()
opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test), batch_size=32)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Validation Set'],loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train','Validation Set'],loc='upper left')

plt.show()
y_pred = model.predict(x_test)



#lets do the inverse one hot encoding

pred = []

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))

    

# also inverse encoding for y_test labels



test = []

for i in range(len(y_test)):

    test.append(np.argmax(y_test[i]))


from sklearn.metrics import accuracy_score

acc = accuracy_score(pred,test)

print("Accuracy of Your Model is = " + str(acc*100))
from tensorflow.keras.models import save_model, load_model

filepath = './input/saved_model'

save_model(model, filepath)
test = pd.read_csv("../input/mobile-price-classification/test.csv")

test.drop(columns= ["id","three_g",'four_g',"m_dep","n_cores","sc_h","pc","wifi","blue",'touch_screen','mobile_wt','clock_speed'],axis=1, inplace=True)
test.head()
test.info()

test.isna().sum()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

test = sc.fit_transform(test)
type(test)
model= load_model(filepath, compile=True)
predictions = model.predict(test)

Price_range = []

for i in range(len(predictions)):

    Price_range.append(np.argmax(predictions[i]))
Price_range

my_sub = pd.DataFrame({'PriceRange':Price_range})

my_sub.to_csv('submission.csv', index=True)