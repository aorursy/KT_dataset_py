## import library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

try:

    !pip install tensorflow-gpu

    import tensorflow as tf

except:

    !pip install tensorflow

    import tensorflow as tf

import seaborn as sns

%matplotlib inline
## import iris data from seaborn data

iris = sns.load_dataset('iris')
iris.head()
from sklearn.preprocessing import LabelEncoder

def encode(df):

    encoder = LabelEncoder()

    target=encoder.fit_transform(df)

    return np.array(target)
target = encode(iris['species'])
target
iris['target'] = np.array(target)
iris_working = iris.drop('species',axis=1)
iris_working.head()
corr = iris_working.corr()
sns.heatmap(corr,cmap='coolwarm',annot=True)
corr2 = iris_working.corr()['target']
corr2.plot()
X = iris_working.drop('target',axis=1)

y = iris_working[['target']]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
def normalize(df):

    result = df.copy()

    for feature_name in df.columns:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result
X_train = normalize(X_train)

X_test = normalize(X_test)
X_train.head()
X_test.head()
def NNmodel():

    

    model = tf.keras.models.Sequential() ## making th sequental model

    ## add layer we have to change the shape with flatten

    model.add(tf.keras.layers.Flatten())

    ## perceptron 128 per layer actication function rectified linear

    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(.25))

    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(.25))

    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

    model.add(tf.keras.layers.Dropout(.25))

    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

    ## final activation function softmax and 3 cause data will be 3 catagory

    model.add(tf.keras.layers.Dense(3,activation=tf.nn.softmax))

    

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model
model = NNmodel()
model.fit(np.array(X_train),np.array(y_train),epochs=30)
loss,acc=model.evaluate(X_test,y_test)
print ("LOSS : "+str(loss))

print ("ACCURACY : "+str(acc))
loss_array=[]

accuracy_array=[]

for epoch in range(1,200):

    tmpmodel = NNmodel()

    tmpmodel.fit(np.array(X_train),np.array(y_train),epochs=epoch)

    loss,acc=tmpmodel.evaluate(X_test,y_test)

    loss_array.append(loss)

    accuracy_array.append(acc)

    

    
x = list(range(1,200))

plt.grid()

sns.lineplot(x, accuracy_array, color='green',label='accuracy', linestyle='-', markersize=12)

sns.lineplot(x, loss_array, color='red', linestyle='--',label='loss', markersize=12)



#lt.legend()

model.fit(np.array(X_train),np.array(y_train),epochs=30)

predicted=model.predict(X_test)
predicted
y_pred=[]

for item in range(len(predicted)):

    y_pred.append(np.argmax(predicted[item]))    
y_pred
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test,y_pred)
print ("MEAN SQUARED LOSS "+str(mse))