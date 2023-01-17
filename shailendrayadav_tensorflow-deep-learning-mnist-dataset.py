import tensorflow as tf

tf.__version__
import numpy as np

import pandas as pd
#loading mnist data 

train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")
train_df.head()
test_df.columns
train_df.columns
y_train=train_df["label"].values

y_train
X_train=train_df.drop(columns="label").values

#X_train=X.values

X_train.shape
#since the data is in range between 0 -255 lets normalize it as a better model practice

X_train=tf.keras.utils.normalize(X_train,axis=1)

X_train[0]  # itxs now range between 0-1
n_cols =X_train.shape[1] # columns
#lets begin the neural network code

model=tf.keras.models.Sequential()#commonly used

#adding layers

#input layer

model.add(tf.keras.layers.Flatten())

#hidden layers with number of neurons and its activation function(RELU is used now a days,sigmoid function are old)

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))



#outputlayer newuros are 10 from 0-9 as output        

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))#not relu as we aredoing probability distribn so using softmax

#running a model 

#neural network always run on a principle of minimising the loss.

#deciding optimizer and loss is major in nueral network

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#training  #neurons fire here

model.fit(X_train,y_train,epochs=5)
test_df.head()

X_test=test_df.values
#since the data is in range between 0 -255 lets normalize it as a better model practice

X_test=tf.keras.utils.normalize(X_test,axis=1)

X_test[0]  # itxs now range between 0-1
#val_loss,val_acc=model.evaluate(X_test,y_test)
predictions=model.predict([X_test])

predictions
#showing X_test 10 predictions

#for i in range(0,10):

 #   plt.matshow(X_test[i])
import numpy as np

for i in range(0,10):

    print(np.argmax(predictions[i]))

    
results = model.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("mnist_data_compertition.csv",index=False)

#save the mdel

model.save("mnist_NN_model")