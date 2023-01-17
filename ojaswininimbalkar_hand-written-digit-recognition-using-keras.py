import tensorflow as tf

from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import normalize

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten

import matplotlib.pyplot as plt #to visualize an image from our dataset

mnist=tf.keras.datasets.mnist  #28x28 image of hand-written digits
(x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.imshow(x_train[0],cmap=plt.cm.binary)
#It's easier for our network to learn this way



x_train=normalize(x_train,axis=1)

x_test=normalize(x_test,axis=1)



#Now, the data has been normalized
#The image will look a bit dull as each pixel stores a value between 0-1, but don't worry..



plt.imshow(x_train[0],cmap=plt.cm.binary)
#Loading the Sequential model



model=Sequential() 
# data is in multidimensional form,so we need to make it into simpler form by flattening it:



model.add(Flatten())

model.add(Dense(128,activation="relu"))  #128 neurons, activation function to fire up the neuron

model.add(Dense(128,activation="relu"))

model.add(Dense(10,activation="softmax")) 



#10 because number of classification is from 0-9 i.e. 10 different types of data





#softmax for probability distribution



#parameters for training of the model:



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])     



#loss is the degree of error


#neural networks don't actually try to increase accuracy but tries to minimize loss

#epochs are just number of iterations the whole model will follow through



model.fit(x_train,y_train,epochs=3)

val_loss,val_acc=model.evaluate(x_test,y_test)

print("\n\n\n\tEvaluation of the Model:\n")

print("Value Accuracy=",val_acc,"\nValue Loss=",val_loss)
#Saving the model as Number_Reader

    

model.save('Number_Reader.model') 
#importing our Number_Reader.model into new_model



new_model = tf.keras.models.load_model('Number_Reader.model')
pred=new_model.predict(x_test)

print(pred)  #these are all probability distributions

import numpy as np
print("MODEL OUTPUT: ")

print(np.argmax(pred[0])) 



#takes out the maximum value, from the probability density given above
print("The Actual Image was of:")

plt.imshow(x_test[0])

plt.show()