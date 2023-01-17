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
X_train=np.load('../input/data/data/train.npy')
Y_train=np.load('../input/data/data/trainlabels.npy')
X_test=np.load('../input/data/data/test.npy')
Y_test=np.load('../input/data/data/testlabels.npy')

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from keras.layers import Embedding,Reshape,Dropout,Dense,Conv2D,Flatten,Lambda,Input,Concatenate
from keras.models import Model 
from keras import backend as K
from keras.preprocessing.text import Tokenizer
import numpy as np 


vocab_size=43297
embedding_dim=300
input_length=2400
n_filters=100

def get_random_model(print_summary=False):

	inp=Input(shape=(input_length,),name='StaticWord2VecInput')
	embed=Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=input_length)(inp)
	embed=Lambda(lambda x: K.expand_dims(x))(embed)
	x1=Conv2D(filters=n_filters,kernel_size=(3,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s3')(embed)
	r1=Reshape((-1,n_filters))(x1)
	maxpool1=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool1')(r1)




	x2=Conv2D(filters=n_filters,kernel_size=(4,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s4')(embed)
	r2=Reshape((-1,n_filters))(x2)
	maxpool2=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool2')(r2)


	x3=Conv2D(filters=n_filters,kernel_size=(5,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s5')(embed)
	r3=Reshape((-1,n_filters))(x3)
	maxpool3=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool3')(r3)



	concatenated=Concatenate(axis=1,name='concatenated')([maxpool1,maxpool2,maxpool3])
	concatenated=Dropout(0.5)(concatenated)

	dense=Dense(1,activation='sigmoid',name='outputlayers')(concatenated)

	model=Model(inputs=[inp],outputs=[dense])

	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

	if print_summary:
		model.summary()

	return model 
model=get_random_model(print_summary=True)
history=model.fit(X_train,Y_train,validation_split=0.1,epochs=7)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Training_Loss')
plt.show()

test_loss,test_accuracy=model.evaluate(X_test,Y_test)
print("Accuracy on test data : {}".format(test_accuracy))
