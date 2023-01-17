
import numpy as np 
import pandas as pd 
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
%matplotlib inline

K.set_image_data_format('channels_last')
# Any results you write to the current directory are saved as output.
# read train and test dataset
# assign 20% of train dataset as validation dataset
train_file = "../input/train.csv"
test_file = "../input/test.csv"
output_file = "submission.csv"

df_train = pd.read_csv(train_file)
df_train.head()
df_train.info()

df_test = pd.read_csv(test_file)
df_test.info()
df_train,df_valid = train_test_split(df_train,test_size=0.2)
# separate labels from train and valid dataset
df_train_labels = df_train.pop('label')
df_valid_labels = df_valid.pop('label')
df_train.info()
df_valid_labels.head()
# convert DataFrame to numpy array and reshape them for Keras accepting format
num_train_samples = df_train.shape[0]
num_features = df_train.shape[1]
image_shape = (num_train_samples,28,28,1)
X_train = df_train.values.reshape( image_shape)
X_train.shape
Y_train = df_train_labels.values
# same as above for valid dataset
number_valid_samples = df_valid.shape[0]
image_shape = (number_valid_samples,28,28,1)
X_valid = df_valid.values.reshape( image_shape)
print(X_valid.shape)
Y_valid = df_valid_labels.values
# for test dataset
num_test_samples = df_test.shape[0]
image_shape = (num_test_samples,28,28,1)
X_test = df_test.values.reshape(image_shape)
X_test.shape
# confirm if reshaping has done correctly by checking image
plt.imshow(X_train[3].reshape(28,28))
plt.gray()
plt.show()
# define CNN model    Input -> padding -> conv2d -> relu -> max pooling -> padding -> conv2d -> relu -> max pooling -> flatten -> dense -> softmax
def MyModel(Input_shape):
    X_input = Input(Input_shape)
    X=ZeroPadding2D((1,1))(X_input)
    X=Conv2D(10,(3,3),strides=(1,1),name='conv0')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2),name='maxpool0')(X)
    X=ZeroPadding2D((2,2))(X)
    X=Conv2D(20,(5,5),strides=(1,1),name='conv1')(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2))(X)
    X=Flatten()(X)
    X=Dense(10,activation='softmax',name='fc')(X)
    model = Model(inputs = X_input, outputs = X, name='MyModel')
    
    return model
# Chose "sparse_categorical_crossentropy" not "categorical_crossentropy" due to not one-hot representation for label 
model = MyModel((28,28,1))
model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
# train the model 10 epochs and save weights for the future use
model.fit(x=X_train,y=Y_train,epochs=10,batch_size=32)
model.save_weights('mnist.h5')
# check accuracy on valid dataset to confirm if overfitting to train dataset
evals = model.evaluate(x=X_valid,y=Y_valid)
print()
print ("Loss = " + str(evals[0]))
print ("Test Accuracy = " + str(evals[1]))
#predict on test dataset and generate digit number of the prediction 
preds = model.predict(X_test)
res = np.argmax(preds,axis=1)
# randomly choose 1 sample and confirm the result by human eye
index =  np.random.randint(X_test.shape[0])
print("Answer is : ",res[index])
img= X_test[index].reshape(28,28)
plt.imshow(img)
plt.gray()
plt.show()

# convert to Pandas DataFrame ,  format to conform  submission file template and write to a csv file
res =res.reshape(res.shape[0],1)
res_df = pd.DataFrame.from_records(res,columns = ['Label'])
res_df['ImageId'] = res_df.index +1
res_df[['ImageId','Label']].to_csv(output_file,index=False)
