import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
#Sklearn OneHot Encoder to Encode categorical integer features
from sklearn.preprocessing import OneHotEncoder
#Sklearn train_test_split to split a set on train and test 
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
#Import Keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
#Load the training  and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.head()
trainX = np.array(train.iloc[:,1:])
trainY = np.array(train.iloc[:,0]).reshape(-1,1)
# Plot some digits from training example
c = np.array([e.tolist() for e in trainX]).reshape(-1,28,28)   # create 3D arrray with digits (42000, 28, 28)
fig, axes = plt.subplots(10, 10, figsize=(6, 6),
subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(c[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(trainY[i]),
    transform=ax.transAxes, color='green')
sns.countplot(train.iloc[:,0])
#Normalization 
trainX = trainX/255
trainX = trainX.astype('float32')

testX = np.array(test)
#Normalization
testX = testX/255
testX = testX.astype('float32')

trainX.shape, trainY.shape, testX.shape
#Encode the Y matrix
enc = OneHotEncoder(sparse=False) 
Y_enc=enc.fit_transform(trainY)
Y_enc.shape
xn_train, xn_test, yn_train, yn_test = train_test_split(trainX, Y_enc, test_size=0.2, random_state=40)
xn_train.shape, xn_test.shape, yn_train.shape, yn_test.shape
model = Sequential()
model.add(Dense(800, input_dim=784, init='normal',activation='relu')) #init = initial weights
model.add(Dense(10, init = 'normal',activation='softmax'))  #activation='relu' activation function
model.compile(loss='categorical_crossentropy',optimizer='SGD', metrics=['accuracy'])
print(model.summary())
model.fit(xn_train, yn_train, batch_size = 300, nb_epoch = 300, verbose = 1) # analyze 200 images, 
#define the gradient direction, change weights,
#verbose = 1   --- print results for each epoch
prediction = model.predict(xn_test)
prediction.shape
y_pred = np.array(np.argmax(prediction, axis=1)) 
y_pred.shape
y_pred
yn_test1 = np.array(np.argmax(yn_test, axis=1)) 
yn_test1
def accuracy(model, testX, testY):      #testY is HotEncoded!!
    m = len(testY)
    prediction = model.predict(testX)
    prediction = np.array(np.argmax(prediction, axis=1)) 
    testY = np.array(np.argmax(testY, axis=1))
    accur = np.array([1 for (a,b) in zip(prediction,testY) if a==b ]).sum()/m
    return accur, prediction
#Check the accuracy of the test validation data
accur, prediction = accuracy(model, xn_test, yn_test)
print(accur)
prediction = model.predict(testX)
prediction = np.array(np.argmax(prediction, axis=1)) 
# Submit the result
#submission_df = {"ImageId": np.linspace(1,len(prediction),len(prediction)).astype(int), "Label": prediction}
#submission = pd.DataFrame(submission_df)
#submission.to_csv("submission_1(NN_Keras).csv",index=False)