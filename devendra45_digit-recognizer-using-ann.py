import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
digits=pd.concat([train,test],axis=0)
train.shape,test.shape
#printing the mergerd data
print(digits.shape)
digits.head()
target=train.label
train.drop(columns=['label'],inplace=True)
y_train=np.asarray(target)
y_train
X_train=np.asarray(train)
X_test=np.asarray(test)
X_train.shape,X_test.shape
X_train=X_train/255
X_test=X_test/255
X_train=X_train.reshape(-1,28,28)
X_test=X_test.reshape(-1,28,28)
X_train.shape,X_test.shape
plt.matshow(X_train[0])
plt.show()
plt.matshow(X_train[1])
plt.show()
plt.matshow(X_train[2])
plt.show()
#Class labels of above image pixels of training data
y_train[0:3]
# test image pixel
plt.matshow(X_test[0])
plt.show()
# test image pixel
plt.matshow(X_test[5])
plt.show()
#class labeles present in datasets
class_labels=list(set(y_train))
class_labels
ann=Sequential()
# input layers of size of 28*28
ann.add(Flatten(input_shape=[28,28]))

# 3 hidden layers containing 100 neurons
ann.add(Dense(512,activation='relu'))
ann.add(Dense(256,activation='relu'))
ann.add(Dense(128,activation='relu'))

#output layers containing 10 neurons to predict each of digit
ann.add(Dense(10,activation='softmax'))
ann.summary()
ann.compile(loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
            
ann.fit(X_train,y_train,batch_size=32,epochs=15)
detail=ann.evaluate(X_train,y_train)
print('loss:',detail[0])
print('accuracy achieved:',round(detail[1]*100,4))
y_pred=ann.predict(X_test)
#predicted digits are output of test data
predicted_digits=[class_labels[np.argmax(y_pred[i])] for i in range(len(y_pred))]
print("first ten outputs of test data:",*predicted_digits[:10])
                  
#checking our output for 2nd digit in test data
plt.matshow(X_test[1])
plt.show()
#above pixel image is 0 and prediction also showing 0
predicted_digits[1]
images=np.random.choice(len(X_test),size=12)
print("")
fig=plt.figure(figsize=(15,9))
fig.suptitle("predicted outputs of random handwritten digits".upper(),fontsize=18)
fig.subplots_adjust(hspace=0.5,wspace=0.5)

for i,num in zip(images,range(1,13)):
    label=class_labels[np.argmax(y_pred[i])]
    ax=fig.add_subplot(3,4,num)
    ax.matshow(X_test[i])
    ax.set_xlabel("Prediction-->{}".format(label),fontsize=16)
images=np.random.choice(len(X_test),size=12)
print("")
fig=plt.figure(figsize=(15,9))
fig.suptitle("predicted outputs of random handwritten digits".upper(),fontsize=18)
fig.subplots_adjust(hspace=0.5,wspace=0.5)

for i,num in zip(images,range(1,13)):
    label=class_labels[np.argmax(y_pred[i])]
    ax=fig.add_subplot(3,4,num)
    ax.matshow(X_test[i])
    ax.set_xlabel("Prediction-->{}".format(label),fontsize=16)
y_train_pred=ann.predict(X_train)
y_train_p=[class_labels[np.argmax(y_train_pred[i])] for i in range(len(y_train_pred))]
y_train_p[0:10]
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
con_mat=confusion_matrix(y_train,y_train_p)
print("Confusion Matrix")
pd.DataFrame(con_mat,columns=class_labels,index=class_labels)
my_submission=pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
my_submission.head()
my_submission['Label']=predicted_digits
my_submission.head(10)
my_submission.to_csv("digits_submission2.csv",index=False)
pd.read_csv("digits_submission2.csv").head(10)
