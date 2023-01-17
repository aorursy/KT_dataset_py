import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
df.shape
df.dtypes
df.isnull().values.any()
df['Class'].unique()
df.describe()
scale=StandardScaler()
df[['Amount','Time']]=scale.fit_transform(np.array(df[['Amount','Time']]))
df.hist(figsize=(25,25))
plt.show()
corrmat=df.corr()
fig=plt.figure(figsize=(12,12))
sns.heatmap(corrmat,vmax=1.0,square=True,linewidths=0.1)
plt.show()
col=df.columns.tolist()
x=df.drop(columns='Class') #input featuures
y=df['Class']  #target or output feature
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=5)
print('xtrain: {0}, ytrain: {1}'.format(xtrain.shape,ytrain.shape))
print('xtest: {0}, ytest: {1}'.format(xtest.shape,ytest.shape))
fraud=(ytrain==1).sum()
valid=(ytrain==0).sum()
print('No. of fraud cases:',fraud)
print('No. of valid cases:',valid)
resample=SMOTE(random_state=5)
x_train,y_train=resample.fit_sample(xtrain,ytrain)
print(x_train.shape,y_train.shape)
re_fraud=(y_train==1).sum()
re_valid=(y_train==0).sum()
print('No. of fraud cases after resampling:',re_fraud)
print('No. of valid cases after resampling:',re_valid)
import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from mlxtend.plotting import plot_confusion_matrix
epoch=10
act_func='relu'
lr=0.0001
dropout_rate=0.5
batchsize=32
loss_func='binary_crossentropy'
model=Sequential()
model.add(Dense(60,input_shape=(30,),activation=act_func))
model.add(Dropout(0.5,seed=0))
model.add(Dense(35,activation=act_func))
model.add(Dropout(0.5,seed=0))
model.add(Dense(1,activation='sigmoid'))
model.summary()
opt=Adam(learning_rate=lr)
model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'] )
history=model.fit(x_train, y_train, batch_size=batchsize, epochs=epoch, validation_split=0.2, shuffle=True)
print(history.history.keys()) 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model accuracy (Training vs Validation)')
plt.legend(['train', 'validation'],loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Model loss (Training vs Validation)')
plt.legend(['train', 'validation'],loc='upper right')
plt.show()
loss, acc=model.evaluate(xtest,ytest,batch_size=batchsize)
print('Loss: ',loss)
print('Accuracy: ',acc)
ypred=model.predict_classes(xtest)
label=['valid','fraud']
con_mat=confusion_matrix(ytest,ypred)
plt.figure()
cm=plot_confusion_matrix(con_mat, class_names=['Valid','Fraud'],colorbar=True)
ax = plt.gca()
ax.set_ylim([1.4,-0.4])
plt.show()
model.save('Fraud_detection.h5')
