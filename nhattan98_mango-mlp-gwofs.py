from keras.models import Sequential
from keras.layers import Dense 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from keras.utils import to_categorical
import random

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
def decode(datum):
    return np.argmax(datum)
df=pd.read_excel('../input/mango-mlp/Traindata2.xlsx',sheet_name='New')

transformer = RobustScaler().fit(df)
df_scaled=pd.DataFrame(transformer.transform(df))
df_scaled.columns=df.columns
df_scaled.Label=df.Label



# df = df_scaled
X = df_scaled.loc[:, df.columns != 'Label']
y = df_scaled.loc[:, 'Label']




y=to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.2)

Selected=np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0,
 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1,
 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
print(Selected)
X_s=X.T[Selected>0]
print(X_s.T.head())
print(X_s.shape,len(X_s))
X_s.T.columns
temp=X_s.T
temp=temp.drop(['MinorAxisLength','EquivDiameter','B_mean','Contrast45','Contrast135','energy45','energy90','energy135','Extent'],axis=1)
print(temp.columns)
new_df=temp
#new_df= new_df.loc[:, (new_df != 0).any(axis=0)]
print(new_df.head(),new_df.shape)
#Retrain with FS
X_s=new_df
print(X_s.shape)

X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
test_size=0.2)


model=Sequential()

model.add(Dense(30,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(50,activation='relu'))
model.add(Dense(35,activation='relu'))
#model.add(Dense(25,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=100, validation_data=(X_val, y_val))

scores_train=model.evaluate(X_train,y_train)
print('Train Acc:',scores_train[1])
score_test=model.evaluate(X_test,y_test)
print('Test Acc:',score_test[1])



SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
fig, ax = plt.subplots(figsize=(8,4))
plt.plot(epochs, acc, 'k', label='Training accurarcy')
plt.plot(epochs, val_acc, 'k--', label='Validation accurarcy')
plt.title('Training and Validation accuracy of ANN')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()
fig.savefig('Train&Val_acc.svg',bbox_inches='tight',pad_inches=0, format='svg', dpi=1200)

fig2,ax=plt.subplots(figsize=(8,4))
#Train and validation loss
plt.plot(epochs, loss,  'k', label='Training loss')
plt.plot(epochs, val_loss, 'k--', label='Validation loss')
plt.title('Training and Validation loss of ANN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()
fig2.savefig('Train&Val_loss.svg',bbox_inches='tight',pad_inches=0, format='svg', dpi=1200)



model.evaluate(new_df,y)
y_pred_test=model.predict_classes(X_test)
print(len(y_pred_test),len(y_test))
print(type(y_pred_test))
print(y_pred_test[:])
decoded_datum=[]
for i in range(y_test.shape[0]):
    datum = y_test[i]
    
    decoded_datum.append(decode(y_test[i]))
print(np.array(decoded_datum))


matrix = confusion_matrix(np.array(decoded_datum), y_pred_test)
print(classification_report(np.array(decoded_datum), y_pred_test))
print(matrix)
model.save('MLPv1_67-58.h5')