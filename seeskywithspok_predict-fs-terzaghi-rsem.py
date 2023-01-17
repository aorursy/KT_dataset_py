#Import library
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import keras
#Import dataset
df = pd.read_csv('../input/fs-terzaghi-finalcsv/FS_TERZAGHI_FINAL.csv')
df.corr()
##Knowing The Data
##Correlation heatmap
corr = df.corr()
sns.heatmap(corr,annot = True,cmap ='coolwarm',fmt='.2f') #เทียบค่าที่ส่งผลกับfs บรรทัดสุดท้าย
plt.title("Correlation Between Variables")
sns.pairplot(df,palette="husl",diag_kind="kde") #ใช้กราฟทแยงเป็นการกระจายของ input 
#Import ANN
# Building ANN As a Regressor
from keras.models import Sequential
from keras.layers import Dense
from keras import backend

#Prepare data
X = df.drop(columns=['FS'])
Y = df['FS']
Xt=X.to_numpy()
Yt=Y.to_numpy()
# Using Test/Train Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xt,Yt, test_size=0.15, random_state = 5)
print ('train set:', X_train.shape)
print ('test set:', X_test.shape)
print ('train set:', y_train.shape)
print ('test set:', y_test.shape)

plt.figure(figsize=(12,4)) #ขนาดกราฟ
plt.subplot(121) #ต้องมี3หลัก แต่ไม่รู้หมายถึงอะไรหาเพิ่ม
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,alpha=0.8)
plt.title('Train set')
plt.subplot(122)
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,alpha=0.8)
plt.title('Test set')

# Feature Scaling
from sklearn.preprocessing import StandardScaler #StandardScaler การแก้ไขvariance ให้เท่ากัน
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Defining Root Mean Square Error As our Metric Function 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


#Building  Layers 
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='relu', input_shape=(7,)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
#output
model.add(keras.layers.Dense(1, activation='relu'))

# Optimize , Compile And Train The Model
model.compile(optimizer='adam',loss='mean_squared_logarithmic_error',metrics=[rmse])
history = model.fit(X_train,y_train,epochs = 1000,batch_size=32,validation_split=0.15)
test=model.evaluate(X_test,y_test,verbose=0)
print(model.summary())
print ('error=',test*100, '%')

y_predict = model.predict(X_test)
print(model)
y_predict
from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))
#plot error
%config TnLineBackend.figure_format='retina'
plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test,y_predict, squared=False))
print('RMSE=', rmse)
plt.scatter(y_test,y_predict)
plt.xlabel('Actual')
plt.ylabel('Predict')
plt.title('Actual VS Predict of Factor of safety')
plt.show()
%config TnLineBackend.figure_format='retina'
# Plotting Loss And Root Mean Square Error For both Training And Test Sets
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('Root Mean Squared Error')
plt.ylabel('rmse')
plt.xlabel('epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('3.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'Validation'], loc='upper left')
plt.savefig('4.png')
plt.show()
model.save('/kaggle/working//PREDICT_FS_TERZAGHI_RMSE.h5')
