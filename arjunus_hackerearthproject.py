# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
resample = SMOTE(sampling_strategy='minority')
data = pd.read_csv('/kaggle/input/CC.csv')
data.shape
Y = data['Class']
X = data.iloc[:,0:31] 
X
plt.hist(Y)
plt.show()
Y.value_counts()
"""plt.subplots(figsize=(20,10))
sns.heatmap(X.corr(),vmin=-1,vmax=1,annot=True)"""
X.describe()
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts,GridSearchCV
normalizer = Normalizer(norm='l2')
X = normalizer.fit_transform(X)
Normalized_data = pd.DataFrame(X)
Normalized_data_resampled,resampled_class = resample.fit_resample(Normalized_data,Y)
resampled_class.value_counts()
plt.hist(resampled_class)
plt.grid()
plt.xlabel('classes')
plt.ylabel('frequency')
plt.show()
train_data = Normalized_data_resampled
train_target = resampled_class
xtrain,xtest,ytrain,ytest = tts(train_data,train_target,test_size=0.2,random_state=0)
import tensorflow as tf
import keras 
model = keras.Sequential([
        keras.layers.Dense(units = 64,input_shape=(31,)),
        keras.layers.Dense(units = 32,activation='relu',kernel_initializer='random_normal'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units = 16,activation='relu',kernel_initializer='random_normal'),
        keras.layers.Dense(units = 8,activation='relu',kernel_initializer='random_normal'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units = 4,activation='relu',kernel_initializer='random_normal'),
        keras.layers.Dense(units = 1,activation='sigmoid'),
        
])
tf.keras.utils.plot_model(model, show_shapes=True)
optimizer = keras.optimizers.Nadam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_loss')<0.3144):
                self.model.stop_training = True
callback = myCallback()
model.fit(xtrain,ytrain,batch_size=2000,epochs=100,verbose=2,validation_split=0.2,callbacks=[callback])
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
prediction = model.predict_classes(xtest,verbose=1)
prediction
len(prediction), ytest.shape
from sklearn.metrics import accuracy_score,classification_report,balanced_accuracy_score
print(accuracy_score(ytest,prediction))
print(classification_report(ytest,prediction))
print(balanced_accuracy_score(ytest,prediction))
model.save('Transaction_fraud_detector.h5')
