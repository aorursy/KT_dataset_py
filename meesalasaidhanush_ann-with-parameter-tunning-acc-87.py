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
df=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isnull().any()
df.describe()
df.info()
cor=df.corr()
cor
import seaborn as sns
sns.heatmap(cor)
t=abs(cor['DEATH_EVENT'])
colls=t[t>0.1]
colls.index
#features=['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
x=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
y=df.DEATH_EVENT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,test_size=0.2)
x_train.shape
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.transform(x_test)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#initializing the model
model=Sequential()

#adding first layer(input layer)
model.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=12))
#adding second layer

model.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
#adding third layer

model.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
#adding fourth layer(output layer)

model.add(Dense(units=1,kernel_initializer='he_uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=30,epochs=300)
pred=model.predict(x_test)
pred=(pred>0.5)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
from keras.wrappers.scikit_learn import KerasClassifier
def create_my_model(batchsize,epochs):
    mymodel=Sequential()
    mymodel.add(Dense(6,input_dim=12,activation='relu'))
    mymodel.add(Dense(6,input_dim=12,activation='relu'))
    mymodel.add(Dense(1,activation='sigmoid'))
    mymodel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return mymodel
model1=KerasClassifier(build_fn=create_my_model)
batchsize=[10,20,30,40,60,80,100]
epochs=[500,1000,1500]
parameter_grid=dict(batch_size=batchsize,epochs=epochs)
from sklearn.model_selection import RandomizedSearchCV
mygrid=RandomizedSearchCV(model,parameter_grid,n_jobs=-1,cv=3)

