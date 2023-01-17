import pandas as pd #Manipulating data

import numpy as np #For Math operations 

import seaborn as sns #For visualization purpose

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split # For Spliting data to training and testing data

from sklearn.preprocessing import MinMaxScaler # MinMaxScaler For fitting the data to the model, it optimize model

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dropout

from sklearn.metrics import classification_report,confusion_matrix #For model evaluation metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



#Importing the data

cancer_data=pd.read_csv('/kaggle/input/breast-cancer-dataset-uci-ml/cancer_classification.csv')
cancer_data.head(5)
cancer_data.info()
cancer_data.describe().transpose()
sns.countplot(x='benign_0__mal_1',data=cancer_data)
cancer_data['benign_0__mal_1'].value_counts()
#Let's see how features correlates to another

sns.heatmap(cancer_data.corr())
cancer_data.corr()['benign_0__mal_1'].sort_values()
cancer_data.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
cancer_data.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
X = cancer_data.drop('benign_0__mal_1',axis=1).values

y = cancer_data['benign_0__mal_1'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)
model=Sequential()



model.add(Dense(units=30, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(units=15, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')



early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(X_train,

          y=y_train,

          epochs=600,

          validation_data=(X_test,y_test),

          verbose=1,

          callbacks=[early_stop]

         

         )
model.summary()
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
predictions = model.predict_classes(X_test)
print('Model Classification Report')

print(classification_report(y_test,predictions))



print('*'*57)

print('Confusion Matrix')

print(confusion_matrix(y_test,predictions))