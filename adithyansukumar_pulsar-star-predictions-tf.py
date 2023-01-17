import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')
df.head()
df.describe().transpose()
df.isnull().sum()
df.info()
sns.countplot(df['target_class'])
df['target_class'].value_counts()
df1=df[df['target_class']==0].head(1639)
df2=df[df['target_class']==1].head(1639)
new_df=pd.concat([df1,df2]).sample(frac=1)
new_df
sns.countplot(new_df['target_class'])
sns.scatterplot(x=' Mean of the integrated profile',y=' Standard deviation of the integrated profile',data=new_df)
sns.scatterplot(x='target_class',y=' Standard deviation of the integrated profile',data=new_df)
sns.scatterplot(x=' Excess kurtosis of the integrated profile',y=' Mean of the DM-SNR curve',data=new_df)
sns.scatterplot(x=' Mean of the integrated profile',y=' Mean of the DM-SNR curve',data=new_df)
sns.jointplot(new_df[' Skewness of the DM-SNR curve'],new_df[' Excess kurtosis of the DM-SNR curve'])
sns.pairplot(new_df)
sns.heatmap(new_df.corr(),annot=True)
from sklearn.model_selection import train_test_split
x=new_df.drop('target_class',axis=1).values
y=new_df['target_class'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
model=Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=700,verbose=1,callbacks=[es])
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
predictions=model.predict_classes(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
