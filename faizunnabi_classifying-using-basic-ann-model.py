import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/data.csv')
df.head()
df.info()
df.describe()
df_pre=df.drop(['id','diagnosis'],axis=1)
plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
sns.countplot(x='diagnosis',data=df)
df_pre.info()
X=df_pre.iloc[:,0:30].values
y=df.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder
lbe=LabelEncoder()
y=lbe.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=15,kernel_initializer='uniform',activation='relu',input_dim=30))
model.add(Dense(units=15,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train,y_train,batch_size=10,epochs=100)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
all_preds = model.predict(X)
all_preds = (all_preds > 0.5)
df['prediction']=all_preds
df.head()
df['prediction'] = df['prediction'].apply(lambda x:'M' if x==True else 'B' )
df.head()