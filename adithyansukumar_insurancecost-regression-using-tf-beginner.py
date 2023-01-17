import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
sns.countplot(df['sex'])
sns.countplot(df['smoker'],hue='sex',data=df)
sns.distplot(df['age'],bins=50)
sns.scatterplot(df['age'],df['bmi'])
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True)
df['sex']=pd.get_dummies(df['sex'],drop_first=False)
df['smoker']=pd.get_dummies(df['smoker'],drop_first=False)
df.drop('region',axis=1,inplace=True)
x=df.drop('charges',axis=1).values
y=df['charges'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_train.shape
x_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=400)
losses=pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions=model.predict(x_test)
mean_absolute_error(y_test,predictions)
np.sqrt(mean_squared_error(y_test,predictions))
explained_variance_score(y_test,predictions)
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')
