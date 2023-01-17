import pandas as pd
df=pd.read_csv('Admission_Predict_Ver1.1.csv')
df.head()
X=df.drop(['Chance of Admit ','Serial No.'],axis=1)
y=df['Chance of Admit ']

from sklearn.model_selection import train_test_split
X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
model_random=RandomForestRegressor(n_estimators=200)
model_random.fit(X_train,y_train)
prd_random=model_random.predict(X_test)
prd=pd.DataFrame(prd_random)
ytest=pd.DataFrame(y_test)
ytest=ytest.reset_index()
compare=pd.concat([ytest,prd],axis=1)
compare
y_test.shape
prd_random.shape
prd
import seaborn as sns
ytest=ytest.reset_index()
compare=pd.concat([ytest,prd],axis=1)
sns.lineplot(x='Chance of Admit ',y=0,data=compare)
compare.columns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Dense(7,activation='relu'))
Dropout(.2)
model.add(Dense(7,activation='relu'))
Dropout(.2)
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
Dropout(.2)
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam',loss='mse')
early_stop=EarlyStopping(patience=20,verbose=1)
model.fit(X_train,y_train.values,epochs=500,validation_data=(X_test,y_test.values))
ytrain=pd.DataFrame(y_train)

ytrain=ytrain.reset_index()
prd_deep=model.predict(X_test)
prd_deep=pd.DataFrame(prd_deep)
compare=pd.concat([ytest,prd_deep],axis=1)
sns.lineplot(x='Chance of Admit ',y=0,data=compare)
loses=pd.DataFrame(model.history.history)
loses.plot()
prd_deep_train=model.predict(X_train)
compare=pd.concat([ytrain,prd_deep_train],axis=1)
