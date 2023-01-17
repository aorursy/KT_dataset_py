import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
covid = pd.read_csv('../input/symptoms-and-covid-presence/Covid Dataset.csv')

covid.head()
covid.info()
covid.describe()
sns.countplot(covid['COVID-19'])
covid = pd.get_dummies(covid,drop_first=True)
covid.info()
plt.figure(figsize=(12,6))
sns.heatmap(covid.corr(),cmap='coolwarm')
plt.figure(figsize=(20,12))
covid.corrwith(covid['COVID-19_Yes'])
from sklearn.model_selection import train_test_split
X = covid.drop('COVID-19_Yes',axis=1)
y = covid['COVID-19_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
X_train.shape
model = Sequential()
model.add(Dense(18 , activation='relu'))

model.add(Dense(9, activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train,
         y=y_train,
         epochs = 30,
         batch_size=256,
         validation_data=(X_test,y_test)
         )
model_loss = model.history.history
losses = pd.DataFrame(model_loss)
losses[['loss','val_loss']].plot()
from sklearn.metrics import confusion_matrix,classification_report
predict = model.predict_classes(X_test)
print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))
