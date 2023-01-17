import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head()
train.describe()
X= train.iloc[:,:-2]

X.head()
y_train= train.iloc[:,-1]

print(f" Classes: {set(y_train)} ")

y_train.value_counts()
time=np.arange(0,len(X),1)

_, axs = plt.subplots(4, figsize=(10,20))

for idx, ax in enumerate(axs):

    sensor=X.iloc[:,idx]

    ax.plot(time, sensor)
modelo= RandomForestClassifier(n_estimators=100)

modelo_GB = GradientBoostingClassifier() 
modelo.fit(X, y_train)
modelo_GB.fit(X,y_train)

test= pd.read_csv("../input/test.csv")

X_test=test.iloc[:,:-2]

y_test= test.iloc[:,-1]



# prediccion= modelo.predict(X_test)
sensor=np.expand_dims(X_test.iloc[0,:],axis=1)

predecir_sensor= modelo.predict(sensor.T)



pred_gb = modelo_GB.predict(X_test)

print(classification_report(y_test,pred_gb))
