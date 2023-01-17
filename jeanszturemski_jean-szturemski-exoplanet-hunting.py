# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv').fillna(0)
df_train.head()
df_train.shape
df_train.LABEL = df_train.LABEL.map({1:0, 2:1})

df_train.head()
plt.title("Distribution de l'intensité", fontsize=10)
plt.xlabel('Temps')
plt.ylabel('Intensité')

# Récupération d'une mesure indiquant la présence d'une exoplanet
positif = df_train.LABEL==1
transit = df_train[positif].iloc[0,]
# Récupération d'une mesure n'indiquant pas la présence d'une exoplanet
negatif = df_train.LABEL==0
nontransit = df_train[negatif].iloc[0,]
plt.plot(transit)# Transit
plt.plot(df_train.iloc[155,])# Pas de transit
df_train.shape
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot('LABEL', data=df_train)
plt.title('Distribution des cas positifs et négatfs(0=Négatif, 1=Positif)')
df_train[positif].LABEL.count()/df_train.LABEL.count()
df_train[positif].LABEL.count()
# Création des dataframes d'entrainement
x_train = df_train.drop(['LABEL'], axis=1)
y_train = df_train['LABEL']
#Resampling 
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
sm = SMOTE (random_state = 27) 
x_train, y_train = sm.fit_sample (x_train, y_train.ravel ())

# Création des dataframes de test
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
y_train.shape
exoplanet=0
nonexoplanet=0
for i in range(len(y_train)-1):
    if y_train[i]==1:
        exoplanet=exoplanet+1
    else:
        nonexoplanet=nonexoplanet+1
    i=i+1   
    
print("Nombre de mesures indiquant la présence d'une exoplanète: ")
print(exoplanet)
print("Nombre de mesures n'indiquant pa la présence d'une exoplanète: ")
print(nonexoplanet)
plt.hist(nontransit, color="blue", bins=250)

plt.hist(transit, color="blue", bins=250)
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(x_train, y_train)
y_rf = rf.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
rf_score = accuracy_score(y_test, y_rf)
print(rf_score)
cm = confusion_matrix(y_test, y_rf)
print(cm)
print(classification_report(y_test, y_rf))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_lr = lr.predict(x_test)
lr_accuracy = accuracy_score(y_test, y_lr)
print(lr_accuracy)
cm = confusion_matrix(y_test, y_lr)
print(cm)
print(classification_report(y_test, y_lr))
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score


model = Sequential()
model.add(Dense(units = 10, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train = model.fit(x_train , y_train , validation_data=(x_test,y_test), epochs=20, verbose=1)

def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()
plot_scores(train)
# Prediction
y_model = model.predict_classes(x_test)
cm = confusion_matrix(y_model,y_test)
print(cm)
plt.figure(figsize = (12,10))
from keras.layers import Dropout

model = Sequential()
model.add(Dense(units = 200, activation = 'relu', input_dim = x_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units = 200, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train = model.fit(x_train , y_train , validation_data=(x_test,y_test), epochs=100, verbose=1)
plot_scores(train)