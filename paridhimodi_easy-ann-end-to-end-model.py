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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df.head()
df['Geography'] = pd.Categorical(df.Geography).codes
df['Gender'] = pd.Categorical(df.Gender).codes
df.info()
x= df.iloc[:, 3:13]
y = df.iloc[:, 13]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 3) 
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten  
from keras.layers  import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
from keras.activations import relu, sigmoid
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units=6, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dense(units=1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_history = classifier.fit(xtrain, ytrain, validation_split= 0.33, batch_size = 10, epochs = 100 )
print(model_history.history.keys())
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test' ], loc = 'upper left')
plt.show()
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test' ], loc = 'upper left')
plt.show()
ypred = classifier.predict(xtest)
ypred = (ypred > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest, ypred)
score = accuracy_score(ypred, ytest)
score
