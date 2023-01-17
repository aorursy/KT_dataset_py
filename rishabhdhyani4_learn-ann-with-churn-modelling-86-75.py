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
# Plotting Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
%matplotlib inline

# Metrics for Classification technique

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Scaler

from sklearn.preprocessing import RobustScaler, StandardScaler

# Cross Validation

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split

# Keras

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LeakyReLU, PReLU, ELU
from keras.activations import relu, sigmoid
from keras.wrappers.scikit_learn import KerasClassifier

# others

!pip install -U dtale
# Importing Data

df = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
data = df.copy()
df.head(6) # Mention no of rows to be displayed from the top in the argument
df.info()
df.describe().transpose()
import dtale
import dtale.app as dtale_app

dtale_app.USE_NGROK = True
d = dtale.show(df)
d.main_url()
# Correlation between the features.

plt.figure(figsize=(20,12))
sns.set_context('notebook',font_scale = 1.3)
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linewidth = 4,linecolor='black')
plt.tight_layout()
X = df.iloc[:, 3:13]
y = df.iloc[:, 13]
#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)
## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

# Adding the Dropout layer
classifier.add(Dropout(0.3))
# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'he_uniform',activation='relu'))

# Adding the Dropout layer
classifier.add(Dropout(0.2))
# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# summary of model

classifier.summary()
# Fitting the ANN to the Training set

classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))
# Function

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)
layers = [(20,),(10,),(30,),(15,10),(30,15),(30,20),(25,10), (40, 20), (45, 30, 15),(50,20,10)]
activations = ['sigmoid', 'relu']
params = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[100])

grid = RandomizedSearchCV(estimator=model, param_distributions =params,cv=5,n_iter=20,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))