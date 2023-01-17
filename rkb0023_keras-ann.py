# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

df.head()
df = df.iloc[:,3:]

df.head()
df.describe()
df.info()
df['Geography'].value_counts()
df['Gender'].value_counts()
df = pd.get_dummies(df, drop_first=True)

df.head()
y = df['Exited']
X = df.drop('Exited', 1)
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
classifier = Sequential()
# adding the input layer and first hidden layer
classifier.add(Dense(units=10, kernel_initializer="he_normal", activation="relu", input_dim=11))
classifier.add(Dropout(0.2))
# adding the second hidden layer
classifier.add(Dense(units=20, kernel_initializer="he_normal", activation="relu"))
classifier.add(Dropout(0.4))
# adding the third hidden layer
classifier.add(Dense(units=15, kernel_initializer="he_normal", activation="relu"))
classifier.add(Dropout(0.3))
# adding the output layer
classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))
# compiling the ANN
classifier.compile(optimizer="Adamax", loss="binary_crossentropy", metrics=['accuracy'])
# fitting the ANN
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)
print(model_history.history.keys())
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

score, acc = classifier.evaluate(X_test, y_test,
                            batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve
y_pred_proba = classifier.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()
#Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(layers, activation, optimizer):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes,input_dim=X.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    
    model.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))
    model.compile(optimizer = optimizer, loss="binary_crossentropy", metrics=['accuracy'])    
    return model
model = KerasClassifier(build_fn=create_model)

parameters = {'layers': [(40,30),(45,30,15)],
              'activation': ['sigmoid', 'relu'],
              'batch_size': [64, 128, 256],
              'epochs': [50, 100],
              'optimizer': ['adam', 'rmsprop']}


grid = GridSearchCV(estimator=model, param_grid=parameters, cv=3, verbose=10)
grid_search = grid.fit(X_train, y_train)
print('Best Parameters after tuning: {}'.format(grid_search.best_params_))
print('Best Accuracy after tuning: {}'.format(grid_search.best_score_))
# Predicting the Test set results
y_pred_grid = grid.predict(X_test)
y_pred_grid = (y_pred_grid > 0.5)

score, acc = classifier.evaluate(X_test, y_test, batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_grid)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print(classification_report(y_test,y_pred_grid))
y_pred_proba
y_pred_proba = grid.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:][:,1])
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()
#Area under ROC curve
roc_auc_score(y_test,y_pred_proba[:][:,1])
