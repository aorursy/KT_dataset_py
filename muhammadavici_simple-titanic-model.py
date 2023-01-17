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
%matplotlib inline
import tensorflow as tf
from tensorflow.keras import models, layers
df_train_raw = pd.read_csv('../input/titanic/train.csv')
df_test_raw = pd.read_csv('../input/titanic/test.csv')
df_sub_raw = pd.read_csv('../input/titanic/gender_submission.csv')
df_train_raw.head()
%config InlineBackend.figure_format = 'png'
ax = df_train_raw['Survived'].value_counts().plot(kind = 'bar',
                                                  figsize = (10, 6), fontsize= 15, rot= 0)
ax.set_ylabel('Counts', fontsize = 15)
ax.set_xlabel('Survived', fontsize = 15)
plt.show()
%config InlineBackend.figure_format = 'png'
ax = df_train_raw['Age'].plot(kind = 'hist',bins = 20, color = 'indigo',
                            figsize = (10, 6), fontsize= 15)
ax.set_ylabel('Frequency', fontsize = 15)
ax.set_xlabel('Age', fontsize = 15)
plt.show()
%config InlineBackend.figure_format = 'png'
ax = df_train_raw.query('Survived == 0')['Age'].plot(kind = 'density',
                                                    figsize = (10, 6), fontsize = 15)
df_train_raw.query('Survived == 1')['Age'].plot(kind = 'density',
                                               figsize = (10, 6), fontsize = 15)
ax.legend(['Survived = 0', 'Survived == 1'], fontsize = 12)
ax.set_ylabel('Density', fontsize = 15)
ax.set_xlabel('Age', fontsize = 15)
plt.show()
def preprocessing(dfdata):
    
    dfresult = pd.DataFrame()
    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult,dfPclass], axis = 1)
    
    #sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)
    
    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')
    
    #SibSP, Parch, Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']
    
    #Cabin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')
    
    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na = True)
    dfEmbarked.columns = ['Embarked' +str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)
    
    return(dfresult)

X_train = preprocessing(df_train_raw)
y_train = df_train_raw['Survived'].values

X_test = preprocessing(df_test_raw)
y_test = df_sub_raw['Survived'].values

print('{} {}'.format(('X_train.shape'),(X_train.shape)))
print('{} {}'.format(('X_test.shape'),(X_test.shape)))
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20, activation = 'relu', input_shape=(15,)))
model.add(layers.Dense(10, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()
model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ["AUC"])

history = model.fit(X_train, y_train,
                   batch_size = 64,
                   epochs = 30,
                   validation_split = 0.2
                   )
#train for 30, 10, 10, 10 epochs
%config InlineBackend.figure_format = 'svg'

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' +metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro--')
    plt.title('Training and validation' +metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(['train_'+metric, 'val_'+metric])
    plt.show()
    

plot_metric(history, 'loss')
plot_metric(history, 'auc')
# model evaluating on test data
model.evaluate(x = X_test, y= y_test)
#predict possibilites
model.predict(X_test[0:10])
#predicting the classes
model.predict_classes(X_test[0:10])
# saving model
model.save('titanic_model.h5')
# you can delete model with "del model" command
del model
#lets load model
model = models.load_model('titanic_model.h5')
model.evaluate(X_test, y_test)
