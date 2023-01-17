# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.utils import np_utils



### modelo 2

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sb



from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dftw= pd.read_csv("../input/TotalWine.csv", delimiter=';')
print(dftw.head(3))

print(dftw.info())

print(dftw.columns)

print(dftw.dtypes)

print(dftw.tail(3))

type(dftw)
pd.options.display.float_format = '{:.2f}'.format

dftw.describe()
Typecount=dftw.groupby ('type')['type'].count()

Typecount
dfrw=dftw[dftw['type'] == 'R']

dfww=dftw[dftw['type'] == 'W']

dfrw.describe()
dfww.describe()
dfrw.drop(['type'],1).hist()

plt.show()
dfww.drop(['type'],1).hist()

plt.show()
#import seaborn as sns

corr = dftw.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
dftw2=dftw 

dftw2['typew']=np.where(dftw2['type']=='R',1,0)

del dftw2['type']

print(dftw2.head())

print(dftw2.tail())
print(dftw2.columns)
#Scaling the continuos variables

df_scale = dftw2.copy()

scaler = preprocessing.StandardScaler()

columns =dftw.columns[0:12]

df_scale[columns] = scaler.fit_transform(df_scale[columns])

df_scale.head()

df_scale = df_scale.iloc[:,0:13]



print(df_scale.head())

print(df_scale.tail())

sample = np.random.choice(df_scale.index, size=int(len(df_scale)*0.8), replace=False)

train_data, test_data = df_scale.iloc[sample], df_scale.drop(sample)



print("Number of training samples is", len(train_data))

print("Number of testing samples is", len(test_data))

print(train_data[:10])

print(test_data[:10])
#train_data

print(train_data.info())

print(test_data.info())
features = train_data.drop('typew', axis=1)

targets = train_data['typew']

features_test = test_data.drop('typew', axis=1)

targets_test = test_data['typew']
# Construccion de modelo

model = Sequential()

model.add(Dense(10, activation='softmax', init='uniform', input_shape=(12,)))

model.add(Dense(8, activation='relu', init='uniform'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(features, targets, epochs=50, batch_size=10, verbose=0)
# evaluamos el modelo

scores = model.evaluate(features, targets)

 

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print (model.predict(features).round())
# evaluamos el modelo

scores_test = model.evaluate(features_test, targets_test)

 

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print (model.predict(features).round())
df_scale.head()
print(df_scale.groupby('typew').size())
X = np.array(df_scale.drop(['typew'],1))

y = np.array(df_scale['typew'])

X.shape
model = linear_model.LogisticRegression()

model.fit(X,y)
validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
name='Logistic Regression'

kfold = model_selection.KFold(n_splits=10, random_state=seed)

cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

print(msg)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(predictions) 
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
import itertools

import pylab as pl

def plot_confusion_matrix(cm, classes,

                          title='Matriz de confusión',

                          cmap=pl.cm.Blues):

    print(cm) # Confusion matrix



    pl.imshow(cm, interpolation='nearest', cmap=cmap) # Pintamos la matriz como una imagen

    pl.title(title)

    pl.colorbar()

    tick_marks = np.arange(len(classes))

    pl.xticks(tick_marks, classes, rotation=45) # Nombre de las clases en X

    pl.yticks(tick_marks, classes) # Nombre de las clases en Y



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        pl.text(j, i, format(cm[i, j], 'd'),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black") 

        # Anotamos cada sección de la imagen con su valor correspondiente en la matriz



    pl.tight_layout()

    pl.ylabel('Valor de verdad')

    pl.xlabel('Valor predicho')
cnf_matrix_Linear_Digits = confusion_matrix(Y_validation, predictions)

np.set_printoptions(precision=2)

pl.figure()

plot_confusion_matrix(cnf_matrix_Linear_Digits, classes=np.unique(Y_validation),title='Matriz de confusión')

pl.show()