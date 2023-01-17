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
wine = pd.read_csv('../input/wine-pca/Wine.csv')

wine
wine.info()
wine.describe()
wine.columns
X = wine.drop(columns=['Customer_Segment'])

Y = wine['Customer_Segment']
X.shape[1]
import matplotlib.pyplot as plt



plt.boxplot(X[0:14])

plt.show()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_scale = scaler.fit_transform(X)
import matplotlib.pyplot as plt



plt.boxplot(X_scale[0:14])

plt.show()
from sklearn.model_selection import train_test_split



x_train_all, x_test, y_train_all, y_test = train_test_split(X_scale,Y,test_size = 0.2)

x_train, x_val, y_train, y_val = train_test_split(x_train_all,y_train_all,test_size = 0.2)
x_test
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# Make Instance

lr = LogisticRegression()



# Grid Search for find best parameter of the model

grid_values = {'C':[0.001,0.01,0.1,1,10,100,1000], 'max_iter':[500,1000,5000],

               'random_state':[0,1,100], 'penalty':['l1','l2','elasticnet'], 'solver':['liblinear','saga','lbfgs']}

gscv = GridSearchCV(lr, param_grid = grid_values, return_train_score=True)

gscv.fit(x_train, y_train)
gscv.best_params_, gscv.best_index_, gscv.best_score_, gscv.best_estimator_
lr_real = LogisticRegression(C=0.1, max_iter=500, penalty='l1',random_state=0, solver='liblinear')

lr_real.fit(x_train, y_train)



pred_lr = lr_real.predict(x_test)



from sklearn.metrics import classification_report

print(classification_report(y_test, pred_lr))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred_lr))
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier



clflog = LogisticRegression(random_state=1)

clfdt = DecisionTreeClassifier(random_state=1)

clfgn = GaussianNB()

eclf_h = VotingClassifier(estimators=[('lr',clflog),('dt',clfdt),('gnb',clfgn)], voting='hard')

eclf_s = VotingClassifier(estimators=[('lr',clflog),('dt',clfdt),('gnb',clfgn)], voting='soft')



models = [clflog, clfdt, clfgn, eclf_h, eclf_s]
c_params = [0.1,5.0,7.0,10.0,15.0,20.0,100.0]

params = {

    'lr__solver': ['liblinear'], 'lr__penalty':['l2'], 'lr__C':c_params,

    'dt__criterion':['gini','entropy'],

    'dt__max_depth':[10,8,7,6,5,4,3,2],

    'dt__min_samples_leaf':[1,2,3,4,5,6,7,8,9]

}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator = eclf_s, param_grid = params, cv=5)

grid = grid.fit(x_train,y_train)

grid.best_params_
from sklearn.metrics import classification_report

eclf_s.fit(x_train, y_train)

pred_vt = eclf_s.predict(x_test)

print(classification_report(y_test, pred_vt))



from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred_vt))
Y_dum = pd.get_dummies(Y)



y_per_train_A, y_per_test = train_test_split(Y_dum, test_size = 0.2)

y_per_train, y_per_val = train_test_split(y_per_train_A, test_size = 0.2)
y_per_train
import tensorflow as tf

from tensorflow.keras import layers



metrics_nm = ['accuracy','categorical_accuracy',]



regular = 0.00001  # regularization amount



model = tf.keras.Sequential()



model.add(layers.Input(shape=x_train.shape[1]))

model.add(layers.Dense(12, activation='relu',

         kernel_regularizer = tf.keras.regularizers.l2(regular),  # Dense Regularization

         activity_regularizer = tf.keras.regularizers.l2(regular)))  # Dense Regularization

model.add(layers.Dense(8, activation='relu'))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(3, activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=metrics_nm)



model.fit(x_train, y_per_train, epochs=100, validation_data=(x_val,y_per_val), batch_size = 16)
hist = model.fit(x_train, y_per_train, epochs=100, validation_data=(x_val,y_per_val), batch_size=16)



hist.history.keys()
import matplotlib.pyplot as plt



weights, biases = model.layers[1].get_weights()

print(weights.shape, biases.shape)



plt.subplot(212)

plt.plot(weights,'x')

plt.plot(biases, 'o')

plt.title('L2 - 0.1')



plt.subplot(221)

plt.plot(hist.history['accuracy'],'^--',label='accuracy')

plt.plot(hist.history['val_accuracy'],'^--', label='v_accuracy')

plt.legend()

plt.title('L2 - 0.1')



plt.subplot(222)

plt.plot(hist.history['loss'],'x--',label='loss')

plt.plot(hist.history['val_loss'],'x--', label='val_loss')

plt.legend()

plt.title('L2 - 0.1')



plt.show()
# x_test = standardScaler.transform(x_test)
model.predict(x_test)
pred_pc = model.predict(x_test)

np.argmax(pred_pc,axis=1)
pred_pc = model.predict(x_test)



test_class = np.argmax(y_per_test.values, axis=1)

pred_class = np.argmax(pred_pc, axis=1)



from sklearn.metrics import classification_report

print(classification_report(test_class, pred_class))
from sklearn.metrics import confusion_matrix



confusion_matrix(test_class, pred_class)