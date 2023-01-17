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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV

import time
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
sns.countplot(train['label'])

# target distribution is not skewed
X = train.drop('label', axis=1)

y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values

y_train = y_train.values

X_test = X_test.values

y_test = y_test.values
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_dt)

print(accuracy)
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)

print(accuracy)
lr_model = LogisticRegression(max_iter=100)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_lr)

print(accuracy)
def evaluate_models(models):

    for model in models:

        print("Evaluation for {}".format(type(model).__name__))

        print("----"*20)

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test,y_pred)

        print("\nConfusion Matrix:\n",cm)

        ac = accuracy_score(y_test,y_pred)

        print("\nAccuracy:\n",ac)

        print("\nClassification Report:\n")

        print(classification_report(y_test,y_pred))
models = [lr_model, dt_model, rf_model]
evaluate_models(models)
def cross_validate_models(models, splits):

    kf = KFold(n_splits=splits,shuffle=True)

    for model in models:

        scores = cross_val_score(model,

                                 X_train,

                                 y_train,

                                 cv=kf,

                                 n_jobs=12,

                                 scoring="accuracy")

        print("Cross-Validation for {}:\n".format(type(model).__name__))

        print("Mean score: ", np.mean(scores))

        print("Variance of score: ", np.std(scores)**2)

        fig = plt.figure(figsize = (10,5))

        ax = fig.add_subplot(111)

        ax = sns.distplot(scores)

        ax.set_xlabel("Cross-Validated Accuracy scores")

        ax.set_ylabel("Frequency")

        ax.set_title('Frequency Distribution of Cross-Validated Accuracy scores for {}'.format(type(model).__name__), fontsize = 15)
model_rf = [rf_model]
cross_validate_models(model_rf,10)
rf_params = {'bootstrap': [True, False],

 'max_depth': [10, 15, 20, 25, 30, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [100, 200, 400, 600, 800, 1000]}
params = [rf_params]

# params_rf = [rf_params]



tuned_models = []
def hyper_param_tuning(models,params,splits,scorer):

    for i in range(len(models)):

        gsearch = RandomizedSearchCV(estimator=models[i],

                               param_distributions=params[i],

                               scoring=scorer,

                               verbose=2,

                               n_jobs=-1,

                               cv=5)

        start = time.time()

        gsearch.fit(X_train,y_train)

        end = time.time()

        

        print("Grid Search Results for {}:\n".format(type(models[i]).__name__))

        print("Time taken for tuning (in secs): \n", end-start)

        print("Best parameters: \n",gsearch.best_params_)

        print("Best score: \n",gsearch.best_score_)

        tuned_models.append(gsearch.best_estimator_)

        print("\n\n")
# hyper_param_tuning(model_rf,params,10,"accuracy")
params_tuned = {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 30, 'bootstrap': False}

rf_tuned_model = RandomForestClassifier(**params_tuned)

rf_tuned_model.fit(X_train, y_train)

y_pred_rf_tuned = rf_tuned_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf_tuned)

print(accuracy)
import itertools

import tensorflow as tf

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
X = X.values.reshape(X.shape[0], 28, 28,1)

X.shape
from keras.utils.np_utils import to_categorical



y = to_categorical(y, num_classes = 10)



from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=101)
mean_px = Xtrain.mean().astype(np.float32)

std_px = Xtrain.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
optimizer = RMSprop(lr=0.001)

model = Sequential([

    Lambda(standardize, input_shape=(28,28,1)),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(10, activation='softmax')

    ])

model.compile(optimizer=optimizer, loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
batch_size = 32

epochs = 25



model.fit(Xtrain, ytrain, batch_size = batch_size, epochs = epochs, 

         validation_data = (Xtest, ytest), verbose = 2, callbacks=[learning_rate_reduction, es])
model.evaluate(Xtest, ytest)
model1 = Sequential()



model1.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape=(28,28,1)))

model1.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model1.add(MaxPool2D(pool_size=(2,2)))

model1.add(Dropout(0.25))





model1.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model1.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model1.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model1.add(Dropout(0.25))





model1.add(Flatten())

model1.add(Dense(256, activation = "relu"))

model1.add(Dropout(0.5))

model1.add(Dense(10, activation = "softmax"))
model1.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
Xtrain = Xtrain/255

Xtest = Xtest/255

epochs1 = 15

batch_size1 = 32

model1.fit(Xtrain, ytrain, batch_size = batch_size1, epochs = epochs1, 

         validation_data = (Xtest, ytest), verbose = 2, callbacks=[learning_rate_reduction, es])
test = pd.read_csv('../input/digit-recognizer/test.csv')
test.shape
test_vals = test
image_id = test.index.values
image_id = image_id + 1
test_vals = test_vals.values.reshape(test_vals.shape[0], 28, 28,1)

test_vals.shape
test_vals = test_vals/255
predictions = model1.predict(test_vals)
predictions = np.argmax(predictions,axis = 1)
predictions
submission1 = pd.DataFrame({

        "ImageId": image_id,

        "Label": predictions

    })
submission1
submission1.to_csv('mysubmission1.csv', index=False)