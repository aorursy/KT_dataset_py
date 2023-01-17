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
# Download train Dataset

train=pd.read_csv('../input/titanic/train.csv')

train.head()
train.columns
# traing Data information

train.info()
# Merge test dataset and gender_submission for labeling Survived column

test=pd.read_csv('../input/titanic/test.csv')

gender_submission=pd.read_csv('../input/titanic/gender_submission.csv')

test1=pd.merge(test,  

                     gender_submission,  

                     on ='PassengerId',  

                     how ='left') 

test1.head()
# Create Age_group column for grouping age 

def age_group(x):

  if x<=10:

    return '<=10'

  elif x>10 and x<=20:

    return '11-20'

  elif x>20 and x<=30:

    return '21-30'

  elif x>30 and x<=40:

    return '31-40'

  elif x>40 and x<=50:

    return '41-50'

  elif x>50:

    return '>=50'

  else:

    return 'No data'



round2 = lambda x: age_group(x)

train['Age_group'] = pd.DataFrame(train['Age'].apply(round2))

train.head()
# Also grouping age for Age_group column

round2 = lambda x: age_group(x)

test1['Age_group'] = pd.DataFrame(test1['Age'].apply(round2))

test1.head()
# function for grouping parameter by Survived

def param_graph(param):    

    train[param].unique()

    dfg=pd.DataFrame(list(train[param].unique()),columns=[param])

    dfg=dfg.dropna()

    

# Counting of parameter group by Survived

    acc2 = train.groupby([param,'Survived'])['Survived'].count()

# Grouping by Survived=0(dead)

    ser_1=acc2[:,0].reset_index()

    pg1=pd.merge(dfg,  

                     ser_1,  

                     on =param,  

                     how ='left') 

    pg1['Survived_0']=pg1['Survived']

    pg1.drop(['Survived'], axis=1, inplace=True)

    

# Grouping by Survived=1(survived)

    ser_2=acc2[:,1].reset_index()

    ser_2['Survived_1']=ser_2['Survived']

    ser_2.drop(['Survived'], axis=1, inplace=True)

    pg2=pd.merge(pg1,  

                     ser_2,  

                     on =param,  

                     how ='left') 

    dfg1=list(pg2[param])

    ser1=list(pg2['Survived_0'])

    ser2=list(pg2['Survived_1'])

# return list of unique value in parameter,

# grouping value's list of Survived=0(dead),

# grouping value's list of Survived=1(survived

    return dfg1,ser1,ser2
# function grouping by Sex

param_graph('Sex')
import matplotlib.pyplot as plt

%matplotlib inline

# function plot 100% stacked bar graph

def stacked_bar_graph(param,width):

  totals = [i+j for i,j in zip(param_graph(param)[1], param_graph(param)[2])]

  S0 = [i / j * 100 for  i,j in zip(param_graph(param)[1], totals)]

  S1 = [i / j * 100 for  i,j in zip(param_graph(param)[2], totals)]

  r = range(len(param_graph(param)[0]))

  barWidth = 1

  plt.figure(figsize=(width,4))

  ax1 = plt.bar(r, S1, bottom=S0, color='#C1FFC1', edgecolor='white', width=barWidth, label='1: Servived')

  ax2 = plt.bar(r, S0, color='#FF9999', edgecolor='white', width=barWidth, label='0: Dead')

  plt.legend()

  plt.xticks(r, param_graph(param)[0], fontweight='bold')

  plt.ylabel("sales")

  def autolabel(rects):

          """Attach a text label above each bar in *rects*, displaying its height."""

          for rect in rects:

              height = round(rect.get_height(),1)

              plt.annotate('{}'.format(height),

                          xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2),

                          xytext=(0, 3),  # 3 points vertical offset

                          textcoords="offset points",

                          ha='center', va='bottom')



  autolabel(ax1)

  autolabel(ax2)

  plt.show()
stacked_bar_graph('Age_group',8)
stacked_bar_graph('Embarked',6)
stacked_bar_graph('Sex',6)
stacked_bar_graph('Pclass',6)
import seaborn as sns

sns.boxplot(x="Pclass", y="Fare", data=train)
sns.boxplot(x="Age_group", y="Fare", data=train)
test.info()
sns.boxplot(x="Embarked", y="Fare", data=train)
train.groupby(['Pclass'])['Fare'].agg(pd.Series.mode)
# Replacing mode value in missing value

values = {'Fare': 8.05}

test1=test1.fillna(value=values)



# Specity train_x by droping columns 

train_x=train.drop(columns=['PassengerId','Survived','Name','Age','Ticket','Cabin'])

# Specify train_y of Survived columns

train_y=train.loc[:,'Survived']

# Specify test_y of Survived columns

test_y=test1.loc[:,'Survived']

test_x=test1.drop(columns=['PassengerId','Survived','Name','Age','Ticket','Cabin'])

print(train_x.shape,train_y.shape)

print(test_x.shape,test_y.shape)
train_x.head(3)
train_x=pd.get_dummies(train_x,columns=['Sex','Embarked','Age_group'])

test_x=pd.get_dummies(test_x,columns=['Sex','Embarked','Age_group'])
train_x.columns
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV



# Setup the parameters of Decision Tree Classification

params = {"max_depth": randint(1, 9),

              "max_features": randint(1, train_x.shape[1]),

#               "max_features": ["auto", "sqrt", "log2","None"],

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"],

              "splitter":["best", "random"]}



# Decision Tree Classification

Dt = DecisionTreeClassifier()



# RandomizedSearchCV of Decision Tree Classification

Dtree_rd = RandomizedSearchCV(Dt, params,cv=None)



# Fit the data

Dtree_rd.fit(train_x,train_y)



# Print the tuning parameters and score

print("Tuned Decision Tree Parameters: {}".format(Dtree_rd.best_params_))

print("Best score is {}".format(Dtree_rd.best_score_))
Dt_best= Dtree_rd.best_estimator_

Dt_best.fit(train_x,train_y)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,plot_confusion_matrix

from sklearn.metrics import precision_recall_fscore_support



# function for evaluating Classification model

def model_evaluation(name,model, Xtest, ytest):

    

# Predict Xtest

    ypred = model.predict(Xtest)

    print('Accuracy of test set is :', accuracy_score(ytest, ypred))

    print("\n")

    

    

# Confusion matrix value

    tn, fp, fn, tp = confusion_matrix(ytest, ypred).ravel()

    print('True Negative = {}\tFalse Positive = {}\nFalse Negative = {}\tTrue Positive={}'

          .format(tn,fp,fn,tp))

# Confusion matrix Table

    class_names=ytest.unique()

    disp = plot_confusion_matrix(model, Xtest,ytest ,

                                display_labels=class_names,

                                cmap=plt.cm.RdPu ,values_format='d')

    disp.ax_.set_title(name)

    

    print("\n")

    print(classification_report(ytest, ypred))

    

model_evaluation('Decision Tree',Dt_best, test_x,test_y)
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import BernoulliNB



NB = BernoulliNB()



parameters = {"alpha":  np.array( [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10,100,1000,10000] )}

NB_GS = GridSearchCV(estimator=NB, param_grid=parameters)

NB_GS.fit(train_x,train_y)

print("\n The best score is: ",NB_GS.best_score_)

print("\n The best estimator is:\n",NB_GS.best_estimator_)

NB_best= NB_GS.best_estimator_

NB_best.fit(train_x,train_y)

model_evaluation('Bernoulli Naïve Bayes',NB_best, test_x,test_y)
import tensorflow as tf

from tensorflow import keras

from keras import models

from keras import layers

from keras import optimizers

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras import regularizers



# Split data for traing dataset and validate dataset

partial_x_train,x_val,  partial_y_train, y_val = train_test_split(train_x,train_y, test_size=0.4)

scaler = StandardScaler()

partial_x_train = scaler.fit_transform(partial_x_train)

x_val = scaler.transform(x_val)

test_x = scaler.transform( test_x)
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8],momentums=0.0,decays=0.0,ndrop=0,lamda1=0,lamda2=0,lamda3=0,lamda4=0):

    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=partial_x_train.shape[1:]))

    for layer in range(n_hidden):

        model.add(keras.layers.Dense(n_neurons, activation="relu", 

                        kernel_regularizer=regularizers.l1_l2(lamda1, l2=lamda4),

                        bias_regularizer=regularizers.l1(lamda2),

                        activity_regularizer=regularizers.l1(lamda3)

                        ))

        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(ndrop))    

    model.add(keras.layers.Dense(2, activation='softmax'))

    optimizer = keras.optimizers.SGD(lr=learning_rate,momentum=momentums,decay=decays)

    model.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=["accuracy"])

    return model
keras_cla = keras.wrappers.scikit_learn.KerasClassifier(build_model)
# !pip install scikit-learn==0.21.2

# import numpy as np

# from scipy.stats import reciprocal

# from sklearn.model_selection import RandomizedSearchCV



# param_distribs = {

#     "n_hidden": (0, 1, 2, 3),

#     "n_neurons": np.arange(1, 50),

#     "learning_rate": (3e-7,3e-6,3e-5),

#     "ndrop" :np.arange(0.0, 0.6,0.1),

#     "lamda1":(1e-6,1e-2,1e-3,1e-4,1e-5),

#     "lamda2":(1e-6,1e-2,1e-3,1e-4,1e-5),

#     "lamda3":(1e-6,1e-2,1e-3,1e-4,1e-5),

#     "lamda4":(1e-6,1e-2,1e-3,1e-4,1e-5),

#     "momentums": np.arange(0.9, 0.99,0.01),

#     "decays": (0.001,0.01,0.1,1,10)

# }



# checkpoint_cb = keras.callbacks.ModelCheckpoint("super-ai2.h5", save_best_only=True)

# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)

# rnd_search_cv = RandomizedSearchCV(keras_cla, param_distribs, n_iter=10, cv=5, verbose=2)

# rnd_search_cv.fit(partial_x_train,partial_y_train, epochs=20,

#                   validation_data=(x_val, y_val),

#                   callbacks=[keras.callbacks.EarlyStopping(patience=10)] )
# load h5 file

from keras.models import load_model

model = load_model('../input/titanic-model-1/titanic.h5')
model.summary()
model.evaluate(test_x,test_y)