#import necessary packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

print(tf.__version__)

from keras.regularizers import l2

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

#from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the "../input/" directory.


print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file_path ="../input/heart.csv"

data = pd.read_csv(file_path) 

print(data.shape)

d=data.transpose()

print(data.describe())



print(data.head())

print(data.columns)
plotdata=data[['age','chol','thalach','trestbps','target']]



plotdata['target'][plotdata['target']==0]='no heart disease present (0)'

plotdata['target'][plotdata['target']==1]='heart disease present (1)'



sns.pairplot(plotdata, hue='target')
sns.relplot(x='trestbps', y="thalach", size="chol", sizes=(15, 200),data=data)
sns.factorplot('cp','age',data=data,

                   hue='target',

                   size=6,

                   aspect=0.8,

                   palette='magma',

                   join=False,

              )


sns.barplot(x=data.age.value_counts()[:15].index,y=data.age.value_counts()[:15].values)

plt.xlabel('Age')

plt.ylabel('Number of people')

plt.show()
plt.figure(figsize=(9,9))

sns.heatmap(data.corr(),annot=True,fmt='.1f',cmap="YlGnBu",annot_kws = {'size':10})

plt.show()
#separate Data versus target

y = data.target

y = y.astype('float64') 

#We drop the sex as it is a binary categorical feature that interferes with the training of the model

X=data.drop(['target','sex'], axis=1)

# 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

#        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'

#Are there missing values?

print('Is there any missing data: %s' %data.isnull().values.any())





#Separate training data from test data

X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True,

                                                      random_state=0)

X_stats=X_train.describe()

X_stats=X_stats.transpose()







#I would like to start with a simple neural network, so lets renormalize the data we will feed to it. 

def norm(x):

    return ((x-X_stats['mean'])/X_stats['std'])

n_X_train=norm(X_train)

n_X_test=norm(X_test)

#We use l2 regularization to prevent overfitting

def Build_Model():

    model=keras.Sequential([

        layers.Dense(64,activation=tf.nn.relu,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), input_shape=[len(n_X_train.keys())]),

        layers.Dense(64,activation=tf.nn.relu,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),

        layers.Dense(1,activation=tf.nn.sigmoid)

    ])





    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])

    return model

model=Build_Model()

model.summary()
EPOCHS=500

history=model.fit(n_X_train,y_train,epochs=EPOCHS,validation_split=0.2,verbose=0)


def plot_history(history):

    hist=pd.DataFrame(history.history)

    hist['epoch']=history.epoch

    

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('cross_entropy')

    plt.plot(hist['epoch'],hist['loss'],label='Train Error')

    plt.plot(hist['epoch'],hist['val_loss'], label='Val Error')

    plt.legend()

    

    plt.figure()

    

plot_history(history)

    
model=Build_Model()

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)

history=model.fit(n_X_train,y_train,epochs=EPOCHS,validation_split=0.2,verbose=0,callbacks=[early_stop])

plot_history(history)
loss, accu=model.evaluate(n_X_test,y_test,verbose=1)

print(accu)

print('Testing accuracy:%0.2f'%(accu))

test_predictions=model.predict_classes(n_X_test)

test_predictions_probability=model.predict(n_X_test)





print('Confusion matrix:')

print(metrics.confusion_matrix(y_test, test_predictions))

confusion=metrics.confusion_matrix(y_test, test_predictions)

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]



#accuracy

print('Accuracy:')

print((TP + TN) / float(TP + TN + FP + FN))

#print(metrics.accuracy_score(y_test, test_predictions))
fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions_probability)



plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for heart disease')

plt.xlabel('False Positive Rate ')

plt.ylabel('True Positive Rate ')

plt.grid(True)

print(metrics.roc_auc_score(y_test, test_predictions_probability))




def Build_Model_and_ROC(model,name,  X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)

    test_predictions_probability=model.predict_proba(X_test)[:,1]

    

    #Get the confusion matrix of the classifier

    confusion=metrics.confusion_matrix(y_test, y_pred)

    TP = confusion[1, 1]

    TN = confusion[0, 0]

    FP = confusion[0, 1]

    FN = confusion[1, 0]

    print('Confusion Matrix for '+ name +':')

    print(confusion)

   

    #ROC curve

    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions_probability)



    #Plotting ROC curve

  

    plt.plot(fpr, tpr, label=name)

    



    print(name+" Number of mislabeled points out of a total %d points  : %d" %(X_test.shape[0],(y_test != y_pred).sum()))

    print( 'Accuracy of '+ name+ ' : %f'%(metrics.accuracy_score(y_test, y_pred)))

    print('auc score of '+ name+ ' : %f'%( metrics.roc_auc_score(y_test, test_predictions_probability)))

    print('\n')

  

   

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.rcParams['font.size'] = 12

    plt.title('ROC curve for heart disease')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.grid(True)

    plt.legend(loc=4) 



models={'XG Boost': XGBClassifier(), 'Logistic Regression': LogisticRegression(), 'Naive Bayes': GaussianNB(),

        'Random Forest':RandomForestClassifier(max_depth=5), 'Decision Tree Bagging' : BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100 ,

                         bootstrap=True, n_jobs=-1,oob_score=True)} 

plt.figure(figsize=(20,10))

for name, model in models.items():

    Build_Model_and_ROC( model, name,  X_train, X_test, y_train, y_test)    



    