# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

np.random.seed(2)

plt.style.use('fivethirtyeight')
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
df.isnull().sum()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
print(df['Class'].unique())
from sklearn.preprocessing import StandardScaler 

df['normalizedAmount']=StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df=df.drop(['Amount'],axis=1)
df.head()
df=df.drop(['Time'],axis=1)

df.head()
X=df.iloc[:,df.columns!='Class']

y=df.iloc[:,df.columns=='Class']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
X_train.shape
X_test.shape
X_train=np.array(X_train)

X_test=np.array(X_test)

y_train=np.array(y_train)

y_test=np.array(y_test)
from keras.models import Sequential

from keras.layers import Dense 

from keras.layers import Dropout
model=Sequential([

    

    Dense(units=16,input_dim=29,activation='relu'),

    Dense(units=24,activation='relu'),

    Dropout(0.5),

    Dense(units=20,activation='relu'),

    Dense(units=24,activation='relu'),

    Dense(units=21,activation='relu'),

    Dense(1,activation='sigmoid')

      ])
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=15,epochs=5)
score=model.evaluate(X_test,y_test)
print(score)
import matplotlib.pyplot as plt

from sklearn import svm,datasets

from sklearn.metrics import confusion_matrix

import itertools   



def plot_confusion_matrix(cm,classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

        print('Normalized confusion matrix')

    else:

        print('Confusion matrix,without normalization')

    print(cm)

    

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation=45)

    plt.yticks(tick_marks,classes)

    fmt='.2f' if normalize else 'd'

    thresh=cm.max()/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment='center',

                color='white' if cm[i,j]> thresh else 'black')

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()                  

    
y_pred=model.predict(X_test)

y_test=pd.DataFrame(y_test)
cnf_matrix=confusion_matrix(y_test,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])

plt.show()
y_pred=model.predict(X)

y_expected=pd.DataFrame(y)

cnf_matrix=confusion_matrix(y_expected,y_pred.round())

plot_confusion_matrix(cnf_matrix,classes=[0,1])
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train.ravel())
y_pred=random_forest.predict(X_test)
random_forest.score(X_test,y_test)
import matplotlib.pyplot as plt

from sklearn import svm,datasets

from sklearn.metrics import confusion_matrix

import itertools   



def plot_confusion_matrix(cm,classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

        print('Normalized confusion matrix')

    else:

        print('Confusion matrix,without normalization')

    print(cm)

    

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation=45)

    plt.yticks(tick_marks,classes)

    fmt='.2f' if normalize else 'd'

    thresh=cm.max()/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment='center',

                color='white' if cm[i,j]> thresh else 'black')

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()         
cnf_matrix=confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=random_forest.predict(X)
cnf_matrix=confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
from sklearn.tree import DecisionTreeClassifier

decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,y_train.ravel())
decision_tree.score(X_test,y_test)
import matplotlib.pyplot as plt

from sklearn import svm,datasets

from sklearn.metrics import confusion_matrix

import itertools   



def plot_confusion_matrix(cm,classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    if normalize:

        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

        print('Normalized confusion matrix')

    else:

        print('Confusion matrix,without normalization')

    print(cm)

    

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation=45)

    plt.yticks(tick_marks,classes)

    fmt='.2f' if normalize else 'd'

    thresh=cm.max()/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment='center',

                color='white' if cm[i,j]> thresh else 'black')

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()                

    
cnf_matrix=confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cnf_matrix,classes=[0,1])
plt.show()
y_pred=decision_tree.predict(X)
cnf_matrix=confusion_matrix(y,y_pred.round())
plot_confusion_matrix(cnf_matrix,classes=[0,1])
fraud_indices=np.array(df[df.Class==1].index)

number_records_fraud=len(fraud_indices)

print(number_records_fraud)
normal_indices=df[df.Class==0].index
random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)

random_normal_indices=np.array(random_normal_indices)

print(len(random_normal_indices))
under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])

print(len(under_sample_indices))