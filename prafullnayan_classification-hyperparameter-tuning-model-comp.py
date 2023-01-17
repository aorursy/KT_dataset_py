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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
data
data.info()
data.describe().transpose()
data.isnull().sum()

# no null values
columns=data.columns

columns
data['class'].value_counts()
(data['class'].value_counts())/len(data['class'])*100
plt.figure(figsize=(12,8))

sns.countplot(data['class'])
data['class']=data['class'].map({'p':0,'e':1})
X=data.drop('class',axis=1)

y=data['class']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

print('size of train data',X_train.shape,y_train.shape)

print('size of test data',X_test.shape,y_test.shape)
columns=[col for col in X_train.columns if X_train[col].dtypes =='object']

columns
category_train=[col for col in X_train.columns if X_train[col].dtypes =='object']

category_test=[col for col in X_test.columns if X_test[col].dtypes =='object']
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

for col in columns:

    X_train[col]=label_encoder.fit_transform(X_train[col])

    X_test[col]=label_encoder.transform(X_test[col])
X_test
data
X_train
X_test
y_train
y_test
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping
X_train.shape
model=Sequential()



model.add(Dense(22,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(16,activation='relu'))

model.add(Dropout(0.2))

 

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=12)
model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[early_stop],epochs=15,batch_size=256)
losses_df=pd.DataFrame(model.history.history)
losses_df.head()
losses_df[['loss','val_loss']].plot()

losses_df[['accuracy','val_accuracy']].plot()
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint
model=RandomForestClassifier(n_jobs=-1)
parameters={'max_depth':[3,5,7,10,None],

           'n_estimators':[100,200,300,400,500],

           'max_features':randint(1,13),

           'criterion':['gini','entropy'],

           'bootstrap':[True,False],

           'min_samples_leaf':randint(1,5)}
def hyperparameter_tuning(model,parameters,n_of_itern,X_train,y_train):

    random_search=RandomizedSearchCV(estimator=model,

                                    param_distributions=parameters,

                                    n_jobs=-1,

                                     n_iter=n_of_itern,

                                     cv=9)

    random_search.fit(X_train,y_train)

    params=random_search.best_params_

    score=random_search.best_score_

    return params,score
final_params,final_score=hyperparameter_tuning(model,parameters,40,X_train,y_train)
#this is our final best parameters for random forest classifier

final_params
# final accuracy with tuned parameters

final_score
# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier



# instantiate the classifier 

rfc = RandomForestClassifier(bootstrap=True,

                             criterion='entropy',

                            max_depth=None,

                            max_features=8,

                            min_samples_leaf=2,

                            n_estimators=200)

                             

                            



# fit the model

rfc.fit(X_train, y_train)



# Predict the Test set results

y_pred = rfc.predict(X_test)



print('Model accuracy is : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
from sklearn.neighbors import KNeighborsClassifier
k=4

model=KNeighborsClassifier(n_neighbors=k)
model.fit(X_train,y_train)

pred=model.predict(X_test)
print(classification_report(pred,y_test))
from sklearn import svm
clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train)
yhat = clf.predict(X_test)

# SVM has perform better than KNN

print(classification_report(yhat,y_test))
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)

log_predict=log_reg.predict(X_test)
print(classification_report(log_predict,y_test))