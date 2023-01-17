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
df = pd.read_csv('../input/hr-data/HR_comma_sep.csv')

df.head()
df.info()
df.describe()
df.left.value_counts()/len(df)
df['average_montly_hours'].plot(kind = 'hist');
df['average_montly_hours_100'] = df['average_montly_hours']/100.0
df['average_montly_hours_100'].plot(kind = 'hist');
df['time_spend_company'].plot(kind = 'hist');
df_dummies = pd.get_dummies(df[['sales','salary']])
df_dummies.head()
df.columns
X = pd.concat([df[['satisfaction_level', 'last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident', 'left',

       'promotion_last_5years','average_montly_hours_100']],df_dummies],axis=1).values

y = df['left'].values
X.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from keras.models import Sequential 

from keras.layers import Dense

from keras.optimizers import Adam,SGD
model = Sequential()

model.add(Dense(1,input_dim=22,activation='sigmoid'))

model.compile(Adam(lr=0.5),'binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,epochs =10)
y_test_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_test_pred)
def pretty_confusion_matrix(y_true,y_pred,labels =["False","True"]):

    cm = confusion_matrix(y_true,y_pred)

    pred_labels = ['Predicted '+ l for l in labels]

    df = pd.DataFrame(cm,index = labels,columns = pred_labels)

    return df
pretty_confusion_matrix(y_test,y_test_pred,['Stay','Leave'])
from sklearn.metrics import precision_score,recall_score,f1_score
print('Precision:\t{:0.3f}'.format(precision_score(y_test,y_test_pred)))

print('Recall:   \t{:0.3f}'.format(recall_score(y_test,y_test_pred)))

print('F1 score: \t{:0.3f}'.format(f1_score(y_test,y_test_pred)))
from sklearn.metrics import classification_report 
print(classification_report(y_test,y_test_pred))
from keras.wrappers.scikit_learn import KerasClassifier

def build_logistic_regression_model():

    model = Sequential()

    model.add(Dense(1,input_dim=22,activation ='sigmoid'))

    model.compile(Adam(lr=0.5),'binary_crossentropy',metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=build_logistic_regression_model,

                        epochs = 10,verbose = 0)
from sklearn.model_selection import KFold,cross_val_score
cv = KFold(5,shuffle=True)

scores = cross_val_score(model,X,y,cv=cv)

scores
print('The cross validation accuracy is {:0.4f} ± {:0.4f}'.format(scores.mean(),scores.std()))
    