# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/HR_comma_sep.csv")
df.info()
df.head()
df.describe()
df.left.value_counts()/len(df) 
df['average_montly_hours100']=df['average_montly_hours']/100.0
plt.figure(figsize=(16,8))

for i, feature in enumerate(['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours100']):
    plt.subplot(1, 4, i+1)
    df[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)
plt.figure(figsize=(16,8))

for i, feature in enumerate(['time_spend_company','Work_accident','left','promotion_last_5years']):
    plt.subplot(1, 4, i+1)
    df[feature].plot(kind='hist', title=feature)
    plt.xlabel(feature)
df_work_task=pd.get_dummies(df[['sales','salary']])
df_work_task.head()
df.columns
X = pd.concat([df[['satisfaction_level', 'last_evaluation', 'number_project', 
              'time_spend_company', 'Work_accident', 
               'promotion_last_5years', 'average_montly_hours100']],df_work_task],axis=1).values
y = df['left'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
model = Sequential()
model.add(Dense(1, input_dim=20,activation = 'sigmoid'))
model.compile(Adam(lr=0.2), 'binary_crossentropy',metrics = ['accuracy'])
model.summary()
model.fit(X_train,y_train,epochs =10)
y_test_prep = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix, classification_report
def pretty_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
    cm = confusion_matrix(y_true, y_pred)
    pred_labels = ['Predicted '+ l for l in labels]
    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
    return df
pretty_confusion_matrix(y_test, y_test_prep, labels=['Stay', 'Leave'])
from keras.wrappers.scikit_learn import KerasClassifier
def build_logistic_regression_model():
    model = Sequential()
    model.add(Dense(1, input_dim=20, activation='sigmoid'))
    model.compile(Adam(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_logistic_regression_model,
                        epochs=15, verbose=0)
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)

print("The cross validation accuracy is {:0.4f} ± {:0.4f}".format(scores.mean(), scores.std()))
scores

