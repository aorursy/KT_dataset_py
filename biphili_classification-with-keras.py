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
df = pd.read_csv('../input/user_visit_duration.csv')

df.head()
df.plot(kind='scatter',x='Time (min)',y='Buy');
from keras.models import Sequential 

from keras.layers import Dense

from keras.optimizers import Adam,SGD
model = Sequential()

model.add(Dense(1,input_shape =(1,),activation='sigmoid'))
model.compile(SGD(lr=0.5),'binary_crossentropy',metrics=['accuracy'])
model.summary()
X = df[['Time (min)']].values

y = df['Buy'].values

model.fit(X,y,epochs =25)
import matplotlib.pyplot as plt

ax = df.plot(kind = 'scatter',x = 'Time (min)',y='Buy',

             title ='Purchase behavior Vs Time spend on the site')

temp = np.linspace(0,4)

ax.plot(temp,model.predict(temp),color='red')

plt.legend(['model','data']);
temp_class = model.predict(temp) >0.5
import matplotlib.pyplot as plt

ax = df.plot(kind = 'scatter',x = 'Time (min)',y='Buy',

             title ='Purchase behavior Vs Time spend on the site')

temp = np.linspace(0,4)

ax.plot(temp,temp_class,color='orange')

plt.legend(['model','data']);
y_pred = model.predict(X)

y_class_pred = y_pred > 0.5
from sklearn.metrics import accuracy_score
print('The accuracy score is {:0.3f}'.format(accuracy_score(y,y_class_pred)))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
params = model.get_weights()

params = [np.zeros(w.shape) for w in params]

model.set_weights(params)
#print('The Accuracy score is {:0.3f}'.format(accuracy_score(y,y_class_pred)))
model.fit(X_train,y_train,epochs =25,verbose = 0)
print('The Accuracy score is {:0.3f}'.format(accuracy_score(y_train,model.predict(X_train)>0.5)))

print('The Accuracy score is {:0.3f}'.format(accuracy_score(y_test,model.predict(X_test)>0.5)))
from keras.wrappers.scikit_learn import KerasClassifier
def build_logistic_regression_model():

    model = Sequential()

    model.add(Dense(1,input_shape=(1,),activation='sigmoid'))

    model.compile(SGD(lr=0.5),

                 'binary_crossentropy',

                 metrics = ['accuracy'])

    return model
model = KerasClassifier(build_fn = build_logistic_regression_model,epochs = 25 )
from sklearn.model_selection import cross_val_score, KFold
cv = KFold(3,shuffle = True)

scores = cross_val_score(model,X,y,cv=cv)
scores
print('The cross validation accuracy is {:0.4f} ± {:0.4f}'.format(scores.mean(),scores.std()))
from sklearn.metrics import confusion_matrix
confusion_matrix(y,y_class_pred)
def pretty_confusion_matrix(y_true,y_pred,labels =["False","True"]):

    cm = confusion_matrix(y_true,y_pred)

    pred_labels = ['Predicted '+ l for l in labels]

    df = pd.DataFrame(cm,index = labels,columns = pred_labels)

    return df
pretty_confusion_matrix(y,y_class_pred,['Not Buy','Buy'])
from sklearn.metrics import precision_score,recall_score,f1_score
print('Precision:\t{:0.3f}'.format(precision_score(y,y_class_pred)))

print('Recall:   \t{:0.3f}'.format(recall_score(y,y_class_pred)))

print('F1 score: \t{:0.3f}'.format(f1_score(y,y_class_pred)))
from sklearn.metrics import classification_report 
print(classification_report(y,y_class_pred))