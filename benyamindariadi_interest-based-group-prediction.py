import numpy as np

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/clustering-categorical-peoples-interests/kaggle_Interests_group.csv')
df.info()

sns.countplot(x='group',data=df)
#see the distinct values per column

for col in df:

    print(df[col].unique())
#see the columns that real missing value, since the dataset also use NaN to indicate the value equal to zero

for col in df:

    if df[col].nunique()>1:

        print(col)

    else:

        pass
#fill the NaN value with zero, only for columns use NaN to indicate the value equal to zero

a=[]

for col in df:

    if df[col].nunique()==1:

        a.append(col)

    else:

        pass

df[a] = df[a].fillna(0)

df.isnull().sum()
#drop the columns with the number of missing value >10% of total data

threshold=0.1*6340



b=[]

for col in df:

    if df[col].isnull().sum()>threshold:

        b.append(col)

    else:

        pass

df=df.drop(b,axis=1)
#drop the rows with the number of missing value from columns with the number of missing value <10% of total data

c=[]

for col in df:

    if (df[col].isnull().sum()<threshold) & (df[col].isnull().sum()>0):

        c.append(col)

    else:

        pass

df = df.dropna(subset=c)
for col in df:

    print(df[col].unique())
df['group'] = df['group'].map({'C':0,'P':1,'R':2,'I':3})
df #cleaned dataset
x=df.drop('group',axis=1).values 

y=df['group'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
scores=[]

for i in range(1,50):

  tree=DecisionTreeClassifier(max_depth = i) 

  tree.fit(x_train, y_train) 

  scores.append(tree.score(x_test,y_test)) 

plt.plot(range(1,50),scores) 

plt.show()

tree=DecisionTreeClassifier(max_depth =5) 

tree.fit(x_train, y_train) 

tree.score(x_test,y_test)
predictions = tree.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier





scores=[]

for i in (np.arange(100,2000,100)):

  classifier = RandomForestClassifier(n_estimators =i, max_depth=10, random_state =101) 

  classifier.fit(x_train, y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.arange(100,2000,100),scores) 

plt.show()
classifier = RandomForestClassifier(n_estimators =700,max_depth=10, random_state =101)

classifier.fit(x_train, y_train)

print(classifier.score(x_test,y_test))
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
from sklearn.neighbors import KNeighborsClassifier



accuracies=[]

for k in range(1,101):

  classifier = KNeighborsClassifier(n_neighbors = k)

  classifier.fit(x_train, y_train)

  accuracies.append(classifier.score(x_test, y_test)) 

  

k_list=list(range(1,101)) 



plt.plot(k_list,accuracies)

plt.show() 
classifier = KNeighborsClassifier(n_neighbors =54)

classifier.fit(x_train, y_train)

classifier.score(x_test, y_test) 
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
from tensorflow.keras.utils import to_categorical

y_cat_train = to_categorical(y_train) 

y_cat_test = to_categorical(y_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Dense(units=160,activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)))



model.add(Dense(units=4,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



model.fit(x_train,y_cat_train,epochs=300,validation_data=(x_test,y_cat_test),verbose=1)
print(model.metrics_names)

print(model.evaluate(x_test,y_cat_test,verbose=0))
losses = pd.DataFrame(model.history.history)

losses[['accuracy','val_accuracy']].plot()

losses[['loss','val_loss']].plot()
from sklearn.metrics import classification_report,confusion_matrix



predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))
