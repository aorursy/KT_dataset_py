# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/machine-learning-starter-program-hackathon-dataset/train.csv')
df.info()
def impute_age(cols):

    Age = cols[0]

    Education = cols[1]

    

    if pd.isnull(Age):

        if Education == 'Bachelors':

            return 39

        elif Education == 'High School Diploma':

            return 35

        elif Education == 'Masters':

            return 43

        elif Education == 'Matriculation':

            return 36

        elif Education == 'No Qualification':

            return 34

    else:

        return Age
df['age'] = df[['age','education']].apply(impute_age,axis=1)
df.dropna(inplace=True)
df.drop('trainee_id',axis=1,inplace=True)

df.drop('program_type',axis=1,inplace=True)

df.drop('test_id',axis=1,inplace=True)

dfprogid = pd.get_dummies(df['program_id'],drop_first=True)

dftesttype = pd.get_dummies(df['test_type'],drop_first=True)

dfdifflvl = pd.get_dummies(df['difficulty_level'],drop_first=True)

dfgen = pd.get_dummies(df['gender'],drop_first=True)

dfed = pd.get_dummies(df['education'],drop_first=True)

dfhand = pd.get_dummies(df['is_handicapped'],drop_first=True)

df = pd.concat([df,dfprogid,dftesttype,dfdifflvl,dfgen,dfed,dfhand],axis=1)

df.drop(['program_id','test_type','difficulty_level','gender','education','is_handicapped'],axis=1,inplace=True)
df.head()
X = df.drop('is_pass',axis=1).values

y = df['is_pass'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout
X_train.shape
model = Sequential()



model.add(Dense(37,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(15,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))





model.compile(loss='binary_crossentropy',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),callbacks=[early_stop])
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))