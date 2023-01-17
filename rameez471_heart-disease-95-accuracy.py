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
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data.head()
data.describe()
data.info()
death = data[data['DEATH_EVENT']==1]

death.head()
non_heart_death = data[data['DEATH_EVENT']==0]

non_heart_death.head()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))

sns.catplot(y='DEATH_EVENT',data=data,kind='count')

plt.title("DEATH DISTRIBUTION")

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(death['age'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['age'],kde=False,label='No Heart Failure Death')

plt.title('Age Distribution')

plt.legend()

plt.show()
plt.figure(figsize=(8,8))

sns.catplot(y='DEATH_EVENT',data=data,kind='count',hue='anaemia')

plt.title('Amnesia Distibution')

plt.show()
label = ['Anaemia Patients','Non Anemia Pateints']

values = [len(death[death['anaemia']==1]),len(death[death['anaemia']==0])]

plt.figure(figsize=(8,8))

plt.pie(values,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(death['creatinine_phosphokinase'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['creatinine_phosphokinase'],kde=False,label='Non Heart Failure Death')

plt.title('Creatinine Phosphokinase Distibution')

plt.legend()

plt.show()
plt.figure(figsize=(8,8))

sns.catplot(y='DEATH_EVENT',hue='diabetes',data=data,kind='count')

plt.title('Diabetes Distibution')

plt.show()
label = ['Diabetic Patients','Non Diabetic Pateints']

values = [len(death[death['diabetes']==1]),len(death[death['diabetes']==0])]

plt.figure(figsize=(8,8))

plt.pie(values,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(death['ejection_fraction'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['ejection_fraction'],kde=False,label='Non Heart Failure Death')

plt.title('Ejection Fraction  Distibution')

plt.legend()

plt.show()
plt.figure(figsize=(8,8))

sns.catplot(y='DEATH_EVENT',data=data,hue='high_blood_pressure',kind='count')

plt.title('High Blood Pressure Distibution')

plt.show()
label = ['Patient with High Blood Pressure','Patient with Normal Blood Pressure']

values = [len(death[death['high_blood_pressure']==1]),len(death[death['high_blood_pressure']==0])]

plt.figure(figsize=(8,8))

plt.pie(values,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(death['platelets'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['platelets'],kde=False,label='Non Heart Failure Death')

plt.title('Platelets Distibution')

plt.legend()

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(death['serum_creatinine'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['serum_creatinine'],kde=False,label='Non Heart Failure Death')

plt.title('Serum Creatinine Distibution')

plt.legend()

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(death['serum_sodium'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['serum_sodium'],kde=False,label='Heart Failure Death')

plt.legend()

plt.title('Serum Sodium Distibution')

plt.show()
plt.figure(figsize=(8,8))

sns.catplot(y='DEATH_EVENT',data=data,hue='sex',kind='count')

plt.title('Sex Distibution')

plt.show()
label = ['Female ','Male']

values = [len(death[death['sex']==1]),len(death[death['sex']==0])]

plt.figure(figsize=(8,8))

plt.pie(values,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
plt.figure(figsize=(8,8))

sns.distplot(death['time'],kde=False,label='Heart Failure Death')

sns.distplot(non_heart_death['time'],kde=False,label='Non Heart Failure Death')

plt.legend()

plt.title('Time Distibution')

plt.show()
plt.figure(figsize=(8,8))

sns.catplot(y='DEATH_EVENT',data=data,hue='smoking',kind='count')

plt.title('SMoking vs Death')

plt.show()
label = ['Smoker ','Non-Smoker']

values = [len(death[death['smoking']==1]),len(death[death['smoking']==0])]

plt.figure(figsize=(8,8))

plt.pie(values,labels=label,startangle=90,autopct='%1.1f%%')

plt.show()
from sklearn.model_selection import train_test_split



y = data['DEATH_EVENT']

X = data.drop('DEATH_EVENT',axis=1)
num_col = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']

cat_col = ['anaemia','diabetes','high_blood_pressure','sex','smoking']
num_data = data[num_col]

cat_col = data[cat_col]
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



num_data_tf = scaler.fit_transform(num_data)

num_data_tf = pd.DataFrame(num_data_tf,columns=num_data.columns)

num_data_tf.head()
data_tf = pd.concat([num_data_tf,cat_col],axis=1)

data_tf.head()
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier()

classifier.fit(X,y)

feature_name = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = classifier.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'],ascending=True)



labels = importance_frame['Features']

values = importance_frame['Importance']



plt.figure(figsize=(8,8))

plt.pie(values,labels=labels)

plt.show()
data_tf = data_tf[['age','ejection_fraction','platelets','serum_creatinine','time','creatinine_phosphokinase','serum_sodium']]
X_np = np.array(data_tf)

y_np = np.array(y)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X_np,y_np,test_size=0.2)
print('Train Data Shape:',X_train.shape)

print('Validation Data Shape:',X_test.shape)
from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score
models= [['Logistic Regression: ',LogisticRegression()],

        ['KNearest Neighbor: ',KNeighborsClassifier()],

        ['Decision Tree Classifier: ',DecisionTreeClassifier()],

        ['Random Forest Classifier: ',RandomForestClassifier()],

        ['Ada Boost: ',AdaBoostClassifier()],

        ['SVM: ',SVC()],

        ['XG Boost:',XGBClassifier()],

        ['Cat Boost',CatBoostClassifier(logging_level='Silent')]]



for name,model in models:

    model = model

    model.fit(X_train,y_train)

    print(name)

    print('Validation Acuuracy: ',accuracy_score(y_test,model.predict(X_test)))

    print('Training Accuracy: ',accuracy_score(y_train,model.predict(X_train)))

    print('############################################')

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
filepath = 'weights_best.h5'

checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
nn = Sequential()

nn.add(Dense(128,activation='relu',input_shape=(7,)))

nn.add(Dropout(0.4))

nn.add(Dense(256,activation='relu'))

nn.add(Dropout(0.4))

nn.add(Dense(512,activation='relu'))

nn.add(Dropout(0.4))

nn.add(Dense(1024,activation='relu'))

nn.add(Dropout(0.4))

nn.add(Dense(512,activation='relu'))

nn.add(Dropout(0.4))

nn.add(Dense(256,activation='relu'))

nn.add(Dropout(0.4))

nn.add(Dense(128,activation='relu'))

nn.add(Dropout(0.4))

nn.add(Dense(1,activation='sigmoid'))



nn.compile(optimizer='adam',metrics=['acc'],loss='binary_crossentropy')



history = nn.fit(X_train,y_train,

                epochs=50,

                validation_data=(X_test,y_test),

                callbacks=[checkpoint])



train_acc = history.history['acc']

test_acc = history.history['val_acc']

epochs = [i for i in range(1,51)]



plt.figure(figsize=(8,8))

plt.plot(epochs,train_acc,label='Training Acuracies')

plt.plot(epochs,test_acc,label='Validation Accuracies')

plt.legend()

plt.show()