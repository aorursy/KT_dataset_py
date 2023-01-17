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
import pandas as pd
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

sample = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
sample.describe()
sample.head()
train.info()
test.info()
train.head()
test.head()
# drop cabin karena terlalu banyak data yang hilang

train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)
# fill missing value dengan median untuk data nominal

# fill missing value dengan modos untuk data kategorik

train['Age'] = train['Age'].fillna(train['Age'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train.head()
train = train.drop(['Name', 'Ticket'], axis=1)

test = test.drop(['Name', 'Ticket'], axis=1)
#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder() 

#train['Cat_Sex']= le.fit_transform(train['Sex']) 

#train['Cat_Embarked']= le.fit_transform(train['Embarked']) 

#test['Cat_Sex']= le.fit_transform(test['Sex']) 

#test['Cat_Embarked']= le.fit_transform(test['Embarked']) 
# mengganti data kategorik ke dalam numerik dengan kolom yang berbeda-beda

male = []

female = []

for num in train['Sex']:

    if num == 'male':

        male.append(1)

        female.append(0)

    else:

        male.append(0)

        female.append(1)

train['male'] = male

train['female'] = female



male = []

female = []

for num in test['Sex']:

    if num == 'male':

        male.append(1)

        female.append(0)

    else:

        male.append(0)

        female.append(1)

test['male'] = male

test['female'] = female



S = []

C = []

Q = []

for num in train['Embarked']:

    if num == 'S':

        S.append(1)

        C.append(0)

        Q.append(0)

    elif num == 'C':

        S.append(0)

        C.append(1)

        Q.append(0)

    else:

        S.append(0)

        C.append(0)

        Q.append(1)

train['S'] = S

train['C'] = C

train['Q'] = Q



S = []

C = []

Q = []

for num in test['Embarked']:

    if num == 'S':

        S.append(1)

        C.append(0)

        Q.append(0)

    elif num == 'C':

        S.append(0)

        C.append(1)

        Q.append(0)

    else:

        S.append(0)

        C.append(0)

        Q.append(1)

test['S'] = S

test['C'] = C

test['Q'] = Q
train.head()
# buang variabel sex dan embarked karena kita sudah tidak memerlukan lagi

train = train.drop(['Sex', 'Embarked'], axis=1)

test = test.drop(['Sex', 'Embarked'], axis=1)
# simpan variabel PassengerId karena nanti akan digunakan di submission sebagai id-nya

PassengerId_test = test['PassengerId']
# karena kita sudah menyimpan data id maka kita akan buang PassengerId karena id tidak berpengaruh apa-apa 

# karena PassengerId hanya sebagai identitas saja

train = train.drop(['PassengerId'], axis=1)

test = test.drop(['PassengerId'], axis=1)
# sebelum mencoba dengan data test, model perlu di coba terlebih dahulu kepada sample

# maka untuk mencari sample nya digunakan train test split

X = train.drop(['Survived'], axis=1)

Y = train['Survived']

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
# mencoba untuk melihat akurasi dengan data sample

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
# melihat hasil 

# confusion_matrix, classification_report berguna untuk melihat akurasi data hasil prediksi

# accuracy_score untuk melihat accurasi

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix,accuracy_score

print(classification_report(Y_test, Y_pred))

print(confusion_matrix(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))
# parameter tuning berguna untuk memaksimalkan akurasi

# pemilihan C berdasarkan akurasi test yang tidak berbeda jauh dengan akurasi train

# hal ini digunakan agar model dapat memprediksi data yang tidak terlihat

# tolak ukur dalam pemilihan model adalah akurasi score +- 10 antara akurasi train dan akurasi test

C_param = [0.001,0.01,0.1,1,10,100,1000]

for num in C_param:

    model=LogisticRegression(C=num)

    

    model.fit(X_train,Y_train)

    Y_pred = model.predict(X_test)

    a = accuracy_score(Y_test, Y_pred)

    

    model.fit(X_test,Y_test)

    Y_pred = model.predict(X_train)

    b = accuracy_score(Y_train, Y_pred)

    

    selisih = abs(b-a)

    print(a,b, selisih, num)
# saya memilih 0.1 karena saya membutuhkan akurasi dan selisih akurasi score train dan test tidak berbeda jauh

# maka kita lakukan pemodelan kedalam data aslinya

model=LogisticRegression(C=0.1)

model.fit(X,Y)

Y_pred = model.predict(test)
# memasukkan hasil prediksi ke dalam data frame sesuai dengan nama kolom sample submissin yang diberikan di awal

submission = pd.DataFrame({

    'PassengerId' : PassengerId_test,

    'Survived' : Y_pred

})
submission.head()