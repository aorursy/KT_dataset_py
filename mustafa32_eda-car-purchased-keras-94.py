



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/prediction-of-purchased-car/Social_Network_Ads.csv")

df.head()
df.shape
sns.countplot(df['Gender'])
df['EstimatedSalary'].describe()
sns.countplot(df['Purchased'])
df['EstimatedSalary'] = pd.cut(df['EstimatedSalary'],bins=10000)

df['Age'] = pd.cut(df['Age'],bins=10)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

df['EstimatedSalary'] = LabelEncoder().fit_transform(df['EstimatedSalary'])

df['Age'] = LabelEncoder().fit_transform(df['Age'])
df = df.drop('User ID',axis=1)

X = df.drop('Purchased',axis=1)

y = df['Purchased']

sc = RobustScaler()

sc.fit(X)

X = sc.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier()

model.fit(X_train,y_train)

predict = model.predict(X_test)

accuracy_score(y_test,predict)
model = ExtraTreeClassifier()

model.fit(X_train,y_train)

predict = model.predict(X_test)

accuracy_score(y_test,predict)
model = LogisticRegression()

model.fit(X_train,y_train)

predict = model.predict(X_test)

accuracy_score(y_test,predict)
from keras.models import Sequential

from keras.layers import Dense

X.shape
model = Sequential()

model.add(Dense(256,input_dim=3,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(12,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X,y,batch_size=1,epochs=10)
_,accuracy = model.evaluate(X,y,verbose=0)
str(round(accuracy*100))+"%"
model.save("onlineShopper.h5")