import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score,mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

import random

from keras.models import Sequential

from scipy import stats

import keras

from keras.layers import Dense

from keras import optimizers

from keras.layers.core import Dropout
data_train=pd.read_csv('../input/train.csv')

data_test=pd.read_csv('../input/test.csv')
print(data_train.shape,data_test.shape)
data_test.head(4)
Cols1=(data_test['PassengerId'])
data_train.head(5)
sns.countplot(x='Survived',data=data_train)
sns.countplot(x='Survived',hue='Embarked',data=data_train)
sns.countplot(x='Survived',hue='Pclass',data=data_train)
data_test.isnull().sum()
data_train.isnull().sum()
sns.heatmap(data_test.isnull())

plt.show()
sns.heatmap(data_train.isnull())

plt.show()
full_data=[data_train,data_test]
for data in full_data:

    data['Family_size']=data['SibSp']+data['Parch']+1

    data['IsAlone']=0

    data.loc[data['Family_size']==1,'IsAlone']=1
data_train.head(5)
for data in full_data:

    data['Embarked']=data['Embarked'].fillna('S')
for data in full_data:

    data['Fare']=data['Fare'].fillna(data['Fare'].median())
random.seed(0)

for data in  full_data:

    avg_age=data['Age'].mean()

    std_age=data['Age'].std()

    data_na_size=data['Age'].isnull().sum()

    data_random_list=np.random.randint(avg_age-std_age,avg_age+std_age,size=data_na_size)

    data['Age'][np.isnan(data['Age'])]=data_random_list

    data['Age']=data['Age'].astype(int)
data_train['Age'].isnull().sum()
data_train['Embarked'].unique()
sex_train= pd.get_dummies(data_train["Sex"],drop_first=True)

sex_test=pd.get_dummies(data_test["Sex"],drop_first=True)
Pcl_train=pd.get_dummies(data_train["Pclass"],drop_first=True)

Pcl_test=pd.get_dummies(data_test["Pclass"],drop_first=True)

Pcl_train.head(9)
embark_train=pd.get_dummies(data_train["Embarked"], drop_first=True)

embark_test=pd.get_dummies(data_test["Embarked"], drop_first=True)

embark_train.head(5)
for data in full_data:

    data.loc[ data['Fare'] <= 7.91, 'Fare']= 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare']= 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']= 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)

    

    data.loc[ data['Age'] <= 16, 'Age']= 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4
drope=['PassengerId','Name','Family_size', 'SibSp', 'Parch','Cabin','Ticket','Sex','Embarked']

data_train=data_train.drop(drope,axis=1)

data_test=data_test.drop(drope,axis=1)

data_train=pd.concat([data_train,sex_train,embark_train,Pcl_train],axis=1)

data_test=pd.concat([data_test,sex_test,embark_test,Pcl_test],axis=1)
data_test.head(20)
sns.heatmap(data_train.isnull())

plt.show()
print(data_train.shape,data_test.shape)
X_train=np.array(data_train.drop('Survived',axis=1))

Y_train=np.array(data_train['Survived'])

X_val=np.array(data_test)
print(X_train.shape,Y_train.shape)

print(type(X_train))
np.random.seed(7)

model = Sequential([Dense(16,kernel_initializer='normal',input_dim=9, activation='relu'),Dropout(0.25),

                    Dense(16,kernel_initializer='normal', input_dim=9, activation='relu'),Dropout(0.35),

                    Dense(1, input_dim=1, activation='sigmoid')])

#sgd = optimizers.SGD(lr=0.005, decay=6e-6, momentum=2.6, nesterov=True)

ac=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1500, batch_size=8)
scores = model.evaluate(X_train, Y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(X_val)

# round predictions

rounded = np.array([round(x[0]) for x in predictions]).astype(int)
submission = pd.DataFrame({"PassengerId": Cols1,"Survived":rounded})
submission.head(5)
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)