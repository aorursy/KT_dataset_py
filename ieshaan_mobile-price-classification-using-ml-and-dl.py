# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/mobile-price-classification/train.csv')

df
df.describe()
df.info()
df.isna().sum()
co = df.corr()

co.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
bplot = sns.boxplot( y = 'ram' ,x ='price_range'  ,data = df  )

_ = plt.setp(bplot.get_xticklabels(), rotation=90)
bplot = sns.boxplot( y = 'n_cores' ,x ='price_range'  ,data = df  )

_ = plt.setp(bplot.get_xticklabels(), rotation=90)
sns.catplot(x="touch_screen",y="battery_power", hue="price_range", kind="bar", data=df)
sns.catplot(x="price_range",y="battery_power", hue="n_cores", kind="bar", data=df)
g = sns.catplot("three_g", col="price_range", col_wrap=4,

                data=df[df.three_g.notnull()],

                kind="count", height=3.5, aspect=.8, 

                palette='tab20')



plt.show()
g = sns.catplot("four_g", col="price_range", col_wrap=4,

                data=df[df.four_g.notnull()],

                kind="count", height=3.5, aspect=.8, 

                palette='tab20')



plt.show()
g = sns.catplot("dual_sim", col="price_range", col_wrap=4,

                data=df[df.dual_sim.notnull()],

                kind="count", height=3.5, aspect=.8, 

                palette='tab20')



plt.show()
X = df.iloc[:,:-1]

Y = df.iloc[:,-1]
sc = StandardScaler()

X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25)
model = LogisticRegression()



# fit the model with the training data

model.fit(X_train,Y_train)



# coefficeints of the trained model

print('Coefficient of model :', model.coef_)

print('                             ')



# intercept of the model

print('Intercept of model',model.intercept_)

print('                             ')

# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 

print('                             ')

# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = DecisionTreeClassifier()



# fit the model with the training data

model.fit(X_train,Y_train)



# depth of the decision tree

print('Depth of the Decision Tree :', model.get_depth())

print('                             ')

# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 

print('                             ')

# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = RandomForestClassifier()



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 

print('                             ')



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
model = GradientBoostingClassifier(n_estimators=100,max_depth=5)



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

print('                                  ')

accuracy_test = accuracy_score(Y_test,predict_test)

print('\naccuracy_score on test dataset : ', accuracy_test)
model = XGBClassifier()



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('\nTarget on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('                                          ')

print('\naccuracy_score on test dataset : ', accuracy_test)
model = SVC()



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the train dataset

predict_train = model.predict(X_train)

print('Target on train data',predict_train) 



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

accuracy_test = accuracy_score(Y_test,predict_test)

print('                            ')

print('accuracy_score on test dataset : ', accuracy_test)
model = GaussianNB()



# fit the model with the training data

model.fit(X_train,Y_train)



# predict the target on the train dataset

predict_train = model.predict(X_train)

print('Target on train data',predict_train) 



# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data',predict_test) 



# Accuracy Score on test dataset

print('                       ')

accuracy_test = accuracy_score(Y_test,predict_test)

print('accuracy_score on test dataset : ', accuracy_test)
Y = pd.get_dummies(Y)

print('One hot encoded array:')

print(Y[0:5])
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25)
model = Sequential()

model.add(Dense(16, input_dim=20, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=100, batch_size=64)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)



print('Test accuracy: %.3f' % (test_acc))