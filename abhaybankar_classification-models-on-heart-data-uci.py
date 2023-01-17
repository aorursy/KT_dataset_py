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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head(4)
df.values.shape
df.columns
df.describe()
df.info()
df.isnull().sum()
sns.countplot(x = 'target', data = df )
sns.pairplot(df)
plt.figure(figsize = (16,7))

sns.heatmap(df.corr(), annot = True)
sns.distplot(df['age'])
sns.distplot(df['oldpeak'])
sns.distplot(df['thalach'])
sns.distplot(df['trestbps'])
sns.distplot(df['chol'])
sns.countplot(x = 'target',data = df, hue = 'sex',)
sns.countplot(x = 'target', data = df, hue = "cp")
plt.figure(figsize = (12,10))

sns.countplot(x = 'age', data = df, hue = 'sex')
df['sex'].value_counts()
plt.figure(figsize = (10,7))

sns.boxplot(data = df[['age', 'trestbps','chol','thalach']])
for i in df[['age','trestbps','chol','thalach']]:

    Q1 = df[i].quantile(0.25)

    Q3 = df[i].quantile(0.75)

    iqr = Q3-Q1

    Upper_limit = Q3+3*iqr

    df = df[df[i]< Upper_limit]

    print(df)     
plt.figure(figsize = (10,7))

sns.boxplot(data = df[['age', 'trestbps','chol','thalach']])
from sklearn.model_selection import train_test_split
X = df.drop('target', axis = 1).values

y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
print("X_train min = {} and max = {}".format(X_train.min(),X_train.max()))
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)



predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print("Logistic regression accuracy is {}".format(accuracy_score(y_test,predictions)))
from sklearn.neighbors import KNeighborsClassifier

Kneighbor = KNeighborsClassifier(n_neighbors=6)

Kneighbor.fit(X_train,y_train)

predict_knn = Kneighbor.predict(X_test)
print(classification_report(y_test,predict_knn))
print(confusion_matrix(y_test,predict_knn))
print('Knn accuracy is {}'.format(accuracy_score(y_test,predict_knn)))
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)
predict_tree = tree.predict(X_test)
print(classification_report(y_test,predict_tree))
print(confusion_matrix(y_test,predict_tree))
print('Decision tree accuracy is {}'.format(accuracy_score(y_test,predict_tree)))
from sklearn.ensemble import RandomForestClassifier

random = RandomForestClassifier(n_estimators = 150)

random.fit(X_train,y_train)
predict_random = random.predict(X_test)
print(classification_report(y_test,predict_random))
print(confusion_matrix(y_test,predict_random))
print('Random forest accuracy is {}'.format(accuracy_score(y_test,predict_random)))
from sklearn.svm import SVC

model_svm = SVC()

model_svm.fit(X_train,y_train)
predict_svm = model_svm.predict(X_test)
print(classification_report(y_test,predict_svm))
print(confusion_matrix(y_test,predict_svm))
print("SVM accuracy is {}".format(accuracy_score(y_test,predict_svm)))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping
model_ai = Sequential()



model_ai.add(Dense(13,activation = 'relu'))

model_ai.add(Dropout(0.2))

model_ai.add(Dense(9,activation = 'relu'))

model_ai.add(Dropout(0.2))



#Binary classification

model_ai.add(Dense(1,activation = 'sigmoid'))



model_ai.compile(loss = 'binary_crossentropy', optimizer = 'adam')

early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose =1, patience = 30)
model_ai.fit(x = X_train,y = y_train, epochs = 600, validation_data=(X_test,y_test),callbacks=[early_stop])
losses = pd.DataFrame(model_ai.history.history)

losses.plot()
prediction_ai = model_ai.predict_classes(X_test)
print(classification_report(y_test,prediction_ai))
print(confusion_matrix(y_test,prediction_ai))
print("Neural Network accuracy is {}".format(accuracy_score(y_test,prediction_ai)))