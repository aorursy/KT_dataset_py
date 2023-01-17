# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics
dataframe= pd.read_csv("/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv")
dataframe.head()
dataframe.tail()
%matplotlib inline

corr=dataframe.corr()

ax=sns.heatmap(corr, vmin=-1, vmax=1, center=0,

               square=True, cmap='YlGnBu')
dataframe.describe()
dataframe.info()
del dataframe['nameDest']

del dataframe['nameOrig']

del dataframe['isFlaggedFraud']
dataframe1=dataframe[0:-1]

dataframe1.head(2)
cnt0=(dataframe['isFraud']==0).sum()

cnt1=(dataframe['isFraud']==1).sum()
somelist=[cnt0,cnt1]

freq_series = pd.Series(somelist)

plt.figure(figsize=(12, 8))

ax = freq_series.plot(kind='bar')

name=['Not Fraud', 'Fraud']

ax.set_title('Distribution')

ax.set_xlabel('Fraud Not Fraud Count')

ax.set_ylabel('Frequency')

ax.set_xticklabels(name)

plt.show()
le=preprocessing.LabelEncoder()

dataframe1['type2']=le.fit_transform(dataframe1['type'])

del dataframe1['type']

shuffled_df=dataframe1.sample(frac=1,random_state=4)

fraud_df=shuffled_df.loc[shuffled_df['isFraud']==1]

non_fraud_df=shuffled_df.loc[shuffled_df['isFraud']==0].sample(8212,random_state=4)

normalized=pd.concat([fraud_df,non_fraud_df])
cnt2=(normalized['isFraud']==0).sum()

cnt3=(normalized['isFraud']==1).sum()
somelist1=[cnt2,cnt3]

freq_series1 = pd.Series.from_array(somelist1)

plt.figure(figsize=(12, 8))

ax = freq_series1.plot(kind='bar')

name=['Not Fraud', 'Fraud']

ax.set_title('Distribution')

ax.set_xlabel('Fraud Not Fraud Count')

ax.set_ylabel('Frequency')

ax.set_xticklabels(name)

plt.show()
normalized1=normalized.sample(frac=1, random_state=4)

X=normalized1.drop(['isFraud'], axis=1)

Y=normalized1['isFraud']

y3=Y.values.reshape(-1,1)

st_sc = StandardScaler()

X = st_sc.fit_transform(X)

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()

Y1=ohe.fit_transform(y3).toarray()

from sklearn.model_selection  import train_test_split

x_train, x_test, y_train,y_test=train_test_split(X,Y1, test_size=0.2, random_state=1)

from sklearn.metrics import confusion_matrix

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.models import load_model

model = Sequential()

model.add(Dense(units = 64, input_dim = 7, activation = "relu"))

model.add(Dense(units = 32, activation = "relu"))

model.add(Dense(units = 64, activation = "relu"))

model.add(Dense(units = 128, activation = "relu"))

model.add(Dropout(0.3))

model.add(Dense(units = 2, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.summary()
model.fit(x_train, y_train, batch_size = 128, epochs = 10, verbose=1)
score = model.evaluate(x_test, y_test)

print(score[1])
scoring=model.predict(x_test)

classes=model.predict_classes(x_test)

classes.reshape(1,-1)
somepredictions=model.predict(x_test)

accdl=metrics.accuracy_score(y_test.argmax(axis=1), somepredictions.argmax(axis=1))

accdl=accdl*100
print(pd.DataFrame(

    confusion_matrix(y_test.argmax(axis=1), somepredictions.argmax(axis=1)),

    columns=['Predicted Not fraud', 'Predicted fraud'],

    index=['True not fraud', 'True fraud']

))
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

k_range=range(1,26)

scores={}

scores_list=[]



for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    y_pred=knn.predict(x_test)

    scores[k]=metrics.accuracy_score(y_test, y_pred)

    scores_list.append(scores[k])
plt.plot(k_range, scores_list)

plt.xlabel("Vlaue of KNN")

plt.ylabel("testing accuracy")
scores_list.index(max(scores_list))
knn1=KNeighborsClassifier(n_neighbors=4)

knn1.fit(x_train, y_train)

y_prediction2=knn1.predict(x_test)

accknn=metrics.accuracy_score(y_test, y_prediction2)

accknn=accknn*100
print(pd.DataFrame(

    confusion_matrix(y_test.argmax(axis=1), y_prediction2.argmax(axis=1)),

    columns=['Predicted Not fraud', 'Predicted fraud'],

    index=['True not fraud', 'True fraud']

))
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score

accdt=accuracy_score(y_test, y_predict)

accdt=accdt*100
print(pd.DataFrame(

    confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1)),

    columns=['Predicted Not fraud', 'Predicted fraud'],

    index=['True not fraud', 'True fraud']

))
acc_list=[accdl,accknn,accdt]

ac_score1 = pd.Series.from_array(acc_list)

plt.figure(figsize=(12, 8))

ax = ac_score1.plot(kind='bar')

name=['DL', 'KNN', 'DTree']

ax.set_title('Comparison')

ax.set_xlabel('ANN vs KNN vs Decision Trees')

ax.set_ylabel('Score')

ax.set_xticklabels(name)

plt.show()