# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import Series, DataFrame

data=pd.read_csv("/kaggle/input/OnlineNewsPopularity.csv")

df=pd.DataFrame(data)

df[:5]
import matplotlib.pyplot as plt

from pylab import rcParams

import seaborn as sb

%matplotlib inline

rcParams['figure.figsize']=20,10

sb.set_style('whitegrid')
df=df[np.abs(df[' shares']-df[' shares'].mean()) <= ((3*df[' shares'].std()))]
df=df.drop(['url',' timedelta'], axis=1)
from sklearn import feature_selection



sel = feature_selection.VarianceThreshold()

train_variance = sel.fit_transform(df)

train_variance.shape
correlated_features = set()

correlation_matrix = df.drop(' shares', axis=1).corr()



for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)

correlated_features
df=df.drop(correlated_features,axis=1)

df
X=df.drop([' shares'], axis=1)

y=[]
for i in df.index:

    if(df[' shares'][i]<=1400):

        y.append(0)

    elif(df[' shares'][i]>1400):

        y.append(1)
plt.hist(x=y)
# Normalize the numerical features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical = [' n_tokens_title', ' n_tokens_content', ' num_hrefs', ' num_self_hrefs', ' num_imgs',' num_videos',\

            ' average_token_length',' num_keywords',' self_reference_min_shares',' self_reference_max_shares']

X[numerical] = scaler.fit_transform(X[numerical])

display(X.head(n = 1))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

# Initialize the three models

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV

forest=RandomForestClassifier(random_state=0)

clf_A = AdaBoostClassifier(random_state=0)

clf_B = LogisticRegression(random_state=0,C=1.0)
selector = RFECV(clf_A, step=1, cv=5)

selector = selector.fit(X, y)

features_ADA = X[X.columns.values[selector.ranking_==1]]

x_train,x_cv,y_train,y_cv= train_test_split(features_ADA,y,test_size=0.2,random_state=42)
clf_A.fit(x_train,y_train)
ada_pred=clf_A.predict(x_cv)

ada_check= pd.DataFrame({'Actual': y_cv,'Predicted': ada_pred.flatten()})

print((y_cv==ada_pred).sum()/len(y_cv))

ada_pred
selector = RFECV(clf_B, step=1, cv=5)

selector = selector.fit(X, y)

features_LR = X[X.columns.values[selector.ranking_==1]]

x_train,x_cv,y_train,y_cv= train_test_split(features_LR,y,test_size=0.2,random_state=42)
clf_B.fit(x_train,y_train)
lr_pred=clf_B.predict(x_cv)

lr_check= pd.DataFrame({'Actual': y_cv,'Predicted': lr_pred.flatten()})

print((y_cv==lr_pred).sum()/len(y_cv))

lr_pred
selector = RFECV(forest, step=1, cv=5)

selector = selector.fit(X, y)

features_forest = X[X.columns.values[selector.ranking_==1]]

x_train,x_cv,y_train,y_cv= train_test_split(features_forest,y,test_size=0.2,random_state=42)
forest.fit(x_train,y_train)
forest_test_pred=forest.predict(x_cv)

forest_test_check= pd.DataFrame({'Actual': y_cv,'Predicted': forest_test_pred.flatten()})

print((y_cv==forest_test_pred).sum()/len(y_cv))

forest_test_pred
from keras.utils import to_categorical

y = to_categorical(y)

x_train,x_cv,y_train,y_cv= train_test_split(X,y,test_size=0.2,random_state=42)
import keras

from keras.models import Sequential

from keras.layers import Dense

# Neural network

model = Sequential()

model.add(Dense(100, input_dim=x_train.shape[1], activation='relu'))

model.add(Dense(70, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=40, batch_size=64)
y_pred = model.predict(x_cv)

#Converting predictions to label

pred = list()

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))

#Converting one hot encoded test label to label

test = list()

for i in range(len(y_cv)):

    test.append(np.argmax(y_cv[i]))
from sklearn.metrics import accuracy_score

a = accuracy_score(pred,test)

print('Accuracy is:', a*100)