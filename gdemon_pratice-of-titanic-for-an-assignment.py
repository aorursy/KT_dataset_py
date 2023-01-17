# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.info()
df.head()
# cabin miss too many, drop it
df.drop("Cabin", axis=1, inplace=True)
df.info()
df.groupby('Embarked').count()

#fill missing Embarked as S because it is most frequent
df['Embarked'] = df['Embarked'].fillna('S')
df.info()
df.groupby('Embarked').count()
#fill missing AGe as mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df.info()
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Sex'].head()
import seaborn as sns
import matplotlib.pyplot as plt
#correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#Survived correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Survived')['Survived'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_1=pd.get_dummies(df, columns=["Pclass","Embarked"])
df_1.info()
#
# Linear Regression (Logistic Regression)
#

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

cols = ["Age","Pclass_1","Pclass_2","Pclass_3","Embarked_C","Embarked_S","Embarked_Q","Sex"] 
x = df_1[cols]
y = df_1['Survived']
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
y_pred_proba = logreg.predict_proba(x_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
#
# Tree
#
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
y_pred_proba = rfc.predict_proba(x_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
#print("Random forest classifier accuracy: %2.3f ")
print('Train/Test split results:')
print(rfc.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(rfc.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(rfc.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout

# create model
tmodel = Sequential()
tmodel.add(Dense(input_dim=x_train.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
tmodel.add(Activation('relu'))

for i in range(0, 8):
    tmodel.add(Dense(units=64, kernel_initializer='normal',
                     bias_initializer='zeros'))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(.25))

tmodel.add(Dense(units=1))
tmodel.add(Activation('linear'))

tmodel.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='rmsprop')
tmodel.fit(x_train, y_train, epochs=600, verbose=2)

# 
# result of NN model
#
score = tmodel.evaluate(x_test, y_test, batch_size=128)
print(tmodel.__class__.__name__+" accuracy is %2.3f" % (score[1]))
#print(score)
