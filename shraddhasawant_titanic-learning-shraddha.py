# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")

print(df)

print(df.columns)

# Any results you write to the current directory are saved as output.



# Treatment check



df.dtypes

df['Pclass'] = df['Pclass'].astype('category')

print(df.dtypes)
# drop by domain knowledge

df.drop(['Name','Ticket','Cabin','PassengerId','Embarked'],axis=1,inplace=True)
df['Age'].unique()

df.isnull().any()

df['Age'] = df['Age'].fillna(df['Age'].median())
# dropping all NaN

df.isnull().any()

#df_v1 = df.dropna()
#one hot encoding

df_v1 = pd.get_dummies(df)
print(df_v1.columns)

len(df_v1)

df_v1.dtypes
# standardize

from sklearn.preprocessing import StandardScaler

def standard(df_v1):

    scaler = StandardScaler()

    scaler.fit(df_v1)

    df_v1_trans = scaler.transform(df_v1)

    df_v1_trans.shape

    df_v1_trans = pd.DataFrame(df_v1_trans)

    df_v1_trans.columns = df_v1.columns

    return df_v1_trans
from sklearn.neighbors import KNeighborsClassifier

X = df_v1.drop('Survived',axis = 1) 

Y = df_v1['Survived']

X = standard(X)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

X_train
y_train
# model call and fit

model = KNeighborsClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# confusion matrix and accuracy score

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
model = KNeighborsClassifier()

model.fit(X_train,y_train)

y_pred_train = model.predict(X_train)

y_pred_test = model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(y_train,y_pred_train)

confusion_matrix(y_test,y_pred_test)
train_accuracy = accuracy_score(y_train,y_pred_train)

print("Train accuracy",train_accuracy)



test_accuracy = accuracy_score(y_test,y_pred_test)

print("Test accuracy",test_accuracy)

metric = pd.DataFrame(columns =['k','accuracy'])

print(metric)

metric_overall = pd.DataFrame(columns =['k','accuracy'])

print(metric_overall)
metric_overall = pd.DataFrame(columns =['k','accuracy'])

for k in range(3,20):

    metric = pd.DataFrame(columns =['k','accuracy'])

    neigh = KNeighborsClassifier(n_neighbors=k)

    neigh.fit(X_train,y_train) 

    yhat = neigh.predict(X_test)

    test_accuracy = accuracy_score(y_test,yhat)

    metric.loc[0,'k'] = k

    metric.loc[0,'accuracy'] = test_accuracy

    metric_overall = metric_overall.append(metric)

    print(metric_overall)

 



metric_overall
max_value = metric_overall['accuracy'].max()

max_frame = metric_overall[metric_overall['accuracy'] == max_value]

#print(max_frame)

best_k = max_frame['k'].min()

#best_k = int(max_frame['k'])

#best_k = int(metric_overall[metric_overall['accuracy'] == metric_overall['accuracy'].max()]['k'])





#neigh.fit(X_t,y_train) 



best_k
test = pd.read_csv("../input/test.csv")

print(test)
test.isnull().sum()

#test[test['Age'] < 10]
print(test.columns)

test.groupby(['Pclass','Age','Embarked']).mean()
pass_id = test['PassengerId']

test['Pclass'] = test['Pclass'].astype('category')

test.drop(['Name','Ticket','Cabin','PassengerId','Embarked'],axis=1,inplace=True)
test
print(df.isnull().any())

#df_v2 = test.dropna()

test['Age'] = test['Age'].fillna(test['Age'].median())
test.columns
#one hot encoding_testing

test = pd.get_dummies(test)
test.columns

len(test)


test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

test.isnull().any()
test = standard(test)

model = KNeighborsClassifier(n_neighbors= best_k)

model.fit(X_train,y_train)

y_pred = model.predict(test)

y_pred = pd.DataFrame(y_pred)

y_pred.columns = ['Survived']

submit = pd.concat([pass_id,y_pred['Survived']],axis =1)

submit
submit.to_csv("submission.csv",index= False)

print("Completed")