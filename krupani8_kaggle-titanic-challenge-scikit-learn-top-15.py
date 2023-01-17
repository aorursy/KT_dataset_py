# from google.colab import drive

# drive.mount('/content/drive')
import os



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt
# os.chdir('drive/My Drive')
!ls ../
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train
df_train['Cabin'][:].tolist()
df_train['Cabin'].isnull().sum()
from string import digits

import math



def get_cabin_class(s):



  if isinstance(s, str):

    remove_digits = str.maketrans('', '', digits)

    res = s.translate(remove_digits)

    res = "".join(set(res))

  else:

    res = ''

  return res
df_train['cabin_class'] = df_train['Cabin'].apply(get_cabin_class)
df_test['cabin_class'] = df_test['Cabin'].apply(get_cabin_class)
df_train
df_train.columns
df_test.columns
import seaborn as sns



corrMatrix = df_train.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
df_train['Pclass'].value_counts()
df_train['Family'] = df_train['SibSp'] + df_train['Parch']
df_test['Family'] = df_test['SibSp'] + df_test['Parch']
df_train
x_train = df_train[['Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare', 'Family']]
x_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare', 'Family']]
y_train = df_train[['Survived']]
x_train['Age'].fillna(-1, inplace=True)
x_test['Age'].fillna(-1, inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
x_train.isna().sum()
x_test.isna().sum()
x_train
x_train = pd.get_dummies(x_train, columns=['Sex'])
x_test = pd.get_dummies(x_test, columns=['Sex'])
x_train
x_train['Age'] = x_train['Age'].astype('int8')
x_test['Age'] = x_test['Age'].astype('int8')
x_train
x_test
x_train.shape
y_train.shape
y_train = y_train.squeeze()
x_train
bins= [0, 13, 20, 40 , 60, 110]



labels = ['Kid','Teen','Young Adult', 'Senior Adult', 'Old']



x_train['Age'] = pd.cut(x_train['Age'], bins=bins, labels=labels, right=False)

x_test['Age'] = pd.cut(x_test['Age'], bins=bins, labels=labels, right=False)
x_train
x_train['Age'].value_counts()
x_train['Age'].isnull().sum()
x_train['Family'].value_counts()
x_train['Pclass'].value_counts()
for i in x_train.index:

  if not isinstance(x_train['Age'][i], str):

    if x_train['Family'][i] > 5 or x_train['Pclass'][i] == 1:

      x_train['Age'][i] = 'Senior Adult'

      print(i)

    else:

      # pass

      x_train['Age'][i] = 'Young Adult'
x_train.isnull().sum()
x_train = pd.get_dummies(x_train, columns=['Age'])
x_train = pd.get_dummies(x_train, columns=['Pclass'])
x_test = pd.get_dummies(x_test, columns=['Age'])
x_test = pd.get_dummies(x_test, columns=['Pclass'])
x_train
x_test
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()



x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
parameters = {'C':[1, 10, 100],

              'gamma':[0.1, 0.01, 0.001],

              'kernel': ['linear', 'poly']

            }
#Import svm model

from sklearn import svm



from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier



#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier, VotingClassifier



#Create a Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



clf2 = DecisionTreeClassifier()



#Create a Gaussian Classifier

clf3 = RandomForestClassifier(n_estimators=100)



# clf = VotingClassifier([('c_svc', clf1), ('dt', clf2), ('rf', clf3)], voting='hard' )
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=clf, param_grid=parameters, cv = 5, n_jobs=-1)
from sklearn.model_selection import RandomizedSearchCV
random = RandomizedSearchCV(estimator=clf, param_distributions=parameters, cv = 5, n_jobs=-1)
#Train the model using the training sets



grid_result = grid.fit(x_train, y_train)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))





#Predict the response for test dataset

y_pred = grid.predict(x_test)
y_pred
y_pred = pd.DataFrame(y_pred, columns=['Survived'])
submission = pd.concat([pd.DataFrame(df_test['PassengerId'], columns=['PassengerId']), y_pred], axis=1)
submission
submission.to_csv('submission.csv', index=False)