

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

ufc_fight = pd.read_csv('/kaggle/input/ufcdata/data.csv', parse_dates = ['date'])
ufc_fight.isnull().sum()
ufc_fight.head()
ufc_fight.shape
R,B,D =ufc_fight['Winner'][ufc_fight['no_of_rounds']==5].value_counts()

sns.countplot(ufc_fight['Winner'][ufc_fight['no_of_rounds']==5],palette = ['Red', 'Blue', 'Green'])

print('For a five round match number of matches won by Red and Blue team Individualy:')

print('                 Red = %d , Blue = %d'%(R, B))
sns.countplot(ufc_fight['Winner'][ufc_fight['no_of_rounds']==3],palette = ['Red', 'Blue', 'Green'])

Re,Bl,Dr = ufc_fight['Winner'][ufc_fight['no_of_rounds']==3].value_counts()

print('For a 3 round match individual contribution of Red and Blue teams')

print('         Red = %d , Blue = %d'%(Re,Bl))
a =ufc_fight['Winner'][ufc_fight['title_bout']== True]

sns.countplot(a, palette = ['Red', 'Blue', 'Green'])

red , blue, draw = a.value_counts()

print('Number of title bout won by Red Fighter : ',red )

print('Number of title bout won by blue fighter: ', blue)
#Setting Date as Index

#Making on extra Column for year.

ufc_fight =ufc_fight.set_index('date')

ufc_fight['year'] = ufc_fight.index.year

sns.relplot(x = 'no_of_rounds', y = 'year',col = 'Winner',kind = 'line',data = ufc_fight)
#Processing the data, remove the unwanted columns like names, country , referee, etc

#Author had already given the processed data

processed_data = pd.read_csv('/kaggle/input/ufcdata/preprocessed_data.csv')

processed_data.head()
#Converting this categorical values (Blue, Red) present in Winner Column to 0 and 1 

#Creating a variable named target (of series datatype).

target =processed_data['Winner'].astype('category').cat.codes

processed_data.drop(['Winner','B_draw','R_draw'], axis = 1, inplace = True)
#Checking for null values.

processed_data.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,f1_score

from sklearn.metrics import accuracy_score
x_train, x_test, y_train,y_test = train_test_split(processed_data,target, test_size = 0.3, random_state = 42)
random = RandomForestClassifier(n_estimators = 350)

random.fit(x_train,y_train)

target_pred = random.predict(x_test)

f1scr = f1_score(y_test,target_pred)*100

accuracy = accuracy_score(y_test,target_pred)*100

print('f1 score for Random Forest Classifier is :',f1scr)

print('Accuracy of the model is :', accuracy)

confu_mat =confusion_matrix(y_test,target_pred)

sns.heatmap(confu_mat, annot = True, fmt = '.2f',annot_kws={'size':16}, cmap = 'coolwarm')

logist = LogisticRegression(max_iter = 500)

logist.fit(x_train,y_train)

log_predict  = logist.predict(x_test)

f1_score = f1_score(y_test, log_predict)*100

acc_score = accuracy_score(y_test,log_predict)*100

print('f1 score for Logistic Regression Classifier is :', f1_score)

print('Accuracy of this model is :', acc_score)

c_m = confusion_matrix(y_test,log_predict)

sns.heatmap(c_m, annot =True, fmt = '.2f', annot_kws={'size':16}, cmap = 'coolwarm')
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import confusion_matrix

model = CatBoostClassifier(iterations = 500,learning_rate = 0.15)

train_x,test_x,train_y,test_y = train_test_split(processed_data, target, test_size = 0.3)

model.fit(train_x,train_y,eval_set = (test_x,test_y))
predict = model.predict(test_x)

f1_scre  = f1_score(test_y,predict)*100

accura = accuracy_score(test_y, predict)*100

print('F1 Score for Cat Boost Algorithm is :', f1_scre)

print('Accuracy of Cat Boost Algorithm is:', accura)

cma =confusion_matrix(test_y, predict)

sns.heatmap(cma, annot = True, fmt = '.2f', annot_kws = {'size': 16}, cmap = 'Blues')