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
import pandas as pd

import numpy as np

import seaborn as sns

import altair as alt

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV 
train = pd.read_csv("../input/titanic/train.csv") 

test = pd.read_csv("../input/titanic/test.csv")
train['flag'] = 1

test['flag'] = 0

df = pd.concat([train,test])

df
df[df['flag']==1].info()
source = df[df['flag']==1]



alt.Chart(source).mark_bar().encode(

    alt.X("Age:Q", bin=True),

    y='count()',

)
alt.Chart(source).mark_bar().encode(

    alt.X("SibSp:Q", bin=True),

    y='count()',

)
alt.Chart(source).mark_bar().encode(

    alt.X("Parch:Q", bin=True),

    y='count()',

)
alt.Chart(source).mark_bar().encode(

    alt.X("Fare:Q", bin=True),

    y='count()',

)
df['norm_fare'] = np.log(df.Fare+1)
source = df[df['flag']==1]
alt.Chart(source).mark_bar().encode(

    alt.X("norm_fare:Q", bin=True),

    y='count()',

)
df[df['flag']==1].Cabin.values
df[df['flag']==1].describe()
df['cabin_adv'] = df.Cabin.apply(lambda x: str(x)[0])
df.Age = df.Age.fillna(df.Age.median())
df.info()
df.dropna(subset=['Embarked'],inplace = True)
df[df['flag']==1].info()
plt.figure(figsize=(10,10))

sns.heatmap(df[df['flag']==1].corr(), annot=True)
df['numeric_ticket'] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

df['ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
df[df['flag']==1].info()
base_model = pd.concat([df,pd.get_dummies(df[['Sex','Embarked','cabin_adv']])],axis=1) 

base_model.columns
df[df['flag']==1].info()
col = ['Survived', 'Pclass', 'Age', 'SibSp',

       'Parch',

       'numeric_ticket', 'norm_fare', 'Sex_female',

       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'cabin_adv_A',

       'cabin_adv_B', 'cabin_adv_C', 'cabin_adv_D', 'cabin_adv_E',

       'cabin_adv_F', 'cabin_adv_G', 'cabin_adv_T', 'cabin_adv_n','flag']
df_final = base_model[col].copy()
plt.figure(figsize=(15,15))

sns.heatmap(df_final.corr(), annot=True)
df_final.drop(columns=["Embarked_C",'Sex_female','Pclass'], inplace=True)
df_final[df_final['flag']==1].info()
source = df_final[df_final['flag']==1]



alt.Chart(source).mark_bar().encode(

    alt.X("Age:Q", bin=True),

    y='count()',

)
df_final[df_final['flag']==1].describe()


scale = StandardScaler()



train_scaled = df_final.copy()



scale.fit(train_scaled[['Age','SibSp','Parch','norm_fare']])



train_scaled[['Age','SibSp','Parch','norm_fare']] = scale.transform(train_scaled[['Age','SibSp','Parch','norm_fare']])

train_scaled.info()
X = train_scaled[train_scaled['flag']==1].drop(columns=['Survived','flag']).copy()

X_predict = train_scaled[train_scaled['flag']==0].drop(columns=['Survived','flag']).copy()

y = train_scaled[train_scaled['flag']==1].Survived.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 456)
def clf_performance(classifier, model_name):

    print(model_name)

    print('Best Score: ' + str(classifier.best_score_))

    print('Best Parameters: ' + str(classifier.best_params_))

    

#função utilizando o tutorial do Kaggle
xgb = XGBClassifier(random_state = 1)



param_grid = {

    'n_estimators': [450,500,550],

    'colsample_bytree': [0.75,0.8,0.85],

    'max_depth': [None],

    'reg_alpha': [1],

    'reg_lambda': [2, 5, 10],

    'subsample': [0.55, 0.6, .65],

    'learning_rate':[0.5],

    'gamma':[.5,1,2],

    'min_child_weight':[0.01],

    'sampling_method': ['uniform']

}



clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True)

best_clf_xgb = clf_xgb.fit(X_train,y_train)

clf_performance(best_clf_xgb,'XGB')
lr = LogisticRegression()

param_grid = {'max_iter' : [2000],

              'penalty' : ['l1', 'l2'],

              'C' : np.logspace(-4, 4, 20),

              'solver' : ['liblinear']}



clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True)

best_clf_lr = clf_lr.fit(X_train,y_train)

clf_performance(best_clf_lr,'Logistic Regression')
knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : [3,5,7,9,15],

              'weights' : ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree','kd_tree'],

              'p' : [1,2]}

clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True)

best_clf_knn = clf_knn.fit(X_train,y_train)

clf_performance(best_clf_knn,'KNN')
svc = SVC(probability = True)

param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],

                                  'C': [.1, 1, 10, 100, 1000]},

                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},

                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]

clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True)

best_clf_svc = clf_svc.fit(X_train,y_train)

clf_performance(best_clf_svc,'SVC')
rf = RandomForestClassifier(random_state = 1)

param_grid =  {'n_estimators': [400,450,500],

               'criterion':['gini','entropy'],

                                  'bootstrap': [True],

                                  'max_depth': [15, 20],

                                  'max_features': ['auto','sqrt', 10],

                                  'min_samples_leaf': [2,3],

                                  'min_samples_split': [2,3]}

                                  

clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True)

best_clf_rf = clf_rf.fit(X_train,y_train)

clf_performance(best_clf_rf,'Random Forest')
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = best_clf_xgb.predict(X_test)



accuracia_modelo = accuracy_score(y_test, y_pred)

mastriz = confusion_matrix(y_test, y_pred)
print(f'O score do modeleo nos dados de teste foi: {(accuracia_modelo*100):.2f}%')

print(f'A matriz de confusão:')

print(f'{mastriz}')
confusion_matrix(y_train, best_clf_svc.predict(X_train))
y_submit = best_clf_xgb.predict(X_predict).astype(int)
submission = pd.concat([test['PassengerId'],pd.DataFrame(y_submit)],axis=1)
submission.rename(columns={0:"Survived"},inplace=True)
submission.to_csv('submission.csv', index =False)