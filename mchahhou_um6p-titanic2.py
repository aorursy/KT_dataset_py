from __future__ import division

import pandas as pd

import numpy as np

np.random.seed(0)



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



from sklearn.preprocessing import StandardScaler, MinMaxScaler
df = pd.read_csv('../input/train.csv')

sub = pd.read_csv('../input/test.csv')

sub['Survived'] = -99



df = df.append(sub).reset_index(drop=True)





df['Age'].fillna(-99, inplace=True)

df['Fare'].fillna(-99, inplace=True)



lb = LabelEncoder()

lb.fit( df['Sex'] )

df['Sex'] = lb.transform( df['Sex'] )





sub = df.loc[df.Survived==-99].reset_index(drop=True)

df  = df.loc[df.Survived!=-99].reset_index(drop=True)



                         

                         

                         
sub = sub[['Fare', 'SibSp', 'Parch', 'Age', 'Sex']].copy()

X = df[['Fare', 'SibSp', 'Parch', 'Age', 'Sex']].copy()

y = df['Survived'].values




sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)



clf = DecisionTreeClassifier()



parameters_dc = { 'max_depth' : range(1,20),

                  'min_samples_leaf': range(2,20)}



gs = GridSearchCV(estimator=clf, param_grid=parameters_dc, cv=sk )

gs.fit(X , y  )

print ( gs.best_params_ )

print ("ERROR : ", 1-gs.best_score_)
results_df = pd.DataFrame()

fold=0



for train_index, test_index in sk.split(X, y):

    X_train = X.loc[train_index]

    y_train = y[train_index]



    X_test = X.loc[test_index]

    y_test = y[test_index]



    model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=2)

    model.fit( X_train,  y_train)



    pred_sub   = model.predict_proba(sub)[:,1]

   

    results_df['fold_'+str(fold)] = pred_sub

    

    fold +=1
#preds = (results_df.sum(axis=1) >=3).astype(int)



preds = (results_df.mean(axis=1) >=0.5).astype(int)
my_final_sub = pd.read_csv('../input/test.csv')[['PassengerId']]

my_final_sub['Survived'] = preds



my_final_sub.to_csv('submission.csv', index=False)

#pd.read_csv('../input/gender_submission.csv').head()
my_final_sub.head()