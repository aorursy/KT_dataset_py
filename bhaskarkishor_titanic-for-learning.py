# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import sklearn

        

# Any results you write to the current directory are saved as output.
#plotting and visualization

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

from xgboost import XGBClassifier



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
path = "/kaggle/input/titanic/"

train = pd.read_csv(path+"train.csv")

test = pd.read_csv(path+"test.csv")

submission = pd.read_csv(path+"gender_submission.csv")
train.describe()
test.describe()
train.isnull().sum()
test.isnull().sum()
#preprocessing



data = [train,test]

for df in data:

    df["Age"].fillna(df['Age'].median(),inplace=True)

    df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

    df["Fare"].fillna(df["Fare"].mean(),inplace=True)

    

    drop_col = ['PassengerId','Cabin', 'Ticket','Name']

    df.drop(drop_col,axis=1,inplace=True)

    

train.isnull().sum()

test.isnull().sum()
#encoding

label = LabelEncoder()

for df in data:

    df['Sex_code'] = label.fit_transform(df['Sex'])

    df['Embarked_Code'] =label.fit_transform(df['Embarked'])

    #df['Title_Code'] = label.fit_transform(df['Title'])

    df['Age_Code'] = label.fit_transform(df['Age'])

    #df['Fare'] =label.fit_transform(df['Fare_Code'])

        

    old_col = ['Sex','Embarked','Age']

    df.drop(old_col,axis=1,inplace=True)

train.info()

test.info()
features = ['Pclass','SibSp','Parch','Fare','Sex_code','Embarked_Code','Age_Code']

target = 'Survived'

X = train[features]

y= train[target]

train_X,valid_X,train_y,valid_y = model_selection.train_test_split(X,y,test_size=0.2,random_state=0)

print(train_X.info())

print(valid_X.info())
train_y.describe()
train_full = pd.concat([train_X,train_y],axis=1)



for x in features:

    if train_full[x].dtype != 'float64':

        print("survival correlation:",x)

        print(train_full[[x,'Survived']].groupby(x, as_index=False).mean())

        print('-'*10)




MLA =[

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    

    gaussian_process.GaussianProcessClassifier(),

    

    linear_model.LogisticRegressionCV(),

    #linear_model.PassiveAgressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #tree.DecissionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    XGBClassifier()

    

]



target = ['Survived']

cv_split =model_selection.ShuffleSplit(n_splits=10,test_size=0.3,train_size=0.6,random_state=0)



MLA_cols = ['MLA name','MLA parameter','MLA train accuracy mean','MLA test accuracy mean','MLA time']

MLA_predict = train_full[target]

MLA_compare = pd.DataFrame(columns = MLA_cols)

row_index =0



for alg in MLA:

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA name'] = MLA_name

    MLA_compare.loc[row_index,'MLA parameter'] = str(alg.get_params())

    

    cv_results =model_selection.cross_validate(alg,train_full[features],train_full[target],cv = cv_split)

    

    MLA_compare.loc[row_index,'MLA index']=cv_results['fit_time'].mean()

    #MLA_compare.loc[row_index,'MLA train accuracy mean']  = cv_results['train_score'].mean()

    MLA_compare.loc[row_index,'MLA test accuracy mean'] = cv_results['test_score'].mean()

    alg.fit(train_full[features],train_full[target])

    MLA_predict[MLA_name] =alg.predict(train_full[features])

    row_index+=1

    

    MLA_compare.sort_values(by = ['MLA test accuracy mean'],ascending = False,inplace=True)

    MLA_compare
#MLA_compare
from sklearn.ensemble import RandomForestClassifier





model = RandomForestClassifier()

prediction = model.fit(train_X[features],train_y)

values = prediction.predict(valid_X[features])

from sklearn.metrics import accuracy_score



score = accuracy_score(valid_y,values)

score
from sklearn.model_selection import cross_val_score



cvscores = cross_val_score(model,train[features],train[target],cv=10)

cvscores.mean()


scorecard = []

estimators = [5,10,50,100,500,1000]

for i in estimators:

    modelX = RandomForestClassifier(random_state=1,n_estimators = i)

    scores = cross_val_score(modelX,train[features],train[target],cv=10)

    scorecard.append(scores.mean())

for i in range(len(scorecard)):

    print(scorecard[i])
scorecard = []

samplesplit = []

for i in range(2,10):

    modelX = RandomForestClassifier(random_state=1,n_jobs=5,n_estimators = 10,min_samples_split=i)

    scores = cross_val_score(modelX,train[features],train[target],cv=10)

    scorecard.append(scores)

for i in range(len(scorecard)):

    print(f'features - {i+2} : {scorecard[i].mean()} - {scorecard[i].std()} - {scorecard[i].var()}')
scorecard = []

maxfeatures = []

for i in range(1,7):

    modelX = RandomForestClassifier(random_state=1,n_jobs=5,n_estimators = 10,min_samples_split=5,max_features = i)

    scores = cross_val_score(modelX,train[features],train[target],cv=10)

    scorecard.append(scores)





for i in range(len(scorecard)):

    print(f'features - {i+1} : {scorecard[i].mean()} - {scorecard[i].std()} - {scorecard[i].var()}')
scorecard = []

maxfeatures = []

for i in range(1,10):

    modelX = RandomForestClassifier(random_state=1,n_jobs=5,n_estimators = 10,min_samples_split=5,max_features = 3,max_depth = i)

    scores = cross_val_score(modelX,train[features],train[target],cv=10)

    scorecard.append(scores)





for i in range(len(scorecard)):

    print(f'features - {i+1} : {scorecard[i].mean()} - {scorecard[i].std()} - {scorecard[i].var()}')
features


modelFinal = RandomForestClassifier(random_state=1,n_jobs=5,n_estimators = 10,min_samples_split=5,max_features = 3,max_depth = 7)

modelFinal.fit(train[features],train[target])

values = modelFinal.predict(test[features])

print(values[:5])
submission.head(10)
test['Survived'] = values

submission['Survived'] = values

submission.to_csv('submission.csv',index=False)

print("done")

submission.head(10)