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



# Any results you write to the current directory are saved as output.
#importing other necessary modules for visualization and label-encoding

import seaborn as sns

import matplotlib.pyplot as plt



#warning suppressing section

import warnings

warnings.filterwarnings('ignore')
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
data1 = data_train.copy()

combo = [data1, test]

print(data1.isnull().sum())

print(data1.describe())
print(test.isnull().sum())

print(test.describe())
data1.head(10)
sns.boxplot(data1.Age)

sns.boxplot(data1.Fare)
for i,d in enumerate(combo):

    d.Age.fillna(d.Age.median(), inplace=True)

    d.Embarked.fillna(d.Embarked.mode()[0], inplace=True)

    d.Fare.fillna(d.Fare.median(),inplace=True)

    d = d.drop(columns=['Cabin','PassengerId','Ticket'], axis=1)              

    print(d.isnull().sum())

    print("-"*20)

    combo[i]=d
data1=combo[0]

data1.groupby('Sex')['Survived'].value_counts(normalize=True)

sns.set_style('darkgrid')

sns.distplot(data1.Age,hist=True,kde=True,bins=25)
print("Total passengers within age group below Five:",data1[data1.Age<5]['Survived'].count())

print("Survived:",data1[(data1.Age<5)&(data1.Survived==1)]['Survived'].count())
data1['title']=d['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]

data1.title.value_counts()
for i,d in enumerate(combo):

    d = d.copy()

    d['family_size']= d['Parch']+d['SibSp']+1

    #name_breakdown

    d['title']=d['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]

    title_tags = d.title.value_counts() <10

    d['title']=d.title.apply(lambda x:'Misc' if title_tags.loc[x]==True else x)

    combo[i]= d
for i,d in enumerate(combo):

    d= d.copy()

    d['Farebin']=pd.qcut(d.Fare.astype('int'), 4)

    d['Agebin']=pd.cut(d.Age.astype('int'), 5)

    combo[i]=d
#label encoding for the train and test data

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

for i,d in enumerate(combo):

    d=d.copy()

    d['sex_code']=enc.fit_transform(d.Sex)

    d['embarked_code']=enc.fit_transform(d.Embarked)

    d['title_code']=enc.fit_transform(d.title)

    d['agebin_code']=enc.fit_transform(d.Agebin)

    d['farebin_code']=enc.fit_transform(d.Farebin)

    combo[i]=d

data_enc = combo[0]

print(data_enc.columns)
#dummy variable creation

y_feature=['Survived']

X_ft=data_enc.columns

X_ft_copy = X_ft.copy()

X_ft_copy = X_ft_copy.drop(['Survived', 'Name','Farebin','Agebin'])

X_dummy = pd.get_dummies(data_enc[X_ft_copy])

print(X_ft_copy)

print(X_dummy.columns)
X_aug_y = X_dummy; X_aug_y['y']= data1['Survived']

plt.figure(figsize=(15,10))

sns.heatmap(X_aug_y.corr(), annot=True, cmap='coolwarm')
to_drop = ['SibSp','Parch','Sex_female','Sex_male','Embarked_Q','embarked_code','title_Misc','y']

X_dummy = X_dummy.drop(to_drop, axis=1)

#X_dummy.columns

y=data1['Survived']
#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbc

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score



def evaluate(estimator, data, features, performCV=True, printFeatureImportance=True, cv=5):

    #Fit the algorithm on the data

    estimator.fit(data[features], y)

        

    #Predict training set:

    predictions = estimator.predict(data[features])

    predprob = estimator.predict_proba(data[features])[:,1]

    

    #Perform cross-validation:

    if performCV:

        cv_score = cross_val_score(estimator, data[features], y, cv=cv, scoring='roc_auc')

    

    #Print model report:

    print("Model Report")

    print("Accuracy : {}".format(accuracy_score(y,predictions)))

    print("F1 Score : {}".format(f1_score(y,predictions)))

    print("AUC Score (Train): {}".format(roc_auc_score(y, predprob)))

    

    if performCV:

        print("CV Score : Mean - {:.7f} | Std - {:.7f} | Min - {:.7f} | Max - {:.7f}" \

              .format(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        

    #Print Feature Importance:

    if printFeatureImportance:

        feat_imp = pd.Series(estimator.feature_importances_, features).sort_values(ascending=False)

        feat_imp.plot(kind='bar',title='Feature Importances')

        plt.ylabel('Feature Importance Score')
#baseline gbm model

features = X_dummy.columns

gbm0=gbc(random_state=20)

evaluate(gbm0, X_dummy, features)
#tuning1:

gbm1=gbc(min_samples_split=10,min_samples_leaf=5,max_depth=3, max_features='sqrt', subsample=0.8, random_state=20)

param_tune1 = {'n_estimators':range(10,51,10)}

gridCV1=GridSearchCV(estimator=gbm1, param_grid=param_tune1, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)

gridCV1.fit(X_dummy,y)

print(gridCV1.best_params_, gridCV1.best_score_)
#tuning2:

gbm2=gbc(n_estimators=40,min_samples_leaf=5,random_state=20,subsample=0.8)

param_tune2 = {'max_depth':range(2,11),

               'min_samples_split':range(5,20),

                'max_features':['auto','sqrt']}

gridCV2 = GridSearchCV(estimator=gbm2,param_grid=param_tune2, scoring='roc_auc',iid=False, n_jobs=-1,cv=5)

gridCV2.fit(X_dummy,y)

print(gridCV2.best_params_, gridCV2.best_score_)
#tuning3:

gbm3=gbc(n_estimators=40,max_depth=3,max_features='sqrt',random_state=20,subsample=0.8)

param_tune3 = {'min_samples_split':range(20,81),

                'min_samples_leaf':range(5,31)}

gridCV3 = GridSearchCV(estimator=gbm3,param_grid=param_tune3, scoring='roc_auc',iid=False, n_jobs=-1,cv=5)

gridCV3.fit(X_dummy,y)

print(gridCV3.best_params_, gridCV3.best_score_)

#evaluation based on current parameter

gbm4=gbc(n_estimators=40,min_samples_split=58, min_samples_leaf=5, max_features='sqrt',

        max_depth=3,subsample=0.8, random_state=20)

evaluate(gbm4, X_dummy, features)
gbm5 = gbc(min_samples_split=58, min_samples_leaf=5, max_features='sqrt',

          max_depth=3, subsample=0.8, random_state=20)

param_tune4 = {'learning_rate':np.arange(0.01,0.1,0.01),

              'n_estimators':range(40,400,10)}

gridCV4 = GridSearchCV(estimator=gbm5,param_grid=param_tune4, scoring='roc_auc',iid=False, n_jobs=-1,cv=5)

gridCV4.fit(X_dummy, y)

print(gridCV4.best_params_, gridCV4.best_score_)

gbm6=gbc(learning_rate=0.02,n_estimators=220, min_samples_split=58, min_samples_leaf=5, max_features='sqrt',

          max_depth=3, subsample=0.8, random_state=20)

#param_tune4={'n_estimators':range(100,1001,100)}

#gridCV4=GridSearchCV(estimator=gbm6, param_grid=param_tune4, scoring='roc_auc', iid=False, n_jobs=-1,

                    #cv=5)

#gridCV4.fit(X_dummy, y)

evaluate(gbm6, X_dummy, features)
test1=combo[1]

#test1.head(10)

ft=test1.columns

ft_copy = ft.copy()

ft_copy = ft_copy.drop(['Name','Farebin','Agebin'])

X_test = pd.get_dummies(test1[ft_copy])
drop = ['SibSp','Parch','Sex_female','Sex_male','Embarked_Q','embarked_code','title_Misc']

X_test=X_test.drop(drop, axis=1)

print(X_dummy.columns)

print(X_test.columns)

#Prediction of Survival Labels

test['Survived'] = gbm6.predict(X_test)

data = {'PassengerId':test.PassengerId.values,

                          'Survived':test.Survived.values}

submission = pd.DataFrame(data)

submission.to_csv('submission.csv', index=False)
test.groupby('Sex')['Survived'].value_counts(normalize=True)

print("Total passengers within age group below Five:",test[test.Age<5]['Survived'].count())

print("Survived:",test[(test.Age<5)&(test.Survived==1)]['Survived'].count())