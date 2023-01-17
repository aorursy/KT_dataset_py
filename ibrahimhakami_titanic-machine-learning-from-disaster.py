import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import seaborn as sns



plt.style.use('fivethirtyeight')



%matplotlib inline

%config InlineBackend.figure_format = 'retina'



# Lines below are just to ignore warnings

import warnings

warnings.filterwarnings('ignore')
# loading training dataset

train = pd.read_csv('../input/titanic/train.csv')
# take a look at the head of the training data set

train.head()
# shape of training dataset

train.shape
# loading test dataset

test = pd.read_csv('../input/titanic/test.csv')
# take a look at the head 

test.head()
# the shape of the test set

test.shape
# merging both dataset to clean both at once, also to get most accurate filling results

df = train.merge(test , how='outer')

df.head()
# checking for nulls in all columns

df.info()
# we can see that the male have high correlation with 0, we can see the gender is somewhat strong predictor

sns.countplot(train['Survived'] , hue = train['Sex'] , orient='v',palette='ocean')
# Pclass 3 consider strong predictor on 0

sns.countplot(train['Survived'] , hue = train['Pclass'] , orient='v',palette='ocean')
# Embarked S also have high corrlation with 0

sns.countplot(train['Survived'] , hue = train.Embarked , orient='v',palette='ocean_r')
# missing values plotting

fig, ax = plt.subplots(figsize = (12, 6))





sns.heatmap(df.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='cividis')

plt.title("Null in the Data", fontsize =15)

plt.xticks(rotation=45)

plt.show()
# checking for nulls

df.isnull().sum()
# we can see 'S' is the most frequent

df.Embarked.value_counts()
# filling with the most frequent Embarked

df.Embarked.fillna('S' , inplace = True)
df.Pclass.value_counts()
df[df.Pclass == 3]['Fare'].mean()
# filling with the mean of Fare of the most frequent Pclass to get more accurate fill

df.Fare.fillna(df[df.Pclass == 3]['Fare'].mean() , inplace =True)
# checking the mean of age in different Pclass

df[['Pclass' , 'Age']].groupby('Pclass').mean()
#defining a function 'impute_age'

def impute_age(age_pclass): # passing age_pclass as ['Age', 'Pclass']

    

    # Passing age_pclass[0] which is 'Age' to variable 'Age'

    Age = age_pclass[0]

    

    # Passing age_pclass[2] which is 'Pclass' to variable 'Pclass'

    Pclass = age_pclass[1]

    

    #applying condition based on the Age and filling the missing data respectively 

    if pd.isnull(Age):



        if Pclass == 1:

            return 38



        elif Pclass == 2:

            return 30



        else:

            return 25



    else:

        return Age
# filling the appropirate age that coresponde to the Pclass

df.Age = df.apply(lambda x :impute_age(x[['Age', 'Pclass']] ) , axis = 1)
df.Cabin.head()
# using the initials of cabin to get class that have correlation with the cabin location

df.Cabin = df.Cabin.astype(str).str[0]
df.Cabin.value_counts()
# checking missing after the fillings

fig, ax = plt.subplots(figsize = (12, 6))



sns.heatmap(df.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='cividis')

plt.title("Null in the Data", fontsize =15)

plt.xticks(rotation=45)

plt.show()
df.isnull().sum()
# creating new feature called familysize

df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1
# creating new feature called is alone 

df['IsAlone'] = df['FamilySize'].apply(lambda x:1 if x==1 else 0)
# take a look at the names

df[['Name']].head()
# creat new column with the title of people

df['Title'] = df['Name'].str.split(',' , expand=True)[1].str.split('.', expand=True)[0]

df['Title'].value_counts()
# drop columns that will not give information

df.drop(columns=['PassengerId' , 'Name' , 'Ticket' , 'SibSp' , 'Parch' ] , inplace=True)
# to get the train data from the meged data set we can use iloc and get all columns, while rows equal to the shape of the train[0]

df.iloc[:train.shape[0],:].head()
# creating dummy variables for all categorical variables in the cleaned and merged dataset

df_d = pd.get_dummies(df , drop_first=True)

df_d.shape
#Survived correlation matrix

corrmat = abs(df_d.iloc[:train.shape[0],:].corr())

plt.figure(figsize=(12, 8))

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Survived')['Survived'].index

cm = np.corrcoef(df_d.iloc[:train.shape[0],:][cols].values.T)

sns.set(font_scale=1.00)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,

                 cmap = 'Blues', linecolor = 'white', linewidth = 1)

plt.title("Correlations between Survived and features including dummy variables", fontsize =15)

plt.xticks(rotation=45)

plt.show()
# getting the target (Survived) column as y

y=pd.DataFrame(df_d.pop('Survived'))
# checking the shape of the train data set, to know from where to cut data set to get training data useing iloc

train.shape[0]
# using iloc on both the target and training data we can get an exact seperation between training and testing datasets

X_train = df_d.iloc[:train.shape[0] , :]

y_train = y.iloc[:train.shape[0]]
# checking train dataset shape to be sure of the correct seperation

print(X_train.shape , y_train.shape)
 # importing test/train split, and use it on training dataset to train the models and score them 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3 , random_state = 101)

# importing scaler, then scale the training and test dataframes

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import RobustScaler



s = StandardScaler()

r = RobustScaler()



X_train_d_s = pd.DataFrame(s.fit_transform(X_train) , columns=X_train.columns)

X_test_d_s = pd.DataFrame(s.transform(X_test) , columns=X_test.columns)
#importing models 

from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
# run different models with default parameters

random_state = 35

model_names = ['LogisticRegression', 'DecisionTreeClassifier',

             'RandomForestClassifier','ExtraTreesClassifier'

             , 'GradientBoostingClassifier','AdaBoostClassifier']

models = [ ('LogisticRegression',LogisticRegression(random_state=random_state)),

         ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=random_state)),

         ('RandomForestClassifier',RandomForestClassifier(random_state=random_state)),

         ('ExtraTreesClassifier',ExtraTreesClassifier(random_state=random_state)),

         ('GradientBoostingClassifier',GradientBoostingClassifier(random_state=random_state)),

         ('AdaBoostClassifier',AdaBoostClassifier(random_state=random_state))

        ]

model_accuracy = []

print ('fitting...')

for k,model in models:



    model.fit(X_train, y_train)

    accuracy = cross_val_score(model, X_train_d_s, y_train, cv=5).mean()

    model_accuracy.append(accuracy)

print('Completed')

# creating dataframe that contain model name and accuracy

Models = pd.concat([pd.Series(model_names), pd.Series(model_accuracy)], axis=1).sort_values(by=1, ascending=False)


Models.rename(columns={0:'model_name',

                      1:'accuracy'}, inplace=True)

Models
# ploting accuracy with model names

a = sns.barplot(Models.accuracy, Models.model_name,palette='ocean_r')

a.set_xlim(0.7,1)

plt.title("accuracy for each model", fontsize =15)

plt.show()
# creating empty list to append each name and accuracy of all models used then compare

O_model_accuracy = []

O_model_name = []
# using grid search on GradientBoostingRegressor model to get the best hyperparameters



param_grid = {'learning_rate': [0.01 ],

 'max_depth': [3 ],

 'max_features': ['auto'],

 'min_samples_leaf': [15 ],

 'min_samples_split': [15],

 'n_estimators': [1500]}



grad = GridSearchCV(GradientBoostingClassifier(),

                           param_grid, cv=5, verbose= 1 , n_jobs=-1)

grad.fit(X_train_d_s , y_train)
grad.best_params_
# getting train score

grad.score(X_train_d_s , y_train)
# getting test scores

grad.score(X_test_d_s , y_test)
O_model_accuracy.append(grad.score(X_test_d_s , y_test))

O_model_name.append('GradientBoostingClassifier')
# confusion matrix, classification report

print(f'confusion matrix for RF\n{confusion_matrix(y_test,grad.predict(X_test_d_s))}\n classification report for RF \n {classification_report(y_test,grad.predict(X_test_d_s))}')
# using RandomGridSerach to find best hyperparametrs for RandomForestRegressor



par = {'bootstrap': [True],

 'max_depth': [20],

 'max_features': ['auto'],

 'min_samples_leaf': [2 ],

 'min_samples_split': [5 ],

 'n_estimators': [ 800 ]}



ra = GridSearchCV(RandomForestClassifier(),

                   par , cv = 5 , verbose= 1  , n_jobs= -1)

ra.fit(X_train_d_s , y_train)
ra.best_params_
# getting train score

ra.score(X_train_d_s , y_train)
# getting test scores

ra.score(X_test_d_s , y_test)
O_model_accuracy.append(ra.score(X_test_d_s , y_test))

O_model_name.append('RandomForestClassifier')
# confusion matrix, classification report

print(f'confusion matrix for RF\n{confusion_matrix(y_test,ra.predict(X_test_d_s))}\n classification report for RF \n {classification_report(y_test,ra.predict(X_test_d_s))}')
# using RandomGridSerach  to fide best hyperparametrs for RandomForestRegressor



par = {'bootstrap': [True],

 'max_depth': [15],

 'max_features': ['auto'],

 'min_samples_leaf': [2],

 'min_samples_split': [5],

 'n_estimators': [100]}



ex = GridSearchCV(ExtraTreesClassifier(),

                   par , cv = 5 , verbose= 1  , n_jobs= -1)

ex.fit(X_train_d_s , y_train)
ex.best_params_
ex.best_score_
ex.score(X_train_d_s , y_train)
ex.score(X_test_d_s , y_test)
O_model_accuracy.append(ex.score(X_test_d_s , y_test))

O_model_name.append('ExtraTreesClassifier')
# confusion matrix, classification report

print(f'confusion matrix for RF\n{confusion_matrix(y_test,ex.predict(X_test_d_s))}\n classification report for RF \n {classification_report(y_test,ex.predict(X_test_d_s))}')
# SVC 

param_grid = {'C': [1000 , 10000],  

              'gamma': [ 0.001,0.01], 

              'kernel': [  'rbf'],}  

  

sv = GridSearchCV(SVC(), param_grid, verbose = 3 , n_jobs=-1 , cv = 5) 

  

# fitting the model for grid search 

sv.fit(X_train_d_s , y_train)
sv.best_params_
sv.score(X_train_d_s , y_train)
sv.score(X_test_d_s , y_test)
O_model_accuracy.append(sv.score(X_test_d_s , y_test))

O_model_name.append('SVC')
# Cconfusion matrix, classification report

print(f'confusion matrix for RF\n{confusion_matrix(y_test,sv.predict(X_test_d_s))}\n classification report for RF \n {classification_report(y_test,sv.predict(X_test_d_s))}')
# LogisticRegression

penalty = ['l1', 'l2']

C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]

solver = ['liblinear', 'saga']



param_grid = dict(penalty=penalty,

                  C=C,

                  class_weight=class_weight,

                  solver=solver)



lo = GridSearchCV(estimator=LogisticRegression(),

                    param_grid=param_grid,

                    scoring='roc_auc',

                    verbose=1,

                    n_jobs=-1 ,

                 cv = 5)

lo.fit(X_train_d_s, y_train)
lo.best_score_
lo.best_estimator_.score(X_train_d_s , y_train)
lo.best_estimator_.score(X_test_d_s , y_test)
O_model_accuracy.append(lo.best_estimator_.score(X_test_d_s , y_test))

O_model_name.append('LogisticRegression')
# confusion matrix, classification report

print(f'confusion matrix for RF\n{confusion_matrix(y_test,lo.predict(X_test_d_s))}\n classification report for RF \n {classification_report(y_test,lo.predict(X_test_d_s))}')
# AdaBoostClassifier 



par={'n_estimators':[500,1000,2000],

             'learning_rate':[.001,0.01,.1]}

ada=GridSearchCV(estimator=AdaBoostClassifier()

                    ,param_grid=par,

                    n_jobs=-1,cv=5 , verbose= 2)

ada.fit(X_train_d_s, y_train)
ada.best_score_
ada.score(X_train_d_s , y_train)
ada.score(X_test_d_s , y_test)
O_model_accuracy.append(ada.score(X_test_d_s , y_test))

O_model_name.append('AdaBoostClassifier')
# confusion matrix, classification report

print(f'confusion matrix for RF\n{confusion_matrix(y_test,ada.predict(X_test_d_s))}\n classification report for RF \n {classification_report(y_test,ada.predict(X_test_d_s))}')
from sklearn.ensemble import VotingClassifier
# Ensembling several models

model1 = LogisticRegression(C=0.1, class_weight={0: 0.6, 1: 0.4}, dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',

                   random_state=None, solver='saga', tol=0.0001, verbose=0,

                   warm_start=False)

model2 = DecisionTreeClassifier()

model3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=30, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=2, min_samples_split=5,

                       min_weight_fraction_leaf=0.0, n_estimators=100,

                       n_jobs=None, oob_score=False,

                       verbose=0, warm_start=False,random_state=42)

model4 = ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini',

                     max_depth=20, max_features='auto', max_leaf_nodes=None,

                     min_impurity_decrease=0.0, min_impurity_split=None,

                     min_samples_leaf=2, min_samples_split=5,

                     min_weight_fraction_leaf=0.0, n_estimators=100,

                     n_jobs=None, oob_score=False, random_state=None, verbose=0,

                     warm_start=False)

model5 = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,

                   learning_rate=0.001, n_estimators=2000, random_state=None)

model6 = GradientBoostingClassifier(criterion='friedman_mse', init=None,

                           learning_rate=0.02, loss='deviance', max_depth=3,

                           max_features='sqrt', max_leaf_nodes=None,

                           min_impurity_decrease=0.0, min_impurity_split=None,

                           min_samples_leaf=15, min_samples_split=15,

                           min_weight_fraction_leaf=0.0, n_estimators=3000,

                           n_iter_no_change=None, presort='auto',

                           subsample=1.0, tol=0.0001,

                           validation_fraction=0.1, verbose=0,

                           warm_start=False,random_state=random_state)

model7 = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)

model8 = GaussianNB()



evc = VotingClassifier(estimators=[('lr', model1), ('dt', model2),

                                     ('rf', model3),('et', model4),

                                     ('adb', model5),('gb', model6),

                                     ('svc', model7),('gnb', model8)],n_jobs=-1)
evc.fit(X_train_d_s,y_train)
evc.score(X_train_d_s,y_train)
evc.score(X_test_d_s,y_test)
O_model_accuracy.append(evc.score(X_test_d_s,y_test))

O_model_name.append('VotingClassifier')
Models = pd.concat([pd.Series(O_model_name), pd.Series(O_model_accuracy)], axis=1).sort_values(by=1, ascending=False)
Models.rename(columns={0:'model_name',

                      1:'accuracy'}, inplace=True)

Models
a = sns.barplot(Models.accuracy, Models.model_name,palette='ocean_r')

a.set_xlim(0.75,0.9)

plt.title("Recomended Models", fontsize =15,)

plt.show()

coef_df = pd.DataFrame({'feature': X_train_d_s.columns,

                        'importance': abs(ra.best_estimator_.feature_importances_), 

                        })



coef_df.head()
# sort by absolute value of coefficient (magnitude)

coef_df.sort_values('importance', ascending=False, inplace=True)

coef_df[:10]
# top features selected by model

plt.xticks(rotation=45)

sns.barplot(coef_df.feature[:7] , coef_df.importance[:7],palette='ocean_r') # top  features

plt.title("Feature imprtance", fontsize =15)

plt.show()
# recreating the training and testing dataset to do the prediction on the testing data

df_d = pd.get_dummies(df  , drop_first=True)

df_d.shape
y=pd.DataFrame(df_d.pop('Survived'))
y_test = y.iloc[train.shape[0]:]

X_test= df_d.iloc[train.shape[0]:,:]

y_train = y.iloc[:train.shape[0]]

X_train = df_d.iloc[:train.shape[0] , :]
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler



s = StandardScaler()

r = RobustScaler()



X_train_d_s = pd.DataFrame(s.fit_transform(X_train) , columns=X_train.columns)

X_test_d_s = pd.DataFrame(s.transform(X_test) , columns=X_test.columns)
# creating the dataframe then save it as csv file before submiting.

sub = pd.DataFrame({

        "PassengerId": test.PassengerId,

        "Survived": evc.predict(X_test_d_s)

})

sub['Survived']= sub['Survived'].astype(int)

sub.head()

sub.to_csv('sub9.csv' , index=False)
pd.read_csv('sub9.csv').head()