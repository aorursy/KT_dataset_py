import pandas as pd

import numpy as np

import pandas_profiling

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import learning_curve

from sklearn.model_selection import validation_curve



import os



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df_copy = train_df.copy()

test_df_copy = test_df.copy()
train_df.head(2)
test_df.head(2)
print('Training Data')

print('\tRows\t',train_df.shape[0])

print('\tColumns\t',train_df.shape[1])

#Train Samples

for col in train_df:

    print(col)

    print('\tDatatype\t:%s '%(train_df[col].dtype))

    print('\tUnique values\t:%s '%(np.size(train_df[col].unique())))  

    print('\tMissing values\t:%d '%(train_df[col].isnull().sum()))
print('Test Data')

print('\tRows\t',test_df.shape[0])

print('\tColumns\t',test_df.shape[1])

#Test Samples

for col in test_df:

    print(col)

    print('\tDatatype\t:%s '%(test_df[col].dtype))

    print('\tUnique values\t:%s '%(np.size(test_df[col].unique())))  

    print('\tMissing values\t:%d '%(test_df[col].isnull().sum()))
#Impute age by median

impute_age = train_df['Age'].median()

train_df['Age'].fillna(impute_age,inplace = True)

test_df['Age'].fillna(impute_age,inplace = True)



#Impute Embarked by mode

impute_embarked = train_df['Embarked'].mode().values[0]

train_df['Embarked'].fillna(impute_embarked,inplace = True)

test_df['Embarked'].fillna(impute_embarked,inplace = True)



#Cabin can be dropped, for now lets create a feature to hold if cabin is available or not

train_df['hasCabin'] = train_df['Cabin'].isnull().astype(int)

test_df['hasCabin'] = test_df['Cabin'].isnull().astype(int)



#fare = 0 means missing value hence needs to be imputed . 

#Fare is based on passanger class. So that can be imputed by median value of fare per class

#Median value wont be impacted by outliers as compared to mean

impute_fare1 = train_df[train_df['Pclass'] == 1]['Fare'].median()

impute_fare2 = train_df[train_df['Pclass'] == 2]['Fare'].median()

impute_fare3 = train_df[train_df['Pclass'] == 3]['Fare'].median()

train_df.loc[((train_df['Fare'] == 0) | (train_df['Fare'].isnull())) & (train_df['Pclass'] == 1),'Fare'] = impute_fare1

train_df.loc[((train_df['Fare'] == 0) | (train_df['Fare'].isnull())) & (train_df['Pclass'] == 2),'Fare'] = impute_fare2

train_df.loc[((train_df['Fare'] == 0) | (train_df['Fare'].isnull())) & (train_df['Pclass'] == 3),'Fare'] = impute_fare3

test_df.loc[((test_df['Fare'] == 0) | (test_df['Fare'].isnull())) & (test_df['Pclass'] == 1),'Fare'] = impute_fare1

test_df.loc[((test_df['Fare'] == 0) | (test_df['Fare'].isnull())) & (test_df['Pclass'] == 2),'Fare'] = impute_fare2

test_df.loc[((test_df['Fare'] == 0) | (test_df['Fare'].isnull())) & (test_df['Pclass'] == 3),'Fare'] = impute_fare3
#Univariate Analysis

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
_ = sns.distplot(train_df['Age'], bins=15, kde=True)
_ = sns.distplot(train_df['Fare'], bins=15, kde=True)
#Bivariate

_ = sns.lmplot(x='Pclass',y='Survived',data=train_df,col='Embarked',hue='Sex')
_ = sns.countplot(x='Survived',data=train_df,hue='Sex')
_ = sns.countplot(x='Survived',data=train_df,hue='Pclass')
_ = sns.countplot(x='Survived',data=train_df,hue='SibSp')
#Outliers

from matplotlib import style

style.use('ggplot')

sns.boxplot(x='Age',data=train_df)
sns.boxplot(x='Fare',data=train_df)
sns.boxplot(x='SibSp',data=train_df)
sns.boxplot(x='Parch',data=train_df)
def detect_outliers(features,n):

    outlier_indices=[]

    for col in features:

        Q1 = np.percentile(train_df[col],25)

        Q3 = np.percentile(train_df[col],75)

        IQR = Q3 - Q1

    

        outlier_indices.extend(train_df[(train_df[col] > (Q3 + (IQR * 1.5)) ) | (train_df[col] < (Q1 - (IQR * 1.5)) )].index)

    

    outlier_indices = Counter(outlier_indices)        

    #multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    multiple_outliers = []

    for k,v in outlier_indices.items():

        if v > n:

            multiple_outliers.append(k)

        

    return multiple_outliers   

outliertodrop = detect_outliers(["Age","SibSp","Parch","Fare"],2)
#Standard scaling of features Age and Fare

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(train_df[['Age']])



train_df['AgeSc'] = sc.transform(train_df[['Age']])

test_df['AgeSc'] = sc.transform(test_df[['Age']])



sc1 = StandardScaler()

sc1.fit(train_df[['Fare']])

train_df['FareSc'] = sc1.transform(train_df[['Fare']])

test_df['FareSc'] = sc1.transform(test_df[['Fare']])
#One hot encoding for Sex, Pclass, Embarked

oheSex = OneHotEncoder(categories='auto')

oheSex.fit(train_df[['Sex']])

f_train = oheSex.transform(train_df[['Sex']]).toarray()

dfSex_train = pd.DataFrame(f_train.astype(int),columns=oheSex.get_feature_names())

train_df = pd.concat([train_df,dfSex_train],axis=1)

f_test = oheSex.transform(test_df[['Sex']]).toarray()

dfSex_test = pd.DataFrame(f_test.astype(int),columns=oheSex.get_feature_names())

test_df = pd.concat([test_df,dfSex_test],axis=1)



#One hot encoding for Pclass

ohePclass = OneHotEncoder(categories='auto')

ohePclass.fit(train_df[['Pclass']])

f_train = ohePclass.transform(train_df[['Pclass']]).toarray()

dfPclass_train = pd.DataFrame(f_train.astype(int),columns=['Pclass_1','Pclass_2','Pclass_3'])

train_df = pd.concat([train_df,dfPclass_train],axis=1)

f_test = ohePclass.transform(test_df[['Pclass']]).toarray()

dfPclass_test = pd.DataFrame(f_test.astype(int),columns=['Pclass_1','Pclass_2','Pclass_3'])

test_df = pd.concat([test_df,dfPclass_test],axis=1)



#One hot encoding for Embarked

oheEmbarked = OneHotEncoder(categories='auto')

oheEmbarked.fit(train_df[['Embarked']])

f_train = oheEmbarked.transform(train_df[['Embarked']]).toarray()

dfEmbarked_train = pd.DataFrame(f_train.astype(int),columns=['EMB_C','EMB_Q','EMB_S'])

train_df = pd.concat([train_df,dfEmbarked_train],axis=1)

f_test = oheEmbarked.transform(test_df[['Embarked']]).toarray()

dfEmbarked_test = pd.DataFrame(f_test.astype(int),columns=['EMB_C','EMB_Q','EMB_S'])

test_df = pd.concat([test_df,dfEmbarked_test],axis=1)
#Features for model

X_ml = train_df[['SibSp' , 'Parch' , 'Fare' , 'hasCabin' , 'AgeSc' , 'FareSc' , 'x0_female' , 'x0_male' , 'Pclass_1' , 'Pclass_2' , 'Pclass_3' , 'EMB_C' , 'EMB_Q' , 'EMB_S']]

y_ml = train_df ['Survived']

X_test1 = test_df[['SibSp' , 'Parch' , 'Fare' , 'hasCabin' , 'AgeSc' , 'FareSc' , 'x0_female' , 'x0_male' , 'Pclass_1' , 'Pclass_2' , 'Pclass_3' , 'EMB_C' , 'EMB_Q' , 'EMB_S']]
#Train Test Split

X_train,X_test,y_train,y_test = train_test_split(X_ml,y_ml)
#kfold cross validation with Logistic Regression with default parameters

n_splits = 10

kfold = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=420)

scores = np.zeros(n_splits)

for k , (train, test) in enumerate(kfold.split(X_train,y_train)):

    #print(train)

    clf_lr = LogisticRegression(penalty='l2')

    clf_lr.fit(X_train.iloc[train],y_train.iloc[train])

    scores[k] = clf_lr.score(X_train.iloc[test],y_train.iloc[test])

    

print('Training Score-->%s'%(scores.mean()))

print('Training Standard Deviation-->%s'%(scores.std()))

print('Logistic Regression Test Accuracy Score --> %s'%(accuracy_score(y_test,clf_lr.predict(X_test)))) 
#parameter tuning for classifier for parameter C

param_range = [0.001,0.01,0.1,1.0,10.0,100.0]

train_score,test_score = validation_curve(clf_lr, 

                                          X = X_train, 

                                          y = y_train, 

                                          param_name = 'C', 

                                          param_range = param_range,  

                                          cv=10, 

                                          n_jobs=1

                                         )



train_mean = np.mean(train_score,axis = 1)

test_mean = np.mean(test_score,axis = 1)



plt.plot(param_range,train_mean,color = 'blue', marker = 'o',label='training accuracy')

plt.plot(param_range,test_mean,color = 'green', marker = 's',label='validation accuracy')

plt.legend(loc = 'best')

plt.xscale('log')

plt.xlabel('Parameter C')

plt.ylabel('Accuracy')
#kfold cross validation with Logistic Regression with Tuned parameters

n_splits = 10

kfold = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=420)

scores = np.zeros(n_splits)

for k , (train, test) in enumerate(kfold.split(X_train,y_train)):

    #print(train)

    clf_lr = LogisticRegression(penalty='l2',C = 10)

    clf_lr.fit(X_train.iloc[train],y_train.iloc[train])

    scores[k] = clf_lr.score(X_train.iloc[test],y_train.iloc[test])

    

print('Training Score-->%s'%(scores.mean()))

print('Training Standard Deviation-->%s'%(scores.std()))

print('Logistic Regression Test Accuracy Score --> %s'%(accuracy_score(y_test,clf_lr.predict(X_test)))) 
#Dignose Bias and variance with Learning curves against multiple samples

train_sizes,train_score,test_score = learning_curve(estimator = clf_lr, 

                                                    X = X_train, 

                                                    y = y_train, 

                                                    train_sizes=np.linspace(0.1  , 1, 10), 

                                                    cv=10, 

                                                    n_jobs=1, pre_dispatch='all', 

                                                    random_state=420)



train_mean = np.mean(train_score,axis = 1)

test_mean = np.mean(test_score,axis = 1)



plt.plot(train_sizes,train_mean,color = 'blue', marker = 'o',label='training accuracy')

plt.plot(train_sizes,test_mean,color = 'green', marker = 's',label='validation accuracy')

plt.legend(loc = 'best')

plt.xlabel('Number of samples')

plt.ylabel('Accuracy')
#kfold cross validation with Random Forest Classifier

n_splits = 10

kfold = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=420)

scores = np.zeros(n_splits)

for k , (train, test) in enumerate(kfold.split(X_train,y_train)):

    #print(train)

    clf_rf = RandomForestClassifier()

    clf_rf.fit(X_train.iloc[train],y_train.iloc[train])

    scores[k] = clf_rf.score(X_train.iloc[test],y_train.iloc[test])

    

print('Training Score-->%s'%(scores.mean()))

print('Training Standard Deviation-->%s'%(scores.std()))

print('Random Forest Test Accuracy Score --> %s'%(accuracy_score(y_test,clf_rf.predict(X_test)))) 
#parameter tuning for classifier for parameter criterion

param_range = ['gini','entropy']

train_score,test_score = validation_curve(clf_rf, 

                                          X = X_train, 

                                          y = y_train, 

                                          param_name = 'criterion', 

                                          param_range = param_range,  

                                          cv=10, 

                                          n_jobs=1

                                         )



train_mean = np.mean(train_score,axis = 1)

test_mean = np.mean(test_score,axis = 1)



plt.plot(param_range,train_mean,color = 'blue', marker = 'o',label='training accuracy')

plt.plot(param_range,test_mean,color = 'green', marker = 's',label='validation accuracy')

plt.legend(loc = 'best')

plt.xlabel('Criterion')

plt.ylabel('Accuracy')
#parameter tuning for classifier for parameter max_depth

param_range = [3,4,5,6,7]

train_score,test_score = validation_curve(clf_rf, 

                                          X = X_train, 

                                          y = y_train, 

                                          param_name = 'max_depth', 

                                          param_range = param_range,  

                                          cv=10, 

                                          n_jobs=1

                                         )



train_mean = np.mean(train_score,axis = 1)

test_mean = np.mean(test_score,axis = 1)
plt.plot(param_range,train_mean,color = 'blue', marker = 'o',label='training accuracy')

plt.plot(param_range,test_mean,color = 'green', marker = 's',label='validation accuracy')

plt.legend(loc = 'best')

plt.xlabel('Depth')

plt.ylabel('Accuracy')
#kfold cross validation with RAndom Forest Classifier with tuned parameters

n_splits = 10

kfold = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=420)

scores = np.zeros(n_splits)

for k , (train, test) in enumerate(kfold.split(X_train,y_train)):

    #print(train)

    clf_rf = RandomForestClassifier(criterion='entropy',

                                         n_estimators=70,

                                         max_depth=7,

                                         min_samples_split=15,

                                         min_samples_leaf=5, 

                                         max_features='auto', 

                                         oob_score=True, 

                                         random_state=420,

                                         n_jobs=-1

                                        )

    clf_rf.fit(X_train.iloc[train],y_train.iloc[train])

    scores[k] = clf_rf.score(X_train.iloc[test],y_train.iloc[test])

    

print('Training Score-->%s'%(scores.mean()))

print('Training Standard Deviation-->%s'%(scores.std()))

print('Random Forest Test Accuracy Score --> %s'%(accuracy_score(y_test,clf_rf.predict(X_test)))) 
#Dignose Bias and variance with Learning curves against multiple samples

train_sizes,train_score,test_score = learning_curve(estimator = clf_rf, 

                                                    X = X_train, 

                                                    y = y_train, 

                                                    train_sizes=np.linspace(0.1  , 1, 10), 

                                                    cv=10, 

                                                    n_jobs=1, pre_dispatch='all', 

                                                    random_state=420)



train_mean = np.mean(train_score,axis = 1)

test_mean = np.mean(test_score,axis = 1)



plt.plot(train_sizes,train_mean,color = 'blue', marker = 'o',label='training accuracy')

plt.plot(train_sizes,test_mean,color = 'green', marker = 's',label='validation accuracy')

plt.legend(loc = 'best')

plt.xlabel('Number of samples')

plt.ylabel('Accuracy')
y_pred = clf_rf.predict(X_test1)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test_df['PassengerId']

submission_df['Survived'] = y_pred

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)