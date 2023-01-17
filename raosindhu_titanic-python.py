# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv("../input/titanic/train.csv")#, index_col= 'PassengerId')

test = pd.read_csv("../input/titanic/test.csv")#, index_col= 'PassengerId')

print(train.info())

print(train.head(5))

print(train.tail())

print(test.head())



all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['Survived'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))



all_data.head()

ntrain = len(train)
train.hist(bins=50, figsize=(30,20))



test.hist(bins=50, figsize=(30,20))
train.head()



#traindata[traindata.Age <=3]

colormap = np.array(['r', 'g'])

plt.scatter(x = train.SibSp, y = train.Age, c=colormap[train.Survived])
#Survival by sex

print('**********Survival by Sex**************')

print(train.groupby('Sex')['Survived'].agg(['count', 'sum']).rename(columns={'Passengers':'Survived'})) 

print(train.groupby('Sex')['Survived'].sum()*100/ train.groupby('Sex')['Survived'].count()) # % of each sex survived



#Survival by class

print('**********Survival by Class**************')

print(train.groupby('Pclass')['Survived'].agg(['count', 'sum']).rename(columns={'Passengers':'Survived'})) 

print(train.groupby('Pclass')['Survived'].sum()*100/train.groupby('Pclass')['Survived'].count()) # % of each sex survived



#Survival by parch

print('**********Survival by Parch**************')

print(train.groupby('Parch')['Survived'].agg(['count', 'sum']))

print(train.groupby('Parch')['Survived'].sum()*100/train.groupby('Parch')['Survived'].count()) # % of each sex survived



#avg fare

print('**********Avg Fare by sex, class**************')

print(train.groupby(['Pclass'])['Fare'].agg(['mean', 'median']))



#Unique values in Embarked column

print('**********Embarked column**************')

print(train['Embarked'].unique())

#Survival by class and gender

print(train.groupby(['Pclass', 'Sex'])['Survived'].agg(['count', 'sum']).rename(columns={'Passengers':'Survived'})) 

train.groupby(['Pclass', 'Sex'])['Survived'].sum()*100/train.groupby(['Pclass', 'Sex'])['Survived'].count() # % of each sex survived

#Survival varies by Gender and Class of passenger
print('******Train data missings*****')

print('There are',train.isnull().any().sum(), 'columns with missing values')

print('The columns are',train.columns[train.isnull().any()].values)

print('Number of missing values in each column are', (train.isnull().sum()/len(train)).sort_values(ascending = False).head(3))

# print(traindata[traindata.isnull().any(axis=1)].head())



print('******Test data missings*****')

print('There are',test.isnull().any().sum(), 'columns with missing values')

print('The columns are',test.columns[test.isnull().any()].values)

print('Number of missing values in each column are', (test.isnull().sum()/len(train)).sort_values(ascending = False).head(3))
#cabin column in traindata has ~78% missing values. Drop the cable column from train and test data

train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)

all_data.drop('Cabin', axis=1, inplace=True)



#Imputing age column

print(all_data.groupby(['Sex', 'Pclass'])['Age'].agg(['mean', 'min', 'max', 'median']))

age_impute = all_data.groupby(['Sex', 'Pclass'])['Age'].median().reset_index()

print(age_impute)

all_data['Age']= all_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))



#Imputing Embarked column

all_data.loc[all_data['Embarked'].isnull(),'Embarked'] = all_data['Embarked'].value_counts().index[0]



all_data.loc[all_data.isnull().any(axis=1), 'Fare'] = 8

all_data.head()

all_data.isnull().any().sum()
all_data['Family_size']= all_data['SibSp']+all_data['Parch']+1

all_data['Fare_pp']= all_data['Fare']/all_data['Family_size']

all_data['Title'] = all_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



print(all_data.groupby('Title').Age.count())

fare = all_data.groupby(['Pclass','Parch','SibSp']).Fare.median()[3][0][0]



for i in range(0, len(all_data)):

    if all_data.Title[i] in ('Mlle', 'Mme', 'Dona', 'Ms', 'Lady', 'the Countess'): 

        all_data.Title[i] = 'Miss'

    elif all_data.Title[i] in ('Jonkheer', 'Don', 'Sir', 'Don', 'Rev'): 

        all_data.Title[i] = 'Mr'
#label encoder

from sklearn.preprocessing import LabelEncoder

lbl= LabelEncoder()

lbl.fit(list(all_data['Title'].values)) 

all_data['Title'] = lbl.transform(list(all_data['Title'].values))



lbl.fit(list(all_data['Embarked'].values)) 

all_data['Embarked'] = lbl.transform(list(all_data['Embarked'].values))



lbl.fit(list(all_data['Sex'].values)) 

all_data['Sex'] = lbl.transform(list(all_data['Sex'].values))



print(all_data.head())
# encoding using get_dummies

# finaldata['Sex'] = pd.get_dummies(finaldata['Sex'], drop_first = True)

# finaldata[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = pd.get_dummies(finaldata['Embarked'], columns="Embarked", prefix="Embarked")

# finaldata.drop(['Ticket', 'Survived', 'Name', 'Embarked'], axis=1, inplace=True)

# finaldata.head()

    #repeating same on test data

# testdata['Sex'] = pd.get_dummies(testdata['Sex'], drop_first = True)

# testdata[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = pd.get_dummies(testdata['Embarked'], columns="Embarked", prefix="Embarked")

# testdata.drop(['Ticket','Name', 'Embarked'], axis=1, inplace=True)

# testdata.head()
from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()



all_data[['Age', 'Fare_pp']] = pd.DataFrame(data=scaler.fit_transform(all_data[['Age', 'Fare']]), columns=['Age', 'Fare'])
#convert Pclass to categorical object

all_data['Pclass'] = pd.Categorical(all_data.Pclass, ordered=True)

all_data.info()



all_data['Sex'] = pd.Categorical(all_data.Sex, ordered=True)

all_data.info()



all_data['Title'] = pd.Categorical(all_data.Title, ordered=True)

all_data.info()
#seperating data

all_data.drop(['Ticket','Name', 'Embarked','Fare'], axis=1, inplace=True)

#seperating data

# all_data.drop(['SibSp','Parch'], axis=1, inplace=True)

traindata = all_data[:ntrain]

testdata = all_data[ntrain:]

traindata.set_index('PassengerId', inplace=True)



target = train.set_index('PassengerId')['Survived']

finaldata=traindata[:]

traindata['Survival'] = target

testdata.set_index('PassengerId', inplace=True)



all_data.info()

finaldata.info()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

from sklearn import model_selection



# implementing train-test-split

X_train, X_test, y_train, y_test = train_test_split(finaldata, target, test_size=0.30, random_state=0)



#Stratified Split

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)

# for train_idex, test_index in split.split(traindata, traindata(['Survival', 'Sex', 'Pclass'])):

#     train_set = traindata.loc[train_idex]

#     test_set = traindata.loc[test_idex]





train_check = X_train[:]

train_check['survived'] = y_train

train_check.head()



#checking the distribution of the survival class in train data and x_train are distributed in the same proportion

#check = ['Sex', 'Pclass']

# for i in check:

#     print(traindata.groupby([i, 'Survived'])['Age'].count())

#     print(traindata.groupby([i, 'Survived'])['Age'].count()/traindata.groupby(i)['Age'].count())

#     print(train_check.groupby([i, 'survived'])['Age'].count())

#     print(train_check.groupby([i, 'survived'])['Age'].count()/train_check.groupby(i)['Age'].count())
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



lg = LogisticRegression()

clf = LogisticRegression().fit(X_train, y_train)

print(clf.score(X_train, y_train))

print(clf.predict(X_test))



importance = clf.coef_[0]

for i,v in enumerate(importance):

    print('Feature: %0d, Score: %.5f' % (i,v))

print(np.round(clf.coef_, decimals=2) >0)



    

clf.predict_proba(X_test)[:, 1]

clf.score(X_test, y_test)



# Accuracy 

accuracy_lg = accuracy_score(y_test, clf.predict(X_test))

print('accuracy of logistic regression',accuracy_lg)

lg_probs = clf.predict_proba(X_test)[:, 1]



lg_auc = roc_auc_score(y_test, lg_probs)

# summarize scores

print('Random Forest: ROC AUC=%.3f' % (lg_auc))

# calculate roc curves

lg_fpr, lg_tpr, _ = roc_curve(y_test, lg_probs)

# plot the roc curve for the model

plt.plot(lg_fpr, lg_tpr, marker='.', label='logistic regression')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
# creating the data set to check the errors

error_check = X_test[:]

error_check['y_test'] = y_test

error_check['predicted'] = clf.predict(X_test)

error_check['proba'] = clf.predict_proba(X_test)[:, 1]

error_check.head()



error_check.query('y_test != predicted')

error_check.query('y_test == predicted')



print(error_check.groupby(['Sex','y_test', 'predicted'])['Age'].count())

print(error_check.groupby(['Pclass', 'y_test', 'predicted'])['Age'].count())



print(error_check.groupby(['Pclass', 'Sex', 'y_test', 'predicted'])['Age'].count())

print(error_check.groupby('Sex')['Age'].count())



# Wrong prediction groups

# sex=0(male) and predicted = 1 and actual = 0

# sex=1(female)  and predicted = 0 and actual = 1

# class = 3 predicted = 0 and actual = 1                                                     

#error_check.query('Pclass == 3 and predicted == 0 and y_test != predicted')
from sklearn.model_selection import cross_val_score



# scores = cross_val_score(lg, X_train, y_train, cv=5)

scores = cross_val_score(lg, finaldata, target, cv=10)

print('Cross-Validation Accuracy Scores', scores)

print(scores.mean())





from sklearn.model_selection import cross_val_predict

# y_pred = cross_val_predict(lg, X_train, y_train, cv=5)

y_pred = cross_val_predict(lg, finaldata, target, cv=5)

crossval_proba = cross_val_predict(lg, finaldata, target, cv=5, method='predict_proba')[:,1] 



# Accuracy 

accuracy_cv = accuracy_score(target, y_pred)

print('accuracy of cross validation for logistic regression',accuracy_cv)



cv_auc = roc_auc_score(target, crossval_proba)

# summarize scores

print('Random Forest: ROC AUC=%.3f' % (cv_auc))

# calculate roc curves

cv_fpr, cv_tpr, _ = roc_curve(target, crossval_proba)

# plot the roc curve for the model

plt.plot(cv_fpr, cv_tpr, marker='.', label='logistic cv regression')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf_class = RandomForestClassifier(oob_score=True, random_state=0,  n_estimators=8, max_depth=5,

                                  max_leaf_nodes=10, min_samples_leaf=4, min_samples_split=4)

rf_class.fit(X_train, y_train)

print('Oob score of random forest classifier is:', rf_class.oob_score_)



accuracy_rf = accuracy_score(y_test, rf_class.predict(X_test))

print('accuracy of random forest', accuracy_rf)



rf_probs = rf_class.predict_proba(X_test)

# keep probabilities for the positive outcome only

rf_probs = rf_probs[:, 1]

# calculate scores

rf_auc = roc_auc_score(y_test, rf_probs)

# summarize scores

print('Random Forest: ROC AUC=%.3f' % (rf_auc))

# calculate roc curves

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# plot the roc curve for the model

plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()

# implementing Grid search

param_grid = [{'n_estimators': np.arange(4,20,2), 'max_features': [3,4,5,6,7,8,9]}]

gs_class = RandomForestClassifier(oob_score=True, random_state=20)

grid_search = model_selection.GridSearchCV(gs_class, param_grid, cv=5, scoring='roc_auc')

grid_search.fit(X_train, y_train)



grid_search.best_params_

gs_bestmodel = grid_search.best_estimator_

gs_predict = gs_bestmodel.predict(X_test)



print('Oob score of random forest classifier with grid search is:', gs_bestmodel.oob_score_)



cvres=grid_search.cv_results_

print(cvres["mean_test_score"])
accuracy_gs = accuracy_score(y_test, gs_predict)

print('accuracy of grid search random forest', accuracy_gs)



gs_probs = gs_bestmodel.predict_proba(X_test)

# keep probabilities for the positive outcome only

gs_probs = gs_probs[:, 1]

# calculate scores

gs_auc = roc_auc_score(y_test, gs_probs)

# summarize scores

print('Random Forest: ROC AUC=%.3f' % (gs_auc))

# calculate roc curves

gs_fpr, gs_tpr, _ = roc_curve(y_test, gs_probs)

# plot the roc curve for the model

plt.plot(gs_fpr, gs_tpr, marker='.', label='Random Forest')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
from sklearn.ensemble import GradientBoostingClassifier



gb_class = RandomForestClassifier(random_state=42,  n_estimators=12, max_depth=4)

gb_class.fit(X_train, y_train)

print('Oob score of Gradient Boosting classifier is:', rf_class.oob_score_)



accuracy_gb = accuracy_score(y_test, gb_class.predict(X_test))

print('accuracy of Gradient Boosting', accuracy_rf)



# keep probabilities for the positive outcome only

gb_probs = gb_class.predict_proba(X_test)[:,1]

# calculate scores

gb_auc = roc_auc_score(y_test, gb_probs)

# summarize scores

print('Gradient Boosting: ROC AUC=%.3f' % (gb_auc))

# calculate roc curves

gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

# plot the roc curve for the model

plt.plot(gb_fpr, gb_tpr, marker='.', label='Gradient Boosting')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()

#from sklearn.ensemble import VotingClassifer

#Logistic regression

lg = LogisticRegression()

clf_p = clf.predict(X_test)    

clf_proba = clf.predict_proba(X_test)[:, 1]



#random forest classifier

rf_class.fit(X_train, y_train)

print('Oob score of voting classifier is:', rf_class.oob_score_)

accuracy_rf = accuracy_score(y_test, rf_class.predict(X_test))

print('accuracy of random forest', accuracy_rf)

rf_probs = rf_class.predict_proba(X_test)[:,1]

rf_p = rf_class.predict(X_test)



# Random forst with grid search

gs_probs = gs_bestmodel.predict_proba(X_test)[:, 1]

gs_p = gs_bestmodel.predict(X_test)



# Gradient Boosting 

gb_probs = gb_class.predict_proba(X_test)[:,1]

gb_p = gb_class.predict(X_test)



# Combining 3 probabilities into a single dataframe

prob_val = pd.DataFrame(data=[clf_proba, rf_probs, gs_probs, gb_probs, clf_p,rf_p,gs_p, gb_p]).T

prob_val.columns=['clf_proba', 'rf_probs', 'gs_probs', 'gb_probs', 'clf_p','rf_p','gs_p', 'gb_p']

print(prob_val)
# Implementing soft voting and hard voting manually

def voting_clf(df):

    hard_voting=[]

    soft_voting=[]

    hard_voting_proba=[]

    soft_voting_proba=[]

    probtest2 = pd.DataFrame(columns=['SoftVoting', 'HardVoting', 'hard_voting_proba', 'soft_voting_proba'])

    print(probtest2)

    i=0

    for i in range(0, len(df)):

        hard_voting_proba.append((df.clf_proba[i]+df.rf_probs[i]+df.gs_probs[i]+df.gb_probs[i])/4)

        soft_voting_proba.append((df.clf_p[i]+df.rf_p[i]+df.gs_p[i]+df.gb_p[i])/4)

        

        hard_voting.append(round((df.clf_proba[i]+df.rf_probs[i]+df.gs_probs[i]+df.gb_probs[i])/4))

        soft_voting.append(round((df.clf_p[i]+df.rf_p[i]+df.gs_p[i]+df.gb_p[i])/4))



    probtest2['HardVoting'] = pd.Series(hard_voting)

    probtest2['SoftVoting'] = pd.Series(soft_voting)

    probtest2['hard_voting_proba'] = pd.Series(hard_voting_proba)

    probtest2['soft_voting_proba'] = pd.Series(soft_voting_proba)

    

    return probtest2





probval2 = voting_clf(prob_val)



probval2.query('SoftVoting != HardVoting')
accuracy_HV= accuracy_score(y_test, probval2['HardVoting'])

print('accuracy of HardVoting', accuracy_HV)



accuracy_SV= accuracy_score(y_test, probval2['SoftVoting'])

print('accuracy of SoftVoting', accuracy_SV)



AUC_HV = roc_auc_score(y_test, probval2['hard_voting_proba'])

print('Hard voting AUC: ROC AUC=%.3f' % (AUC_HV))



AUC_SV = roc_auc_score(y_test, probval2['soft_voting_proba'])

print('Soft voting AUC: ROC AUC=%.3f' % (AUC_SV))



# calculate roc curves for soft voting

sv_fpr, sv_tpr, _ = roc_curve(y_test, probval2['SoftVoting'])

# plot the roc curve for the model

plt.plot(sv_fpr, sv_tpr, marker='.', label='SoftVoting')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()





# calculate roc curves for hard voting

hv_fpr, hv_tpr, _ = roc_curve(y_test, probval2['HardVoting'])

# plot the roc curve for the model

plt.plot(hv_fpr, hv_tpr, marker='.', label='HardVoting')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
testdata.head()

# testdata.set_index('PassengerId', inplace=True)

# testdata.head()



#predicting using logistic regression model

tclf_p = clf.predict(testdata)    

tclf_proba = clf.predict_proba(testdata)[:, 1]



#predicting using random forest model

trf_probs = rf_class.predict_proba(testdata)[:,1]

trf_p = rf_class.predict(testdata)



#predicting using the best model in grid search

tgs_probs = gs_bestmodel.predict_proba(testdata)[:,1]

tgs_p = gs_bestmodel.predict(testdata)





#predicting using gradient Boosting

tgb_probs = gb_class.predict_proba(testdata)[:,1]

tgb_p = gb_class.predict(testdata)





prob_test = pd.DataFrame(data=[tclf_proba, trf_probs, tgs_probs, tgb_probs, tclf_p, trf_p, tgs_p, tgb_p]).T

prob_test.columns=['clf_proba', 'rf_probs', 'gs_probs', 'gb_probs', 'clf_p','rf_p','gs_p','gb_p']

print(prob_test)



prob_test2 = voting_clf(prob_test)



prob_test2.head(25)



prob_test2.info()
  

submission_hv = pd.DataFrame(columns = ['PassengerId', 'Survived'])

submission_hv['Survived'] = prob_test2['HardVoting']

submission_hv['Survived'] = submission_hv['Survived'].astype('int64')

submission_hv['PassengerId'] = testdata.index

print(submission_hv.head())

submission_hv.to_csv('submission_HardVoting.csv', index=False)





submission_sv = pd.DataFrame(columns = ['PassengerId', 'Survived'])

submission_sv['Survived'] = prob_test2['SoftVoting']

submission_sv['Survived'] = submission_sv['Survived'].astype('int64')

submission_sv['PassengerId'] = testdata.index

print(submission_sv.head())

submission_sv.to_csv('submission_SoftVoting.csv', index=False)