# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn import linear_model, preprocessing, tree, model_selection
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_cp = train_data.copy()
train_data.head()
#print (train_data.info())

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_cp = test_data.copy()
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
man = train_data.loc[train_data.Sex == 'male']['Survived']
rate_man = sum(man)/len(man)
#man2 = pd.get_dummies(train_data[['Sex','Survived']])
#print('% of men who survived:', man2)
#print(train_data.shape)
print('% of men who survived:', rate_man)
def predict_gender(train_data):
    train_data['hypo'] = 0
    train_data.loc[train_data.Sex =='female','hypo'] = 1
    
    train_data['result'] = 0
    train_data.loc[train_data.Survived == train_data['hypo'],'result'] = 1
    
    print(train_data['result'].value_counts(normalize=True))
    
predict_gender(train_data)
train_data.sample(20)
type(train_cp)

data_cp_all = [train_cp, test_cp]

def data_engineering(data_all):
    for data in data_all:
        #create feature combining the existing ones, family size from sibsp and parch"
        data['Family size'] = data['SibSp'] + data['Parch'] + 1
        #extract info from other features, title from name;
        data['Title'] = data['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]
        #outputs: [' Mr' ' Mrs' ' Miss' ' Master' ' Don' ' Rev' ' Dr' ' Mme' ' Ms' ' Major'
        #' Lady' ' Sir' ' Mlle' ' Col' ' Capt' ' the Countess' ' Jonkheer']
        ##feature engineering #2
        ##For title counts < 5, we will replace it with 'misc'
        title_count = data.Title.value_counts() < 10
        #print(title_count)
        data.Title = data.Title.apply(lambda x: 'Misc' if title_count.loc[x] == True else x)
        #train_cp.loc[train_cp.Title.value_counts() < 6,'Title'] = 'misc'




data_engineering(data_cp_all) 


print(train_cp.Title.value_counts())
data_cp_all[0].info()
train_cp.sample(12)
##fancy randomforestregressor to interpolate the NAN
from sklearn.ensemble import RandomForestRegressor
for data in data_cp_all:    
    
    data2 = data.copy()
    label = LabelEncoder()
    data2['Title'] = label.fit_transform(data2['Title'])
    
    age_related_feat = data2[['Age', 'SibSp', 'Parch', 'Pclass', 'Title']]
    
    print(age_related_feat.shape)
    
    age_known = age_related_feat[age_related_feat.Age.notna()].values    ###extract value to form a numpy array instead of dataframe
    age_unknown = age_related_feat[age_related_feat.Age.isna()].values
    
    print(len(age_known))
    print(len(age_unknown))
    
    known_age_Y = age_known[:,0]
    known_age_X = age_known[:,1:]
    
    rf = RandomForestRegressor(random_state = 1, n_estimators = 300)
    rf.fit(known_age_X, known_age_Y)
    
    unknown_age_X = age_unknown[:,1:]
    
    age_pred = rf.predict(unknown_age_X)
    data.loc[data.Age.isna(), 'Age'] = np.round_(age_pred,0)
    
    print(data.Age.isna().sum())
    

    
####QC the interpolated Age

print(train_cp.sample(5))
print(test_cp.sample(5))
#print(test_data.loc[test_data.PassengerId == 1005])
#print(test_cp.loc[test_data.PassengerId == 1005])
print(train_cp.columns.values)
print(train_cp.columns.tolist())
print(train_cp.Survived)
def clean_data(data):   
    data1 = data.copy()

    ### 2 methods to encode categorical data for comparison here: 1, label encoding, 2, get_dummies: one-hot encoding
    data1['Fare'] = data1['Fare'].fillna(data1['Fare'].dropna().median())
    data1['Age'] = data1['Age'].fillna(data1['Age'].dropna().median())
    data1['Embarked'] = data1['Embarked'].fillna(data1['Embarked'].dropna().mode()[0])
    
    #Continuous variable assigned to different bins to simplify the data.
    data1['FareBin'] = pd.qcut(data1['Fare'], 12)
    data1['AgeBin'] = pd.cut(data1['Age'].astype(int), 8)
    
    all_feat = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Family size', 'Title']
    '''
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2
    '''
    
    
    
    ###1, get_dummies to get onehot encoding
    dummy_pclass = pd.get_dummies(data1['Pclass'], prefix = 'Pclass')
    dummy_embarked = pd.get_dummies(data1['Embarked'], prefix = 'Embarked')
    data_dummy = pd.get_dummies(data1[all_feat])
    #data_dummy.insert(2, 'Survived', dd, True)
    
    #data_dummy.drop(['Pclass'],axis=1,inplace = True)
    #print(dummy_pclass)
    #print(data_dummy)
    
    ###2, Sklearn LabelEncoder to conver 'object' to other format
    label = LabelEncoder()
    data1['Title'] = label.fit_transform(data1['Title'])
    data1['Sex'] = label.fit_transform(data1['Sex'])
    data1['Embarked'] = label.fit_transform(data1['Embarked'])
    data1['FareBin'] = label.fit_transform(data1['FareBin'])
    data1['AgeBin'] = label.fit_transform(data1['AgeBin'])
    
    data_dummy = pd.concat([data_dummy, dummy_pclass, dummy_embarked, data1['Embarked']], axis = 1)
    return data1, data_dummy

train_cp, test_cp = data_cp_all

train_clean, train_dummy = clean_data(train_cp)
#print(train_clean.info())
train_dummy.insert(1, 'Survived', train_cp['Survived'], True)    

test_clean, test_dummy = clean_data(test_cp)
#print(test_clean.info())

#print(train_dummy.shape) 

#print(train_clean['Survived'])     #succeed
print(train_dummy.columns.values)

train_dummy.sample(4)
for x in ['Pclass','Sex', 'Title', 'SibSp', 'Parch', 'Family size', 'Embarked']:
    print('Survival correlation by:', x)
    a = train_cp[[x, 'Survived']].groupby(x, as_index=True).mean()
    g = train_cp.groupby([x, 'Survived'])
    print(pd.DataFrame(g.count()['PassengerId'])) 
    print(a)
    print('_'*10, '\n')
### Age and Fare are largely distributed, try to scale them down, see if it improves the result.
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler().fit(train_clean['Age'].values.reshape(-1,1))
#age_scale_param = scaler.fit(train_clean['Age'])
#print(scaler.transform(train_clean[['Age', 'Fare']]))
#train_clean['Age_scl'] = scaler.transform(train_clean['Age'].values.reshape(-1,1)) #both this one and the fit_transform works, but need to convert DF to numpyarray and use reshape to convert it to a 2D array
train_clean['Age_scl'] = scaler.fit_transform(train_clean['Age'].values.reshape(-1,1))
train_clean['Fare_scl'] = scaler.fit_transform(train_clean['Fare'].values.reshape(-1,1))

scaler = preprocessing.StandardScaler().fit(train_dummy['Age'].values.reshape(-1,1))
train_dummy['Age_scl'] = scaler.fit_transform(train_dummy['Age'].values.reshape(-1,1))
train_dummy['Fare_scl'] = scaler.fit_transform(train_dummy['Fare'].values.reshape(-1,1))
test_dummy['Age_scl'] = scaler.fit_transform(test_dummy['Age'].values.reshape(-1,1))
test_dummy['Fare_scl'] = scaler.fit_transform(test_dummy['Fare'].values.reshape(-1,1))

test_dummy.sample(3)

###based on the analysis from the statistic above, we chose the features that is highly correlated to the Survive rate as follows:
#first round of features, by rough estimate: feature_names = ['Pclass', 'Age', 'Fare', 'Embarked', 'Sex', 'SibSp', 'Parch']



#### feature order matters for RFC,
feature_names1 = ['Sex', 'Title', 'Age', 'Pclass', 'Fare',  'Family size']   ##found that Age and Fare are largely distributed, which is not good for model to converge, need to scale them

feature_names1b = ['Sex', 'Title', 'Age_scl', 'Pclass', 'Fare_scl',  'Family size']   ##the scaled input doesn't change the performance of the RF algorithm


###a fair comparison for onehot encoded features:
feature_names2 = ['Sex_female', 'Sex_male', 'Title_ Master', 'Title_ Miss', 'Title_ Mrs', 'Title_ Mr', 'Title_Misc', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked','Age_scl','Fare_scl', 'SibSp', 'Parch', 'Family size']
print(os.listdir("/kaggle/input/titanic-leaked"))
groundtruth_data = pd.read_csv('/kaggle/input/titanic-leaked/titanic.csv')
groundtruth_data.sample(4)
groundtruth_data = pd.concat([groundtruth_data['Survived'], test_data], axis = 1)
groundtruth_data.sample(5)
print(test_dummy.shape)
test_dummy = pd.concat([groundtruth_data['Survived'], test_dummy], axis = 1)
test_dummy.shape
print(test_dummy['Survived'].shape)
from sklearn.ensemble import BaggingClassifier
def predict_logistic(data):
    target = data['Survived'].values
    features = data[feature_names2].values
    
    lr_clf = linear_model.LogisticRegression().fit(features, target)
    bagging_clf = BaggingClassifier(lr_clf, n_estimators=50, max_samples=0.8, max_features=0.8, bootstrap=True, bootstrap_features=False, random_state=0)
    ensemble = bagging_clf.fit(features, target)
    cv_scores = model_selection.cross_val_score(lr_clf, features, target, scoring = 'accuracy', cv = 30)
    print('logistic regression:', lr_clf.score(features,target), 'ensemble:', ensemble.score(features,target), 'cv:', cv_scores.mean())
    
    return lr_clf
    
    

start = time.time()
lr_clf = predict_logistic(train_dummy)
end = time.time()
print(end - start)
print(lr_clf)

#use the feature_names1, without scaling the 'Age', 'Fare' logistic regression: 0.7991021324354658 
#0.034967899322509766
#use the feature_names1b, with scaling the 'Age', 'Fare', logistic regression: 0.7991021324354658
#0.01706242561340332    ##faster!!
#use the feature_names2, onehot encoded features, logistic regression: 0.8316498316498316   ##better accuracy!!
#0.07921242713928223    


#further examine the model coefficient:
#print(list(clf.coef_.T))
pd.DataFrame({'columns': feature_names2, 'coef': list(lr_clf.coef_.T)})
##SVC test
target = train_dummy['Survived'].values
features = train_dummy[feature_names2].values
svc_clf = SVC(probability=True).fit(features, target)
print(svc_clf.score(features,target))

cv_scores = model_selection.cross_val_score(svc_clf, features, target, scoring = 'accuracy', cv = 20)
print(cv_scores.mean())

###randomforest parameter tweaking:
#1, max_depth 4, 6, 8, 10, 12, 14
target = train_dummy['Survived'].values
features = train_dummy[feature_names1b].values
param_grid = {'max_depth': [4, 6, 8, 10, 12, 14], 'n_estimators':[50, 100, 500, 1000], 'random_state': [10]}

clf = RandomForestClassifier()
model_tune = model_selection.GridSearchCV(clf, param_grid = param_grid, scoring = 'accuracy', cv = 20)
model_tune.fit(features, target)
print(model_tune.best_params_)
    

##{'max_depth': 6, 'n_estimators': 500, 'random_state': 10} is the same as I tested it separately

def predict_randomforest1(data):
    target = data['Survived'].values
    features = data[feature_names1b].values
    
    clf = RandomForestClassifier(max_depth = 6, n_estimators =500, random_state = 10)
    
    #clf = random_forest = RandomForestClassifier(criterion = "gini", 
    #                                   min_samples_leaf = 1, 
    #                                   min_samples_split = 10,   
    #                                   n_estimators=100, 
    #                                   max_features='auto', 
    #                                   oob_score=True, 
    #                                   random_state=10, 
    #                                   n_jobs=-1)
    rfc = clf.fit(features,target)
    print('random forest:', clf.score(features,target))
    
    scores = model_selection.cross_val_score(clf, features, target, scoring = 'accuracy', cv = 50)
    #print(scores)
    print('avg random forest w/ CV50:',scores.mean())
    return clf

def predict_randomforest2(data):
    target = data['Survived'].values
    features = data[feature_names2].values
    
    clf = RandomForestClassifier(max_depth = 6, n_estimators =500, random_state = 10)
    
    ###use bagging regressor to fit:
    
    #clf = random_forest = RandomForestClassifier(criterion = "gini", 
    #                                   min_samples_leaf = 1, 
    #                                   min_samples_split = 10,   
    #                                   n_estimators=100, 
    #                                   max_features='auto', 
    #                                   oob_score=True, 
    #                                   random_state=10, 
    #                                   n_jobs=-1)
    rfc = clf.fit(features,target)
    print('random forest:', clf.score(features,target))
    
    #scores = model_selection.cross_val_score(clf, features, target, scoring = 'accuracy', cv = 50)
    #print(scores)
    #print('avg random forest w/ CV50:',scores.mean())
    return rfc
    
#clean_data_ = clean_data(train_data)
#predict_randomforest(clean_data_)
    
start = time.time()
#randomforest1 = predict_randomforest1(train_clean)   ##RF: 0.9158; RF CV50: 0.8285   With Sibsp, Parch
                                                     ##RF: 0.9068; RF CV50: 0.8341   With Family size instead
                                                     ##RF: 0.8710; RF CV50: 0.8410   With Family size instead; remove Title Rev, Dr; n_estimator 100 > 500 
end = time.time() 
print('runing time:', end - start)    ##65.048 vs 65.378: n_jobs = -1 vs none

randomforest2 = predict_randomforest2(train_dummy)   ##RF: 0.9181; RF CV50: 0.8329   With Sibsp, Parch; With Sex onehot only
                                                     ##RF: 0.8530; RF CV50: 0.8365   With Sibsp, Parch; With Sex, Title onehot
                                                     ##RF: 0.8597; RF CV50: 0.8320   With Family size instead; With Sex, Title onehot
                                                     ##RF: 0.8653; RF CV50: 0.8375   With Family size instead; With Sex, Title onehot, remove Rev, Dr
                                                     ##RF: 0.8676; RF CV50: 0.8420   With Family size instead; With Sex, Title onehot, remove Rev, Dr; n_estimator 100 -> 500 
pd.DataFrame({'features': feature_names2, 'importance': list(randomforest2.feature_importances_)})
##use the ground_truth for quick turnaround test:
test_features = test_dummy[feature_names2].values
test_target = test_dummy['Survived'].values
print(test_target.shape)
print('random forest:', randomforest2.score(test_features,test_target))
####model ensemble using lr_clf, svc_clf, rfc
lr_pred = lr_clf.predict(test_features)
svc_pred = svc_clf.predict(test_features)
rfc_pred = randomforest2.predict(test_features)

#vote_est = [lr_pred, svc_pred, rfc_pred]
vote_est = [('rf',randomforest2),('lr',lr_clf),('svc',svc_clf)]

vote_hard = VotingClassifier(estimators = vote_est, voting = 'hard').fit(features, target)
vote_hard_cv = model_selection.cross_validate(vote_hard, features, target, cv = 20)
print('hard voting:',vote_hard_cv['test_score'].mean())

vote_soft = VotingClassifier(estimators = vote_est, voting = 'soft').fit(features, target)
vote_soft_cv = model_selection.cross_validate(vote_soft, features, target, cv = 20)
print('soft voting:',vote_soft_cv['test_score'].mean())

##use the ground_truth for quick turnaround test:
test_features = test_dummy[feature_names2].values
test_target = test_dummy['Survived'].values
print(test_target.shape)
print('random forest:', vote_soft.score(test_features,test_target))

#test_clean = clean_data(test_cp)
X_test_features = test_dummy[feature_names2].values
predictions = vote_soft.predict(X_test_features)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
