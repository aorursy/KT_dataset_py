# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



training['train_test'] = 1

test['train_test'] = 0

test['Survived'] = np.NaN

all_data = pd.concat([training,test])



%matplotlib inline

all_data.columns
# Understand nature of the data .info() .describe()

# Histograms and boxplots 

# Value counts 

# Missing data 

# Correlation between the metrics 

# Explore interesting themes 

    # Wealthy survive? 

    # By location 

    # Age scatterplot with ticket price 

    # Young and wealthy Variable? 

    # Total spent? 

# Feature engineering 

# preprocess data together or use a transformer? 

    # use label for train and test   

# Scaling?



# Model Baseline 

# Model comparison with CV 
#quick look at our data types & null counts 

training.info()
# to better understand the numeric data, we want to use the .describe() method. This gives us an understanding of the central tendencies of the data 

training.describe()
#quick way to separate numeric columns

training.describe().columns
# look at numeric and categorical values separately 

df_num = training[['Age','SibSp','Parch','Fare']]

df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
#distributions for all numeric variables 

for i in df_num.columns:

    plt.hist(df_num[i])

    plt.title(i)

    plt.show()
print(df_num.corr())

sns.heatmap(df_num.corr())
# compare survival rate across Age, SibSp, Parch, and Fare 

pd.pivot_table(training, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
for i in df_cat.columns:

    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)

    plt.show()\

    
# Comparing survival and each of these categorical variables 

print(pd.pivot_table(training, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))

print()

print(pd.pivot_table(training, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))

print()

print(pd.pivot_table(training, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))
df_cat.Cabin

training['cabin_multiple'] = training.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

# after looking at this, we may want to look at cabin by letter or by number. Let's create some categories for this 

# letters 

# multiple letters 

training['cabin_multiple'].value_counts()
pd.pivot_table(training, index = 'Survived', columns = 'cabin_multiple', values = 'Ticket' ,aggfunc ='count')
#creates categories based on the cabin letter (n stands for null)

#in this case we will treat null values like it's own category



training['cabin_adv'] = training.Cabin.apply(lambda x: str(x)[0])

#comparing surivial rate by cabin

print(training.cabin_adv.value_counts())

pd.pivot_table(training,index='Survived',columns='cabin_adv', values = 'Name', aggfunc='count')
#understand ticket values better 

#numeric vs non numeric 

training['numeric_ticket'] = training.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

training['ticket_letters'] = training.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)

training['numeric_ticket'].value_counts()
#lets us view all rows in dataframe through scrolling. This is for convenience 

pd.set_option("max_rows", None)

training['ticket_letters'].value_counts()

#difference in numeric vs non-numeric tickets in survival rate 

pd.pivot_table(training,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')
#survival rate across different tyicket types 

pd.pivot_table(training,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')
#feature engineering on person's title 

training.Name.head(50)

training['name_title'] = training.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

#mr., ms., master. etc
training['name_title'].value_counts()
#create all categorical variables that we did above for both training and test sets 

all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))

all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])

all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)

all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)

all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())



#impute nulls for continuous data 

#all_data.Age = all_data.Age.fillna(training.Age.mean())

all_data.Age = all_data.Age.fillna(training.Age.median())

#all_data.Fare = all_data.Fare.fillna(training.Fare.mean())

all_data.Fare = all_data.Fare.fillna(training.Fare.median())



#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 

all_data.dropna(subset=['Embarked'],inplace = True)



#tried log norm of sibsp (not used)

all_data['norm_sibsp'] = np.log(all_data.SibSp+1)

all_data['norm_sibsp'].hist()



# log norm of fare (used)

all_data['norm_fare'] = np.log(all_data.Fare+1)

all_data['norm_fare'].hist()



# converted fare to category for pd.get_dummies()

all_data.Pclass = all_data.Pclass.astype(str)



#created dummy variables from categories (also can use OneHotEncoder)

all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])



#Split to train test again

X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)

X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)





y_train = all_data[all_data.train_test==1].Survived

y_train.shape
# Scale data 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

all_dummies_scaled = all_dummies.copy()

all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])

all_dummies_scaled



X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)

X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)



y_train = all_data[all_data.train_test==1].Survived

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
#I usually use Naive Bayes as a baseline for my classification tasks 

gnb = GaussianNB()

cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
lr = LogisticRegression(max_iter = 2000)

cv = cross_val_score(lr,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
lr = LogisticRegression(max_iter = 2000)

cv = cross_val_score(lr,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
dt = tree.DecisionTreeClassifier(random_state = 1)

cv = cross_val_score(dt,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
dt = tree.DecisionTreeClassifier(random_state = 1)

cv = cross_val_score(dt,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
knn = KNeighborsClassifier()

cv = cross_val_score(knn,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
knn = KNeighborsClassifier()

cv = cross_val_score(knn,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
rf = RandomForestClassifier(random_state = 1)

cv = cross_val_score(rf,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
rf = RandomForestClassifier(random_state = 1)

cv = cross_val_score(rf,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
svc = SVC(probability = True)

cv = cross_val_score(svc,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state =1)

cv = cross_val_score(xgb,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
#Voting classifier takes all of the inputs and averages the results. For a "hard" voting classifier each classifier gets 1 vote "yes" or "no" and the result is just a popular vote. For this, you generally want odd numbers

#A "soft" classifier averages the confidence of each of the models. If a the average confidence is > 50% that it is a 1 it will be counted as such

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
voting_clf.fit(X_train_scaled,y_train)

y_hat_base_vc = voting_clf.predict(X_test_scaled).astype(int)

basic_submission = {'PassengerId': test.PassengerId, 'Survived': y_hat_base_vc}

base_submission = pd.DataFrame(data=basic_submission)

base_submission.to_csv('base_submission.csv', index=False)
from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import RandomizedSearchCV 
#simple performance reporting function

def clf_performance(classifier, model_name):

    print(model_name)

    print('Best Score: ' + str(classifier.best_score_))

    print('Best Parameters: ' + str(classifier.best_params_))
lr = LogisticRegression()

param_grid = {'max_iter' : [2000],

              'penalty' : ['l1', 'l2'],

              'C' : np.logspace(-4, 4, 20),

              'solver' : ['liblinear']}



clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_lr = clf_lr.fit(X_train_scaled,y_train)

clf_performance(best_clf_lr,'Logistic Regression')
knn = KNeighborsClassifier()

param_grid = {'n_neighbors' : [3,5,7,9],

              'weights' : ['uniform', 'distance'],

              'algorithm' : ['auto', 'ball_tree','kd_tree'],

              'p' : [1,2]}

clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_knn = clf_knn.fit(X_train_scaled,y_train)

clf_performance(best_clf_knn,'KNN')
svc = SVC(probability = True)

param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],

                                  'C': [.1, 1, 10, 100, 1000]},

                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},

                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]

clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_svc = clf_svc.fit(X_train_scaled,y_train)

clf_performance(best_clf_svc,'SVC')
#Because the total feature space is so large, I used a randomized search to narrow down the paramters for the model. I took the best model from this and did a more granular search 

"""

rf = RandomForestClassifier(random_state = 1)

param_grid =  {'n_estimators': [100,500,1000], 

                                  'bootstrap': [True,False],

                                  'max_depth': [3,5,10,20,50,75,100,None],

                                  'max_features': ['auto','sqrt'],

                                  'min_samples_leaf': [1,2,4,10],

                                  'min_samples_split': [2,5,10]}

                                  

clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid, n_iter = 100, cv = 5, verbose = True, n_jobs = -1)

best_clf_rf_rnd = clf_rf_rnd.fit(X_train_scaled,y_train)

clf_performance(best_clf_rf_rnd,'Random Forest')"""
rf = RandomForestClassifier(random_state = 1)

param_grid =  {'n_estimators': [400,450,500,550],

               'criterion':['gini','entropy'],

                                  'bootstrap': [True],

                                  'max_depth': [15, 20, 25],

                                  'max_features': ['auto','sqrt', 10],

                                  'min_samples_leaf': [2,3],

                                  'min_samples_split': [2,3]}

                                  

clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_rf = clf_rf.fit(X_train_scaled,y_train)

clf_performance(best_clf_rf,'Random Forest')
best_rf = best_clf_rf.best_estimator_.fit(X_train_scaled,y_train)

feat_importances = pd.Series(best_rf.feature_importances_, index=X_train_scaled.columns)

feat_importances.nlargest(20).plot(kind='barh')
"""xgb = XGBClassifier(random_state = 1)



param_grid = {

    'n_estimators': [20, 50, 100, 250, 500,1000],

    'colsample_bytree': [0.2, 0.5, 0.7, 0.8, 1],

    'max_depth': [2, 5, 10, 15, 20, 25, None],

    'reg_alpha': [0, 0.5, 1],

    'reg_lambda': [1, 1.5, 2],

    'subsample': [0.5,0.6,0.7, 0.8, 0.9],

    'learning_rate':[.01,0.1,0.2,0.3,0.5, 0.7, 0.9],

    'gamma':[0,.01,.1,1,10,100],

    'min_child_weight':[0,.01,0.1,1,10,100],

    'sampling_method': ['uniform', 'gradient_based']

}



#clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

#best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)

#clf_performance(best_clf_xgb,'XGB')

clf_xgb_rnd = RandomizedSearchCV(xgb, param_distributions = param_grid, n_iter = 1000, cv = 5, verbose = True, n_jobs = -1)

best_clf_xgb_rnd = clf_xgb_rnd.fit(X_train_scaled,y_train)

clf_performance(best_clf_xgb_rnd,'XGB')"""
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



clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)

clf_performance(best_clf_xgb,'XGB')

y_hat_xgb = best_clf_xgb.best_estimator_.predict(X_test_scaled).astype(int)

xgb_submission = {'PassengerId': test.PassengerId, 'Survived': y_hat_xgb}

submission_xgb = pd.DataFrame(data=xgb_submission)

submission_xgb.to_csv('xgb_submission3.csv', index=False)
best_lr = best_clf_lr.best_estimator_

best_knn = best_clf_knn.best_estimator_

best_svc = best_clf_svc.best_estimator_

best_rf = best_clf_rf.best_estimator_

best_xgb = best_clf_xgb.best_estimator_



voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 

voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft') 

voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft') 

voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')



print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))

print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean())



print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5))

print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5).mean())



print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=5))

print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=5).mean())



print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))

print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())

#in a soft voting classifier you can weight some models more than others. I used a grid search to explore different weightings

#no new results here

params = {'weights' : [[1,1,1],[1,2,1],[1,1,2],[2,1,1],[2,2,1],[1,2,2],[2,1,2]]}



vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = 5, verbose = True, n_jobs = -1)

best_clf_weight = vote_weight.fit(X_train_scaled,y_train)

clf_performance(best_clf_weight,'VC Weights')

voting_clf_sub = best_clf_weight.best_estimator_.predict(X_test_scaled)

#Make Predictions 

voting_clf_hard.fit(X_train_scaled, y_train)

voting_clf_soft.fit(X_train_scaled, y_train)

voting_clf_all.fit(X_train_scaled, y_train)

voting_clf_xgb.fit(X_train_scaled, y_train)



best_rf.fit(X_train_scaled, y_train)

y_hat_vc_hard = voting_clf_hard.predict(X_test_scaled).astype(int)

y_hat_rf = best_rf.predict(X_test_scaled).astype(int)

y_hat_vc_soft =  voting_clf_soft.predict(X_test_scaled).astype(int)

y_hat_vc_all = voting_clf_all.predict(X_test_scaled).astype(int)

y_hat_vc_xgb = voting_clf_xgb.predict(X_test_scaled).astype(int)
#convert output to dataframe 

final_data = {'PassengerId': test.PassengerId, 'Survived': y_hat_rf}

submission = pd.DataFrame(data=final_data)



final_data_2 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_hard}

submission_2 = pd.DataFrame(data=final_data_2)



final_data_3 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_soft}

submission_3 = pd.DataFrame(data=final_data_3)



final_data_4 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_all}

submission_4 = pd.DataFrame(data=final_data_4)



final_data_5 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_xgb}

submission_5 = pd.DataFrame(data=final_data_5)



final_data_comp = {'PassengerId': test.PassengerId, 'Survived_vc_hard': y_hat_vc_hard, 'Survived_rf': y_hat_rf, 'Survived_vc_soft' : y_hat_vc_soft, 'Survived_vc_all' : y_hat_vc_all,  'Survived_vc_xgb' : y_hat_vc_xgb}

comparison = pd.DataFrame(data=final_data_comp)
#track differences between outputs 

comparison['difference_rf_vc_hard'] = comparison.apply(lambda x: 1 if x.Survived_vc_hard != x.Survived_rf else 0, axis =1)

comparison['difference_soft_hard'] = comparison.apply(lambda x: 1 if x.Survived_vc_hard != x.Survived_vc_soft else 0, axis =1)

comparison['difference_hard_all'] = comparison.apply(lambda x: 1 if x.Survived_vc_all != x.Survived_vc_hard else 0, axis =1)

comparison.difference_hard_all.value_counts()
#prepare submission files 

submission.to_csv('submission_rf.csv', index =False)

submission_2.to_csv('submission_vc_hard.csv',index=False)

submission_3.to_csv('submission_vc_soft.csv', index=False)

submission_4.to_csv('submission_vc_all.csv', index=False)

submission_5.to_csv('submission_vc_xgb2.csv', index=False)