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
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing and helper libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

#LOADING DIFFERENT ML MODELS
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression



train_data  = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
gender_sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
Id = test_data["PassengerId"]
train_data.head()
train_data.info()
test_data.info()
#printing only top 5 values
train_data.head()
#Storing both dataFrames as list dsp that they can be iterated over
data = [train_data, test_data]
print("Training data with null values:\n",train_data.isnull().sum())
print("\n\n\nTesting data with null values:\n",test_data.isnull().sum())
for data_ in data:
    # filling median of age at null values , and not mean because mean can get affected by outliers which are very few elderly people in this case
    data_["Age"].fillna(data_["Age"].median(), inplace=True)
    # Filling null cabin with mode  (which are most frequent ie cabins type in which most of people out of all passengers are staying are staying)
    data_["Cabin"].fillna(data_["Cabin"].mode()[0], inplace=True)
    # Filling fare with mean as fares are continous not discrete  
    data_["Fare"].fillna(data_["Fare"].mean(), inplace=True)
    # Filling embarked with mode ie same as  from where most people have embarked from.
    data_["Embarked"].fillna(data_["Embarked"].mode()[0], inplace=True)

    
print("Training data with null values:\n",train_data.isnull().sum())
print("\n\n\nTesting data with null values:\n",test_data.isnull().sum())
train_data.describe()



for data_ in data:
    
    #adding sibling, cousins, parents and the person to get the size of the family onboard
    data_["familysize"] = data_["SibSp"] + data_["Parch"] + 1;
    
    #Dividing age into bins of age-gap as it will be more categorical as compared to a continuos value like age
    data_["agebin"] = pd.cut(data_["Age"].astype(int), 5)
    
    #Creating bins for fares also 
    data_["farebin"] = pd.qcut(data_["Fare"],5)
    
    #Assigning 1 to everyone ie everyone is alone and then cheking if its family size if >1 then it is not alone so the value is set to 0 
    data_["alone"] = 1
    data_["alone"].loc[data_["familysize"] > 1] = 0

    
#Printing newly created features along with their values   
print("FamilySize :\n", data[0].familysize.value_counts())
print("\n\n\nAlone :\n", data[0].alone.value_counts())
print("\n\n\nAgeBin :\n", data[0].agebin.value_counts())
print("\n\n\nFareBin :\n", data[1].farebin.value_counts())
data[0].head()
data[1].head()
label = LabelEncoder()

for data_ in data:
    data_["Sex"] = label.fit_transform(data_["Sex"])
    data_["Embarked"] = label.fit_transform(data_["Embarked"])
    data_["Cabin"] = label.fit_transform(data_["Cabin"])
    data_["agebin"] = label.fit_transform(data_["agebin"])
    data_["farebin"] = label.fit_transform(data_["farebin"])
data[0].head()
print("Training data with null values:\n",train_data.info())
print("\n\n\nTesting data with null values:\n",test_data.info())


#Plotting histograms for various features

plt.figure(figsize=(10,10))

plt.subplot(221)
plt.hist(x = [train_data[train_data['Survived']==1]['familysize'], train_data[train_data['Survived']==0]['familysize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(222)
plt.hist(x = [train_data[train_data['Survived']==1]['farebin'], train_data[train_data['Survived']==0]['farebin']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('FareBin Histogram by Survival')
plt.xlabel('FareBin(#)')
plt.ylabel('# of Passengers')
plt.legend()


plt.subplot(223)
plt.hist(x = [train_data[train_data['Survived']==1]['Age'], train_data[train_data['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (#)')
plt.ylabel('# of Passengers')
plt.legend()


plt.subplot(224)
plt.hist(x = [train_data[train_data['Survived']==1]['Sex'], train_data[train_data['Survived']==0]['Sex']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Sex Histogram by Survival')
plt.xlabel('Sex (#)')
plt.ylabel('# of Passengers')
plt.legend()

plt.figure(figsize = (16,10))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
train_data.columns
clean_train = np.array( train_data.drop(columns=['PassengerId', 'Survived', 'Name','Ticket']))
clean_target = np.array( train_data["Survived"] )
clean_test_data = np.array( test_data.drop(columns=['PassengerId','Name','Ticket']))
print(clean_train.shape, clean_test_data.shape)

#SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(clean_train, clean_target, test_size=0.25, random_state=42)
print(X_train.shape,y_train.shape, X_test.shape, y_test.shape)
random_forest_clf = RandomForestClassifier(n_estimators=10)
extra_trees_clf = ExtraTreesClassifier(n_estimators=10)
lg_reg_clf = LogisticRegression()
bag_clf = BaggingClassifier(n_estimators=10)
ada_clf = AdaBoostClassifier(n_estimators=10)
grad_clf = GradientBoostingClassifier(n_estimators=10)
xgb_clf = XGBClassifier(n_estimators=10)
# Helping_Function to show Cross Val Scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
# Estimators
estimators = [random_forest_clf, extra_trees_clf, lg_reg_clf, bag_clf, ada_clf, grad_clf, xgb_clf]

# STORING CROSS VAL-SCORE IN A LIST SCORES
scores= []
for estimator in estimators:
    scores.append(cross_val_score(estimator, X_train, y_train, cv=7))

# PRINTING ACCURACY AND MEAN OF EACH ESTIMATORS
i = 0
for score in scores:
    print("-"*10 +estimators[i].__class__.__name__ + "-"*10 + "\n")
    display_scores(score)
    print("\n\n")
    i = i+1
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("lg_reg_clf", lg_reg_clf),
    ("bag_clf", bag_clf),
    ("ada_clf", ada_clf),
    ("grad_clf", grad_clf),
    ("xgb_clf", xgb_clf),
    
]
# As we have seen above , the accuracy of some models are low as compared to others and they might be reducing the accuracy of your voting model so I have created a new list containing only top 4 classifiers.
selected_named_estimators = [
    ("bag_clf", bag_clf),
    ("ada_clf", ada_clf),
    ("grad_clf", grad_clf),
    ("xgb_clf", xgb_clf),
    
]
# Both voting model , one with all estimators and one with selected estimators

all_estimators_voting_clf = VotingClassifier(named_estimators, voting="soft")
selected_estimators_voting_clf = VotingClassifier(selected_named_estimators, voting = "soft")
all_estimators_voting_scores = cross_val_score(all_estimators_voting_clf, X_train ,y_train, cv=7)
selected_estimators_voting_scores = cross_val_score(selected_estimators_voting_clf, X_train ,y_train, cv=7)
print("VOTING WITH ALL ESTIMATORS & WITHOUT TUNING :\n")
display_scores(all_estimators_voting_scores)

print("\n\n"+"%"*100+"\n\n\nVOTING WITH SELECTED ESTIMATORS & WITHOUT TUNING :\n")
display_scores(selected_estimators_voting_scores)

# Setting values for all the parameters

grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

# You can use google and  references for getting all parameters 
grid_param = [
             [{
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }],
            
            
            [{
            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'random_state': grid_seed
             }],
            
            [{
            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
            'fit_intercept': grid_bool, #default: True
            #'penalty': ['l1','l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
            'random_state': grid_seed
             }],
            
      
            [{
            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio, #default=1.0
            'random_state': grid_seed
             }],

    
            [{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }],
    
            [{
            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            #'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth, #default=3   
            'random_state': grid_seed
             }],

            [{
            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
            'learning_rate': grid_learn, #default: .3
            'max_depth': [1,2,4,6,8,10], #default 2
            'n_estimators': grid_n_estimator, 
            'seed': grid_seed  
             }]   
    
            
        ]

# for clf, param in zip (named_estimators, grid_param):
#     best_search = GridSearchCV(estimator = clf[1], param_grid = param, cv = 10, scoring = 'roc_auc')
#     best_search.fit(X_train, y_train)
#     best_param = best_search.best_params_
#     print('The best parameter for {} is {} '.format(clf[1].__class__.__name__, best_param))

best_random_forest_clf = RandomForestClassifier( criterion="entropy", n_estimators=100, max_depth=8, oob_score=True, random_state=0)
best_extra_trees_clf = ExtraTreesClassifier( criterion="entropy", n_estimators=300, max_depth=6, random_state=0)
best_lg_reg_clf = LogisticRegression( fit_intercept= True, random_state= 0, solver= 'newton-cg')
best_bag_clf = BaggingClassifier( max_samples = 0.25, n_estimators= 300,  random_state=0)
best_ada_clf = AdaBoostClassifier( learning_rate=0.1, n_estimators=100, random_state=0)
best_grad_clf = GradientBoostingClassifier( learning_rate = 0.05, max_depth= 4, n_estimators=300, random_state=0)
best_xgb_clf = XGBClassifier( learning_rate=0.25, max_depth=6, n_estimators=100, seed=0)
all_estimators_tuned = [
    ("random_forest_clf", best_random_forest_clf),
    ("extra_trees_clf", best_extra_trees_clf),
    ("lg_reg_clf", best_lg_reg_clf),
    ("bag_clf", best_bag_clf),
    ("ada_clf", best_ada_clf),
    ("grad_clf", best_grad_clf),
    ("xgb_clf", best_xgb_clf),
]
estimators = [best_random_forest_clf, best_extra_trees_clf, best_lg_reg_clf, best_bag_clf, best_ada_clf, best_grad_clf, best_xgb_clf]

scores= []
for estimator in estimators:
    scores.append(cross_val_score(estimator, X_train, y_train, cv=7))
i = 0
for score in scores:
    print("-"*10 +estimators[i].__class__.__name__ + "-"*10 + "\n")
    display_scores(score)
    print("\n\n")
    i = i+1

selected_estimators_tuned = [
    ("random_forest_clf", best_random_forest_clf),
    ("extra_trees_clf", best_extra_trees_clf),
    ("bag_clf", best_bag_clf),
    ("grad_clf", best_grad_clf),

]
all_estimators_tuned_voting_clf = VotingClassifier(all_estimators_tuned, voting="soft")
selected_estimators_tuned_voting_clf = VotingClassifier(selected_estimators_tuned, voting="soft")
#Their cross-val scores calculation
all_tuned_voting_clf_scores = cross_val_score(all_estimators_tuned_voting_clf,X_train, y_train,cv=7)
selected_tuned_voting_clf_scores = cross_val_score(selected_estimators_tuned_voting_clf,X_train, y_train,cv=7)
print("VOTING WITH ALL ESTIMATORS & WITHOUT TUNING :\n")
display_scores(all_estimators_voting_scores)

print("\n\n"+"%"*100+"\n\n\nVOTING WITH SELECTED ESTIMATORS & WITHOUT TUNING :\n")
display_scores(selected_estimators_voting_scores)

print("\n\n"+"%"*100+"\n\n\nVOTING WITH ALL ESTIMATORS & WITH TUNING :\n")
display_scores(all_tuned_voting_clf_scores)

print("\n\n"+"%"*100+"\n\n\nVOTING WITH SELECTED ESTIMATORS & WITH TUNING :\n")
display_scores(selected_tuned_voting_clf_scores)


all_estimators_voting_clf.fit(X_train, y_train)
selected_estimators_voting_clf.fit(X_train, y_train)
all_estimators_tuned_voting_clf.fit(X_train, y_train)
selected_estimators_tuned_voting_clf.fit(X_train, y_train)
pred_allestimators_without_tuning = all_estimators_voting_clf.predict(clean_test_data)
pred_selectedestimators_without_tuning = selected_estimators_voting_clf.predict(clean_test_data)
pred_allestimators_with_tuning = all_estimators_tuned_voting_clf.predict(clean_test_data)
pred_selectedestimators_with_tuning = selected_estimators_tuned_voting_clf.predict(clean_test_data)

submission = pd.DataFrame({'PassengerId':Id, 'Survived':pred_allestimators_without_tuning})
submission.to_csv('submission_pred_allestimators_without_tuning.csv', index = False) 
submission = pd.DataFrame({'PassengerId':Id, 'Survived':pred_selectedestimators_without_tuning})
submission.to_csv('submission_pred_selectedestimators_without_tuning.csv', index = False)

submission = pd.DataFrame({'PassengerId':Id, 'Survived':pred_allestimators_with_tuning})
submission.to_csv('submission_pred_allestimators_with_tuning.csv', index = False)

submission = pd.DataFrame({'PassengerId':Id, 'Survived':pred_selectedestimators_with_tuning})
submission.to_csv('submission_pred_selectedestimators_with_tuning.csv', index = False)


