import pandas as pd



train = pd.read_csv("/kaggle/input/titanic/train.csv")



test = pd.read_csv("/kaggle/input/titanic/test.csv")
import pandas_profiling



train.profile_report()
train.info()
test.info()
train.head()
num_features = ['Age', 'SibSp', 'Parch', 'Fare']

cat_features = ['Survived', 'Pclass', 'Sex', 'Embarked']

cat_other = ['PassengerId', 'Name', 'Ticket', 'Cabin']



train[num_features].describe()
%matplotlib inline

import matplotlib as plt



train[num_features].hist(bins=8, figsize=(10,10))
num_passengers_in_train_set = 891



num_pass_without_child_or_parent = train.loc[train.Parch == 0]['Parch'].count()

percent_pass_without_child_or_parent = 100 * num_pass_without_child_or_parent / num_passengers_in_train_set

print("%2d passengers (%2d%%) travelled without a parent or child" %(num_pass_without_child_or_parent, percent_pass_without_child_or_parent) )



num_pass_without_sib_or_spouse = train.loc[train.SibSp == 0]['SibSp'].count()

percent_pass_without_sib_or_spouse = 100 * num_pass_without_sib_or_spouse / num_passengers_in_train_set

print("%2d passengers (%2d%%) travelled without a sibling or spouse" %(num_pass_without_sib_or_spouse, percent_pass_without_sib_or_spouse) )
train[cat_features].describe(include=['O']) # include strings in the white list of features
train[cat_other].describe(include=['O']) # include strings in the white list of features
# who resided in cabins C23, C25 and C27?

train.loc[train.Cabin == 'C23 C25 C27'][['Name', 'Ticket']].values
# who had ticket 347082? 

train.loc[train.Ticket == '347082'][['Name', 'Ticket', 'Cabin', 'Fare', 'Pclass']].values
correlation_matrix = train.corr() # Compute pairwise correlation of columns, excluding NA/null values.

correlation_matrix['Survived']
train_children = train.loc[train.Age <= 12].copy()

num_children = train_children.shape[0]

num_children_survived = train_children[train_children.Survived == 1].shape[0]

percent_children_survived = 100 * num_children_survived / num_children



num_children_in_pc1 = train_children.loc[train_children.Pclass == 1].shape[0]

num_children_in_pc2 = train_children.loc[train_children.Pclass == 2].shape[0]

num_children_in_pc3 = train_children.loc[train_children.Pclass == 3].shape[0]



print('%d (%d%%) of the %d children (<=12 years) survived' %(num_children_survived, percent_children_survived, num_children ))

print('%d, %d and %d children travelled in 1st, 2nd and 3rd class respectively' %(num_children_in_pc1,num_children_in_pc2, num_children_in_pc3 ))





train_children[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass', ascending=True)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) # return object with group labels as the index, sorted by Survived.
# Determine which passengers have missing Embarkation data

train.loc[pd.isnull(train.Embarked)]
train.loc[train.Cabin == 'B28']
train.loc[train.Ticket == '113572']
# calculate median values for train and test sets

#

# Using a SimpleImputer is overkill in this circumstance as it calculates the median/mode values for all features. We only need two in total in each dataset

from sklearn.impute import SimpleImputer



# Store PassengerIds for the test data as we'll need them for submission

test_passenger_ids = test['PassengerId']



# Drop the Cabin column

for data in [train, test]: 

    # Note: this drop must be performed inplace else the contents of train and test will not be updated.

    data.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)



# print(test.loc[pd.isnull(test.Fare)])       

for data in [train, test]: 

    imputer = SimpleImputer(strategy='median')

    data_trans = pd.DataFrame(imputer.fit_transform(data[['Fare']]), columns=['Fare'], index=data.index) #DataFrame

    data.update(data_trans) # update the original DataFrame with imputed values



# Replace missing Embarked data with most frequent (mode) imputed values

for data in [train, test]:

    imputer = SimpleImputer(strategy='most_frequent') # most_frequent == mode

    data_trans = pd.DataFrame(imputer.fit_transform(data[['Embarked']]), columns=['Embarked'], index=data.index) #DataFrame

    data.update(data_trans) # update the original DataFrame with imputed values
# create dummy variables for Embarked

train_dummies = pd.get_dummies(train, columns=['Embarked'], prefix=['Embarked'])

train = train_dummies



test_dummies = pd.get_dummies(test, columns=['Embarked'], prefix=['Embarked'])

test = test_dummies



# map male and female values to 1 and 0 respectively    

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)  

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)  



train.info()
# The following code is based on notebook https://www.kaggle.com/startupsci/titanic-data-science-solutions

#

for data in [train, test]:

    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train['Title'], train['Sex'])
for data in [train, test]:

    data['Title'].replace('Mlle', 'Miss', inplace=True)

    data['Title'].replace('Ms', 'Miss', inplace=True)

    data['Title'].replace('Mme', 'Mrs', inplace=True)   



train[['Title', 'Survived']].groupby(['Title'], as_index=False).count()
for data in [train, test]:

    data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other', inplace=True)



print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).count())
train.head()
train_dummies = pd.get_dummies(train, columns=['Title'], prefix=['Title'])

train = train_dummies



test_dummies = pd.get_dummies(test, columns=['Title'], prefix=['Title'])

test = test_dummies





for data in [test, train]:

    data.drop(['Name'], axis=1, inplace=True)
train.head()
#

# Replace missing Age data with iteratively imputed values

#

#

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor



for data in [test, train]:

    

    #iterative_regressor = IterativeImputer(random_state=0, max_iter=100, estimator=DecisionTreeRegressor(max_features=None, random_state=0))

    iterative_regressor = IterativeImputer(random_state=0, max_iter=100, estimator=KNeighborsRegressor(n_neighbors=15))



    iterative_regressor.fit(data[['Age', 'Sex', 'Pclass', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Other']])



    data_imp = iterative_regressor.transform(data[['Age', 'Sex', 'Pclass', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Other']])

    data_temp = pd.DataFrame(data_imp, columns=['Age', 'Sex', 'Pclass', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Other'], index=data.index) #DataFrame

    

    # replace Age with imputed values

    data['Age'] = data_temp['Age']

train.info()

test.info()
train['FamilySize'] = 0

test['FamilySize'] = 0





for data in [train, test]:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

    data['PassengerFare'] = data.Fare/data.FamilySize
train.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
# Even though PassengerFare is a better measure of what each passenger paid, It's Pearson correlation with Survived is less than Fare. 

# Fare and Passenger and of course highly correlated so the new Feature PasengerFare should be dropped.

#

for data in [train, test]:

    data.drop(['PassengerFare','Embarked_Q'], axis=1, inplace=True)
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(random_state=1, n_estimators=100)



#dtc_rfe = RFECV(model, step = 1, scoring = 'accuracy', cv = cv_split)

rfc_rfe = RFECV(model, step = 1, scoring = 'accuracy', cv = 10)

rfc_rfe.fit(train.drop(['Survived'], axis=1), train['Survived'].copy()) # remove the target 'Survived' before performing RFE



print('The optimal number of features :', rfc_rfe.n_features_)

print('The optimal features are:', (train.drop(['Survived'], axis=1)).columns[rfc_rfe.support_])
#reduce train and test dataframes. Select only the optimal features.

train_survived = train['Survived'].copy()

train.drop(['Survived'], axis=1, inplace=True)

train = pd.DataFrame(train.values[:,rfc_rfe.support_], columns=train.columns[rfc_rfe.support_], index=train.index)

train['Survived'] = train_survived # re-add Survived



test = pd.DataFrame(test.values[:,rfc_rfe.support_], columns=test.columns[rfc_rfe.support_], index=test.index)



train.head()
train_X = train.drop(['Survived'], axis=1)

train_Y = train['Survived'].copy()

test_X = test # there are no labels for test
# Enable when required by a new model.

#

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()



# note we perform a fit and transform on train_X. We then use the same fit to transform test_X.

train_X = standard_scaler.fit_transform(train_X)



test_X = standard_scaler.transform(test_X)
# import modules required by each model building code segment.

import numpy as np 

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.model_selection import ShuffleSplit



import pprint



pp = pprint.PrettyPrinter(indent=4)



# store scores in a Dict for later comparison

models = {}



# an alternative to simply specifying the number of folds for cv in each model fit operation

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

from sklearn.linear_model import LogisticRegressionCV



model_name = 'lrc_model'

models[model_name] = {}



# create an instance of lrc model with untuned parameters

model = LogisticRegressionCV(random_state=1, cv=10, max_iter=1000)



# fit model to training data with cross-validation

model.fit(train_X, train_Y)



# store model

models[model_name]['model'] = model



# store score

models[model_name]['crossval_score'] = model.score(train_X, train_Y, sample_weight=None)



pp.pprint(models[model_name])
from sklearn.tree import DecisionTreeClassifier



model_name = 'dtc_model'

models[model_name] = {}



# create an instance of dtc model with untuned parameters

model = DecisionTreeClassifier(random_state=1)



# fit model using all training data

model.fit(train_X, train_Y)



# store model

models[model_name]['model'] = model



# store non-cross validation score. This represents the mean accuracy with overfitting.

models[model_name]['score'] = model.score(train_X, train_Y, sample_weight=None)



# store cross-val_score

models[model_name]['cross_val_score'] = cross_val_score(model, train_X, \

                                                             train_Y, scoring="accuracy", cv=10).mean()



pp.pprint(models[model_name])
from sklearn.ensemble import RandomForestClassifier



model_name = 'rfc_model'

models[model_name] = {}



# create an instance of rfc model with untuned parameters

model = RandomForestClassifier(random_state=1)



# fit model using all training data

model.fit(train_X, train_Y)



# store model

models[model_name]['model'] = model



# store non-cross validation score. This represents the mean accuracy with overfitting.

models[model_name]['score'] = model.score(train_X, train_Y, sample_weight=None)



# store cross-val_score

models[model_name]['cross_val_score'] = cross_val_score(model, train_X, \

                                                             train_Y, scoring="accuracy", cv=10).mean()



pp.pprint(models[model_name])
#

# TODO This needs to be revised to reflext lrcCV and not LRC. The scoring needs to be revised too.

#

model_name = 'lrc_model'



models[model_name]['param_grid'] = [

        {'max_iter':[1000]}]



grid_search = GridSearchCV(models[model_name]['model'], models[model_name]['param_grid'], cv=10,

                           scoring='accuracy', return_train_score=True)



grid_search.fit(train_X, train_Y)





# store best model and params

models[model_name]['best_params'] = grid_search.best_params_

models[model_name]['best_estimator'] = grid_search.best_estimator_

#

# Modify this!!!

#

models[model_name]['best_score'] = grid_search.best_score_





pp.pprint(models)
model_name = 'dtc_model'



models[model_name]['param_grid'] = [

        {'max_features': ['auto'], 'max_depth': [7,8,9], 'criterion': ['gini', 'entropy']}

]



#

#grid_search = GridSearchCV(models[model_name]['model'], models[model_name]['param_grid'], cv=10,

#                           scoring='accuracy', return_train_score=True)

    

grid_search = GridSearchCV(models[model_name]['model'], models[model_name]['param_grid'], cv=cv_split,

                           scoring='accuracy', return_train_score=True)    



grid_search.fit(train_X, train_Y)





# store best model and params

models[model_name]['best_params'] = grid_search.best_params_

models[model_name]['best_estimator'] = grid_search.best_estimator_

models[model_name]['best_score'] = grid_search.best_score_



pp.pprint(models[model_name])
model_name = 'rfc_model'



models[model_name]['param_grid'] = [

        {'n_estimators': [100], 'oob_score': [True], 'max_features': ['auto'], 'max_depth': [7,8,9], 'criterion': ['gini', 'entropy']}

]



#

#grid_search = GridSearchCV(models[model_name]['model'], models[model_name]['param_grid'], cv=10,

#                           scoring='roc_auc', return_train_score=True)



grid_search = GridSearchCV(models[model_name]['model'], models[model_name]['param_grid'], cv=cv_split,

                           scoring='roc_auc', return_train_score=True)







grid_search.fit(train_X, train_Y)





# store best model and params

models[model_name]['best_params'] = grid_search.best_params_

models[model_name]['best_estimator'] = grid_search.best_estimator_

models[model_name]['best_score'] = grid_search.best_score_



pp.pprint(models[model_name])
import datetime



today = datetime.datetime.now()

file_name = today.strftime("%Y%m%d%H%M%S")



best_score = 0

# Determine the best model

for key, value in models.items():

    if (value['best_score'] > best_score):

        best_score = value['best_score']

        best_model = key

        best_estimator = value['best_estimator']



print("The best model is",best_model, "with a score of", best_score)        

predictions = best_estimator.predict(test_X)



output_predictions = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})

output_predictions.to_csv(file_name+"prediction.csv", index=False)



f = open(file_name+"models.txt","w")    

pp = pprint.PrettyPrinter(indent=4, stream=f)

pp.pprint(models)

f.close()