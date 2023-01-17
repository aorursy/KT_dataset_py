import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# read individual CSVs into pandas DataFrames
train = pd.read_csv('../input/train.csv', index_col = 'PassengerId')
test = pd.read_csv('../input/test.csv', index_col = 'PassengerId')
# add "Survived" column to test & combine into a single DataFrame
import numpy as np

test['Survived'] = np.nan
data = train.append(test)
data.head()
# look at counts for target variable 'survived'
pd.crosstab(index = data['Survived'], columns = 'Count')
data.isnull().sum()
data.Cabin.head(10)
# Define new variable 'Deck' from first character of the string in 'Cabin'
data['Deck'] = data['Cabin'].astype(str).str[0]

# replace n with NaNs
data['Deck'] = data.Deck.replace('n', np.nan)

# check again
data.Deck.head(10)
# Adding 1 (the passenger) + Parch (# of parents & children traveling with) + SibSp (# of siblings & spouses traveling with)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# check new variable
data.FamilySize.head(10)
# crosstab of FamilySize
pd.crosstab(data['FamilySize'], data['Survived'])
# make a dataframe of just FamilySize and Survived
famsizes = data[['FamilySize','Survived']]

sns.set_context('talk')
sns.barplot(x = 'FamilySize', y = 'Survived', data = famsizes, color = 'green')
# new variables (fun with list comprehensions!)
data['Alone'] = [1 if familysize == 1 else 0 for familysize in data['FamilySize']]
data['LargeFamily'] = [1 if familysize >= 5 else 0 for familysize in data['FamilySize']]
data['SmallFamily'] = [1 if familysize >= 2 and familysize < 5 else 0 for familysize in data['FamilySize']]
# crosstab of FamilySize
pd.crosstab(data['FamilySize'], 'Count')
# crosstabs of new variables
pd.crosstab(data['Alone'], 'Count')
pd.crosstab(data['LargeFamily'], 'Count') # 22 + 25 + 16 + 8 + 11 = 82
pd.crosstab(data['SmallFamily'], 'Count') # 235 + 159 + 43 = 437
data.Name.head()
# Split off last name into new column
data['LastName'] = data['Name'].str.split(',').str.get(0)
data['LastName'] = data['LastName'].str.strip() # strip whitespace from ends

# Split off Title into new column
data['Title'] = data['Name'].str.split('.').str.get(0)
data['Title'] = data['Title'].str.split(',').str.get(1)
data['Title'] = data['Title'].str.strip()

# Excise parenthetical full maiden names
data['MaidenName'] = data['Name'].str.split('(').str.get(1)
data['MaidenName'] = data['MaidenName'].str.split(')').str.get(0)

# Excise nicknames
data['Nickname'] = data['Name'].str.split('"').str.get(1)
data['Nickname'] = data['Nickname'].str.strip()

# Get first name
data['FirstName'] = data['Name'].str.split('.').str.get(1)
data['FirstName'] = data['FirstName'].str.split(' ').str.get(1)
# get maiden first names
data['MaidenFirstName'] = data['MaidenName'].str.split(' ').str.get(0)
# Replace FirstName with MaidenFirstName except with the NaNs filled with 
# FirstName and strip
data['FirstName'] = data['MaidenFirstName'].fillna(data['FirstName'])
data['FirstName'] = data['FirstName'].str.strip()
# drop MaidenFirstName
data = data.drop(['MaidenFirstName'], axis = 1)
# Get MaidenLastName from MaidenName
data['MaidenLastName'] = data['MaidenName'].str.rsplit(' ', expand = True, n=1)[1]
# replace 'None' with NaN and strip MaidenLastName
data['MaidenLastName'] = data.MaidenLastName.replace('None', np.nan)
data['MaidenLastName'] = data['MaidenLastName'].str.strip()
# Drop MaidenName
data = data.drop('MaidenName', axis =1)
data.Ticket.head()
# Let's define a ticket prefix as all the letters coming before the first 
# space. Some have '.' or '/' in the letters so we first have to remove those.
data['Ticket'] = data.Ticket.str.replace('.','')
data['Ticket'] = data.Ticket.str.replace('/','')

data.Ticket.head()
# Now split at the spaces
data['TicketPrefix'] = data.Ticket.str.split().str.get(0)

# list comprehension to replace numeric values of TicketPrefix with 'None'    
data['TicketPrefix'] = ['None' if prefix.isnumeric() == True else prefix for prefix in data.TicketPrefix]

# take a look again at the cleaned up TicketPrefix Variable
data['TicketPrefix'].tail(25)
# counts of TicketPrefix
data['TicketPrefix'].value_counts()
# Replace 'SCParis' with 'SCPARIS'
data.TicketPrefix = data.TicketPrefix.str.replace('SCParis', 'SCPARIS')
# combine prefix categories with single members into new cateogry, 'Unique'.
prefixes = data['TicketPrefix'].value_counts()
uniquePrefixes = list(prefixes[prefixes == 1].index)
data.TicketPrefix = ['Unique' if prefix in uniquePrefixes else prefix for prefix in data.TicketPrefix]

# look at value counts again
data['TicketPrefix'].value_counts()
# make dummies for TicketPrefix
prefixDummies = pd.get_dummies(data['TicketPrefix'], prefix = 'TicketPrefix')
data = pd.concat([data, prefixDummies], axis = 1)
# create flag variables for each column with missing values where 1 means value was imputed and 0 means value was not imputed
for column in data.columns:
    if data[column].isnull().sum() != 0:
        data[column + '_M'] = data[column].isnull()
        data[column + '_M'] = data[column + '_M'].astype('int64').replace('True', 1)
        data[column + '_M'] = data[column + '_M'].astype('int64').replace('False', 0)

# Rename the 'Survived_M' variable as Test since it identifies the members of the test set
data.rename(columns = {'Survived_M':'Test'}, inplace = True)

data.columns
data = data.drop(['Cabin', 'Cabin_M'], axis = 1)
# check number of missing values for each variable again
data.isnull().sum()
# Look at Embarked
pd.crosstab(index = data['Embarked'], columns = 'Count') # 914 'S'
# replace those with the most common value ('S')
data.Embarked = data.Embarked.astype('str').replace('nan', 'S')

# Check again
pd.crosstab(index = data['Embarked'], columns = 'Count') # 916 'S'
# Make dummies for the Embarked variable
embarkedDummies = pd.get_dummies(data['Embarked'], prefix = 'Embarked')
data = pd.concat([data, embarkedDummies], axis = 1)
# Find the Pclass value for the passenger missing an entry for Fare
data['Pclass'][data['Fare'].isnull() == True] # Pclass = 3
# Set missing Fare value equal to the average for 3rd Class
data['Fare'] = data['Fare'].fillna(data.groupby('Pclass').Fare.mean()[3])
# replace female/male with 1/0
data['Sex'] = data['Sex'].replace('female', 1)
data['Sex'] =data['Sex'].replace('male', 0)
pd.crosstab(data['Title'], 'Count')
# replace non-English, non-honorific titles with English versions
data.Title = data.Title.str.replace('Don', 'Mr')
data.Title = data.Title.str.replace('Dona', 'Mrs')
data.Title = data.Title.str.replace('Mme', 'Mrs')
data.Title = data.Title.str.replace('Ms', 'Mrs')
data.Title = data.Title.str.replace('Mra', 'Mrs')
data.Title = data.Title.str.replace('Mlle', 'Miss')

pd.crosstab(data['Title'], 'Count')
data['Title_Dr'] = [1 if title in ['Dr'] else 0 for title in data['Title']]

data['Title_Rev'] = [1 if title in ['Rev'] else 0 for title in data['Title']]
militaryTitles = ['Capt', 'Col', 'Major']
data['MilitaryTitle'] = [1 if title in militaryTitles else 0 for title in data['Title']]

nobleTitles = ['Jonkheer', 'Lady', 'Sir', 'the Countess']
data['NobleTitle'] = [1 if title in nobleTitles else 0 for title in data['Title']]
data.loc[data['Title'].isin(['Jonkheer'])]['Age']    # 38
# replace male special titles with Mr.
male = dict.fromkeys(['Dr','Rev', 'Capt', 'Col', 'Major', 'Jonkheer', 'Sir'], 'Mr')
data['Title'] = data.Title.replace(male)
# check ages of the Lady and the Countess
data.loc[data['Title'].isin(['Lady'])]['Age']   # 48
data.loc[data['Title'].isin(['the Countess'])]['Age']    # 33
# replace Lady and the Countess with Mrs in Title column
female = dict.fromkeys(['Lady', 'the Countess'], 'Mrs')
data['Title'] = data.Title.replace(female)
sns.set_context('talk')
sns.set_style('darkgrid')

# boxplot of Age by Title
sns.boxplot(x='Title', y='Age', data=data)
# two histogram plots, one for males + one for females
gents = ['Master', 'Mr']
colGents = ['blue', 'green']
for i in range(len(gents)):
    a_gents = sns.distplot(data[data['Title'] == gents[i]].Age.dropna(), 
                                label = gents[i],
                                color = colGents[i])
a_gents.legend()
ladies = ['Miss', 'Mrs']
colLadies = ['orange', 'red']
for i in range(len(ladies)):
    a_ladies = sns.distplot(data[data['Title'] == ladies[i]].Age.dropna(),
                            label = ladies[i],
                            color = colLadies[i])
a_ladies.legend()
# Masters
masters = pd.DataFrame(data[data.Title == 'Master'].Age, columns = ['Age'])
masters.Age = masters.fillna(masters.median())
data = data.combine_first(masters)

# Miss's
misses = pd.DataFrame(data[data.Title == 'Miss'].Age, columns = ['Age'])
misses.Age = misses.fillna(misses.median())
data = data.combine_first(misses)

# Mr's
misters = pd.DataFrame(data[data.Title == 'Mr'].Age, columns = ['Age'])
misters.Age = misters.fillna(misters.median())
data = data.combine_first(misters)

# Mrs's
missuses = pd.DataFrame(data[data.Title == 'Mrs'].Age, columns = ['Age'])
missuses.Age = missuses.fillna(missuses.median())
data = data.combine_first(missuses)

# check missing values
data.Age.isnull().sum()
sns.boxplot(x='Title', y='Age', data=data)
# get dummies
titledummies = pd.get_dummies(data['Title'], prefix = 'Title')
# add to dataset
data = pd.concat([data, titledummies], axis=1)
# crosstab of Deck
pd.crosstab(data['Deck'], columns = 'Count')
data[data.Deck == 'T'].Name
data[data.Deck == 'T'].Pclass
# crosstab of Pclass and Deck
pd.crosstab(data['Deck'], data['Pclass'])
data['Deck'] = data.Deck.str.replace('T', 'A')
# fill nan's with 'missing'
data.Deck = data.Deck.fillna('missing')
# crosstab
pd.crosstab(data[data['Pclass'] == 3]['Deck'], 'Count', margins = True)
data.loc[data['Pclass'] == 3, 'Deck'] = data.loc[data['Pclass'] == 3, 'Deck'].str.replace('missing', 'S')

# crosstab of Pclass and Deck
pd.crosstab(data['Deck'], data['Pclass'], margins = True)
# isolate train and test subsets for Deck imputation model
train_deck = data[data['Deck'] != 'missing'].drop(['Deck_M', 'Embarked', 'Fare_M',
                 'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
                 'Survived', 'Test', 'Ticket', 'TicketPrefix', 'Title'], axis = 1)

# drop the ticket prefix indicators for now
train_deck = train_deck[['Age', 'Age_M', 'Alone', 'Deck', 'Embarked_C', 'Embarked_M', 'Embarked_Q', 'Embarked_S', 
                         'FamilySize', 'Fare', 'LargeFamily', 'MaidenLastName_M', 'MilitaryTitle', 'Nickname_M', 
                         'NobleTitle', 'Parch', 'Pclass', 'Sex', 'SibSp', 'SmallFamily', 'Title_Dr', 'Title_Master', 
                         'Title_Mr', 'Title_Mrs', 'Title_Miss']]

# since we only have 1st and 2nd class passengers to predict Deck for, we'll only use 1st and 2nd class passengers to build 
# the model
train_deck = train_deck[train_deck.Pclass != 3] 

missing_deck = data[data['Deck'] == 'missing'].drop(['Deck_M', 'Embarked', 'Fare_M',
                 'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
                 'Survived', 'Test', 'Ticket', 'TicketPrefix', 'Title'], axis = 1)

missing_deck = missing_deck[['Age', 'Age_M', 'Alone', 'Deck', 'Embarked_C', 'Embarked_M', 'Embarked_Q', 'Embarked_S', 
                             'FamilySize', 'Fare', 'LargeFamily', 'MaidenLastName_M', 'MilitaryTitle', 'Nickname_M', 
                             'NobleTitle', 'Parch', 'Pclass', 'Sex', 'SibSp', 'SmallFamily', 'Title_Dr', 'Title_Master', 
                             'Title_Mr', 'Title_Mrs', 'Title_Miss']]

# separate into inputs and outputs
X_train_deck = train_deck.drop('Deck', axis = 1)
X_missing_deck = missing_deck.drop('Deck', axis = 1)
y_train_deck = train_deck['Deck']

# feature names
X_names = X_train_deck.columns
# take a look at correlations among predictors for Deck
corrDeckX = X_train_deck.corr()

# plot the heatmap
fig, ax = plt.subplots()
fig.set_size_inches(14, 10)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.heatmap(corrDeckX, 
            xticklabels=corrDeckX.columns,
            yticklabels=corrDeckX.columns,
            cmap = 'PiYG')
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV

# Set up a dict with values to test for each parameter/argument in the model object
deck_grid = {'max_depth'         : np.arange(1,30),
             'min_samples_split' : np.arange(2,20),
             'min_samples_leaf'  : np.arange(1,20)}

# Construct random forest object
tree_deck = DecisionTreeClassifier(random_state = 538, 
                                   max_features = 'sqrt',
                                   presort = True)
treeGrid = GridSearchCV(tree_deck, deck_grid)
# fit trees
treeGridFit = treeGrid.fit(X_train_deck,y_train_deck)
treeGrid.best_score_
treeGrid.best_params_
# Build the best tree model
treeDeckBest = DecisionTreeClassifier(random_state = 538, 
                                      max_features = 'sqrt',
                                      presort = True,
                                      max_depth = 14,
                                      min_samples_split = 8,
                                      min_samples_leaf = 1)

treeDeckBestFit = treeDeckBest.fit(X_train_deck, y_train_deck)
# calculate test accuracy estimate for best model (use default 3-fold CV)
cv_error_tree = np.mean(cross_val_score(treeDeckBest, X_train_deck, y_train_deck,
                                      scoring = 'accuracy'))
print('Est. Test Accuracy: ', cv_error_tree)
# Predict the missing values of Deck
missing_deck['Deck'] = treeDeckBest.predict(X_missing_deck)

# replace 'missing' entries in data 'Deck' column with NaNs again
data.Deck = data.Deck.replace('missing', np.nan)

# use combine_first to replace missing values in data with imputed values now
# in missing_deck
data = data.combine_first(missing_deck)
# crosstab of Pclass and Deck
pd.crosstab(data['Deck'], data['Pclass'], margins = True)
data.isnull().sum()
# Make dummies for the imputed Deck variable
deckDummies = pd.get_dummies(data['Deck'], prefix = 'Deck')
data = pd.concat([data, deckDummies], axis = 1)
data.columns
# Separate out into train and test sets; drop text variables
test_model = data[data['Survived'].isnull()].drop(['Deck', 'Embarked', 'Fare_M',
            'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
            'Test', 'Ticket', 'TicketPrefix', 'Title', 'Deck_M'], axis = 1)
train_model = data[data['Survived'].notnull()].drop(['Deck', 'Embarked', 'Fare_M',
            'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
            'Test', 'Ticket', 'TicketPrefix', 'Title', 'Deck_M'], axis = 1)
# take a look at correlations among the final set of predictors and the target, Survived
corrModelX = train_model.corr()

fig, ax = plt.subplots()
sns.set_context('talk')
sns.set_style('darkgrid')
fig.set_size_inches(15,10)

# plot the heatmap
sns.heatmap(corrModelX, 
            xticklabels=corrModelX.columns,
            yticklabels=corrModelX.columns,
            cmap = 'PiYG')
# separate into inputs and outputs
X_train = train_model.drop(['Survived'], axis = 1)
y_train = train_model['Survived']
X_test = test_model.drop(['Survived'], axis = 1)

# Feature names
features = X_train.columns
# Create a dataframe to hold model results
model_results = pd.DataFrame(columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
# set up 5-fold and 10-fold cross-validation schemes for test error estimation
from sklearn.model_selection import KFold
cv_5fold = KFold(n_splits = 5, shuffle = True, random_state = 237) 

cv_10fold = KFold(n_splits = 10, shuffle = True, random_state = 237)
# First, LogisticRegression:
from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(penalty = 'l1',
                         random_state = 555,
                         solver = 'liblinear')

# fit the model
lr1_fit = lr1.fit(X_train, y_train)

# make predictions on the training set
lr1_preds = lr1.predict(X_train)

# confusion matrix for training set
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_train, lr1_preds)
# classification report
print(classification_report(y_train, lr1_preds,
                            target_names = ['Died', 'Survived']))
# training accuracy
lr1_trainAcc = lr1.score(X_train, y_train) # 0.8395
lr1_trainAcc
# calculate test accuracy estimate
lr1_testErrEst = np.mean(cross_val_score(lr1, X_train, y_train,
                                         scoring = 'accuracy', 
                                         cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', lr1_testErrEst)    # 0.8283
# construct a coefficent table
lr1_coefs = [coef for coef in lr1.coef_[0]]
featuresList = list(features)
lr1_coefs = pd.DataFrame(list(zip(featuresList, lr1_coefs)), columns = ['Feature', 'Coef'])
lr1_coefs
lr1_nonzeroCoef = lr1_coefs[lr1_coefs.Coef != 0]
lr1_nonzeroCoef
# add metrics to model_results dataframe
lr1_results = pd.DataFrame([['l1LogReg', lr1_trainAcc, lr1_testErrEst]],
                           columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(lr1_results)
model_results
# make predictions on test set
lr1_test = pd.DataFrame(lr1.predict(X_test).astype(int), 
                        columns = ['Survived'],
                        index = test.index)

# write to csv
lr1_test.to_csv('L1lr_test.csv')
from sklearn.linear_model import SGDClassifier

# Set up a dict with values to test for each parameter/argument in the model object
en1_grid = {'alpha'    : [0.005, 0.01, 0.015], 
            'l1_ratio' : np.arange(0, 1, 0.05)}

# SGD Classifier object; log loss makes this logistic regression
en1 = SGDClassifier(loss = 'log',
                    penalty = 'elasticnet',
                    random_state = 237,
                    learning_rate = 'optimal',
                    max_iter = 500)

# set up the grid search
en1_GridSearch = GridSearchCV(en1, en1_grid)

# fit trees
en1_fit = en1_GridSearch.fit(X_train,y_train)
en1_GridSearch.best_score_
en1_GridSearch.best_params_
# best model
en1_best = SGDClassifier(alpha = 0.005,
                         l1_ratio = 0.15,
                         loss = 'log',
                         penalty = 'elasticnet',
                         random_state = 237,
                         learning_rate = 'optimal',
                         max_iter = 500)

en1BestFit = en1_best.fit(X_train, y_train)
# make prediction on the training set
en1_preds = en1_best.predict(X_train)

# confusion matrix for training set
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_train, en1_preds)
# classification report
print(classification_report(y_train, en1_preds,
                            target_names = ['Died', 'Survived']))
# training accuracy
en1_trainAcc = en1_best.score(X_train, y_train)

# calculate test accuracy estimate for best model using 5-fold CV
en1_testErrEst = np.mean(cross_val_score(en1_best, X_train, y_train,
                                        scoring = 'accuracy',
                                        cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', en1_testErrEst)
# add metrics to model_results dataframe
en1_results = pd.DataFrame([['ElasticNet', en1_trainAcc, en1_testErrEst]],
                           columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(en1_results)
model_results
# make predictions on test set
en1_test = pd.DataFrame(en1_best.predict(X_test).astype(int), 
                       columns = ['Survived'],
                       index = test.index)

# write to csv
en1_test.to_csv('en1_test.csv')
# Set up a dict with values to test for each parameter/argument in the model object
rf1_grid = {'n_estimators'      : [20, 50, 100],
            'max_depth'         : np.arange(1,5),
            'min_samples_split' : np.arange(6,20,2),
            'min_samples_leaf'  : np.arange(3,15,3),
            'max_leaf_nodes'    : [5, 10, 15]}

# Construct random forest object
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(random_state = 555)

# set up the grid search
rf1_GridSearch = GridSearchCV(rf1, rf1_grid)

# fit trees
rf1_fit = rf1_GridSearch.fit(X_train,y_train)
rf1_GridSearch.best_score_
rf1_GridSearch.best_params_
# Build the best RF model with hyperparameters determined above
from sklearn.ensemble import RandomForestClassifier
rf1_best = RandomForestClassifier(max_depth = 3,
                                  max_leaf_nodes = 15,
                                  min_samples_leaf = 3,
                                  min_samples_split = 18,
                                  n_estimators = 20)

r1fBestFit = rf1_best.fit(X_train, y_train)

# make prediction on the training set
rf1_preds = rf1_best.predict(X_train)

# confusion matrix for training set
confusion_matrix(y_train, rf1_preds)
# classification report
print(classification_report(y_train, rf1_preds,
                            target_names = ['Died', 'Survived']))
# training accuracy
rf1_trainAcc = rf1_best.score(X_train, y_train)

# calculate test accuracy estimate for best model using 5-fold CV
rf1_testErrEst = np.mean(cross_val_score(rf1_best, X_train, y_train,
                                        scoring = 'accuracy',
                                        cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', rf1_testErrEst)
# add metrics to model_results dataframe
rf1_results = pd.DataFrame([['RandomForest', rf1_trainAcc, rf1_testErrEst]],
                           columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(rf1_results)

model_results
# make predictions on test set
rf1_test = pd.DataFrame(rf1_best.predict(X_test).astype(int), 
                       columns = ['Survived'],
                       index = test.index)

# write to csv
rf1_test.to_csv('rf1_test.csv')
# look at feature importances
rf1_importances = rf1_best.feature_importances_

headers_rf1 = ["Variable", "Importance"]
values_rf1 = pd.DataFrame(sorted(zip(X_train.columns, rf1_importances), key=lambda x: x[1] * -1), columns = headers_rf1)

# horizontal bar plot of importances
fig, ax = plt.subplots()
fig.set_size_inches(12, 13)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.barplot(x = 'Importance', y = 'Variable', data = values_rf1, orient = 'h', color = 'green')
from sklearn.ensemble import GradientBoostingClassifier
# Set up a dict with values to test for each parameter/argument in the model
# object
gbc_grid = {'n_estimators'        : np.arange(20,100,20),
            'learning_rate'       : [0.05, 0.1, 0.15],
            'max_features'        : np.arange(1,6),
            'max_depth'           : np.arange(1,4),
            'min_samples_split'   : np.arange(10,20,5),
            'min_samples_leaf'    : np.arange(3,21,7),
            'subsample'           : [0.3, 0.5, 0.7],
            'max_leaf_nodes'      : np.arange(5,20,5)}


# Construct random forest object
gbc = GradientBoostingClassifier(random_state = 555)

# set up the grid search
gbc_GridSearch = GridSearchCV(gbc, gbc_grid)

# fit learners
gbc_fit = gbc_GridSearch.fit(X_train,y_train)
gbc_GridSearch.best_score_
gbc_GridSearch.best_params_
from sklearn.ensemble import GradientBoostingClassifier
gbc_best = GradientBoostingClassifier(n_estimators = 40,
                                      learning_rate = 0.15,
                                      max_depth = 3,
                                      max_features = 5,
                                      min_samples_leaf = 3,
                                      min_samples_split = 10,
                                      max_leaf_nodes = 10,
                                      subsample = 0.5,
                                      random_state = 555)

gbcBestFit = gbc_best.fit(X_train, y_train)

# make prediction on the training set
gbc_preds = gbc_best.predict(X_train)

# confusion matrix for training set
confusion_matrix(y_train, gbc_preds)
# classification report
print(classification_report(y_train, gbc_preds,
                            target_names = ['Died', 'Survived']))
# training accuracy
gbc_trainAcc = gbc_best.score(X_train, y_train)

# calculate test accuracy estimate for best model using 5-fold CV
gbc_testErrEst = np.mean(cross_val_score(gbc_best, X_train, y_train,
                                         scoring = 'accuracy',
                                         cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', gbc_testErrEst)
# add metrics to model_results dataframe
gbc_results = pd.DataFrame([['GradientBoostedTree', gbc_trainAcc, 
                             gbc_testErrEst]],
                             columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(gbc_results)

model_results
# make predictions on test set
gbc_test = pd.DataFrame(gbc_best.predict(X_test).astype(int), 
                        columns = ['Survived'],
                        index = test.index)

# write to csv
gbc_test.to_csv('gbc_test.csv')
# look at feature importances
gbc_importances = gbc_best.feature_importances_

headers_gbc = ["Variable", "Importance"]
values_gbc = pd.DataFrame(sorted(zip(X_train.columns, gbc_importances), key=lambda x: x[1] * -1), columns = headers_gbc)

# horizontal bar plot of importances
fig, ax = plt.subplots()
fig.set_size_inches(12, 13)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.barplot(x = 'Importance', y = 'Variable', data = values_gbc, orient = 'h',color = 'green')
# let's take the top 25 predictors
reducedFeatures = values_gbc.Variable.head(25)

# Extract only those features from the test and train sets
test_model_red = test_model[reducedFeatures]
train_model_red = train_model[reducedFeatures]
train_model_red.loc[:,'Survived'] = train_model['Survived']
# Correlations in the new training set
corrRedModelX = train_model_red.corr()

fig, ax = plt.subplots()
sns.set_context('talk')
sns.set_style('darkgrid')
fig.set_size_inches(15,10)

# plot the heatmap
sns.heatmap(corrRedModelX, 
            xticklabels=corrRedModelX.columns,
            yticklabels=corrRedModelX.columns,
            cmap = 'PiYG')
# separate into inputs and outputs
X_train_red = train_model_red.drop(['Survived'], axis = 1)
y_train_red = train_model_red['Survived']
X_test_red = test_model_red
# Set up a dict with values to test for each parameter/argument in the model
# object
gbcRed_grid = {'n_estimators'        : [50, 75, 100],
               'learning_rate'       : [0.05, 0.1, 0.15],
               'max_features'        : np.arange(1,4),
               'max_depth'           : np.arange(1,4),
               'min_samples_split'   : np.arange(8,20,4),
               'min_samples_leaf'    : [5, 7, 9],
               'subsample'           : [0.3, 0.5, 0.7],
               'max_leaf_nodes'      : [10, 15, 20]}

# Construct random forest object
gbcRed = GradientBoostingClassifier(random_state = 555)

# set up the grid search
gbcRed_GridSearch = GridSearchCV(gbcRed, gbcRed_grid)

# fit trees
gbcRed_fit = gbcRed_GridSearch.fit(X_train_red,y_train_red)
gbcRed_GridSearch.best_score_
gbcRed_GridSearch.best_params_
# Build the best model
gbcRed_best = GradientBoostingClassifier(n_estimators = 80,
                                         learning_rate = 0.15,
                                         max_depth = 3,
                                         max_features = 4,
                                         min_samples_leaf = 3,
                                         min_samples_split = 15,
                                         max_leaf_nodes = 10,
                                         subsample = 0.7,
                                         random_state = 555)

gbcRedBestFit = gbcRed_best.fit(X_train_red, y_train_red)

# make prediction on the training set
gbcRed_preds = gbcRed_best.predict(X_train_red)

# confusion matrix for training set
confusion_matrix(y_train_red, gbcRed_preds)
# classification report
print(classification_report(y_train_red, gbcRed_preds,
                            target_names = ['Died', 'Survived']))
# training accuracy
gbcRed_trainAcc = gbcRed_best.score(X_train_red, y_train_red)

# calculate test accuracy estimate for best model using 1-fold CV
gbcRed_testErrEst = np.mean(cross_val_score(gbcRed_best, X_train_red, y_train_red,
                                         scoring = 'accuracy',
                                         cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', gbcRed_testErrEst)
# add metrics to model_results dataframe
gbcRed_results = pd.DataFrame([['GBC_ReducedX', gbcRed_trainAcc, 
                                 gbcRed_testErrEst]],
                              columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(gbcRed_results)

model_results
# make predictions on test set
gbcRed_test = pd.DataFrame(gbcRed_best.predict(X_test_red).astype(int), 
                        columns = ['Survived'],
                        index = test.index)

# write to csv
gbcRed_test.to_csv('gbcRed_test.csv')
# look at feature importances
gbcRed_importances = gbcRed_best.feature_importances_

headers_gbcRed = ["Variable", "Importance"]
values_gbcRed = pd.DataFrame(sorted(zip(X_train_red.columns, gbcRed_importances), key=lambda x: x[1] * -1), 
                             columns = headers_gbcRed)

# horizontal bar plot of importances
fig, ax = plt.subplots()
fig.set_size_inches(12, 13)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.barplot(x = 'Importance', y = 'Variable', data = values_gbcRed, orient = 'h', color = 'green')
# Add a column with Kaggle public leaderboard results
model_results['PublicLeaderboard'] = [0.77511, 0.77950, 0.79425, 0.77033, 0.75598]
model_results