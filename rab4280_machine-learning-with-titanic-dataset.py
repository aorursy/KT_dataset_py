import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# Evaluating algorithms

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_absolute_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
X = pd.read_csv('/kaggle/input/titanic/train.csv') 

X.head()
# Statistics by Gender

print('Who survived???')

print('---------------')

rate = sum(X['Survived'])  / len(X)

print('{rate:2.0f}% of {total} passengers survived'.format(rate=rate*100, total=len(X)))

women = X.loc[X.Sex=='female']['Survived']

rate_women = sum(women) / len(women)

men = X.loc[X.Sex=='male']['Survived']

rate_men = sum(men) / len(men)

print('{rate:2.0f}% of {total} women survived'.format(rate=rate_women*100, total=len(X.loc[X.Sex=='female'])))

print('{rate:2.0f}% of {total} men survived'.format(rate=rate_men*100, total=len(X.loc[X.Sex=='male'])))



# Statistics by Class

first_class = X.loc[X.Pclass==1]['Survived']

rate_first = sum(first_class) / len(first_class)

second_class = X.loc[X.Pclass==2]['Survived']

rate_second= sum(second_class) / len(second_class)

print('{rate:2.0f}% of 2nd class passengers survived'.format(rate=rate_second*100))

third_class = X.loc[X.Pclass==3]['Survived']

rate_third= sum(third_class) / len(third_class)

print('{rate:2.0f}% of 3rd class passengers survived'.format(rate=rate_third*100))



# Combined class and gender

first_class_women = X.loc[X.Pclass==1][X.Sex=='female']['Survived']

rate_fw = sum(first_class_women) / len(first_class_women)

print('{rate:2.0f}% of 1st class women survived!'.format(rate=rate_fw*100))
X.describe()
# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['Survived'], inplace=True)

y = X.Survived

X.drop(['Survived'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X.info()
# All categorical columns

object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == "object"]

numeric_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in ['int64', 'float64']]

# All numeric columns

print('Categorical columns:', object_cols)

print('Numeric columns:', numeric_cols)



# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train_full[col].nunique(), object_cols))

numeric_nunique = list(map(lambda col: X_train_full[col].nunique(), numeric_cols))

d1 = dict(zip(object_cols, object_nunique))

d2 = dict(zip(numeric_cols, numeric_nunique))



# Print number of unique entries by column, in ascending order

print('\n Unique values:')

print(sorted(d1.items(), key=lambda x: x[1]))

print(sorted(d2.items(), key=lambda x: x[1]))



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train_full.isnull().sum())

print('\n Missing values:')

print(missing_val_count_by_column[missing_val_count_by_column > 0])
first_class_age = X_train_full.loc[X_train_full.Pclass==1]['Age'].mean()

second_class_age = X_train_full.loc[X_train_full.Pclass==2]['Age'].mean()

third_class_age = X_train_full.loc[X_train_full.Pclass==3]['Age'].mean()

print('Median age:', X_train_full['Age'].median() )

print('Median fare:', X_train_full['Fare'].median() )



    

def replace_age_by_class(age, pclass, display=False):

    # All ages that are NaN

    isnan = np.isnan(age)    

    # Replace first class passagers without age value with class mean

    ind1 = (pclass==1) & isnan    

    age[ind1] = first_class_age     

    # Replace second class passagers without age value with class mean

    ind2 = (pclass==2) & isnan    

    age[ind2] = second_class_age   

    # Replace third class passagers without age value with class mean

    ind3 = (pclass==3) & isnan

    age[ind3] = third_class_age

    

    if display:

        print('NaNs:', sum(isnan))

        print('1st class NaNs:', np.sum(ind1))

        print('2nd class NaNs:', np.sum(ind2))

        print('3rd class NaNs:', np.sum(ind3))    

    return age



def custom_preprocesor(dataFrame, OH_encoder=None, medianAge=29, medianFare=14.45):

    

    columns = dataFrame.columns

    if ('Name' == columns).any():

        dataFrame.drop(['Name'], axis=1, inplace=True)

    

    if ('Cabin' == columns).any():

        dataFrame.drop(['Cabin'], axis=1, inplace=True)

        

    if ('Ticket' == columns).any():

        dataFrame.drop(['Ticket'], axis=1, inplace=True)



    if ('PassengerId' == columns).any():

        dataFrame.drop(['PassengerId'], axis=1, inplace=True)

        

    if ('Survived' == columns).any():

        dataFrame.drop(['Survived'], axis=1, inplace=True)        

        

    # Replace SibSp (number of siblings and spouses) amd Parch (number of parents and childern) 

    # to a single family member column

    nColumns = len(dataFrame.columns)-2

    familyMembers = dataFrame['SibSp']+dataFrame['Parch']

    dataFrame.drop(['SibSp'], axis=1, inplace=True)

    dataFrame.drop(['Parch'], axis=1, inplace=True)

    dataFrame.insert(nColumns,'FamilyMembers',familyMembers)

    

    # Replace missing age by mean of class

    age = dataFrame['Age'].to_numpy()

    isnan = np.isnan(age)

    age[isnan] = medianAge

    

    # Quanitze Age

    # Create the column age_categories '0-9', '10-19','20-29', '30-39', '40-49','>=50'

    age_cat = pd.cut(age, [0, 10, 20, 30, 40, 50, np.inf], 

                                    labels=[1, 2, 3, 4, 5, 6], include_lowest=True, right=False).astype(int)

    dataFrame['Age'] = age_cat

    

    # Quantize Fare

    # Create the column fare_categories '0-7.99', '8-13.99','14-30.99', '31-98.99', '99-249.99', '>=250'

    fare = dataFrame['Fare'].to_numpy()

    isnan = np.isnan(fare)

    fare[isnan] = medianFare

    

    fare_cat = pd.cut(fare, [0, 8, 14, 31, 99, 250, np.inf], 

                                    labels=[0, 1, 2, 3, 4, 5], include_lowest=True, right=False).astype(int)

    dataFrame['Fare'] = fare_cat

    

    # One-hot encoder for Sex and Embarked categories

    embarked = dataFrame['Embarked'].copy()

    isnan = dataFrame['Embarked'].isnull().to_numpy()

    embarked[isnan] = 'S'

    dataFrame['Embarked'] = embarked

    if OH_encoder:

        OH_cols =  pd.DataFrame( OH_encoder.transform( dataFrame[ ['Sex', 'Embarked'] ] ), index=dataFrame.index )

    else:        

        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        OH_cols =  pd.DataFrame( OH_encoder.fit_transform( dataFrame[ ['Sex', 'Embarked'] ] ), index=dataFrame.index )



    dataFrame.drop(['Sex'], axis=1, inplace=True)

    dataFrame.drop(['Embarked'], axis=1, inplace=True)

    processedFrame = pd.concat([dataFrame, OH_cols], axis=1)





    return processedFrame, OH_encoder

X_train, OH_encoder = custom_preprocesor(X_train_full.copy())

X_valid, _  = custom_preprocesor(X_valid_full.copy(), OH_encoder=OH_encoder)





xgb_model = XGBClassifier(n_estimators=200, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], random_state=0)



rf_model = RandomForestClassifier(n_estimators=250, random_state=0)

rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))





xgb_model.fit(X_train, y_train)

preds = xgb_model.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))

# Create a data frame showing the feature importance as given by the random forest model

feature_importance = pd.DataFrame({'Feature':X_train.columns,'Importance':np.round(rf_model.feature_importances_,3)})

feature_importance.sort_values('Importance',ascending=False).reset_index().drop('index', axis=1)
# We will use RandomizedSearchCV for hyperparameter tuning so first we need to create a parameter grid



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]



# Number of features to consider at every split

max_features = ['auto', 'sqrt', 'log2']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]



# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

random_grid
# Random search of parameters using 3 fold cross validation and 100 different combinations

random_forest_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, verbose=2, cv = 3, random_state=42, n_jobs = -1)



# Fit the random search model

random_forest_random.fit(X_train, y_train)
# Let's have a look at the best parameters from fitting the random search

random_forest_random.best_params_
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=2, min_samples_split=10, max_features='log2',bootstrap=True, random_state=0)

rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
from sklearn.model_selection import cross_val_score



def get_score(n_estimators):

    my_pipeline = get_rf_pipeline(n_estimators=n_estimators)

    scores = -1 * cross_val_score(my_pipeline, X[predict_cols], y, cv=3, scoring='neg_mean_absolute_error')  

    return scores.mean()



est = np.arange(50, 400, 50)

results = np.zeros( (len(est),1) )

#for i in range( len(est) ):

#    results[i] = get_score(est[i])

#plt.plot( est, results, '.-')
X = pd.read_csv('/kaggle/input/titanic/train.csv') 

y = X.Survived

X, OH_encoder = custom_preprocesor(X.copy())

X_test = pd.read_csv('/kaggle/input/titanic/test.csv') 

PassengerId = X_test.PassengerId

X_test, _  = custom_preprocesor(X_test.copy(), OH_encoder=OH_encoder)



rf_model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=2, min_samples_split=10, max_features='log2',bootstrap=True, random_state=0)

rf_model.fit(X, y)

predictions = rf_model.predict(X_test)



output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")