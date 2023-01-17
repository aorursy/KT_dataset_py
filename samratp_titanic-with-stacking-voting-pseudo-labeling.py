import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import ensemble
from sklearn import model_selection

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import scipy.stats as st

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
def drop_col_not_req(df, cols):
    df.drop(cols, axis = 1, inplace = True)
def fare_category(fare):
    if (fare <= 4):
        return 'Very_Low_Fare'
    elif (fare <= 10):
        return 'Low_Fare'
    elif (fare <= 30):
        return 'Med_Fare'
    elif (fare <= 45):
        return 'High_Fare'
    else:
        return 'Very_High_Fare'
def pclass_fare_category(df, Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare):
    if (df['Pclass'] == 1):
        if (df['Fare'] <= Pclass_1_mean_fare):
            return 'Pclass_1_Low_Fare'
        else:
            return 'Pclass_1_High_Fare'
    elif (df['Pclass'] == 2):
        if (df['Fare'] <= Pclass_2_mean_fare):
            return 'Pclass_2_Low_Fare'
        else:
            return 'Pclass_2_High_Fare'
    elif (df['Pclass'] == 3):
        if (df['Fare'] <= Pclass_3_mean_fare):
            return 'Pclass_3_Low_Fare'
        else:
            return 'Pclass_3_High_Fare'
def family_size_category(family_size):
    if (family_size <= 1):
        return 'Single'
    elif (family_size <= 3):
        return 'Small_Family'
    else:
        return 'Large_Family'
def age_group_cat(age):
    if (age <= 1):
        return 'Baby'
    if (age <= 4):
        return 'Toddler'
    elif(age <= 12):
        return 'Child'
    elif (age <= 19):
        return 'Teenager'
    elif (age <= 30):
        return 'Adult'
    elif (age <= 50):
        return 'Middle_Aged'
    elif(age < 60):
        return 'Senior_Citizen'
    else:
        return 'Old'
def name_len_category(name_len):
    if (name_len <= 20):
        return 'Very_Short_Name'
    elif (name_len <= 28):
        return 'Short_Name'
    elif (name_len <= 45):
        return 'Medium_Name'
    else:
        return 'Long_Name'
def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis = 1)
    missing_age_y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis = 1)
    
    gbm_reg = ensemble.GradientBoostingRegressor(random_state = 42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [3], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv = 5, n_jobs = 25, verbose = 1, scoring = 'neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_y_train)
    
    print("Age feature Best GB Params: " + str(gbm_reg_grid.best_params_))
    print("Age feature Best GB Score: " + str(gbm_reg_grid.best_score_))
    print("GB Train Error for 'Age' Feature Regressor: " + str(gbm_reg_grid.score(missing_age_X_train, missing_age_y_train)))
    
    missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    
    lrf_reg = LinearRegression()
    lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
    lrf_reg_grid = model_selection.GridSearchCV(lrf_reg, lrf_reg_param_grid, cv = 5, n_jobs = 25, verbose = 1, scoring = 'neg_mean_squared_error')
    lrf_reg_grid.fit(missing_age_X_train, missing_age_y_train)
    
    print("Age feature Best LR Params: " + str(lrf_reg_grid.best_params_))
    print("Age feature Best LR Score: " + str(lrf_reg_grid.best_score_))
    print("LR Train Error for 'Age' Feature Regressor: " + str(lrf_reg_grid.score(missing_age_X_train, missing_age_y_train)))
    
    missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_LRF'][:4])
    
    missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LRF']].mean(axis = 1)
    print(missing_age_test['Age'][:4])
    drop_col_not_req(missing_age_test, ['Age_GB', 'Age_LRF'])

    return missing_age_test
def get_top_n_features(titanic_train_data_X, titanic_train_data_y, top_n_features):
    rf_est = RandomForestClassifier(random_state = 42)
    rf_param_grid = {'n_estimators' : [500], 'min_samples_split':[2, 3], 'max_depth':[20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_y)
    
    print("Top N Features Best RF Params: " + str(rf_grid.best_params_))
    print("Top N Features Best RF Score: " + str(rf_grid.best_score_))
    print("Top N Features RF Train Error: " + str(rf_grid.score(titanic_train_data_X, titanic_train_data_y)))

    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending = False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print("Sample 25 Features from RF Classifier:")
    print(str(features_top_n_rf[:25]))
    
    ada_est = ensemble.AdaBoostClassifier(random_state = 42)
    ada_param_grid = {'n_estimators' : [500], 'learning_rate': [0.5, 0.6]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_y)
    
    print("Top N Features Best Ada Params: " + str(ada_grid.best_params_))
    print("Top N Features Best Ada Score: " + str(ada_grid.best_score_))
    print("Top N Features Ada Train Error: " + str(ada_grid.score(titanic_train_data_X, titanic_train_data_y)))
    
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending = False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print("Sample 25 Features from Ada Classifier:")
    print(str(features_top_n_ada[:25]))
    
    et_est = ensemble.ExtraTreesClassifier(random_state = 42)
    et_param_grid = {'n_estimators' : [500], 'min_samples_split':[3, 4], 'max_depth':[15]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs = 25, cv = 10, verbose = 1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_y)
    
    print("Top N Features Best ET Params: " + str(et_grid.best_params_))
    print("Top N Features Best ET Score: " + str(et_grid.best_score_))
    print("Top N Features ET Train Error: " + str(et_grid.score(titanic_train_data_X, titanic_train_data_y)))
    
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending = False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print("Sample 25 Features from ET Classifier:")
    print(str(features_top_n_et[:25]))
    
    #### Merge top_n_features from all three models
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et], ignore_index = True).drop_duplicates()
    
    return features_top_n
# Stacking Ensemble
def Stacking_Ensemble(models, X_train, y_train, X_test, n_folds):
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    Stacking_train = np.zeros((X_train.shape[0], (len(models) * 2)))
    Stacking_test = np.zeros((X_test.shape[0], (len(models) * 2)))

    Strat_KFold = model_selection.StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 42)
    skf = list(Strat_KFold.split(X_train, y_train))

    ### Loop all the Base Models
    for model_counter, model in enumerate(models):
        print("Model_Counter" + str(model_counter))
        S_test_temp_p1 = np.zeros((titanic_test_data_X.shape[0], n_folds))
        S_test_temp_p2 = np.zeros((titanic_test_data_X.shape[0], n_folds))
        
        ### Loop across the Folds
        for fold_counter, (tr_index, te_index) in enumerate(skf):
            print("Fold : " + str(fold_counter))
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            model.fit(X_tr, y_tr)
            print("----------------------")
            print("Model : " + str(model_counter) + "Fold : " + str(fold_counter) + "Score is - " + str(model.score(X_tr, y_tr)))
            print("----------------------")
            tr_pred = model.predict_proba(X_te)
            Stacking_train[te_index, model_counter * 2] = tr_pred[:,0]
            Stacking_train[te_index, (model_counter * 2) + 1] = tr_pred[:,1]
            
            te_pred = model.predict_proba(X_test)
            S_test_temp_p1[:, fold_counter] = te_pred[:,0]
            S_test_temp_p2[:, fold_counter] = te_pred[:,1]
            
        Stacking_test[:, model_counter * 2] = np.mean(S_test_temp_p1, axis = 1)
        Stacking_test[:, (model_counter * 2) + 1] = np.mean(S_test_temp_p2, axis = 1)
    
    return (Stacking_train, Stacking_test)

train_data_orig = pd.read_csv('../input/train.csv')
test_data_orig = pd.read_csv('../input/test.csv')
train_data_orig.shape
train_data_orig.info()
train_data_orig.describe()
train_data_orig.head()
test_data_orig.shape
test_data_orig.info()
test_data_orig.describe()
test_data_orig.head()
test_data_orig['Survived'] = 0
combined_train_test = train_data_orig.append(test_data_orig)
combined_train_test.shape
combined_train_test.info()
combined_train_test.describe()
print(combined_train_test.groupby(['Survived', 'Embarked'])['Survived'].count())
print(combined_train_test['PassengerId'].groupby(by = combined_train_test['Embarked']).count().sort_values(ascending = False))
print(combined_train_test['Fare'].groupby(by = combined_train_test['Embarked']).mean().sort_values(ascending = False))

if (combined_train_test['Embarked'].isnull().sum() != 0):
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    combined_train_test.info()
    
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],
                                prefix = combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis = 1)
print(combined_train_test['Sex'].groupby(by = combined_train_test['Sex']).count().sort_values(ascending = False))
print(combined_train_test.groupby(['Survived', 'Sex'])['Survived'].count())
'''
lb_sex = preprocessing.LabelBinarizer()
lb_sex.fit(np.array(['male', 'female']))
combined_train_test['Sex'] = lb_sex.transform(combined_train_test['Sex'])
'''
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'],
                                prefix = combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis = 1)
combined_train_test['Title'] = combined_train_test['Name'].str.extract('.+,(.+)').str.extract('^(.+?)\.').str.strip()
print(combined_train_test['Title'].unique())
print(combined_train_test['Title'].groupby(by = combined_train_test['Title']).count().sort_values(ascending = False))
title_Dict = {}
title_Dict.update(dict.fromkeys(["Capt", "Col", "Major", "Dr", "Rev"], "Officer"))
title_Dict.update(dict.fromkeys(["Jonkheer", "Don", "Sir", "the Countess", "Dona", "Lady"], "Royalty"))
title_Dict.update(dict.fromkeys(["Mme", "Ms", "Mrs"], "Mrs"))
title_Dict.update(dict.fromkeys(["Mlle", "Miss"], "Miss"))
title_Dict.update(dict.fromkeys(["Mr", "Ms"], "Mr"))
title_Dict.update(dict.fromkeys(["Master"], "Master"))
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
print(combined_train_test['Title'].groupby(by = combined_train_test['Title']).count().sort_values(ascending = False))

title_dummies_df = pd.get_dummies(combined_train_test['Title'],
                                prefix = combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis = 1)
combined_train_test['Name_Length'] = combined_train_test['Name'].str.len()
print(combined_train_test['Name_Length'].groupby(by = combined_train_test['Name_Length']).count().sort_values(ascending = False)[:5])
combined_train_test['Name_Length_Category'] = combined_train_test['Name_Length'].map(name_len_category)
print(combined_train_test['Name_Length_Category'].groupby(by = combined_train_test['Name_Length_Category']).count().sort_values(ascending = False))

le_fare = LabelEncoder()
le_fare.fit(np.array(['Very_Short_Name', 'Short_Name', 'Medium_Name', 'Long_Name', 'Very_High_Fare']))
combined_train_test['Name_Length_Category'] = le_fare.transform(combined_train_test['Name_Length_Category'])

print(combined_train_test[['Name_Length_Category', 'Survived']].corr())

first_name_dummies_df = pd.get_dummies(combined_train_test['Name_Length_Category'],
                                prefix = combined_train_test[['Name_Length_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, first_name_dummies_df], axis = 1)
combined_train_test['First_Name'] = combined_train_test['Name'].str.extract('^(.+?),').str.strip()
print(combined_train_test['First_Name'].groupby(by = combined_train_test['First_Name']).count().sort_values(ascending = False)[:5])

first_name_dummies_df = pd.get_dummies(combined_train_test['First_Name'],
                                prefix = combined_train_test[['First_Name']].columns[0])
combined_train_test = pd.concat([combined_train_test, first_name_dummies_df], axis = 1)
combined_train_test['Last_Name'] = combined_train_test['Name'].str.split("\.").str[1].str.strip()
combined_train_test['Last_Name'] = combined_train_test['Last_Name'].str.strip("\([^)]*\)")
combined_train_test['Last_Name'].fillna(combined_train_test['Name'].str.split("\.").str[1].str.strip())
print(combined_train_test['Last_Name'].groupby(by = combined_train_test['Last_Name']).count().sort_values(ascending = False)[:5])

last_name_dummies_df = pd.get_dummies(combined_train_test['Last_Name'],
                                prefix = combined_train_test[['Last_Name']].columns[0])
combined_train_test = pd.concat([combined_train_test, last_name_dummies_df], axis = 1)
combined_train_test['Original_Name'] = combined_train_test['Name'].str.split("\((.*?)\)").str[1].str.strip("\"").str.strip()
print(combined_train_test['Original_Name'].groupby(by = combined_train_test['Original_Name']).count().sort_values(ascending = False)[:5])

last_name_dummies_df = pd.get_dummies(combined_train_test['Original_Name'],
                                prefix = combined_train_test[['Original_Name']].columns[0])
combined_train_test = pd.concat([combined_train_test, last_name_dummies_df], axis = 1)
if (combined_train_test['Fare'].isnull().sum() != 0):
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform('mean'))
    combined_train_test.info()
combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by = combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare']/combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis = 1, inplace = True)
if (sum(n == 0 for n in combined_train_test.Fare.values.flatten()) > 0):
    combined_train_test.loc[combined_train_test.Fare == 0, 'Fare'] = np.nan
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform('mean'))

combined_train_test['Fare'].describe()
combined_train_test['Fare_Category'] = combined_train_test['Fare'].map(fare_category)
le_fare = LabelEncoder()
le_fare.fit(np.array(['Very_Low_Fare', 'Low_Fare', 'Med_Fare', 'High_Fare', 'Very_High_Fare']))
combined_train_test['Fare_Category'] = le_fare.transform(combined_train_test['Fare_Category'])

fare_cat_dummies_df = pd.get_dummies(combined_train_test['Fare_Category'],
                                prefix = combined_train_test[['Fare_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, fare_cat_dummies_df], axis = 1)

print(combined_train_test['Fare_Category'].groupby(by = combined_train_test['Fare_Category']).count().sort_values(ascending = False))
print(combined_train_test['Fare'].groupby(by = combined_train_test['Pclass']).mean())
Pclass_1_mean_fare = combined_train_test['Fare'].groupby(by = combined_train_test['Pclass']).mean().get([1]).values[0]
Pclass_2_mean_fare = combined_train_test['Fare'].groupby(by = combined_train_test['Pclass']).mean().get([2]).values[0]
Pclass_3_mean_fare = combined_train_test['Fare'].groupby(by = combined_train_test['Pclass']).mean().get([3]).values[0]
combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare), axis = 1)
print(combined_train_test['Pclass_Fare_Category'].groupby(by = combined_train_test['Pclass_Fare_Category']).count().sort_values(ascending = False))

le_fare = LabelEncoder()
le_fare.fit(np.array(['Pclass_1_Low_Fare', 'Pclass_1_High_Fare', 'Pclass_2_Low_Fare', 'Pclass_2_High_Fare', 'Pclass_3_Low_Fare', 'Pclass_3_High_Fare']))
combined_train_test['Pclass_Fare_Category'] = le_fare.transform(combined_train_test['Pclass_Fare_Category'])
print(combined_train_test['Fare'].groupby(by = combined_train_test['Pclass']).mean().sort_values(ascending = True))
combined_train_test['Pclass'].replace([1, 2, 3],[Pclass_1_mean_fare, Pclass_2_mean_fare, Pclass_3_mean_fare], inplace = True)
combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
print(combined_train_test['Family_Size'].groupby(by = combined_train_test['Family_Size']).count().sort_values(ascending = False))

combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

print(combined_train_test['Family_Size_Category'].groupby(by = combined_train_test['Family_Size_Category']).count().sort_values(ascending = False))
print(combined_train_test.groupby(['Survived', 'Family_Size_Category'])['Survived'].count())

le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

fam_size_cat_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
                                prefix = combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, fam_size_cat_dummies_df], axis = 1)
print(combined_train_test['Age'].groupby(by = combined_train_test['Title']).mean().sort_values(ascending = True))
combined_train_test['Age_Null'] = combined_train_test['Age'].apply(lambda x: 1 if(pd.notnull(x)) else 0)
missing_age_df = pd.DataFrame(combined_train_test[['Age', 'Parch', 'Sex', 'SibSp', 'Family_Size', 'Family_Size_Category', 'Title', 'Fare']])
missing_age_df = pd.get_dummies(missing_age_df, columns = ['Title', 'Family_Size_Category', 'Sex'])
missing_age_df.shape
missing_age_df.info()

missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test  = missing_age_df[missing_age_df['Age'].isnull()]
combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
if (sum(n < 0 for n in combined_train_test.Age.values.flatten()) > 0):
    combined_train_test.loc[combined_train_test.Age < 0, 'Age'] = np.nan
    combined_train_test['Age'] = combined_train_test[['Age']].fillna(combined_train_test.groupby('Title').transform('mean'))
print(combined_train_test['Age'].groupby(by = combined_train_test['Title']).mean().sort_values(ascending = True))
combined_train_test['Age_Category'] = combined_train_test['Age'].map(age_group_cat)
le_age = LabelEncoder()
le_age.fit(np.array(['Baby', 'Toddler', 'Child', 'Teenager', 'Adult', 'Middle_Aged', 'Senior_Citizen', 'Old']))
combined_train_test['Age_Category'] = le_age.transform(combined_train_test['Age_Category'])

age_cat_dummies_df = pd.get_dummies(combined_train_test['Age_Category'],
                                prefix = combined_train_test[['Age_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, age_cat_dummies_df], axis = 1)
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: np.NaN if x.isnumeric() else x)
combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
combined_train_test['Ticket_Number'].fillna(0, inplace = True)
combined_train_test = pd.get_dummies(combined_train_test, columns = ['Ticket', 'Ticket_Letter'])
combined_train_test.shape
combined_train_test['Cabin_Letter'] = combined_train_test['Cabin'].apply(lambda x: str(x)[0]  if(pd.notnull(x)) else x)
combined_train_test = pd.get_dummies(combined_train_test, columns = ['Cabin', 'Cabin_Letter'])
combined_train_test.shape
scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare']])
combined_train_test[['Age', 'Fare']] = scale_age_fare.transform(combined_train_test[['Age', 'Fare']])
combined_train_test.drop(['Name', 'PassengerId', 'Embarked', 'Sex', 'Title', 'Fare_Category', 'Family_Size_Category', 'Age_Category', 'First_Name', 'Last_Name', 'Original_Name', 'Name_Length_Category'], axis = 1, inplace = True)
train_data = combined_train_test[:891]
test_data = combined_train_test[891:]

titanic_train_data_X = train_data.drop(['Survived'], axis = 1)
titanic_train_data_y = train_data['Survived']

titanic_test_data_X = test_data.drop(['Survived'], axis = 1)
features_to_pick = 150
features_top_n = get_top_n_features(titanic_train_data_X, titanic_train_data_y, features_to_pick)

print("Total Features: " + str(combined_train_test.shape))
print("Picked Features: " + str(features_top_n.shape))

titanic_train_data_X = titanic_train_data_X[features_top_n]
titanic_train_data_X.shape
titanic_train_data_X.info()

titanic_test_data_X = titanic_test_data_X[features_top_n]
titanic_test_data_X.shape
titanic_test_data_X.info()
base_models = [ensemble.RandomForestClassifier(n_estimators = 750, criterion = 'gini', max_features = 'sqrt', max_depth = 3, min_samples_split = 4, min_samples_leaf = 2, n_jobs = 15, random_state = 42, verbose = 1),
               ensemble.GradientBoostingClassifier(n_estimators = 900, learning_rate = 0.001, loss = 'exponential', min_samples_split = 3, min_samples_leaf = 2, max_features ='sqrt', max_depth = 3,  random_state = 42, verbose = 1),
               ensemble.ExtraTreesClassifier(n_estimators = 750, max_depth = 35,  n_jobs = 15, criterion = 'entropy', random_state = 42, verbose = 1),
               ensemble.AdaBoostClassifier(n_estimators = 500, learning_rate = 0.5, random_state = 42),
               XGBClassifier(n_estimators = 500, max_depth = 2, eta = 0.05, n_jobs = 15, random_state = 42),
               LogisticRegression(C = 0.5, max_iter = 200, random_state = 42),
               ensemble.BaggingClassifier(n_estimators = 500, max_samples = 5, max_features = 5, random_state = 42),
               DecisionTreeClassifier(max_features = 5, max_depth = 5, min_samples_split = 3, min_samples_leaf =2, random_state =42),
               GaussianNB(),
               KNeighborsClassifier(n_neighbors = 2),
               SVC(C = 0.4, probability = True, random_state = 42)]

Stacked_train, Stacked_test = Stacking_Ensemble(base_models, titanic_train_data_X, titanic_train_data_y, titanic_test_data_X, n_folds = 7)
rf_est = ensemble.RandomForestClassifier(n_estimators = 750, criterion = 'gini', max_features = 'sqrt', max_depth = 3, min_samples_split = 4, min_samples_leaf = 2, n_jobs = 15, random_state = 42, verbose = 1)
gbm_est = ensemble.GradientBoostingClassifier(n_estimators = 900, learning_rate = 0.001, loss = 'exponential', min_samples_split = 3, min_samples_leaf = 2, max_features ='sqrt', max_depth = 3,  random_state = 42, verbose = 1)
et_est = ensemble.ExtraTreesClassifier(n_estimators = 750, max_depth = 35,  n_jobs = 15, criterion = 'entropy', random_state = 42, verbose = 1)

voting_est = ensemble.VotingClassifier(estimators = [('rf', rf_est),('gbm', gbm_est),('et', et_est)],
                                       voting = 'hard',
                                       n_jobs = 35)
                                       
voting_est.fit(Stacked_train, titanic_train_data_y)
print("VotingClassifier Score: " + str(voting_est.score(Stacked_train, titanic_train_data_y)))
print("VotingClassifier Estimators: " + str(voting_est.estimators_))
titanic_test_data_X['Survived'] = voting_est.predict(Stacked_test)
#### Pseudo Labeling
Stacked_train_df = pd.DataFrame(Stacked_train)
Stacked_test_df = pd.DataFrame(Stacked_test)

titanic_comb_X = pd.concat([Stacked_train_df, Stacked_test_df])
titanic_comb_y = pd.concat([titanic_train_data_y, titanic_test_data_X['Survived']])
#### Build the models
rf_est = ensemble.RandomForestClassifier(n_estimators = 750, criterion = 'gini', max_features = 'sqrt', max_depth = 3, min_samples_split = 4, min_samples_leaf = 2, n_jobs = 35, random_state = 42)
gbm_est = ensemble.GradientBoostingClassifier(n_estimators = 900, learning_rate = 0.001, loss = 'exponential', min_samples_split = 3, min_samples_leaf = 2, max_features ='sqrt', max_depth = 3,  random_state = 42)
et_est = ensemble.ExtraTreesClassifier(n_estimators = 750, max_depth = 35,  n_jobs = 15, criterion = 'entropy', random_state = 42)

voting_est = ensemble.VotingClassifier(estimators = [('rf', rf_est),('gbm', gbm_est),('et', et_est)],
                                       voting = 'hard',
                                       n_jobs = 35)
                                       
voting_est.fit(titanic_comb_X, titanic_comb_y)
print("VotingClassifier Score: " + str(voting_est.score(titanic_comb_X, titanic_comb_y)))
print("VotingClassifier Estimators: " + str(voting_est.estimators_))

titanic_test_data_X['Survived_new'] = voting_est.predict(Stacked_test_df)
#### Prepare submission file
submission = pd.DataFrame({'PassengerId': test_data_orig.loc[:, 'PassengerId'],
                           'Survived': titanic_test_data_X.loc[:, 'Survived_new']})
submission.to_csv("../working/submission.csv", index = False)