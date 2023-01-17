# data manipulation

import pandas as pd

import numpy as np



# sklearn helper functions

from sklearn.base import clone

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import ShuffleSplit, cross_validate, GridSearchCV, cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix, classification_report, plot_precision_recall_curve, plot_roc_curve



# ML algorithms

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier
df_train_raw = pd.read_csv("../input/titanic/train.csv")

df_test_raw = pd.read_csv("../input/titanic/test.csv")
df1 = df_train_raw.copy()



print(df_train_raw.info())

df_train_raw.sample(10)
# check null values in train and test set

print("Null value count in train set\n", df1.isnull().sum())

print("-"*30)

print("Null value count in test set\n", df_test_raw.isnull().sum())

print("-"*30)



# summary statistics for train set

df1.describe(include="all")
# Remove target variable from training set and keep a copy of it

train_labels = df1['Survived'].copy()

df1.drop('Survived', axis=1, inplace=True)
# convert 'Pclass' from int to string

df1['Pclass'] = df1['Pclass'].astype(str)
df1.sample(10)
# lists for different type of preprocessing

preprocess_features1 = ['PassengerId', 'Name', 'Ticket', 'Cabin']

preprocess_features2 = ['Pclass', 'Sex']

preprocess_features3 = ['Age']

preprocess_features4 = ['SibSp', 'Parch']

preprocess_features5 = ['Fare']

preprocess_features6 = ['Embarked']



# creating transformation pipelines for different features

transformer1 = 'drop'



transformer2 = Pipeline(steps=[

    ('onehot', OneHotEncoder())

])



transformer3 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



transformer4 = Pipeline(steps=[

    ('scaler', StandardScaler())

])



transformer5 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



transformer6 = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())

])



# final transfomer

preprocessor = ColumnTransformer(

    transformers=[

        ('t1', transformer1, preprocess_features1),

        ('t2', transformer2, preprocess_features2),

        ('t3', transformer3, preprocess_features3),

        ('t4', transformer4, preprocess_features4),

        ('t5', transformer5, preprocess_features5),

        ('t6', transformer6, preprocess_features6)])
# apply full pipeline to train data

df1 = preprocessor.fit_transform(df1)
MLA = [

    RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1),

    

    ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1),

    

    AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 

                       n_estimators=200, 

                       algorithm='SAMME.R', 

                       learning_rate=0.5),

    

    XGBClassifier(learning_rate=0.05, max_depth=4, n_estimators=50, reg_alpha=0.01, reg_lambda=0.3, seed=0),

    

    BaggingClassifier(base_estimator=DecisionTreeClassifier(), 

                      n_estimators=500,

                      max_samples=50,

                      bootstrap=True,

                      n_jobs=-1),    # Bagging

    

    BaggingClassifier(base_estimator=DecisionTreeClassifier(), 

                      n_estimators=500,

                      max_samples=50,

                      bootstrap=False,

                      n_jobs=-1),    # Pasting

    

    BaggingClassifier(base_estimator=DecisionTreeClassifier(), 

                      n_estimators=500,

                      max_samples=100,

                      bootstrap=True,

                      max_features=5,

                      bootstrap_features=True,

                      n_jobs=-1),    # Random Patches Method

    

    BaggingClassifier(base_estimator=DecisionTreeClassifier(), 

                      n_estimators=500,

                      bootstrap=False,

                      max_features=5,

                      bootstrap_features=True,

                      n_jobs=-1),    # Random Subspaces Method

    

    GaussianNB(),

    

    SVC(kernel='linear', C=1),

    

    SVC(kernel='poly', degree=3, coef0=0.01, C=5),

    

    SVC(kernel='rbf', gamma=0.1, C=2),

    

    DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)

]
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)



MLA_columns = ['MLA Name', 'MLA Parameters', 'Train F1 Score Mean', 'Test F1 Score Mean', 'Test F1 Score 3*STD' , 'Test Precision Mean', 'Test Recall Mean', 'Training Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



MLA_predict = train_labels.copy()



row_index = 0

for alg in MLA:



    # set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    # cross validation

    cv_results = cross_validate(alg, df1, train_labels, cv=cv_split, scoring=['f1','precision', 'recall'], return_train_score=True)

    

    MLA_compare.loc[row_index, 'Training Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'Train F1 Score Mean'] = cv_results['train_f1'].mean()

    MLA_compare.loc[row_index, 'Test F1 Score Mean'] = cv_results['test_f1'].mean()

    MLA_compare.loc[row_index, 'Test F1 Score 3*STD'] = cv_results['test_f1'].std()*3

    MLA_compare.loc[row_index, 'Test Precision Mean'] = cv_results['test_precision'].mean()

    MLA_compare.loc[row_index, 'Test Recall Mean'] = cv_results['test_recall'].mean()



    # save MLA predictions

    alg.fit(df1, train_labels)

    MLA_predict[MLA_name] = alg.predict(df1)

    

    row_index+=1

    

#print and sort table

MLA_compare.sort_values(by = ['Test F1 Score Mean'], ascending = False, inplace = True)

MLA_compare
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)



shortlisted_model = XGBClassifier()

base_results = cross_validate(shortlisted_model, df1, train_labels, cv=cv_split, scoring='f1', return_train_score=True)



print('BEFORE GridSearch Parameters: ', shortlisted_model.get_params())

print("BEFORE GridSearch Training F1 score mean: {:.2f}". format(base_results['train_score'].mean()))

print("BEFORE GridSearch Test F1 score mean: {:.2f}". format(base_results['test_score'].mean()))

print("BEFORE GridSearch Test F1 score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*3))

print('-'*10)



param_grid = [

    {

        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.25, 0.3],

         'max_depth': [1,2,4,6,8,10],

         'n_estimators': [10, 50, 100, 300],

         'reg_alpha': [0.01, 0.1, 0.3],

         'reg_lambda': [0.1, 0.2, 0.3, 0.5],

         'seed': [0]

    }

]



grid_search = GridSearchCV(shortlisted_model, param_grid=param_grid, scoring='f1', cv=cv_split, return_train_score=True)

grid_search.fit(df1, train_labels)



print('AFTER GridSearch Parameters: ', grid_search.best_params_)

print("AFTER GridSearch Training F1 score mean: {:.2f}". format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))

print("AFTER GridSearch Test F1 score mean: {:.2f}". format(grid_search.cv_results_['mean_test_score'][grid_search.best_index_]))

print("AFTER GridSearch Test F1 score 3*std: +/- {:.2f}". format(grid_search.cv_results_['std_test_score'][grid_search.best_index_]*3))

print('-'*10)
# predictors for voting classifier

predictor1 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1)

predictor2 = SVC(kernel='rbf', gamma=0.1, C=2, probability=True)

predictor3 = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)

predictor4 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),

                               n_estimators=500,

                               bootstrap=False,

                               max_features=5,

                               bootstrap_features=True,

                               n_jobs=-1)

predictor5 = GaussianNB()

predictor6 = XGBClassifier(learning_rate=0.05, max_depth=4, n_estimators=50, reg_alpha=0.01, reg_lambda=0.3, seed=0)

predictor7 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 

                                n_estimators=200, 

                                algorithm='SAMME.R', 

                                learning_rate=0.5)
voting_clf = VotingClassifier(

    estimators=[('pred1', predictor1), ('pred2', predictor2), ('pred3', predictor3), ('pred4', predictor4), ('pred5', predictor5), ('pred6', predictor6), ('pred7', predictor7)],

    voting='soft'

)



cross_val_score(voting_clf, df1, train_labels, cv=5, scoring='f1').mean()
rfc = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1).fit(df1, train_labels)



feature_names = preprocess_features1 + preprocess_features2 + preprocess_features3 + preprocess_features4 + preprocess_features5 + preprocess_features6



print("Feature Name\t", "Feature Importance")

print("-"*40)

for feature_name, score in zip(feature_names, rfc.feature_importances_):

    print(feature_name,"\t", score)
eval_model = clone(voting_clf)

eval_model.fit(df1, train_labels)



eval_predictions = cross_val_predict(eval_model, df1, train_labels, cv=3)
confusion_matrix(train_labels, eval_predictions)
print(classification_report(train_labels, eval_predictions))
plot_precision_recall_curve(eval_model, df1, train_labels, response_method='predict_proba')
plot_roc_curve(eval_model, df1, train_labels, response_method='predict_proba')
# copy test set 'Id'

submission_Ids = df_test_raw['PassengerId'].copy()



df_test_raw['Pclass'] = df_test_raw['Pclass'].astype(str)



# apply preprocessing pipeline to test set

df_test_prepared = preprocessor.transform(df_test_raw)
# FINAL MODEL

final_model = clone(voting_clf)

final_model.fit(df1, train_labels)



# submission predictions

final_pred = final_model.predict(df_test_prepared)
# create submission file

my_submission = pd.DataFrame({'PassengerId': submission_Ids, 'Survived': final_pred})

my_submission.to_csv("submission.csv", index=False)