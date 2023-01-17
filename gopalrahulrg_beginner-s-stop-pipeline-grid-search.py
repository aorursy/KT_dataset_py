# importing library to handle time

import time



# importing library to handle arrays

import numpy as np 



# importing library to handle dataframes

import pandas as pd



# importing library for preprocessing and machine learning

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.utils.class_weight import compute_class_weight



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



from sklearn.model_selection import GridSearchCV



from sklearn.metrics import f1_score, make_scorer
# loading train, test and sample files

df_train = pd.read_csv('../input/titanic/train.csv')

df_test  = pd.read_csv('../input/titanic/test.csv')

df_sample= pd.read_csv('../input/titanic/gender_submission.csv')
# defining function to encode features and get dummies

def feat_enc(df):

    

    df = df.drop(['Name','Ticket','Cabin'], axis=1)

    

    # encoding features to get dummies for training

    sex    = pd.get_dummies(df['Sex'], drop_first=True)

    embark = pd.get_dummies(df['Embarked'], drop_first=True)



    # adding encoded features to dataframes for training

    df = pd.concat([df,sex,embark], axis=1)



    # dropping columns after getting dummies for training

    df = df.drop(['Sex','Embarked'], axis=1)

    

    return df
# processing both traing and test dataframes

df_train_proc = feat_enc(df_train) 

df_test_proc = feat_enc(df_test) 
# features and label separation for training

features = df_train_proc.drop('Survived',axis=1).columns

X,y   = df_train_proc[features], df_train_proc['Survived']
# getting class weights

weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))



# setting classifier names

clf_names = ["Logistic_Regression_GrSrch", "SVC_GrSrch"]



# list of classifiers

classifiers = [LogisticRegression(class_weight=weights), SVC(class_weight=weights)]



# defining imputer and scaling functions

imputer  = SimpleImputer()

scaler   = StandardScaler()



# defining parameters for grid search

parameters = [

              {'imputer__strategy': ('mean', 'median'),

               'clf__penalty': ('l1', 'l2'),

               'clf__C': [0.1, 1],

               'clf__solver': ('liblinear', 'saga')},

              {'imputer__strategy': ('mean', 'median'),

               'clf__C': [0.1, 1],

               'clf__kernel': ('rbf', 'sigmoid'),

               'clf__gamma': ('scale', 'auto')}

             ]



# iterating through parameters for imputers/scalers & classifiers

time_list = []

y_pred_list = []

score_list = []



for name, classifier, params in zip(clf_names, classifiers, parameters):

    clf_pipe = Pipeline([('imputer', imputer), ('scaler', scaler),

                             ('clf', classifier)])

        

    t0 = time.time()

    gs_clf = GridSearchCV(clf_pipe, param_grid=params, scoring=make_scorer(f1_score), 

                              n_jobs=-1, cv=5, verbose=1)

        

    clf = gs_clf.fit(X, y)

        

    y_pred_list.append(clf.predict(df_test_proc))

        

    t1 = time.time()

    time_list.append(round(t1-t0,3))

    

    # getting best score for grid search

    score_list.append(round(gs_clf.best_score_, 3))



# exporting metrics to dataframe

metrics_df = pd.DataFrame.from_dict({'Classifier': clf_names,

                                     'Evaluation_Time': time_list,

                                     'Score': score_list,

                                     'Predictions': y_pred_list

                                     }).sort_values(['Score'], ascending=False)

    

print("Evaluation Metrics for Critical Reviews:", "\n")

print(metrics_df[['Classifier', 'Score', 'Evaluation_Time']])
# exporting results

y_pred = pd.DataFrame({'Survived': metrics_df.iloc[0][3]})

y_pred['PassengerId'] = df_test['PassengerId']



y_pred.to_csv('titanic_pred_sub.csv',index=False)
# cross checking shape of test output

y_pred.shape