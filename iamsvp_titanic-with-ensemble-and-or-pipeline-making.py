import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB 

import catboost
test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')
def how_much_empty(*args): # calculate the percentage of empty values

    print('column '.ljust(13,' '), end='\t')

    for i in range(len(args)):

        print(i+1,'df ', end='\t\t')

    print('\n', '='*20*len(args), end='\n\n')

    

    cols_df = []

    for df in args:

        cols_df.append([(df[col].isnull().sum()/len(df))*100 for col in df.columns])    

    

    for i in range(len(cols_df[0])):

        print(str(df.columns[i]).ljust(13,' '), end = '')

        for j in cols_df:

            print(str(round(j[i], 3)).rjust(6,' '), end = '%\t\t') 

        print('\n')
how_much_empty(train)
def pipelining_preprocessor(df1, dropcolls = None, target = None): # return processed  dataFrame

    df = df1.copy()

    

    if target:

        y = df[target]

        df.drop(target, axis=1, inplace=True)

    

    if dropcolls:

        df.drop(dropcolls, axis=1, inplace=True)

    

    categoric = df.select_dtypes(include='object').columns

    numeric = df.select_dtypes(exclude='object').columns

    

    imputer_numeric = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 

                                      ('normalizer', Normalizer()), 

                                     ])

    imputer_categoric = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 

                                        ('onehot', OneHotEncoder(drop='if_binary')), 

                                       ])

    

    preprocessor = ColumnTransformer(transformers=[('imputer_numeric', imputer_numeric, numeric), ('imputer_categoric', imputer_categoric, categoric)], 

                                     n_jobs=-1, verbose=True)

    

    if target:

        return (pd.DataFrame(preprocessor.fit_transform(df)), y)

    else:

        return pd.DataFrame(preprocessor.fit_transform(df))
def pipelining_preprocessor2(df1, dropcolls = None, target = None): # return processed  dataFrame

    df = df1.copy()

    

    if target:

        y = df[target]

        df.drop(target, axis=1, inplace=True)

    

    if dropcolls:

        df.drop(dropcolls, axis=1, inplace=True)

    

    def normalizer(df):

        return pd.DataFrame(Normalizer().fit_transform(df), columns = df.columns)

    

    def fill_empty_by_pop(df): # filler for empty values in cols by most popular values (order columns by original)

        df_obj = df.loc[:, df.dtypes == 'object']

        pop_obj = df_obj.describe().loc['top',:]

        

        df_digits = df.loc[:, df.dtypes != 'object']

        pop_digits = df_digits.median()

        

        

#         return df_obj.fillna('Unknown').join(normalizer(df_digits.fillna(pop_digits)))[df.columns]

        return df_obj.fillna(pop_obj).join(normalizer(df_digits.fillna(pop_digits)))[df.columns]

    

    ohe = pd.get_dummies(fill_empty_by_pop(df), drop_first=True, prefix_sep=': ',)

    

    if target:

        return ohe, y

    else:

        return ohe
X, y = pipelining_preprocessor2(train, 

                                dropcolls = ['PassengerId', 'Name', 'Cabin', 'Ticket'], 

                                target='Survived')
X_test = pipelining_preprocessor2(test, dropcolls = ['PassengerId', 'Name', 'Cabin', 'Ticket'])
X
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
full_pool = catboost.Pool(X, y)
params = {'learning_rate': [0.87, 0.9, 0.93],

         }



cat_model = catboost.CatBoostClassifier(early_stopping_rounds=25, 

                                        eval_metric='Accuracy',

                                        loss_function='Logloss', 

#                                         task_type = 'GPU', 

                                        depth=4, 

                                        l2_leaf_reg=12, 

                                        max_ctr_complexity = 1

                                       )
grid_search_results = cat_model.grid_search(params, full_pool, 

                                            partition_random_seed=2, cv = skf, 

                                            plot=True, 

                                            verbose=False)
grid_search_results['params']
rf_model = RandomForestClassifier(n_estimators=1000,

                                  random_state=2, 

                                  max_features = 'sqrt', 

                                  oob_score = True, 

                                  criterion='gini', 

                                  n_jobs=-1,)



params = {'max_depth': range(19, 23)}
forest_results = GridSearchCV(rf_model, 

                              params, cv=skf, 

                              n_jobs=-1, verbose=1)



# forest_results.fit(X, y)
# forest_results.best_params_
cat_final = catboost.CatBoostClassifier(early_stopping_rounds=25, 

                                        eval_metric='Accuracy',

                                        loss_function='Logloss', 

#                                         task_type = 'GPU', 

                                        learning_rate = 0.9, 

                                        depth=4, 

                                        l2_leaf_reg=12, 

                                        max_ctr_complexity = 1

                                       )



rf_final = RandomForestClassifier(n_estimators=1000,

                                  random_state=2, 

                                  max_features = 'sqrt', 

                                  oob_score = True, 

                                  criterion='gini', 

                                  max_depth=25, 

                                  n_jobs=-1,)



final_model_hard = VotingClassifier(estimators=[('rf', rf_final), 

                                           ('cat', cat_final), 

                                           ('LogisticRegression', LogisticRegression(max_iter = 30000)), 

                                           ('SVC', SVC()), 

                                           ('GaussianNB', GaussianNB()), 

                                           ('DecisionTreeClassifier', DecisionTreeClassifier()), 

                                           ], voting = 'hard')



final_model_soft = VotingClassifier(estimators=[('rf', rf_final), 

                                           ('cat', cat_final), 

                                           ('LogisticRegression', LogisticRegression(max_iter = 30000)), 

                                           ('SVC', SVC(probability=True)), 

                                           ('GaussianNB', GaussianNB()), 

                                           ('DecisionTreeClassifier', DecisionTreeClassifier()), 

                                           ], voting = 'soft')



final_model_lil = VotingClassifier(estimators=[('rf', rf_final), 

                                           ('cat', cat_final), 

                                           ('DecisionTreeClassifier', DecisionTreeClassifier()), 

                                           ], voting = 'hard')
cross_val_score(final_model_hard, X, y, cv=skf, n_jobs=-1).mean()
cross_val_score(final_model_soft, X, y, cv=skf, n_jobs=-1).mean()
cross_val_score(final_model_lil, X, y, cv=skf, n_jobs=-1).mean()
cross_val_score(rf_final, X, y, cv=skf, n_jobs=-1).mean()
cross_val_score(cat_final, X, y, cv=skf, n_jobs=-1).mean()
final_model_soft.fit(X,y)

y_pred = final_model_soft.predict(X_test)



# rf_final.fit(X,y)

# y_pred = rf_final.predict(X_test)
output = pd.DataFrame({'PassengerId': pd.read_csv('../input/titanic/test.csv').PassengerId, 'Survived': y_pred})

output.to_csv('submission_ensemble_soft.csv', index=False)

output