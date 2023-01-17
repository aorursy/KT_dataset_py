import numpy as np 

import pandas as pd

import os



# CSV loaderfunction

def load_data(path, file_name):

    csv_path = os.path.join(path, file_name)

    return pd.read_csv(csv_path)



# Extract X_test

X_test = load_data('/kaggle/input/titanic-machine-learning-from-disaster','test.csv')



# y_test

y_test = load_data('/kaggle/input/y-test-titanic','y_test.csv')



# Extract X_train

X_train = load_data('/kaggle/input/titanic-machine-learning-from-disaster','train.csv')



# Extract predictions (y_train) from X_train

y_train = X_train[['Survived']]

y_train.append(y_test[['Survived']])



# Drop the predictions from the training set

X_train.drop('Survived', axis=1, inplace=True)



# Merge x_train and x_test

X_merged = X_train.append(X_test)
from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer





# COMBINE AND CLEAN THE DATA - Using a Class



class Modifier():

    

    def transform(self, df=X_merged, 

                  company=True, title=True, 

                  cabin_letter=True, ticket_num=True,

                  age=True, pipeline=True):

        if company:

            df['Company'] = df['SibSp'] + df['Parch']

        if title:

            df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.')

            

            df['Title'] = df['Title'].replace(

                                            ['Lady','Countess',

                                             'Capt','Col',

                                             'Don','Dr',

                                             'Major','Rev',

                                             'Sir','Jonkheer',

                                             'Dona'],

                                                    'Unusual')

            

            df['Title'] = df['Title'].replace('Mlle','Miss')

            df['Title'] = df['Title'].replace('Ms','Miss')

            df['Title'] = df['Title'].replace('Mme','Mrs')

        if cabin_letter:

            df['Cabin_letter'] = df.Cabin.str.extract('([A-Z])') #Takes the letter of the Cabin

        if ticket_num:

            df['Ticket_num'] = df.Ticket.str.extract('([0-9]+)') #Takes the number on the ticket

        if age:

            df['Age_strat'] = pd.cut(df['Age'],

                                     bins=[

                                             0, 11, 

                                             18, 22,

                                             27, 33,

                                             40, 66,

                                             np.inf

                                                     ],

                                     

                                     labels=[i for i in range(1,9)]

                                    )

            

        return df

    

# ATTRIBUTES



# Numeric

num_attr = ['Fare']



# Categorical Alphabetic

cat_attr = ['Embarked', 'Title',

            'Cabin_letter','Sex']



# Categorical Numeric

ord_attr = ['Pclass','Company','Age_strat']





#PIPELINE TO CLEAN AND TRANSFORM THE DATA 



# For numeric attributes

num_pipeline = Pipeline(

    [

        ('imputer', SimpleImputer(strategy='median')),

        ('StdScaler', StandardScaler())

    ]

)



# For categorical alphabetic attributes

cat_pipeline = Pipeline(

    [

        ('imputer', SimpleImputer(strategy='constant', fill_value = 'Z')),

        ('OneVeryHot', OneHotEncoder())

    ]

)



# For categorical alphabetic numeric

ord_pipeline = Pipeline(

    [

        ('imputer', SimpleImputer(strategy='constant',

                                  fill_value = 0)),

        ('OneVeryHot', OneHotEncoder())

    ]    

)





# MERGE THE THREE PIPELINES IN ONE



full_pipeline = ColumnTransformer(

    [('num', num_pipeline, num_attr),

    ('cat', cat_pipeline, cat_attr),

    ('ord', ord_pipeline, ord_attr)]

)



# Instanciateing

attr_mod = Modifier()

mod_df = attr_mod.transform(ticket_num=False) #ticket_num set to false since it is probably not useful for the algorithm



# Our dataframe transformed and output as a spare matrix --- READY TO BE TO PASS IT TO THE ALGORITHM 

X = full_pipeline.fit_transform(mod_df)
# Split into training data and test data 



X, X_test = X[:891], X[891:]
# Function to nicely print the cross val results

def cross_val_results(cross_val_score):

    print('---------CROSS VALIDATION - THREE-FOLD---------\n')

    print('Scores: \t', [round(i,3) for i in cross_val_score])

    print('Mean:   \t', round(cross_val_score.mean(),2))

    print('Std dev:\t', round(cross_val_score.std(),6))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



# List of attributes after pipeline

list_of_attr = [

                    'Fare','Embarked_S','Embarked_C',

                    'Embarked_Q','Embarked_Nan' ,'Mr',

                    'Mrs','Miss','Master','Unusual',

                    'Cabin_NaN','Cabin_C','Cabin_E', 

                    'Cabin_G','Cabin_D','Cabin_A',

                    'Cabin_B','Cabin_F','Cabin_T',

                    'male','female','Class_3',

                    'Class_1','Class_2','Company_1',

                    'Company_0','Company_4','Company_2',

                    'Company_6','Company_5','Company_3',

                    'Company_7','Company_10','Age_3',

                    'Age_6', 'Age_4', 'Age_NaN','Age_7',

                    'Age_1','Age_2','Age_5','Age_8'

                                                   ]



# Create a function that returns a data frame of which its columns are: attributes, coeficients and Odds Ratio

def coef(model, list_of_attr):

    list_of_coef = list(model.coef_[0,:])

    

    coef_df = pd.DataFrame(

            {

            'Attributes':list_of_attr, 

            'Coefficients': list_of_coef, 

            'Odds_Ratio': [np.exp(i) for i in list_of_coef]

            }

        )

    

    return coef_df



log_reg = LogisticRegression()

log_reg_cross_val = cross_val_score(

                                log_reg, 

                                X,

                                np.ravel(y_train),

                                cv=3

                                    )



# Print cross val results

cross_val_results(log_reg_cross_val)



log_reg.fit(X, np.ravel(y_train))



coef_df = coef(log_reg, list_of_attr)



# Print positive coefficients, in descending order

coef_df.loc[coef_df['Coefficients'] < 0].sort_values(

                                                    ascending=False,

                                                    by='Coefficients'

                                                            ).head()
from sklearn.tree import DecisionTreeClassifier



tree_clf = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=7)

tree_clf.fit(X, np.ravel(y_train))



cross_val_results(

    cross_val_score(

        tree_clf,

        X,

        np.ravel(y_train),

        cv=3

    )

)



# Extract the decision tree as a .dot file



from sklearn.tree import export_graphviz



export_graphviz(

    tree_clf,

    out_file='/kaggle/working/tree_clf.dot',

    feature_names=list_of_attr,

    class_names= ['Died','Survived'],

    rounded=True,

    filled=True

)



# Convert the decision tree .dot file into an understandable image .png



import pydot



(graph,) = pydot.graph_from_dot_file('tree_clf.dot')

graph.write_png('tree_clf.png')
from sklearn.ensemble import RandomForestClassifier

from  sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC



# Instanciate

log_clf = LogisticRegression()

rnd_clf = RandomForestClassifier(n_estimators=300,

                                 max_leaf_nodes=15,

                                 n_jobs=-1)

svm_clf = SVC(probability=True)



voting_clf = VotingClassifier(

                    estimators=[

                                    ('lr',log_clf),

                                    ('rf',rnd_clf),

                                    ('svc',svm_clf)                         

                                ],

                    voting='soft'

)



cross_val_results(

    cross_val_score(

                    voting_clf,

                    X,

                    np.ravel(y_train),

                    cv=3)

                            )
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



param_grid = [

    {

     'n_estimators': [200, 300, 500],

     'max_leaf_nodes': [15,17,23]

    }

]



rnd_clf = RandomForestClassifier(n_jobs=-1, oob_score=True) #All available CPU's





grid_search = GridSearchCV(rnd_clf, param_grid, cv=3, return_train_score=True)



grid_search.fit(X, np.ravel(y_train))

grid_search.best_params_



cvres = grid_search.cv_results_



for scores, params in zip(cvres['mean_test_score'],cvres['params']):

    print(scores, params)



grid_search.best_params_
rfc_2 = RandomForestClassifier(n_estimators=300, max_leaf_nodes=15, n_jobs=-1) 



cross_val_results(

    cross_val_score(

                    rfc_2,

                    X,

                    np.ravel(y_train),

                    cv=3)

                            )
from sklearn.ensemble import ExtraTreesClassifier



etc_clf = ExtraTreesClassifier(n_jobs=-1, oob_score=True) #All available CPU's





grid_search_2 = GridSearchCV(

                             etc_clf,

                             param_grid,

                             cv=3,

                             return_train_score=True

                                )



grid_search.fit(X, np.ravel(y_train))



cvres = grid_search.cv_results_



for scores, params in zip(cvres['mean_test_score'],cvres['params']):

    print(scores, params)
# RANDOM FOREST CLASSIFIER



random_forest = RandomForestClassifier(

                                        n_estimators=300,

                                        max_leaf_nodes=15,

                                        n_jobs=-1

                                                    )

random_forest.fit(X, np.ravel(y_train))

y_pred = random_forest.predict(X_test)



from sklearn.metrics import accuracy_score



accuracy_score(y_pred, y_test['Survived'])
from sklearn.metrics import precision_score, recall_score



precision = precision_score(y_pred, y_test['Survived'])

recall = recall_score(y_pred, y_test['Survived'])



print('Precision: {}\nRecall: {}'.format(round(precision,2),

                                         round(recall,2)))
# EXTREMELY RANDOMIZED TREES 



etc = ExtraTreesClassifier(max_leaf_nodes=17, n_estimators=200, n_jobs=-1)

etc = etc.fit(X, np.ravel(y_train))

ex_tr_cl_pred = etc.predict(X_test)



accuracy_score(ex_tr_cl_pred, y_test['Survived'])
# SOFT VOTING CLASSIFIER 



voting_clf.fit(X, np.ravel(y_train))

p = voting_clf.predict(X_test)



accuracy_score(p, y_test['Survived'])
# SUBMISSION



submission = pd.DataFrame(

                    {

                        'PassengerId': y_test['PassengerId'],

                        "Survived": y_pred

                    }

                            )



submission.to_csv('/kaggle/working/submission.csv', index=False)
