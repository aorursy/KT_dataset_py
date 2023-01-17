# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model, svm

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV



from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import make_scorer, accuracy_score

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# show plots in the notebook

%matplotlib inline



# Any results you write to the current directory are saved as output.
def import_csv_datasets(file):

    ds = pd.read_csv('../input/' + file, sep=',', header=0)

    

    return ds 
train_origin = import_csv_datasets('train.csv')

test_origin = import_csv_datasets('test.csv')



train_origin.columns = map(str.lower, train_origin.columns)

test_origin.columns = map(str.lower, test_origin.columns)
sns.barplot(x="embarked", y="survived", hue="sex", data=train_origin);

train_origin[train_origin.isnull().any(axis=1)]
from sklearn.preprocessing import StandardScaler  



def simplify_ages(df):

    df["age"] = df["age"].fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    df['age'] = pd.cut(df.age, bins, labels=group_names)

    

    return df



def simplify_cabins(df):

    df.cabin = df.cabin.fillna('N')

    df.cabin = df.cabin.apply(lambda x: x[0])

    

    return df



def simplify_fares(df):    

    for pclass in [1,2,3]:

        fare_S = train_origin.loc[(train_origin['pclass'] ==pclass) & (train_origin['embarked'].isin(['S'])), ['fare']]

        fare_C = train_origin.loc[(train_origin['pclass'] ==pclass) & (train_origin['embarked'].isin(['C'])), ['fare']]

        fare_Q = train_origin.loc[(train_origin['pclass'] ==pclass) & (train_origin['embarked'].isin(['Q'])), ['fare']]



        df.fare = df.fare.fillna(fare_S.mean())

        df.fare = df.fare.fillna(fare_C.mean())

        df.fare = df.fare.fillna(fare_Q.mean())

        

    df.fare = df.fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    df.fare = pd.cut(df.fare, bins, labels=group_names)

    

    return df



def format_name(df):

    df['lname'] = df.name.apply(lambda x: x.split(' ')[0])

    df['name_prefix'] = df.name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

    return df.drop(['ticket', 'name'], axis=1)



def discret_family(df):

    family_size = df['sibsp']+df['parch'] + 1

    df['family_size'] = pd.cut(family_size, (0,1,5, 20), labels=["single", "small", "big"], include_lowest = True, right=True)

    

    return df



def simplify_embarked(df):

    df.embarked.fillna(-1)

    df.embarked = pd.factorize(df.embarked)[0]

    

    return df



def transform_features(df):    

    df = simplify_ages(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = discret_family(df)

    df = simplify_cabins(df)

    df = simplify_embarked(df)



    df = drop_features(df)

    

    return df
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['fare', 'cabin', 'age', 'sex', 'lname', 'name_prefix', 'family_size']

    df_combined = pd.concat([df_train[features], df_test[features]])

        

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

            

    return df_train, df_test
# Transofrming features

data_train = transform_features(train_origin)

data_test = transform_features(test_origin)
data_train
data_train, data_test = encode_features(data_train, data_test)
X_all = data_train.drop(['passengerid', 'survived'], axis=1, errors='ignore')

y_all = data_train.loc[:,'survived']



X_train, X_test, y_train, y_test =  train_test_split(X_all, y_all, test_size=0.2)
def scale_sets(X_train, X_test):

    scaler = StandardScaler()

    scaler.fit(X_train)

    

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    

    return X_train, X_test
def test_many_nn_models(X_train, Y_train, X_test, Y_test):

    parameters = {

        'hidden_layer_sizes':  [

            (5,3,2),

            (8,2),

            (10,5,3),

            (5,5,5),

            (11,8,5),

            (5,5),

            (13,8,5,5),

            (5, 5, 2),

            (11,8),

            (13,8),

            (13,17,2),

            (13,5,2)

        ], 

        "max_iter": [

            1000, 1500, 5000, 10000

        ],

        "learning_rate": [ "adaptive"],

        "alpha":[1e-8],

    }



    # Type of scoring used to compare parameter combinations

    acc_scorer = make_scorer(accuracy_score)



    # Run the grid search

    grid_obj = GridSearchCV(MLPClassifier(), parameters, scoring=acc_scorer, n_jobs=5)

    grid_obj = grid_obj.fit(X_train, y_train)



    # Set the clf to the best combination of parameters

    clf = grid_obj.best_estimator_



    # Fit the best algorithm to the data. 

    clf.fit(X_train, y_train)

    

    best_model = clf

    best_model_score = grid_obj.best_score_



    return best_model, best_model_score
def test_logistic_regression_models(X_train, Y_train, X_test, Y_test):    

    classifiers = [

        SVC(kernel="linear"),

        SVC(kernel="poly"),

        RandomForestClassifier(criterion='entropy',

            max_depth=5, max_features='log2', max_leaf_nodes=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=9),

        #XGBClassifier(),

        GaussianProcessClassifier(1.0 * RBF(1.0)),

        MLPClassifier(alpha=1),

    ]

    

    model_scores = []

    best_model = None

    best_model_score = None

    

    for classificator in classifiers:

        nn_test_model =classificator

        nn_test_model.fit(X_train,Y_train)

        

        predicted = nn_test_model.predict(X_test)

        score = metrics.accuracy_score(y_test, predicted)

        

        model_scores.append((classificator, score))

        

        if(best_model_score is None or score > best_model_score):

            best_model = nn_test_model

            best_model_score = score



    return best_model, best_model_score, model_scores
#X_train, X_test = scale_sets(X_train, X_test)

#best_model_nn, best_model_score_nn = test_many_nn_models(X_train, y_train, X_test, y_test)



#print(best_model_nn)

#print(best_model_score_nn)
def test_xgb(X_train, Y_train, X_test, Y_test):

    #inspiration by https://www.kaggle.com/phunter/xgboost-with-gridsearchcv

    

    xgb_model = XGBClassifier()



    parameters = {'nthread':[4],

                  'objective':['binary:logistic'],

                  'learning_rate': [0.01, 0.1], #so called `eta` value

                  'max_depth': [6, 7,8],

                  'min_child_weight': [11],

                  'silent': [1],

                  'subsample': [0.8],

                  'colsample_bytree': [0.7],

                  'n_estimators': [5, 100, 1000], #number of trees, change it to 1000 for better results

                  'missing':[-999],

                  'seed': [1337]}



    grid_obj= GridSearchCV(xgb_model, parameters, n_jobs=5,scoring='roc_auc', refit=True)

    grid_obj = grid_obj.fit(X_train, y_train)

    

    # Set the clf to the best combination of parameters

    clf = grid_obj.best_estimator_

    

    best_model = clf

    best_model_score = grid_obj.best_score_



    return best_model, best_model_score
best_model, best_model_score = test_xgb(X_train, y_train, X_test, y_test)

print(best_model)

print(best_model_score)



#evaluate the model using 10-fold cross-validation

scores = cross_val_score( best_model, X_all, y_all, scoring='accuracy', cv=10)

print(scores.mean()) 
#best_model2, best_model_score2, model_scores = test_logistic_regression_models(X_train, y_train, X_test, y_test)



#if best_model_score2 > best_model_score:

    #best_model = best_model2

    #best_model_score = best_model_score2



print(best_model)



#predictions = best_model.predict(X_test)

#print(metrics.classification_report(y_test,predictions))

#print(accuracy_score(y_test, predictions))



#evaluate the model using 10-fold cross-validation

scores = cross_val_score( best_model, X_all, y_all, scoring='accuracy', cv=10)

print(scores.mean())

from sklearn.cross_validation import KFold



def run_kfold(clf):

    kf = KFold(891, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(best_model)
##Saving output

X_out = data_test.drop(['passengerid'], axis=1, errors='ignore')

final_predictions =  best_model.predict(X_out.as_matrix())



submission = pd.DataFrame({

        "PassengerId": data_test['passengerid'],

        "Survived": final_predictions

    })

submission.to_csv('titanic_predictions.csv', index=False)

#submission.export()

#print(check_output(["ls", "../"]).decode("utf8"))

print(check_output(["ls", "../working"]).decode("utf8"))