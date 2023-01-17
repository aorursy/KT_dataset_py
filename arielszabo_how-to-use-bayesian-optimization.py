import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import category_encoders

from sklearn.model_selection import train_test_split, cross_validate, KFold

from sklearn.pipeline import Pipeline

from sklearn import metrics

from hyperopt import hp, tpe, fmin, space_eval

import os



from sklearn.decomposition import PCA

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.cluster import FeatureAgglomeration



np.random.seed(123)

train = pd.read_csv(os.path.join('..', 'input', 'train.csv'), index_col='ID')

train.head()
test = pd.read_csv(os.path.join('..', 'input', 'test.csv'), index_col='ID')

test.head()
train.info()
def test_model(x_train, x_test, y_train, y_test, model):

    """ fit the model and print the train and test result """

    np.random.seed(1)

    model.fit(x_train, y_train)

    print('train score: ', model.score(x_train, y_train))

    print('test score: ', model.score(x_test, y_test))
# Split to X and y and then to train and test sets:

X = train.drop('y', axis=1)

y = train['y']

x_train, x_test, y_train, y_test = train_test_split(X, y)
# One hot encoding to the categorical columns in the data:

one_hot = category_encoders.OneHotEncoder(cols=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'], drop_invariant=True, use_cat_names=True)

x_train_one_hot = one_hot.fit_transform(x_train)

x_test_one_hot = one_hot.transform(x_test)
test_model(x_train_one_hot, x_test_one_hot, y_train, y_test, model=SVR())
def get_model(args):

    """Construct the mode based on the args choosen in the current step of the bayesian optimization process"""

    feature_selector = args['selection']

        

    model = Pipeline([

        ('scaler', args['scaler']()),

        ('selection', feature_selector['selection_algo'](**feature_selector['selection_params'])),

        ('clf', args['clf'](**args['clf_params']))

    ])



    return model
def objective_func(args, x_train=x_train_one_hot, y_train=y_train):

    """

    Run a cross validation on the train data and return the mean test score.

    This function output will be value the bayesian optimization process will try to minimize.

    """

    np.random.seed(123)

    model = get_model(args)



    cv_results = cross_validate(estimator=model, X=x_train, y=y_train, n_jobs=-1, scoring='r2',

                                cv=KFold(n_splits=4))

    return - cv_results['test_score'].mean() # minus is because we optimize to the minimum
search_space = {

    'scaler': hp.choice('scaler', [StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler]),

    'selection':  hp.choice('selection',[

        {

        'selection_algo': SelectKBest,

        'selection_params': 

            {

            'k': hp.choice('k', ['all'] + list(range(1, x_train_one_hot.shape[1]))),

            'score_func': hp.choice('score_func', [f_regression, mutual_info_regression])

            }

        },

        {

            'selection_algo': PCA,

            'selection_params': {'n_components': hp.uniformint('n_components', 1, x_train_one_hot.shape[1])}

        },

        {

            'selection_algo': FeatureAgglomeration,

            'selection_params': {'n_clusters': hp.uniformint('n_clusters', 1, x_train_one_hot.shape[1])}

        }

    ]),



    'clf': SVR,

    'clf_params': 

        {

            'kernel': hp.choice('kernel', ['rbf', 'poly', 'linear']),

            'C': hp. uniform('C', 0.0001, 30)

        }



}
np.random.seed(123)

best_space = fmin(objective_func, space=search_space, algo=tpe.suggest, max_evals=100)

best_model =  get_model(space_eval(search_space, best_space))

print(best_model)
space_eval(search_space, best_space)
test_model(x_train_one_hot, x_test_one_hot, y_train, y_test, model=best_model)
# Run on the real test

# X_one_hot = one_hot.fit_transform(X)

# test_one_hot = one_hot.transform(test)



# best_model.fit(X_one_hot, y)

# pd.DataFrame({'ID':test.index, 'y': best_model.predict(test_one_hot)}).to_csv(r'subs.csv', index=False)