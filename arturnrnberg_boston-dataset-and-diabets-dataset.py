# importação de bibliotecas



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn import datasets

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV, train_test_split



%matplotlib inline
boston = datasets.load_boston()



print(boston.DESCR)
# Transformando os dados em data frame



boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)

boston_df['MEDV'] = boston.target



boston_df.head()
boston_corr = boston_df.corr().MEDV.drop('MEDV').sort_values(ascending = False)



boston_corr
boston_df.describe()
n_boston_df = (boston_df.iloc[:, :-1] - boston_df.iloc[:, :-1].mean())/boston_df.iloc[:, :-1].std()



n_boston_df.head()
X_train, X_test, y_train, y_test = train_test_split(n_boston_df,

                                                    boston.target,

                                                    test_size = 0.3,

                                                    random_state = 1)



models = {

    'random_forest': {'model': RandomForestRegressor()},

    'decision_tree': {'model': DecisionTreeRegressor()},

    'linear_regression': {'model': LinearRegression()},

    'k_nearest_neighbor': {'model': KNeighborsRegressor()}

}





for key in models.keys():

    models[key]['model'].fit(X_train, y_train)

    prediction = models[key]['model'].predict(X_test)

    models[key]['MSE'] = mean_squared_error(prediction, y_test)

    models[key]['MSA'] = mean_absolute_error(prediction, y_test)

    models[key]['r2'] = r2_score(y_test, prediction)

    print(models[key])
plt.scatter(y_test, models['random_forest']['model'].predict(X_test))

plt.xlabel('y test')

plt.ylabel('y predict')

plt.show()
columns = boston_corr.abs().sort_values(ascending = False).index



models_columns = dict()



for i in range(1, len(columns) + 1):

    X_train, X_test, y_train, y_test = train_test_split(n_boston_df.iloc[:, :i],

                                                        boston.target,

                                                        test_size = 0.3,

                                                        random_state = 1)

    

    

    models = {

        'random_forest': {'model': RandomForestRegressor()},

        'decision_tree': {'model': DecisionTreeRegressor()},

        'linear_regression': {'model': LinearRegression()},

        'k_nearest_neighbor': {'model': KNeighborsRegressor()}

    }

    MSA = dict()



    for key in models.keys():

        models[key]['model'].fit(X_train, y_train)

        prediction = models[key]['model'].predict(X_test)

        MSA[key] = mean_squared_error(prediction, y_test)

        

        

    models_columns[i] = MSA.copy()

    MSA.clear

  

models_MSA = pd.DataFrame(models_columns).T

models_MSA
plt.figure(figsize = (12,6))

plt.plot(models_MSA.index, models_MSA.random_forest, label = 'Random Forest')

plt.plot(models_MSA.index, models_MSA.decision_tree, label = 'Decision Tree')

plt.plot(models_MSA.index, models_MSA.linear_regression, label = 'Linear Regression')

plt.plot(models_MSA.index, models_MSA.k_nearest_neighbor, label = 'K Nearest Neighbor')

plt.title('MSA em função do número de colunas')

plt.xlabel('Numero de colunas')

plt.ylabel('MSA')

plt.legend()

plt.show()
diabets = datasets.load_diabetes()



print(diabets.DESCR)
diabets_df = pd.DataFrame(diabets.data, columns = diabets.feature_names)

diabets_df['target'] = diabets.target



diabets_df.head()
diabets_df.describe()
diabets_corr = diabets_df.corr().target.drop('target').sort_values(ascending = False)



diabets_corr
def select_model(features, target):

    models = [

        {

            'name': 'Random Forest',

            'estimator': RandomForestRegressor(),

            'hyperparameters': {

                'n_estimators': [1, 5, 10, 20, 50, 100], # 1 estimator é igual a arvore de decisão

                'max_depth': [2, 5, 10],

                "min_samples_leaf": [1, 5, 8],

                "min_samples_split": [2, 3, 5],

                'n_jobs': [-1]

            }

        },

        {

            'name': 'Linear Regression',

            'estimator': LinearRegression(),

            'hyperparameters': {

                'normalize': [True, False]

            }

        },

        {

            'name': 'K Nearest Neighbor',

            'estimator': KNeighborsRegressor(),

            'hyperparameters':{

                "n_neighbors": range(1,20,2),

                "weights": ["distance", "uniform"],

                "algorithm": ["ball_tree", "kd_tree", "brute"],

                "p": [1,2]

            }

        }

    ]

    

    best_parameters = dict()

    best_estimator = dict()

    best_score = dict()

    

    for model in models:

        grid = GridSearchCV(model['estimator'],

                            param_grid = model['hyperparameters'],

                            cv = 5)

        grid.fit(features, target)

        best_estimator[model['name']] = grid.best_estimator_

        best_parameters[model['name']] = grid.best_params_

        best_score[model['name']] = grid.best_score_

        

    return best_estimator, best_parameters, best_score
best_estimator, best_parameters, best_score = select_model(diabets_df.drop('target', axis = 1), diabets.target)



print(best_score)

print(best_estimator)
columns = diabets_corr.abs().sort_values(ascending = False).index



r2 = dict()



for i in range(1, len(columns) + 1):

    X_train, X_test, y_train, y_test = train_test_split(diabets_df.iloc[:, :i],

                                                        diabets.target,

                                                        test_size = 0.3,

                                                        random_state = 1)

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    r2[i] = r2_score(y_test, prediction)



r2
plt.scatter(y_test, prediction)

plt.xlabel('y predict')

plt.ylabel('y test')

plt.show()