# importação de bibliotecas



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV



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
m_base = models['random_forest']['model']

columns = boston_df.drop('MEDV', axis = 1).columns

values = m_base.feature_importances_



columns_importances = pd.Series(index = columns, data = values).sort_values(ascending = False)

columns_importances
important_columns = columns_importances[columns_importances > 0.005].index

boston_df_clean = boston_df[list(important_columns) + ['MEDV']]
def dendogram_spearmanr(df, tags):



    import scipy.cluster.hierarchy

    import scipy.stats

    

    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)

    corr_condensed = scipy.cluster.hierarchy.distance.squareform(1-corr)

    z = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')

    fig = plt.figure(figsize=(18,8))

    dendrogram = scipy.cluster.hierarchy.dendrogram(z, labels=tags, orientation='left', leaf_font_size=16)

    plt.show()

    

dendogram_spearmanr(boston_df_clean, boston_df_clean.columns)
X_train, X_test, y_train, y_test = train_test_split(boston_df_clean.drop('MEDV', axis = 1),

                                                    boston_df_clean.MEDV,

                                                    test_size=0.3,

                                                    random_state = 1)
%%time



m_base = RandomForestRegressor(oob_score = True,

                               n_jobs=-1,

                               random_state=0,

                               n_estimators=120,

                               min_samples_leaf=1,

                               max_features=0.90,

                               max_depth=8,)

m_base.fit(X_train, y_train)

r2_score(y_test, m_base.predict(X_test))
!pip install treeinterpreter

!pip install waterfallcharts
from treeinterpreter import treeinterpreter as ti

import waterfall_chart
row = X_test.values[np.newaxis,0]



prediction, bias, contributions = ti.predict(m_base, row)



idxs = np.argsort(contributions[0])

[o for o in zip(X_test.columns[idxs], X_test.iloc[0][idxs], contributions[0][idxs])]
waterfall_chart.plot(X_test.columns, contributions[0], threshold=0.08, 

                     rotation_value=90,formatting='{:,.3f}');
diabets = datasets.load_diabetes()



print(diabets.DESCR)
diabets_df = pd.DataFrame(diabets.data, columns = diabets.feature_names)

diabets_df['target'] = diabets.target



diabets_df.head()
diabets_df.describe()
diabets_corr = diabets_df.corr().target.drop('target').sort_values(ascending = False)



diabets_corr
def dendogram_spearmanr(df, tags):



    import scipy.cluster.hierarchy

    import scipy.stats

    

    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)

    corr_condensed = scipy.cluster.hierarchy.distance.squareform(1-corr)

    z = scipy.cluster.hierarchy.linkage(corr_condensed, method='average')

    fig = plt.figure(figsize=(18,8))

    dendrogram = scipy.cluster.hierarchy.dendrogram(z, labels=tags, orientation='left', leaf_font_size=16)

    plt.show()

    

dendogram_spearmanr(diabets_df, diabets_df.columns)
def select_model(features, target):

    models = [

        {

            'name': 'Random Forest',

            'estimator': RandomForestRegressor(),

            'hyperparameters': {

                'n_estimators': [1, 5, 10, 20, 50, 100], # 1 estimator é igual a arvore de decisão

                'max_depth': [2, 5, 10],

                "min_samples_leaf": [1, 5, 8],

                "min_samples_split": [2, 3, 5]

            }

        },

        {

            'name': 'Linear Regression',

            'estimator': LinearRegression(),

            'hyperparameters': {

                'normalize': [True, False]

            }

        },

#         {

#             'name': 'K Nearest Neighbor',

#             'estimator': KNeighborsRegressor(),

#             'hyperparameters':{

#                 "n_neighbors": range(1,20,2),

#                 "weights": ["distance", "uniform"],

#                 "algorithm": ["ball_tree", "kd_tree", "brute"],

#                 "p": [1,2]

#             }

#         }

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
X_train, X_test, y_train, y_test = train_test_split(diabets_df.drop(['target'], axis = 1),

                                                    diabets_df.target,

                                                    test_size=0.3,

                                                    random_state = 0)





linear_model = LinearRegression()

linear_model.fit(X_train, y_train)



forest_model = RandomForestRegressor(max_depth=10, min_samples_leaf=8, min_samples_split=3,

                      n_estimators=50)

forest_model.fit(X_train, y_train)



print('Linear Model R2: ',r2_score(y_test, linear_model.predict(X_test)))

print('Random Forest Model R2: ',r2_score(y_test, forest_model.predict(X_test)))
columns =diabets_df.drop(['target'], axis = 1).columns

values = forest_model.feature_importances_



columns_importances = pd.Series(index = columns, data = values).sort_values(ascending = False)

columns_importances
columns_importances = pd.Series(index = columns, data = linear_model.coef_).abs().sort_values(ascending = False)

columns_importances
row = X_test.values[np.newaxis,0]



prediction, bias, contributions = ti.predict(m_base, row)



idxs = np.argsort(contributions[0])

[o for o in zip(X_test.columns[idxs], X_test.iloc[0][idxs], contributions[0][idxs])]
waterfall_chart.plot(X_test.columns, contributions[0], threshold=0.08, 

                     rotation_value=90,formatting='{:,.3f}');