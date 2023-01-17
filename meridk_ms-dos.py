# unzip data
!unzip /kaggle/input/restaurant-revenue-prediction/test.csv.zip
!unzip /kaggle/input/restaurant-revenue-prediction/train.csv.zip
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
def preparing_data(df):
    
    # drop Id as useless column
    df.drop('Id', axis=1, inplace=True)

    # remove revenue outstands
    df = df[~df['revenue'].ge(1.25 * 10**7)]
    
    # convert Open Date to datetime
    df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y')
    
    # create OpenDays column
    df['OpenDays'] = (df['Open Date'].max() - df['Open Date']).astype('timedelta64[D]').astype(int) + 1
    
    # log on OpenDays
    df['OpenDays'] = np.log(df['OpenDays'])

    # drop Open Date column
    df.drop('Open Date', axis=1, inplace=True)

    # in test data there are 57 unique cities and int train only 34, so we drop this column
    df.drop('City', axis=1, inplace=True)

    # get dummies of City Group
    df = pd.get_dummies(df, columns=['City Group'], drop_first=True)

    # change MB to DT type
    df['Type'].replace('MB', 'DT', inplace=True)

    # get dummies of Type
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    
    # log revenue
    df['revenue'] = np.log(df['revenue'])
    
    return df
# read data and drop Id column as useless column
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# connect data for preprocessing
df = pd.concat([train, test], axis=0)
plt.plot(df['revenue'], '.')
plt.show()
df['Open Date'].max()
# prepare data
df = preparing_data(df)
# find columns with high correlation
# r = df.corr()['revenue']
# r[r > 0.15]
# use columns with high correlation
# df = df[['P2', 'P6', 'P23', 'P28', 'OpenDays', 'revenue']]
df.head()
# split df on test and train
train = df[~df['revenue'].isnull()]
test = df[df['revenue'].isnull()]
# split test and train on test_X, test_y, train_X
train_X, train_y = train.drop('revenue', axis=1), train['revenue']
test_X = test.drop('revenue', axis=1)
def grid_func(model, train_X, train_y, test_X, file_name):
    
    parameters = [{'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
               'normalize': [True, False]}]
    
    grid = GridSearchCV(model, parameters)
    
    grid.fit(train_X, train_y)
    
    print(file_name, ':', grid.best_estimator_)
    
    res = grid.predict(test_X)
    
    pd.DataFrame(np.e ** res, columns=['Prediction']).to_csv(file_name, index=True, index_label='Id')
def linear_regression(train_X, train_y, test_X):
    lr = LinearRegression(normalize=True).fit(train_X, train_y)
    pd.DataFrame(np.e ** lr.predict(test_X), columns=['Prediction']).to_csv('linear_regression.csv', index=True, index_label='Id')

linear_regression(train_X, train_y, test_X)
grid_func(Ridge(), train_X, train_y, test_X, 'ridge_grid.csv')
grid_func(Lasso(), train_X, train_y, test_X, 'lasso_grid.csv')
grid_func(ElasticNet(), train_X, train_y, test_X, 'elastic_grid.csv')
def ridge_bayesian(train_X, train_y, test_X):
    ridge_b = BayesianRidge(normalize=True)
    ridge_b.fit(train_X, train_y)
    print(ridge_b.alpha_)
    
    res = ridge_b.predict(test_X)
    
    pd.DataFrame(np.e ** res, columns=['Prediction']).to_csv('ridge_bayesian.csv', index=True, index_label='Id')

ridge_bayesian(train_X, train_y, test_X)
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

neg_rmsle = make_scorer(rmsle, greater_is_better=False)

params = { 'alpha': (0.001, 1), 'fit_intercept': (0,1.99) }
def target(**params):

    model = Lasso(alpha = params['alpha'])
    
    scores = cross_val_score(model, train_X, train_y, scoring=neg_rmsle, cv=3)
    return scores.mean()

lasso_alpha = BayesianOptimization(target, params, random_state=101)

lasso_alpha.maximize(init_points=5, n_iter=15, acq='ucb', kappa=2)
lasso1 = Lasso(alpha=0.02945).fit(train_X, train_y)
pd.DataFrame(np.e ** lasso1.predict(test_X), columns=['Prediction']).to_csv('lasso_bayesian1.csv', index=True, index_label='Id')

lasso2 = Lasso(alpha=0.07492).fit(train_X, train_y)
pd.DataFrame(np.e ** lasso2.predict(test_X), columns=['Prediction']).to_csv('lasso_bayesian2.csv', index=True, index_label='Id')
def target(**params):
    fit_intercept = int(params['fit_intercept'])
    fit_intercept_dict = { 0: False, 1: True }

    model = ElasticNet(alpha = params['alpha'],
                    fit_intercept = fit_intercept_dict[fit_intercept],
                    copy_X = True)
    
    scores = cross_val_score(model, train_X, train_y, scoring=neg_rmsle, cv=3)
    return scores.mean()

elastic_alpha = BayesianOptimization(target, params, random_state=101)

elastic_alpha.maximize(init_points=5, n_iter=15, acq='ucb', kappa=2)
el1 = ElasticNet(alpha=0.3077).fit(train_X, train_y)
pd.DataFrame(np.e ** el1.predict(test_X), columns=['Prediction']).to_csv('el1.csv', index=True, index_label='Id')

el2 = ElasticNet(alpha=0.2426).fit(train_X, train_y)
pd.DataFrame(np.e ** el2.predict(test_X), columns=['Prediction']).to_csv('el2.csv', index=True, index_label='Id')

el3 = ElasticNet(alpha=0.04738).fit(train_X, train_y)
pd.DataFrame(np.e ** el3.predict(test_X), columns=['Prediction']).to_csv('el3.csv', index=True, index_label='Id')
forest = RandomForestRegressor().fit(train_X, train_y)
forest_res = np.e ** forest.predict(test_X)

pd.DataFrame(forest_res, columns=['Prediction']).to_csv('random_forest.csv', index=True, index_label='Id')
