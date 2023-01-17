import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

pd.options.display.max_columns = 50
data = pd.read_csv('../input/house-price-prediction-challenge/train.csv')

print(data.shape)

data.head()
data.describe()
DEPENDENT_VARIABLE = 'TARGET(PRICE_IN_LACS)'

CATEGORICAL_INDEPENDENT_VARIABLES = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK', 'READY_TO_MOVE', 'RESALE']

CONTINUOUS_INDEPENDENT_VARIABLES = ['SQUARE_FT', 'LONGITUDE', 'LATITUDE']
fig, ax = plt.subplots(figsize=(20, 6))

sns.heatmap(data.isnull(), cbar=False, yticklabels=False)

fig.suptitle('Missing Values', fontsize=18)
fig = plt.figure(figsize=(20, 6))

fig.suptitle('Dependent Variable Distribution', fontsize=24)    

sns.distplot(data[DEPENDENT_VARIABLE], kde=False) 
for col in CATEGORICAL_INDEPENDENT_VARIABLES:

    fig = plt.figure(figsize=(20, 6))

    fig.suptitle(col, fontsize=18)

    ax =[

        plt.subplot2grid((1, 12), (0,0), colspan=4), plt.subplot2grid((1, 12), (0,4), colspan=8),

    ]

    data[col].value_counts().plot.pie(ax=ax[0], autopct="%1.1f%%")

    ax[0].legend()

    for val in data[col].unique():

        sns.distplot(data[data[col] == val][DEPENDENT_VARIABLE], ax=ax[1], label=val, kde=False)

    ax[1].legend()
for col in CONTINUOUS_INDEPENDENT_VARIABLES:

    fig, ax =plt.subplots(1, 2, figsize=(20, 6))

    fig.suptitle(col, fontsize=18)

    sns.distplot(data[col], ax=ax[0], kde=False)

    data.plot.scatter(x=col, y=DEPENDENT_VARIABLE, ax=ax[1])
import folium



map = folium.Map(location=[22.00,78.00], tiles='cartodbpositron', zoom_start=6)





for i in range(len(data)):

    folium.Circle(

        location=[data.iloc[i]['LONGITUDE'], data.iloc[i]['LATITUDE']],

        radius=100,

        color='blue').add_to(map)



map
X = data[CONTINUOUS_INDEPENDENT_VARIABLES+CATEGORICAL_INDEPENDENT_VARIABLES]

y = data[DEPENDENT_VARIABLE]



dtypes = {k: str for k in CATEGORICAL_INDEPENDENT_VARIABLES}

X = X.astype(dtypes)    



X.info()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import make_column_selector, make_column_transformer

from sklearn.pipeline import make_pipeline



def get_column_names(feature_name, columns):

    val = feature_name.split('_')[1]

    col_idx = int(feature_name.split('_')[0][1:])

    return f'{columns[col_idx]}_{val}'



class Preprocessor():

    

    def __init__(self, return_df=True):

        self.return_df = return_df

        

        self.impute_median = SimpleImputer(strategy='median')

        self.impute_const = SimpleImputer(strategy='constant')

        self.ss = StandardScaler()

        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        

        self.num_cols = make_column_selector(dtype_include='number')

        self.cat_cols = make_column_selector(dtype_exclude='number')

        

        self.preprocessor = make_column_transformer(

            (make_pipeline(self.impute_median, self.ss), self.num_cols),

            (make_pipeline(self.impute_const, self.ohe), self.cat_cols),

        )

        

    def fit(self, X):

        return self.preprocessor.fit(X)

        

    def transform(self, X):

        if self.return_df:

            return pd.DataFrame(

                self.preprocessor.transform(X),

                columns=self.num_cols(X)+[

                    get_column_names(_, self.cat_cols(X)) for _ in self.preprocessor.transformers_[1][1][1].get_feature_names()

                ]

            )

        return X

        

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
preprocessor = Preprocessor()

X = preprocessor.fit_transform(X)

X.head()
from sklearn.dummy import DummyRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.metrics import mean_squared_error



kf = KFold(random_state=19, shuffle=True)

X, y = X.values, y.values
%%time

r2scores = []

rmse = []

for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = DummyRegressor(strategy='mean').fit(X_train, y_train)

    r2scores.append(model.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    

print('Mean r2 score', np.mean(r2scores))

print('Mean rmse', np.mean(rmse))
%%time

r2scores = []

rmse = []

for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = DecisionTreeRegressor(random_state=19).fit(X_train, y_train)

    r2scores.append(model.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    

print('Mean r2 score', np.mean(r2scores))

print('Mean rmse', np.mean(rmse))
%%time



from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from xgboost import XGBRFRegressor, XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor



trees = [

    ('Random Forest', RandomForestRegressor), ('Extra Trees', ExtraTreesRegressor), ('LightGBM', LGBMRegressor),

    ('Gradient Boosting', GradientBoostingRegressor), ('XGBoost', XGBRegressor), ('XGBoostRF', XGBRFRegressor),

    ('CatBoost', CatBoostRegressor)

]
performance = {'rmse':[], '100* r2':[]}



for name, algo in trees:

    scores = []

    rmse = []

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        if name == 'CatBoost':

            model = algo(random_state=19, silent=True).fit(X_train, y_train)

        else:

            model = algo(random_state=19).fit(X_train, y_train)

        r2scores.append(model.score(X_test, y_test))

        rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))



    print(name)

    print('Mean r2 score', np.mean(r2scores))

    print('Mean rmse', np.mean(rmse))

    print()

    performance['100* r2'].append(100*np.mean(r2scores))

    performance['rmse'].append(np.mean(rmse))
fig, ax = plt.subplots(figsize=(20, 6))

ax = [ax]



labels = [x[0] for x in trees]





color = 'tab:orange'

ax[0].set_ylabel('RMSE', color=color, fontsize=12)

ax[0].plot(labels, performance['rmse'], color=color)

ax[0].tick_params(axis='y', labelcolor=color)



ax.append(ax[0].twinx())



color = 'tab:blue'

ax[1].set_ylabel('100*R2', color=color, fontsize=12)

ax[1].plot(labels, performance['100* r2'], color=color)

ax[1].tick_params(axis='y', labelcolor=color)



fig.suptitle('Comparison: Tree Ensembles', fontsize=18)

plt.show()
%%time

r2scores = []

rmse = []

for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = VotingRegressor([

        ('rf', RandomForestRegressor(random_state=19)),

        ('et', ExtraTreesRegressor(random_state=19)),

        ('lgbm', LGBMRegressor(random_state=19)),

        ('gb', GradientBoostingRegressor(random_state=19)),

        ('xgb', XGBRegressor(random_state=19)),

        ('xgbrf', XGBRFRegressor(random_state=19)),

        ('cb', CatBoostRegressor(random_state=19, silent=True)),

    ], weights=performance['100* r2']).fit(X_train, y_train)

    r2scores.append(model.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    

print('Mean r2 score', np.mean(r2scores))

print('Mean rmse', np.mean(rmse))
%%time



params = {

    'n_estimators': [10, 50, 100],

    'max_depth': [5, 10, 15, 25, 30, None],

    'min_samples_split': [2, 5, 10, 15, 25],

    'min_samples_leaf': [1, 2, 5, 10],

}



model = RandomizedSearchCV(ExtraTreesRegressor(random_state=19), params, cv=kf, n_iter=30, random_state=19).fit(X, y)

model.best_estimator_
%%time

r2scores = []

rmse = []

for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = ExtraTreesRegressor(max_depth=25, min_samples_split=5, random_state=19).fit(X_train, y_train)

    r2scores.append(model.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    

print('Mean r2 score', np.mean(r2scores))

print('Mean rmse', np.mean(rmse))
model = ExtraTreesRegressor(max_depth=25, min_samples_split=5, random_state=19).fit(X, y)
test = pd.read_csv('../input/house-price-prediction-challenge/test.csv')

print(test.shape)

test.head()
X_test = test[CONTINUOUS_INDEPENDENT_VARIABLES+CATEGORICAL_INDEPENDENT_VARIABLES]



dtypes = {k: str for k in CATEGORICAL_INDEPENDENT_VARIABLES}

X_test = X_test.astype(dtypes)



X_test = preprocessor.transform(X_test)

X_test.head()
pd.DataFrame(model.predict(X_test), columns=[DEPENDENT_VARIABLE]).to_csv('predictions.csv', index=False)