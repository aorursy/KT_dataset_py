!pip install joblib
from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest

from sklearn.ensemble import *

from sklearn.linear_model import *

from xgboost import XGBRegressor #ML

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.svm import SVR

from sklearn.linear_model import ElasticNet

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from joblib import dump, load

import warnings

warnings.filterwarnings(action='ignore')
features = ['NU_NOTA_MT','NU_NOTA_COMP1','NU_NOTA_COMP2','NU_NOTA_COMP4','NU_NOTA_COMP5','NU_NOTA_COMP3','NU_NOTA_REDACAO','NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN']

data = pd.read_csv('../input/enem-2016/microdados_enem_2016_coma.csv', encoding='latin-1', sep=',', usecols=features, nrows=200000)
data.head()
data[['NU_NOTA_MT']].info()
data.describe()
col = data.columns       # .columns gives columns names in data 

print(col)
features = ['NU_NOTA_MT','NU_NOTA_COMP1','NU_NOTA_COMP2','NU_NOTA_COMP4','NU_NOTA_COMP5','NU_NOTA_COMP3','NU_NOTA_REDACAO','NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN']

target = "NU_NOTA_MT"
total = data[features].isnull().sum().sort_values(ascending = False)

percent = (data[features].isnull().sum()/data[features].isnull().count()*100).sort_values(ascending = False)

missing  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing
data_map = data[[target]]

data_map[features] = data[features]

plt.figure(figsize=(15,15))

sns.heatmap(data_map.corr(), annot=True, square=True, cmap='coolwarm')

plt.show()
for column in features:

    plt.figure(figsize = (20, 3))

    data.plot(kind='scatter', x=column, y=target)
duplicated_data = data.duplicated()

data[duplicated_data]
train = data.copy()

train = train.loc[:, features]

train.dropna(subset=[target], inplace=True)
y = train[target]

X = train.drop([target], axis=1)
numerical_columns = list(X._get_numeric_data().columns)

categorical_columns = list(set(X.columns) - set(numerical_columns))
numerical_pipeline = Pipeline([

        ('data_filler', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])
categorical_pipeline = Pipeline([

        ('data_filler', SimpleImputer(strategy='most_frequent')),

        ('encoder', OneHotEncoder(handle_unknown='ignore'))

    ])
transformer = ColumnTransformer([

    ("numerical", numerical_pipeline, numerical_columns),

    ("categorical", categorical_pipeline, categorical_columns)

])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
def train_ensemble_models(X, y):

    clf1 = LinearRegression()

    clf2 = Lasso(alpha=.5)

    clf3 = Ridge(alpha=.1)

    clf4 = LassoLars(alpha=.1)

    clf5 = AdaBoostRegressor()

    clf6 = SVR(kernel='rbf',gamma='scale',C=100)

    clf7 = GradientBoostingRegressor()



    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7], ['Linear Regression', 'Lasso', 'Ridge','Lasso Lars','Ada Boost Regressor','SVR', 'Gradient Boosting Regressor']):

        execute_pipeline(clf, X, y, label)
def execute_pipeline(clf, X, y, title):

    

    pipe = Pipeline([

        ('transformer', transformer),

        ('reduce_dim', 'passthrough'),

        ('classify', clf)

    ])



    N_FEATURES_OPTIONS = [2, 4, 8]

    

    param_grid = [

        {

            'reduce_dim': [PCA()],

            'reduce_dim__n_components': N_FEATURES_OPTIONS

        },

        {

            'reduce_dim': [SelectKBest()],

            'reduce_dim__k': N_FEATURES_OPTIONS

        },

    ]

    reducer_labels = ['PCA', 'KBest']



    grid = GridSearchCV(pipe,  param_grid=param_grid, scoring='r2', cv=10, verbose=1, n_jobs=-1, return_train_score=True)

    grid.fit(X, y)



    mean_train_scores = np.array(grid.cv_results_['mean_train_score'])

    mean_scores = np.array(grid.cv_results_['mean_test_score'])

    mean_scores = mean_scores.reshape(2, len(N_FEATURES_OPTIONS))

    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + .5)



    plt.figure()

    COLORS = 'bgrcmyk'

    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):

        plt.bar(bar_offsets + i, mean_train_scores[i], label='{} train'.format(label),alpha=.7)

        plt.bar(bar_offsets + i, reducer_scores, label='{} test'.format(label), color=COLORS[i])



    plt.title(title)

    plt.xlabel('Number of features')

    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)

    plt.ylabel('Classification accuracy')

    plt.ylim((0, 1))

    plt.legend(loc='upper left')



    plt.show()
grid_result = train_ensemble_models(X_train, y_train)
transformer = transformer

reduction = SelectKBest(k=8)

model = GradientBoostingRegressor()



X_train_transformer = transformer.fit_transform(X_train)

X_test_transformer = transformer.transform(X_test)



X_train_reduction_transformer = reduction.fit_transform(X_train_transformer, y_train)

X_test_reduction_transformer = reduction.transform(X_test_transformer)



model.fit(X_train_reduction_transformer, y_train)



y_predict = model.predict(X_test_reduction_transformer)



rmse = (np.sqrt(mean_squared_error(y_test, y_predict)))

r2 = r2_score(y_test, y_predict)

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))
cols = reduction.get_support(indices=True)

new_features = []

for bool, feature in zip(cols, X_train.columns):

    if bool:

        new_features.append(feature)

        

dataframe = pd.DataFrame(X_train, columns=new_features)

dataframe
dataframe['target'] = y_train
plt.figure(figsize=(15,15))

sns.heatmap(dataframe.corr(), annot=True, square=True, cmap='coolwarm')

plt.show()
persistence = {}

persistence['transformer'] = transformer

persistence['reduction'] = reduction

persistence['model']  = model

dump(persistence, 'persist.joblib')
persistence = load('persist.joblib')



transformer = persistence['transformer']

reduction = persistence['reduction']

model = persistence['model']



dataset_test_transformer = transformer.transform(dataset_test)

dataset_test_reduction_transformer = reduction.transform(dataset_test_transformer)



predictions = model.predict(dataset_test_reduction_transformer)
output = pd.DataFrame({'NU_INSCRICAO': nuInscricao, 'NU_NOTA_MT': predictions})
output.to_csv('answer.csv', index=False)

print("Your submission was successfully saved!")