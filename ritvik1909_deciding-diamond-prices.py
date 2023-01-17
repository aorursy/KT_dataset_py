import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('darkgrid')

pd.options.display.max_columns = 150
data = pd.read_csv('../input/diamonds/diamonds.csv')

print(data.shape)

data.head()
data.drop('Unnamed: 0', axis=1, inplace=True)
data.describe()
data = data[(data['x'] > 0) & (data['y'] > 0) & (data['z'] > 0)].reset_index(drop=True)

print(len(data))
fig = plt.figure(figsize=(20, 6))

sns.heatmap(data.isnull(), yticklabels=False, cbar=False)
fig = plt.figure(figsize=(20, 6))

sns.distplot(data['price'], kde=False)
for col in ['cut', 'color', 'clarity']:

    fig, ax =plt.subplots(1, 2, figsize=(20, 6))

    fig.suptitle(col, fontsize=18)

    data[col].value_counts().plot.pie(ax=ax[0], autopct="%1.1f%%")

    ax[0].legend()

    for val in data[col].unique():

        sns.distplot(data[data[col] == val]['price'], ax=ax[1], label=val, kde=False)

    ax[1].legend()

    plt.show()
for col in ['carat', 'depth', 'table', 'x', 'y', 'z']:

    fig, ax =plt.subplots(1, 2, figsize=(20, 6))

    fig.suptitle(col, fontsize=18)

    sns.distplot(data[col], ax=ax[0], kde=False)

    data[[col]+['price']].plot.scatter(x=col, y='price', ax=ax[1])

    plt.show()
sns.catplot(data=data, x='clarity', hue='cut', y='price', kind='point', aspect=3)
sns.catplot(data=data, x='color', hue='clarity', y='price', kind='point', aspect=3)
sns.catplot(data=data, x='color', hue='cut', y='price', kind='point', aspect=3)
df = pd.DataFrame()

df['Volume'] = data[['x', 'y', 'z']].apply(lambda row: row['x'] * row['y'] * row['z'], axis=1)

df['Mass'] = data['carat']

df['Density'] = df['Mass'] / df['Volume']

df['Price'] = data['price']
df.head()
df.describe()
for col in ['Volume', 'Density']:

    fig, ax =plt.subplots(1, 2, figsize=(20, 6))

    fig.suptitle(col, fontsize=18)

    sns.distplot(df[col], ax=ax[0], kde=False)

    df[[col]+['Price']].plot.scatter(x=col, y='Price', ax=ax[1])

    plt.show()
fig, ax = plt.subplots(1, 3, figsize=(20, 6))



df.plot.scatter(x='Density', y='Price', ax=ax[0])

df.plot.scatter(x='Mass', y='Price', ax=ax[1])

df.plot.scatter(x='Volume', y='Price', ax=ax[2])
sns.heatmap(df.corr(), annot=True, center=0, cmap='RdYlGn')
X = data.drop(['price'], axis=1)

y = data['price']
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

        self.ohe = OneHotEncoder(handle_unknown='ignore')

        

        self.num_cols = make_column_selector(dtype_include='number')

        self.cat_cols = make_column_selector(dtype_exclude='number')

        

        self.preprocessor = make_column_transformer(

            (make_pipeline(self.impute_median, self.ss), self.num_cols),

            (make_pipeline(self.impute_const, self.ohe), self.cat_cols),

        )

        

    def fit(self, X):

        return self.preprocessor.fit(X)

        

    def transform(self, X):

        Xtransformed = self.preprocessor.transform(X)

        try:

            Xtransformed = Xtransformed.todense()

        except:

            pass

        if self.return_df:

            return pd.DataFrame(

                Xtransformed,

                columns=self.num_cols(X)+list(map(

                    lambda x: get_column_names(x, self.cat_cols(X)),

                    self.preprocessor.transformers_[1][1][1].get_feature_names()

                ))

            )

        return X

        

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)
X = Preprocessor().fit_transform(X)

print(X.shape)

X.head()
features = X.columns

X = X.values

y= y.values
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import mean_squared_error
kf = KFold(random_state=19, shuffle=True)
%%time

r2scores = []

rmse = []

for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = LinearRegression().fit(X_train, y_train)

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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from xgboost import XGBRFRegressor, XGBRegressor

from lightgbm import LGBMRegressor



trees = [

    ('Random Forest', RandomForestRegressor), ('Extra Trees', ExtraTreesRegressor), ('LightGBM', LGBMRegressor),

    ('Gradient Boosting', GradientBoostingRegressor), ('XGBoost', XGBRegressor), ('XGBoostRF', XGBRFRegressor),

]
%%time

for name, algo in trees:

    r2scores = []

    rmse = []

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]



        model = algo(random_state=19).fit(X_train, y_train)

        r2scores.append(model.score(X_test, y_test))

        rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))



    print(name)

    print('Mean r2 score', np.mean(r2scores))

    print('Mean rmse', np.mean(rmse))

    print()
%%time

r2scores = []

rmse = []

for train_index, test_index in kf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    model = LGBMRegressor(random_state=19).fit(X_train, y_train)

    r2scores.append(model.score(X_test, y_test))

    rmse.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))

    

print('Mean r2 score', np.mean(r2scores))

print('Mean rmse', np.mean(rmse))
fig, ax = plt.subplots(figsize=(20, 6))

fig.suptitle('Feature Importance')

pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).plot.bar(ax=ax)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))