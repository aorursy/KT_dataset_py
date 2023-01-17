import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, Normalizer

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
## Forcing pandas to display any number of elements

def set_pandas_options() -> None:

    pd.options.display.max_columns = 1000

    pd.options.display.max_rows = 1000

    pd.options.display.max_colwidth = 199

    pd.options.display.width = None

    pd.options.display.precision = 8

    pd.options.display.float_format = '{:,.3f}'.format

set_pandas_options()
# importing data

df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head()
df.info()
# distribtion of price variable

plt.figure(figsize=(15,5))

sns.distplot(df['price'], bins=20, kde=False)
# by looking the values count of these variable we see that these are categorial varable. 

cat_features = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']
for feature in cat_features:

    plt.figure(figsize=(15,8))

    sns.boxplot(y='price', x=feature, data=df)
num_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15',

              'yr_built', 'yr_renovated', 'lat', 'long', 'price']
sns.pairplot(df[num_feature])
# making target variable first column

def target_to_start(df, target):

    feature = list(df)

    feature.insert(0, feature.pop(feature.index(target)))

    df = df.loc[:, feature]

    return df

    

df = target_to_start(df, 'price')
# removing id column because it not relevant.

df.drop('id', axis=1, inplace=True)
# coverting columns into integer from float.

df.price = df.price.astype(int)

df.bathrooms = df.bathrooms.astype(int)

df.floors = df.floors.astype(int)
#I remove bedrooms above 11 because price is not as high as no.of bedroom e.g with 33 bedrooms price is less than 9 bed rooms

df = df[df['bedrooms']<11]
df = df[(df['bathrooms'] !=4) & (df['price'] != 7062500)]
df = df[(df['sqft_living']<13000) & (df['price']!=2280000)]
# i think yr_built is not so useful feature so i convert this into age_of_house.

df['age_of_house'] = df['date'].apply(lambda x: int(x[:4])) - df['yr_built']

df.drop('yr_built', axis=1, inplace=True)
# droping data columns because we do not need anymore.

df.drop('date', axis=1, inplace=True)
# convert yr_renovated into categorical variable.

df['renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)

df.drop('yr_renovated', axis=1, inplace=True)
# Performing log transformation of numrical variable to get normal distribation.

df['price_log'] = np.log(df['price'])

df['sqft_living_log'] = np.log(df['sqft_living'])

df['sqft_lot_log'] = np.log(df['sqft_lot'])

df['sqft_above_log'] = np.log(df['sqft_above'])

df['sqft_living15_log'] = np.log(df['sqft_living15'])

df['sqft_lot15_log'] = np.log(df['sqft_lot15'])
plt.figure(figsize=(15,5))

sns.distplot(df['price_log'], bins=20, kde=False)
df.to_pickle('clean_dataset')
df = pd.read_pickle('clean_dataset')
# All features

feature1 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition',

            'grade', 'sqft_above', 'sqft_basement', 'zipcode', 'lat', 'long', 'sqft_living15','sqft_lot15',

            'age_of_house', 'renovated']



# features that correlation is greater than "0.2"

feature2 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view',

            'grade', 'sqft_above', 'sqft_basement', 'lat', 'sqft_living15']



# numerical features with log_transform

feature3 = ['bedrooms', 'bathrooms', 'sqft_living_log', 'sqft_lot_log', 'floors','waterfront', 'view', 'condition',

            'grade', 'sqft_above', 'sqft_basement', 'zipcode', 'lat', 'long', 'sqft_living15_log','sqft_lot15_log',

            'age_of_house', 'renovated']



# numerical features with log_transform where correlation is greater that "0.2"

feature4 = ['bedrooms', 'bathrooms', 'sqft_living_log', 'floors', 'view',

            'grade', 'sqft_above', 'sqft_basement', 'lat', 'sqft_living15_log']
def correlation_of_each_feature(dataset, features):

    # get correlations of each features in dataset

    features.append('price_log')

    corrmat = dataset[features].corr()

    top_corr_features = corrmat.index

    plt.figure(figsize=(20,20))

    #plot heat map

    sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")



correlation_of_each_feature(df, feature1.copy())
# Evaluation Matrix 

evaluation_df = pd.DataFrame(columns=['Name of Model','Feature Set', 'Target', 'R^2 of Training', 'R^2 of Testing', 'Mean Squaued Error Training',

                                      'Mean Squaued Error Testing'])
# function to split data into training and testing set

def feature_target(features, target):

    X = df[features]

    y = df[target]

    feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return feature_train, feature_test, label_train, label_test
def model_xyz(model, feature_train, feature_test, label_train, label_test, model_name='Linear Regression', feature_set=1,

              target='price'):

    model.fit(feature_train, label_train)  

    y_pred_train = model.predict(feature_train)

    y_pred_test = model.predict(feature_test)

    r2_train = r2_score(label_train, y_pred_train)

    r2_test = r2_score(label_test, y_pred_test)

    rmse_train = mean_squared_error(label_train, y_pred_train)

    rmse_test = mean_squared_error(label_test, y_pred_test)

    

    r = evaluation_df.shape[0]

    evaluation_df.loc[r] = [model_name, feature_set, target, r2_train, r2_test, rmse_train, rmse_test]
feature_train, feature_test, label_train, label_test = feature_target(feature1, 'price')

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 1,

          target='price')
feature_train, feature_test, label_train, label_test = feature_target(feature2, 'price')

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 2,

          target='price')
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 3,

          target='price_log')
feature_train, feature_test, label_train, label_test = feature_target(feature4, 'price_log')

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 4,

          target='price_log')
evaluation_df
feature_train, feature_test, label_train, label_test = feature_target(feature1, 'price')

polyfeat = PolynomialFeatures(degree = 2)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2',

          feature_set= 1, target='price')
feature_train, feature_test, label_train, label_test = feature_target(feature2, 'price')

polyfeat = PolynomialFeatures(degree = 2)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2', 

          feature_set= 2, target='price')
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

polyfeat = PolynomialFeatures(degree = 2)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2', 

          feature_set= 3, target='price_log')
feature_train, feature_test, label_train, label_test = feature_target(feature4, 'price_log')

polyfeat = PolynomialFeatures(degree = 2)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2', 

          feature_set= 4, target='price_log')
evaluation_df
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

polyfeat = PolynomialFeatures(degree = 3)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 3',

          feature_set= 3, target='price_log')
feature_train, feature_test, label_train, label_test = feature_target(feature4, 'price_log')

polyfeat = PolynomialFeatures(degree = 3)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = LinearRegression()

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 3',

          feature_set= 4, target='price_log')
evaluation_df.iloc[6:,]
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

polyfeat = PolynomialFeatures(degree = 3)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = Lasso(alpha=10)

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Lasso Regression with degree 3',

          feature_set= 3, target='price_log')
evaluation_df.iloc[8:,]
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

polyfeat = PolynomialFeatures(degree = 3)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = Lasso()

search_grid={'alpha':[0.001,0.01,0.05,1,10,20]}

search=GridSearchCV(estimator=lr, param_grid=search_grid, 

                    scoring='neg_mean_squared_error', n_jobs=1, cv=5)
search.fit(feature_train, label_train)
search.best_params_
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

polyfeat = PolynomialFeatures(degree = 3)

feature_train = polyfeat.fit_transform(feature_train)

feature_test = polyfeat.fit_transform(feature_test)

lr = Lasso(alpha=20)

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Lasso Regression with degree 3 with alpha=20',

          feature_set= 3, target='price_log')
evaluation_df
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

lr =  DecisionTreeRegressor()

search_grid={'max_depth':[6,7,8,9,10,11,12,13,14,15]}

search=GridSearchCV(estimator=lr, param_grid=search_grid, 

                    scoring='neg_mean_squared_error', n_jobs=1, cv=3)
search.fit(feature_train, label_train)
search.best_params_
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

lr =  DecisionTreeRegressor(max_depth=9)

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Decision Tree Regressor with alpha 9',

          feature_set= 3, target='price_log')
evaluation_df
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9))



search_grid={'n_estimators':[200,300,400,500],'learning_rate':[0.05, 0.1, 0.3, 1]}



search=GridSearchCV(estimator=ada, param_grid=search_grid, 

                    scoring='neg_mean_squared_error', n_jobs=1, cv=3)
search.fit(feature_train, label_train)
search.best_params_
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

lr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), learning_rate=1, n_estimators=500)

model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Decision Tree Regressor with alpha 9',

          feature_set= 3, target='price_log')
evaluation_df
np.exp(0.032)
X = df[feature3]

y = df['price_log']

lr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), learning_rate=1, n_estimators=500)

scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)
scores.mean()
feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')

lr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), learning_rate=1, n_estimators=500)

model = lr.fit(feature_train, label_train)

pred = model.predict(feature_test)
validation_df  = pd.DataFrame()
validation_df['actual'] = np.exp(label_test)
validation_df['predetion'] = np.exp(pred)
validation_df