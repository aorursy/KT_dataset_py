# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import all libraries

import numpy as np



import pandas as pd



from matplotlib import pyplot as plt



import seaborn as sns

sns.set_style('darkgrid')
# get red wine dataset

df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
# stats of all the features

print(df.shape)
# get first 10 and last 10 values of the dataset

df.head(10)

df.tail(10)

# have a look at the values of all the features, all make sense, no data currupt
# check the unique values of out target feature

df.quality.unique()
# check the numeric and catagorical features in the dataset

df.dtypes

# all the features except target are float64

# target feature is int64
# Display stats of all numeric features

df.describe()

# observations

# 1. Quality(our target feature) has min 3 and max 8 value

# 2. There is no null values in any of the feature

# 3. count of all the features are same with the len(df)

# 4. all the features with their values makes sense

# 5. all the mean, min, max and std for all the features make sense
# display distributions of numeric features is by using Pandas histograms

df.hist(figsize=(10,10), xrot=45)

plt.show()

# observe all the features how all they distributed based on a scale of 0 to 1000 and 0.00 to 1.00
# check the count of some features like quality

sns.countplot(x='quality', data=df)

plt.show()
# get some correlations between our numeric features

corr = df.corr()

sns.set_style('whitegrid')

plt.figure(figsize=(10,10))

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = 1

sns.heatmap(corr, cmap='RdBu_r', annot=True, cbar=False, mask=mask)

plt.show()

# observations

# fixed acid is +ve correlated with density and -ve correlated with citric acid and pH

# quality is +ve correlated with alcohal
# Lets see how our target feature related with other features

# quality with fixed acidity

sns.boxplot(x='quality', y='fixed acidity', data=df)

plt.show()
# quality with pH

sns.boxplot(x='quality', y='pH', data=df)

plt.show()
# quality with density

sns.boxplot(x='quality', y='density', data=df)

plt.show()
# quality with alcohol

sns.boxplot(x='quality', y='alcohol', data=df)

plt.show()
# quality with citric acid

sns.boxplot(x='quality', y='citric acid', data=df)

plt.show()
# check is there any outliers with fixed acidity using violinplot

sns.violinplot(x='quality',y='fixed acidity' ,data=df)

plt.show()
# check is there any outliers with density using violinplot

sns.violinplot(x='quality',y='density' ,data=df)

plt.show()
# check the relation of fixed acidity, alcohal and quality

sns.lmplot(x='fixed acidity', y='alcohol', hue='quality', fit_reg=False, data=df)

plt.show()

# some observations

# there are some values might be outliers where fixed acidity > 14
# check the data where fixed aacidity > 14

df[df['fixed acidity'] > 14]

# in those records we found one record where alcohal is high and fixed acidity is high but quality is 5, it might be not

# good for our model, but we need to pick
# remove all the duplicates

df = df.drop_duplicates()

df.shape

# there are 240 duplicates in the dataset, removed all duplicate records
# import all algorithms, processing and metrics 

# algorithms

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# model selection

from sklearn.model_selection import train_test_split, GridSearchCV

# scaling and tuning

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

# evaluation

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# split datasets into test and train

X = df.drop('quality', axis=1)

y= df.quality

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

print(len(X_train), len(X_test), len(y_train), len(y_test))
# scale datasets manually

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
# create a function to fit and eveluate scores for default

def fit_and_evaluate_default(model, name):

    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    print('------------- {} default -------------'.format(name))

    print('R^2 score - ', r2_score(y_test, pred))

    print('MSE score - ', mean_squared_error(y_test, pred))

    print('MAE score - ', mean_absolute_error(y_test, pred))
# get some scores

# linear models

fit_and_evaluate_default(LinearRegression(), 'LinearRegression')

fit_and_evaluate_default(Lasso(random_state=123), 'Lasso')

fit_and_evaluate_default(Ridge(random_state=123), 'Ridge')

fit_and_evaluate_default(ElasticNet(random_state=123), 'ElasticNet')

# bagging and boosting

fit_and_evaluate_default(RandomForestRegressor(random_state=123), 'RandomForestRegressor')

fit_and_evaluate_default(GradientBoostingRegressor(random_state=123), 'GradientBoostingRegressor')

# observations

# in Linear models

#  - Lasso and ElasticNet is doing very poor

#  - LinearRegression and Ridge is doing ok and same

# in bagging and boosting

#  - Random Forest is worst than linear model

#  - Boosting is the winner with minimum MAE, MSE and good R2 score
# make pipeline and hyperparameters

pipeline = {

    'lasso': make_pipeline(StandardScaler(), Lasso(random_state=123)),

    'ridge': make_pipeline(StandardScaler(), Ridge(random_state=123)),

    'enet': make_pipeline(StandardScaler(), ElasticNet(random_state=123)),

    'rf': make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),

    'gb': make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))

}

lasso_hyperparameters = {

    'lasso__alpha' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

}

ridge_hyperparameters = {

     'ridge__alpha' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

}

enet_hyperparameters = {

    'elasticnet__alpha' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],

    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]

}

rf_hyperparameters = {

    'randomforestregressor__n_estimators': [100, 200],

    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],

    'randomforestregressor__min_samples_leaf': [1, 3, 5, 10]

}

gb_hyperparameters = {

    'gradientboostingregressor__n_estimators': [100, 200],

    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],

    'gradientboostingregressor__max_depth': [1, 3, 5]

}

hyperparameters = {

    'lasso': lasso_hyperparameters,

    'ridge': ridge_hyperparameters,

    'enet': enet_hyperparameters,

    'rf': rf_hyperparameters,

    'gb': gb_hyperparameters

}
# fit, tune and evaluate the models with hyperparameters

fitted_models = {}

for name, pip in pipeline.items():

    model = GridSearchCV(pip, hyperparameters[name], cv=10, n_jobs=-1)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print('-------------- {} HyperParameters --------------'.format(name))

    fitted_models[name] = model

    print('R^2 Score - ', r2_score(y_test, pred))

    print('MSE Score - ', mean_squared_error(y_test, pred))

    print('MAE Score - ', mean_absolute_error(y_test, pred))

# observations

# As we can see all the models with hyperparameters is doing great than default

#	Default parameters evaluation	                        ||	Hyper parameters evaluation

#	------------- LinearRegression default -------------	||	

#	R^2 score -  0.34950893422968987	||	

#	MSE score -  0.40176510124316495	||	

#	MAE score -  0.49586161191064304	||	



#	------------- Lasso default -------------	            ||	-------------- lasso HyperParameters --------------

#	R^2 score -  -9.415671320400776e-06	                    ||	R^2 Score -  0.343088183112235

#	MSE score -  0.6176393578219748	                        ||	MSE Score -  0.4057307725006274

#	MAE score -  0.6701018723956923	                        ||	MAE Score -  0.4988708354252194



#	------------- Ridge default -------------	            ||	-------------- Ridge HyperParameters --------------

#	R^2 score -  0.3495702136164457	                        ||	R^2 Score -  0.351485902258056

#	MSE score -  0.40172725303844764	                    ||	MSE Score -  0.40054405947661825

#	MAE score -  0.49582317783992663	                    ||	MAE Score -  0.49512066432203716



#	------------- ElasticNet default -------------	        ||	-------------- ElasticNet HyperParameters --------------

#	R^2 score -  -9.415671320400776e-06	                    ||	R^2 Score -  0.3410166818641156

#	MSE score -  0.6176393578219748	                        ||	MSE Score -  0.4070102011545636

#	MAE score -  0.6701018723956923	                        ||	MAE Score -  0.5007868122635286



#	------------- RandomForestRegressor default ----------	||	-------------- RandomForestRegressor HyperParameters --------------

#	R^2 score -  0.31135397745924054	                    ||	R^2 Score -  0.4151273888148237

#	MSE score -  0.42533088235294114	                    ||	MSE Score -  0.3612369426917526

#	MAE score -  0.5040441176470588	                        ||	MAE Score -  0.47408980640988907



#	----------- GradientBoostingRegressor default --------	||	-------------- GradientBoostingRegressor HyperParameters --------------

#	R^2 score -  0.3791890903531926	                        ||	R^2 Score -  0.3855813826539055

#	MSE score -  0.3834336412779906	                        ||	MSE Score -  0.37948554714032473

#	MAE score -  0.48444528046587343	                    ||	MAE Score -  0.4793086003776446



# GradientBoostingRegressor did good in their default parameters

# but RandomForestRegressor improved more than GradientBoostingRegressor with hyperparameters



# Winner model is "RandomForestRegressor HyperParameters"

#----------------------------------------

#R^2 Score -  0.4151273888148237

#MSE Score -  0.3612369426917526

#MAE Score -  0.47408980640988907

#----------------------------------------

# RandomForestRegressor best estimation by setting those parameters:

#  - max_features='sqrt', n_estimators=200, min_samples_leaf=3
# finally get the best estimator of the model

print(fitted_models['rf'].best_estimator_)

# RandomForestRegressor best estimation by setting those parameters:

#  - max_features='sqrt', n_estimators=200, min_samples_leaf=3
# plot our predicted values with our sample test values

pred = fitted_models['rf'].predict(X_test)

plt.scatter(pred, y_test)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# observations

# 5 and 6 are predicted well almost close
# Save the final model

import pickle

with open('final_model.pkl', 'wb') as f:

    pickle.dump(fitted_models['rf'].best_estimator_, f)