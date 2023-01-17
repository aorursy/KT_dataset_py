import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score



from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
sns.set_style('darkgrid')
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.shape
df.head()
# Serial No. is not needed. let's drop it

df.drop('Serial No.', axis=1, inplace=True)



# make sure it is dropped

df.head()
df.shape
# check data types of all features

df.dtypes
# is there any nulls?

df.isnull().sum()
# what about NA?

df.isna().sum()
df.rename(columns={'GRE Score':'GRE', 

                   'TOEFL Score':'TOEFL', 

                   'University Rating':'uni_rating',

                   'Chance of Admit ':'admit_prob'}, inplace=True)

df.head(10)
df.tail(10)
# let's check summary of all feature distribution

df.describe()
# Plot Admission chance vs GRE score

plt.scatter(df.GRE, df.admit_prob, alpha=0.25)

plt.title('Admission Chance vs GRE Score')

plt.xlabel('GRE Score')

plt.ylabel('Admission Chance')

plt.show()
# Plot Admission chance vs TOEFL score

plt.scatter(df.TOEFL, df.admit_prob, alpha=0.25)

plt.title('Admission Chance vs TOEFL Score')

plt.xlabel('TOEFL Score')

plt.ylabel('Admission Chance')

plt.show()
# Plot Admission chance vs CGPA

plt.scatter(df.CGPA, df.admit_prob, alpha=0.25)

plt.title('Admission Chance vs CGPA')

plt.xlabel('CGPA')

plt.ylabel('Admission Chance')

plt.show()
# Let's look at all distributions

df.hist(figsize=(9,9), xrot=45)

plt.show()
sns.lmplot(x='GRE', y='admit_prob', hue='Research', data=df, fit_reg=False, 

           scatter_kws={'alpha':0.25, 's':100})

plt.xlabel('GRE Score')

plt.ylabel('Admission Chance')

plt.show()
sns.lmplot(x='TOEFL', y='admit_prob', hue='Research', data=df, fit_reg=False,

           scatter_kws={'alpha':0.25, 's':100})

plt.xlabel('TOEFL Score')

plt.ylabel('Admission Chance')

plt.show()
sns.lmplot(x='CGPA', y='admit_prob', hue='Research', data=df, fit_reg=False,

           scatter_kws={'alpha':0.25, 's':100})

plt.xlabel('CGPA')

plt.ylabel('Admission Chance')

plt.show()
df.corr()
sns.set_style('white')

plt.figure(figsize=(7,6))



df_corr = df.corr()



bool_mask = np.zeros_like(df_corr)

bool_mask[np.triu_indices_from(df_corr)]=1



sns.heatmap(df_corr, cmap='RdBu_r', annot=True, mask=bool_mask, cbar=False)



plt.show()
# Check if there are any duplicate records

df.duplicated().sum()
# let's create X and y to separate input data from target variable

y = df.admit_prob

X = df.drop('admit_prob', axis=1)
# split total data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=1234)
print(len(X_train), len(X_test), len(y_train), len(y_test))
X_train.head()
y_train.head()
# Create pipelines dictionary

pipelines = {

    'lasso': make_pipeline(StandardScaler(), Lasso(random_state=1234)),

    'ridge': make_pipeline(StandardScaler(), Ridge(random_state=1234)),

    'enet': make_pipeline(StandardScaler(), ElasticNet(random_state=1234)),

    'rf': make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1234)),

    'gb': make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1234))

}
# Lasso hyperparameters

lasso_hyperparameters = {

    'lasso__alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

}



# Ridge hyperparameters

ridge_hyperparameters = {

    'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

}



# Elastic Net hyperparameters

enet_hyperparameters = {

    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],

    'elasticnet__l1_ratio': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

}



# RandomForest hyperparameters

rf_hyperparameters = {

    'randomforestregressor__n_estimators': [100, 200, 500, 1000],

    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33, 0.25]

}



# GradientBoost hyperparameters

gb_hyperparameters = {

    'gradientboostingregressor__n_estimators': [100, 200, 500],

    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2, 0.3],

    'gradientboostingregressor__max_depth': [1, 3, 5, 7]

}
# hyperparameters dictionary

hyperparameters = {

    'lasso': lasso_hyperparameters,

    'ridge': ridge_hyperparameters,

    'enet': enet_hyperparameters,

    'rf': rf_hyperparameters,

    'gb': gb_hyperparameters

}
import time



# Create empty dictionary

fitted_models = {}



# Loop through model pipelines, tuning each one and storing in fitted_models

for name,pipeline in pipelines.items():

    start = time.time()

    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    

    # Fit model on X_train, y_train

    model.fit(X_train, y_train)

    

    # Store a model in fitted_models

    fitted_models[name] = model

    

    end = time.time()

    print(name, ' has been fitted. Took ', np.around(end - start, decimals=2), ' seconds')
# let's check that the models have been fitted correctly

from sklearn.exceptions import NotFittedError



for name, model in fitted_models.items():

    try:

        pred = model.predict(X_test)

        print(name, 'has been fitted.')

    except NotFittedError as e:

        print(repr(e))
# let's evaluate fitted models and print performance scores

for name,model in fitted_models.items():

    pred = model.predict(X_test)

    print()

    print(name)

    print('------------')

    print('R2 :', np.around(r2_score(y_test, pred), decimals=4))

    print('MSE :', np.around(mean_squared_error(y_test, pred), decimals=4))

    print('Holdout R2 :', np.around(model.best_score_, decimals=4))
sns.set_style('darkgrid')

plt.scatter(fitted_models['ridge'].predict(X_test), y_test, alpha=0.25, s=100)

plt.title('Predicted vs Actual')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
fitted_models['ridge'].best_params_
fitted_models['ridge'].best_estimator_