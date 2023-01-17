# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns



# import pystan model class

import pystan



# import sklearn data preprocessing

from sklearn.preprocessing import RobustScaler



# import sklearn model class

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



# import sklearn model evaluation classification metrics

from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
# acquiring training and testing data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# combine training and testing dataframe

df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_test.insert(1, 'target', np.nan)

df_data = pd.concat([df_train, df_test], ignore_index=True)
def swarmplot(categorical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a swarm plot applied for categorical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        categorical_x (list or str): The categorical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    categorical_x, numerical_y = [categorical_x] if type(categorical_x) == str else categorical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(categorical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.swarmplot(x=vj, y=vi, data=data, ax=axes[i*len(categorical_x) + j], rasterized=True) for i, vi in enumerate(numerical_y) for j, vj in enumerate(categorical_x)]

    return fig
# describe training and testing data

df_data.describe(include='all')
# convert dtypes numeric to object

col_convert = ['target']

df_data[col_convert] = df_data[col_convert].astype('object')
# list all features type number

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))



# list all features type object

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))
# feature exploration: histogram of all numeric features

_ = df_data.hist(bins=20, figsize=(200, 150))
# feature exploration: target

col_number = df_data.select_dtypes(include=['number']).columns.drop(['id']).tolist()

_ = swarmplot('target', col_number, df_data)
# feature extraction: target

df_data['target'] = df_data['target'].fillna(-1)
# convert category codes for data dataframe

df_data = pd.get_dummies(df_data, columns=['datatype'], drop_first=True)
# convert dtypes object to numeric for data dataframe

col_convert = ['target']

df_data[col_convert] = df_data[col_convert].astype(int)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# select all features to evaluate the feature importances

x = df_data[df_data['datatype_training'] == 1].drop(['id', 'target', 'datatype_training'], axis=1)

y = df_data.loc[df_data['datatype_training'] == 1, 'target']
# set up lasso regression to find the feature importances

lassoreg = Lasso(alpha=1e-5).fit(x, y)

feat = pd.DataFrame(data=lassoreg.coef_, index=x.columns, columns=['feature_importances']).sort_values(['feature_importances'], ascending=False)
# plot the feature importances

feat[(feat['feature_importances'] < -1e-3) | (feat['feature_importances'] > 1e-3)].dropna().plot(y='feature_importances', figsize=(20, 5), kind='bar')

plt.axhline(-0.05, color="grey")

plt.axhline(0.05, color="grey")
# list feature importances

model_feat = feat[(feat['feature_importances'] < -0.05) | (feat['feature_importances'] > 0.05)].index
# select the important features

x = df_data.loc[df_data['datatype_training'] == 1, model_feat]

y = df_data.loc[df_data['datatype_training'] == 1, 'target']
# create scaler to the features

scaler = RobustScaler()

x = scaler.fit_transform(x)
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.25, random_state=58)
# linear regression model setup

model_linreg = LinearRegression()



# linear regression model fit

model_linreg.fit(x_train, y_train)



# linear regression model prediction

model_linreg_ypredict = model_linreg.predict(x_validate)



# linear regression model metrics

model_linreg_rocaucscore = roc_auc_score(y_validate, model_linreg_ypredict)

model_linreg_cvscores = cross_val_score(model_linreg, x, y, cv=20, scoring='roc_auc')

print('linear regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_linreg_rocaucscore, model_linreg_cvscores.mean(), 2 * model_linreg_cvscores.std()))
# lasso regression model setup

model_lassoreg = Lasso(alpha=0.01)



# lasso regression model fit

model_lassoreg.fit(x_train, y_train)



# lasso regression model prediction

model_lassoreg_ypredict = model_lassoreg.predict(x_validate)



# lasso regression model metrics

model_lassoreg_rocaucscore = roc_auc_score(y_validate, model_lassoreg_ypredict)

model_lassoreg_cvscores = cross_val_score(model_lassoreg, x, y, cv=20, scoring='roc_auc')

print('lasso regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_rocaucscore, model_lassoreg_cvscores.mean(), 2 * model_lassoreg_cvscores.std()))
# specify the hyperparameter space

params = {

    'alpha': np.logspace(-4, -2, base=10, num=50),

}



# lasso regression grid search model setup

model_lassoreg_cv = GridSearchCV(model_lassoreg, params, iid=False, cv=5)



# lasso regression grid search model fit

model_lassoreg_cv.fit(x_train, y_train)



# lasso regression grid search model prediction

model_lassoreg_cv_ypredict = model_lassoreg_cv.predict(x_validate)



# lasso regression grid search model metrics

model_lassoreg_cv_rocaucscore = roc_auc_score(y_validate, model_lassoreg_cv_ypredict)

model_lassoreg_cv_cvscores = cross_val_score(model_lassoreg_cv, x, y, cv=20, scoring='roc_auc')

print('lasso regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_lassoreg_cv_rocaucscore, model_lassoreg_cv_cvscores.mean(), 2 * model_lassoreg_cv_cvscores.std()))

print('  best parameters: %s' %model_lassoreg_cv.best_params_)
# ridge regression model setup

model_ridgereg = Ridge(alpha=35)



# ridge regression model fit

model_ridgereg.fit(x_train, y_train)



# ridge regression model prediction

model_ridgereg_ypredict = model_ridgereg.predict(x_validate)



# ridge regression model metrics

model_ridgereg_rocaucscore = roc_auc_score(y_validate, model_ridgereg_ypredict)

model_ridgereg_cvscores = cross_val_score(model_ridgereg, x, y, cv=20, scoring='roc_auc')

print('ridge regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_ridgereg_rocaucscore, model_ridgereg_cvscores.mean(), 2 * model_ridgereg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, 4, base=10, num=50)}



# ridge regression grid search model setup

model_ridgereg_cv = GridSearchCV(model_ridgereg, params, iid=False, cv=5)



# ridge regression grid search model fit

model_ridgereg_cv.fit(x_train, y_train)



# ridge regression grid search model prediction

model_ridgereg_cv_ypredict = model_ridgereg_cv.predict(x_validate)



# ridge regression grid search model metrics

model_ridgereg_cv_rocaucscore = roc_auc_score(y_validate, model_ridgereg_cv_ypredict)

model_ridgereg_cv_cvscores = cross_val_score(model_ridgereg_cv, x, y, cv=20, scoring='roc_auc')

print('ridge regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_ridgereg_cv_rocaucscore, model_ridgereg_cv_cvscores.mean(), 2 * model_ridgereg_cv_cvscores.std()))

print('  best parameters: %s' %model_ridgereg_cv.best_params_)
# elastic net regression model setup

model_elasticnetreg = ElasticNet(alpha=0.01, l1_ratio=0.9)



# elastic net regression model fit

model_elasticnetreg.fit(x_train, y_train)



# elastic net regression model prediction

model_elasticnetreg_ypredict = model_elasticnetreg.predict(x_validate)



# elastic net regression model metrics

model_elasticnetreg_rocaucscore = roc_auc_score(y_validate, model_elasticnetreg_ypredict)

model_elasticnetreg_cvscores = cross_val_score(model_elasticnetreg, x, y, cv=20, scoring='roc_auc')

print('elastic net regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_elasticnetreg_rocaucscore, model_elasticnetreg_cvscores.mean(), 2 * model_elasticnetreg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, -2, base=10, num=10),

          'l1_ratio': np.linspace(0.1, 0.9, num=5),

}



# elastic net regression grid search model setup

model_elasticnetreg_cv = GridSearchCV(model_elasticnetreg, params, iid=False, cv=5)



# elastic net regression grid search model fit

model_elasticnetreg_cv.fit(x_train, y_train)



# elastic net regression grid search model prediction

model_elasticnetreg_cv_ypredict = model_elasticnetreg_cv.predict(x_validate)



# elastic net regression grid search model metrics

model_elasticnetreg_cv_rocaucscore = roc_auc_score(y_validate, model_elasticnetreg_cv_ypredict)

model_elasticnetreg_cv_cvscores = cross_val_score(model_elasticnetreg_cv, x, y, cv=20, scoring='roc_auc')

print('elastic net regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_elasticnetreg_cv_rocaucscore, model_elasticnetreg_cv_cvscores.mean(), 2 * model_elasticnetreg_cv_cvscores.std()))

print('  best parameters: %s' %model_elasticnetreg_cv.best_params_)
# kernel ridge regression model setup

model_kernelridgereg = KernelRidge(alpha=0.0001, kernel='polynomial', degree=4)



# kernel ridge regression model fit

model_kernelridgereg.fit(x_train, y_train)



# kernel ridge regression model prediction

model_kernelridgereg_ypredict = model_kernelridgereg.predict(x_validate)



# kernel ridge regression model metrics

model_kernelridgereg_rocaucscore = roc_auc_score(y_validate, model_kernelridgereg_ypredict)

model_kernelridgereg_cvscores = cross_val_score(model_kernelridgereg, x, y, cv=20, scoring='roc_auc')

print('kernel ridge regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_kernelridgereg_rocaucscore, model_kernelridgereg_cvscores.mean(), 2 * model_kernelridgereg_cvscores.std()))
# specify the hyperparameter space

params = {'alpha': np.logspace(-4, -2, base=10, num=10),

          'degree': [1, 2, 3, 4, 5],

}



# kernel ridge regression grid search model setup

model_kernelridgereg_cv = GridSearchCV(model_kernelridgereg, params, iid=False, cv=5)



# kernel ridge regression grid search model fit

model_kernelridgereg_cv.fit(x_train, y_train)



# kernel ridge regression grid search model prediction

model_kernelridgereg_cv_ypredict = model_kernelridgereg_cv.predict(x_validate)



# kernel ridge regression grid search model metrics

model_kernelridgereg_cv_rocaucscore = roc_auc_score(y_validate, model_kernelridgereg_cv_ypredict)

model_kernelridgereg_cv_cvscores = cross_val_score(model_kernelridgereg_cv, x, y, cv=20, scoring='roc_auc')

print('kernel ridge regression grid search\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_kernelridgereg_cv_rocaucscore, model_kernelridgereg_cv_cvscores.mean(), 2 * model_kernelridgereg_cv_cvscores.std()))

print('  best parameters: %s' %model_kernelridgereg_cv.best_params_)
# decision tree regression model setup

model_treereg = DecisionTreeRegressor(splitter='best', min_samples_split=5)



# decision tree regression model fit

model_treereg.fit(x_train, y_train)



# decision tree regression model prediction

model_treereg_ypredict = model_treereg.predict(x_validate)



# decision tree regression model metrics

model_treereg_rocaucscore = roc_auc_score(y_validate, model_treereg_ypredict)

model_treereg_cvscores = cross_val_score(model_treereg, x, y, cv=20, scoring='roc_auc')

print('decision tree regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_treereg_rocaucscore, model_treereg_cvscores.mean(), 2 * model_treereg_cvscores.std()))
# random forest regression model setup

model_forestreg = RandomForestRegressor(n_estimators=100, min_samples_split=3, random_state=58)



# random forest regression model fit

model_forestreg.fit(x_train, y_train)



# random forest regression model prediction

model_forestreg_ypredict = model_forestreg.predict(x_validate)



# random forest regression model metrics

model_forestreg_rocaucscore = roc_auc_score(y_validate, model_forestreg_ypredict)

model_forestreg_cvscores = cross_val_score(model_forestreg, x, y, cv=20, scoring='roc_auc')

print('random forest regression\n  roc auc score: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestreg_rocaucscore, model_forestreg_cvscores.mean(), 2 * model_forestreg_cvscores.std()))
# stan model setup

model_code = """

    data {

        int N; // the number of training data

        int N2; // the number of testing data

        int K; // the number of features

        int y[N]; // the response variable

        matrix[N,K] X; // the training matrix

        matrix[N2,K] X_test; // the testing matrix

    }

    parameters {

        vector[K] alpha;

        real beta;

    }

    transformed parameters {

        vector[N] y_linear;

        y_linear = beta + X * alpha;

    }

    model {

        alpha ~ cauchy(0, 10); // cauchy distribution

        for (i in 1:K)

            alpha[i] ~ student_t(1, 0, 0.03); // student t distribution

        y ~ bernoulli_logit(y_linear); // bernoulli distribution, logit parameterization

    }

    generated quantities {

        vector[N2] y_pred;

        y_pred = beta + X_test * alpha;

    }

"""



model_data = {

    'N': 250,

    'N2': 19750,

    'K': 300,

    'y': df_data.loc[df_data['datatype_training'] == 1, 'target'],

    'X': df_data[df_data['datatype_training'] == 1].drop(['id', 'target', 'datatype_training'], axis=1),

    'X_test': df_data[df_data['datatype_training'] == 0].drop(['id', 'target', 'datatype_training'], axis=1),

}



model_stan = pystan.StanModel(model_code=model_code)



# stan model fit

model_stan_fitted = model_stan.sampling(data=model_data, seed=58)
# prepare testing data and compute the observed value

x_test = df_data[df_data['datatype_training'] == 0]

y_test = pd.DataFrame(np.mean(model_stan_fitted.extract(permuted=True)['y_pred'], axis=0), columns=['target'], index=df_data.loc[df_data['datatype_training'] == 0, 'id'])
# submit the results

out = pd.DataFrame({'id': y_test.index, 'target': y_test['target']})

out.to_csv('submission.csv', index=False)