import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from IPython.display import HTML, display

import warnings

warnings.filterwarnings('ignore')

green_diamond = dict(markerfacecolor='g', marker='D')

mpl.style.use('seaborn')



%matplotlib inline
#Loading the different data into pandas dataframe objects.

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", sep = ',')

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", sep = ',')

#Show the train data

train_data.head(5)
#First we check the shape of the data

train_data.shape
def checkNanValues(dataframe):

    for columnName in dataframe.columns:

        nulls = dataframe[columnName].isna().sum()

        if nulls > 0:

            print(columnName + " : " + str(nulls) + ", " + str(nulls/dataframe.shape[0]) + "%")



checkNanValues(train_data)
#Drop all the features that we mentioned

train_data = train_data.drop(columns = ['Id','LotFrontage', 'Alley', 'BsmtQual', 'BsmtCond',

                                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',

                                        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',

                                       'PoolQC', 'Fence', 'MiscFeature'])

train_data['MasVnrType'].fillna((train_data['MasVnrType'].mode()[0]), inplace=True)

train_data['MasVnrArea'].fillna((train_data['MasVnrArea'].median()), inplace=True)

checkNanValues(train_data)

train_data.dropna(inplace=True)
plt.title('SalePrice Boxplot');

plt.boxplot(train_data['SalePrice'].values, flierprops=green_diamond);

plt.xticks([1],['SalePrice']);
sns.distplot(train_data['SalePrice']);
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

for name in train_data.columns:

    if train_data[name].dtype == type(object):

        train_data[name] = lb.fit_transform(train_data[name])



train_data.head(10)
#Visualizing the continuous variables

train_data.hist(figsize=(50,50), layout=(20,4));
for name in train_data.columns:

    if len(np.unique(train_data[name].values)) == 1:

        print(name)

        

train_data['Street'].value_counts()
fig, ax = plt.subplots(2, 2, figsize = (10,10));

ax[0,0].boxplot(train_data['3SsnPorch'].values, flierprops=green_diamond);

ax[0,0].set_title('3SnPorch');

ax[1,0].boxplot(train_data['MiscVal'].values, flierprops=green_diamond);

ax[1,0].set_title('MiscVal');

ax[0,1].boxplot(train_data['PoolArea'].values, flierprops=green_diamond);

ax[0,1].set_title('PoolArea');

ax[1,1].boxplot(train_data['ScreenPorch'].values, flierprops=green_diamond);

ax[1,1].set_title('ScreenPorch');
corr = train_data.corr()

fig = plt.figure(figsize=(20,20))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(train_data.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(train_data.columns)

ax.set_yticklabels(train_data.columns)

plt.show()
for i in range(0, len(corr.columns)-1):

    for j in range(0, i):

        if corr.iloc[i][j] < (-0.8) or corr.iloc[i][j] > 0.8:

            print(corr.columns[i])
train_data = train_data.drop(columns = ['Exterior2nd', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'])
test_data.head(10)
#First we are going to do a MinMaxScaler to all the feature values except for the salePrice and

#then standardize them.

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

X_train = train_data.drop(columns = ['SalePrice'])

y_train = train_data['SalePrice']

train_data.drop(columns = ['SalePrice'], inplace = True)

X_test = test_data.drop(columns = ['Id','LotFrontage', 'Alley', 'BsmtQual', 'BsmtCond',

                                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',

                                        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',

                                       'PoolQC', 'Fence', 'MiscFeature', 'Exterior2nd', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'])



test_data.drop(columns = ['Id','LotFrontage', 'Alley', 'BsmtQual', 'BsmtCond',

                                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',

                                        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',

                                       'PoolQC', 'Fence', 'MiscFeature', 'Exterior2nd', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea'], inplace=True)

print(X_test.isna().sum())



#Fill the nans of the test with the median or the mode

for name in X_test.columns:

    if X_test[name].dtype == type(object):

        X_test[name].fillna((X_test[name].mode()[0]), inplace=True)

    else:

        X_test[name].fillna((X_test[name].median()), inplace=True)

#Or better fill them with 0, as we do not know their value

#Encode the categorical data with the label encoder as in the training set

lb = LabelEncoder()

for name in X_test.columns:

    if X_test[name].dtype == type(object):

        X_test[name] = lb.fit_transform(X_test[name])



#Standardize the values of the variables for making them fit to a normal distribution

standard = StandardScaler()

X_train = standard.fit_transform(X_train)

X_train = pd.DataFrame(X_train)

X_train.columns = train_data.columns

X_test = standard.fit_transform(X_test)

X_test = pd.DataFrame(X_test)

X_test.columns = test_data.columns
paramsRF = {

    'n_estimators' : [100, 500, 1000],

    'criterion' : ('mse', 'mae')

}

paramsLasso = {

    'alpha' : [0, 0.1, 0.5, 1],

    'max_iter' : [100, 500, 1000, 2000, 5000]

}
#Define the RMSE function

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer



def rmse(y_true, y_pred):

    "Returns the rmse of a prediction"

    return np.sqrt(abs(mean_squared_error(y_true, y_pred)))



score = make_scorer(rmse, greater_is_better=True)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



estimator = RandomForestRegressor()

#We will make a 5 k fold partition to test our models in the training set

clf = GridSearchCV(estimator, paramsRF, cv = 5, scoring = score);

clf.fit(X_train.values, y_train.values);



print("Best score and params for Random Forest")

print(clf.best_score_)

print(clf.best_params_)
randomForestclf = RandomForestRegressor(n_estimators = clf.best_params_['n_estimators'])



randomForestclf.fit(X_train.values, y_train.values)

#Plot the importance of the variables

feature_imp = pd.Series(randomForestclf.feature_importances_,index=train_data.columns.values).sort_values(ascending=False);

fig = plt.figure(figsize=(10,10));

ax = fig.add_subplot(111);

sns.barplot(x=feature_imp, y=feature_imp.index);

# Añadimos nombres al gráfico

plt.xlabel('Feature importance');

plt.ylabel('Features');

plt.title("Feature importance");

plt.figure(figsize = (10,10));

plt.show();





predictionsRandomForest = randomForestclf.predict(X_test)

f = open("randomForestPredictions.csv", "w")

f.write("Id,SalePrice\n")

for i in range(0, predictionsRandomForest.shape[0]):

    f.write(str(i+1461) + "," + str(predictionsRandomForest[i]) + "\n")

f.close()
X_selected = X_train[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '2ndFlrSF',

                    'BsmtFinSF1', 'GarageCars', 'LotArea', 'YearBuilt', 'YearRemodAdd',

                    'Neighborhood', 'FullBath']]

X_test_selected = X_test[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '2ndFlrSF',

                    'BsmtFinSF1', 'GarageCars', 'LotArea', 'YearBuilt', 'YearRemodAdd',

                    'Neighborhood', 'FullBath']]

estimator = RandomForestRegressor()

#We will make a 5 k fold partition to test our models in the training set

clf = GridSearchCV(estimator, paramsRF, cv = 5, scoring = score);

clf.fit(X_selected.values, y_train.values);



print("Best score and params for Random Forest")

print(clf.best_score_)

print(clf.best_params_)



randomForestclf.fit(X_selected.values, y_train.values)

predictionsRandomForest = randomForestclf.predict(X_test_selected.values)

print(predictionsRandomForest)

f = open("randomForestPredictionsFS.csv", "w")

f.write("Id,SalePrice\n")

for i in range(0, predictionsRandomForest.shape[0]):

    f.write(str(i+1461) + "," + str(predictionsRandomForest[i]) + "\n")

f.close()
from sklearn.linear_model import Lasso



estimator = Lasso();

clf = GridSearchCV(estimator, paramsLasso, cv = 5, scoring = score);

clf.fit(X_train.values, y_train.values);



print("Best score and params for Lasso Regressor")

print(clf.best_score_)

print(clf.best_params_)
lassoclf = Lasso(alpha = clf.best_params_['alpha'])



lassoclf.fit(X_train, y_train.values)



predictionsLasso = lassoclf.predict(X_test)



f = open("lassoPredictions.csv", "w")

f.write("Id,SalePrice\n")

for i in range(0, predictionsLasso.shape[0]):

    f.write(str(i+1461) + "," + str(predictionsLasso[i]) + "\n")

f.close()


estimator = Lasso();

clf = GridSearchCV(estimator, paramsLasso, cv = 5, scoring = score);

clf.fit(X_selected.values, y_train.values);



print("Best score and params for Lasso Regressor")

print(clf.best_score_)

print(clf.best_params_)



lassoclf = Lasso(alpha = clf.best_params_['alpha'])



lassoclf.fit(X_selected.values, y_train.values)



predictionsLasso = lassoclf.predict(X_test_selected.values)



f = open("lassoPredictionsFS.csv", "w")

f.write("Id,SalePrice\n")

for i in range(0, predictionsLasso.shape[0]):

    f.write(str(i+1461) + "," + str(predictionsLasso[i]) + "\n")

f.close()