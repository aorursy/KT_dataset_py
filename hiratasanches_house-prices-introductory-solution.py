import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# load train and test data sets

dfTrain = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

dfTest = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
# print dfTrain and dfTest shape and info about dfTrain columns

print("Shape of train and test DataFrames:", dfTrain.shape, dfTest.shape)

dfTrain.info()
# print description of train data set numeric features

dfTrain.describe()
# lists with kind of the column data

numeric = list(dfTrain.select_dtypes(include=['int64', 'float64']).columns.values)

integer = list(dfTrain.select_dtypes(include=['int64']).columns.values)

real = list(dfTrain.select_dtypes(include=['float64']).columns.values)

string = list(dfTrain.select_dtypes(include=['object']).columns.values)



print("Numeric features:", len(numeric))

print("Numeric integer features:", len(integer))

print("Numeric real features:", len(real))

print("String features:", len(string))
# plot number of categories per string feature

nu = dfTrain[string].nunique().reset_index()

nu.columns = ['feature','nunique']

plt.figure(figsize=(20,5))

sns.barplot(x='feature', y='nunique', data=nu)

size = len(string)

plt.xticks(np.linspace(0,size+1,size+2), dfTrain[string].columns.values, rotation=45, ha="right")

plt.xlim(-0.7,size-0.3)

plt.title("Number of categories per feature", fontsize=18)

plt.ylabel("Number of categories")

plt.show()
# plot percentage of missing data in train data set

total = dfTrain.isnull().sum().sort_values(ascending=False)

percent = (dfTrain.isnull().sum()/dfTrain.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total.head(20), percent.head(20)], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='45', ha="right")

sns.barplot(x=missing_data.index, y=missing_data.Percent)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

plt.show()

missing_data.head(10)
# plot heatmap with correlation between numeric features

dfNum = dfTrain.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(20,20))

sns.heatmap(data=dfNum.corr(), vmin=-1, vmax=1, cmap='bwr', square=True)

plt.xticks(rotation='45', ha="right")

plt.show()
# print correlation with SalePrice

dfNum.corr()['SalePrice'].sort_values()
# correlation matrix containing only numeric features with higher correlation with SalePrice

main_features = ['SalePrice', 'Fireplaces', 'MasVnrArea', 'YearRemodAdd', 'YearBuilt', 'TotRmsAbvGrd', 

              'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual', 

              'OpenPorchSF', '2ndFlrSF', 'WoodDeckSF', 'BsmtFinSF1']

target = ['SalePrice']

size = dfNum[main_features].shape[1]

plt.figure(figsize=(9,9))

sns.heatmap(data=dfNum[main_features].corr(), vmin=-1, vmax=1, cmap='bwr', annot=True, fmt = ".1f")

plt.xticks(rotation='45', ha="right")

plt.title("Correlation Matrix with the most correlated with SalePrice features", fontsize=18)

plt.show()
# plot of OverallQual x SalePrice

sns.scatterplot(x='OverallQual', y='SalePrice', data=dfTrain)

plt.title("Relation between OverallQual and SalePrice", fontsize=15)

plt.show()
# plot of GrLivArea x SalePrice

sns.scatterplot(x='GrLivArea', y='SalePrice', data=dfTrain)

plt.title("Relation between GrLivArea and SalePrice", fontsize=15)

plt.show()
# plot of GarageCars x SalePrice

sns.scatterplot(x='GarageCars', y='SalePrice', data=dfTrain)

plt.title("Relation between GarageCars and SalePrice", fontsize=15)

plt.show()
# plot of YearBuilt x SalePrice

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=dfTrain)

plt.title("Relation between TotalBsmtSF and SalePrice", fontsize=15)

plt.show()
# plot of YearBuilt x SalePrice

sns.scatterplot(x='YearBuilt', y='SalePrice', data=dfTrain)

plt.title("Relation between YearBuilt and SalePrice", fontsize=15)

plt.show()
# plot SalePrice distribution

from scipy import stats



sns.distplot(dfTrain['SalePrice'] , fit=stats.norm);





# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(dfTrain['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(dfTrain['SalePrice'], plot=plt)

plt.show()

# explore presence of outliers

plt.figure(figsize=(25,5))

dfNorm = (dfNum - dfNum.mean()) / (dfNum.max() - dfNum.min())



sns.boxplot(data = dfNorm)

size = dfNorm.shape[1]

plt.xticks(np.linspace(0,size+1,size+2), dfNorm.columns.values, rotation=45, ha="right")

plt.xlim(-0.7,size-0.3)

plt.show()
# fill the five columns with most missing data with 'None'

for df in [dfTrain, dfTest]:

    df["PoolQC"] = df["PoolQC"].fillna("None")

    df["MiscFeature"] = df["MiscFeature"].fillna("MiscFeature")

    df["Alley"] = df["Alley"].fillna("None")

    df["Fence"] = df["Fence"].fillna("None")

    df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
# fill numeric missing values by median and string missing values with mode

numeric_new = numeric.copy()

numeric_new.remove('SalePrice')

for df in [dfTrain, dfTest]:

    df.loc[:,numeric_new] = df.loc[:,numeric_new].fillna(df.loc[:,numeric_new].median())

    df.loc[:,string] = df.loc[:,string].fillna(df.loc[:,string].mode().to_dict('records')[0])
# create total area joining data from basement 1st floor and 2nd floor

for df in [dfTrain, dfTest]:

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# apply boxcox1p in the most skewed features

from scipy.stats import skew

from scipy.special import boxcox1p



for df in [dfTrain, dfTest]:

    skewed_feats = df.loc[:,numeric_new].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew' :skewed_feats})

    skewness = skewness[abs(skewness) > 0.75]

    skewed_features = skewness.index

    lam = 0.15

    for feat in skewed_features:

        df.loc[:,feat] = boxcox1p(df.loc[:,feat], lam)
dfTrain.describe()
print(dfTrain.info())

print(dfTest.info())
# prepare data

from sklearn.model_selection import cross_validate

from sklearn import metrics

from sklearn.compose import TransformedTargetRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



target = ['SalePrice']

X = pd.get_dummies(dfTrain, columns=string, drop_first=True).drop(columns=target)

y = dfTrain[target]
# Linear Regression

from sklearn.linear_model import LinearRegression



lr = TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1)

scores = cross_validate(lr, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_lr = round(scores['test_r2'].mean(), 3)

rmse_lr = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 2)

rmsle_lr = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)

print('R2:', r2_lr)

print('Root Mean Squared Error:', rmse_lr)

print('Root Mean Squared Log Error:', rmsle_lr)
# Lasso Regression

from sklearn.linear_model import Lasso



lasso = make_pipeline(RobustScaler(), TransformedTargetRegressor(Lasso(alpha =0.0005),

                                                 func=np.log1p, inverse_func=np.expm1))

scores = cross_validate(lasso, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_lasso = round(scores['test_r2'].mean() , 3)

rmse_lasso = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 2)

rmsle_lasso = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)

print('R2:', r2_lasso)

print('Root Mean Squared Error:', rmse_lasso)

print('Root Mean Squared Log Error:', rmsle_lasso)
# ElasticNet Regression

from sklearn.linear_model import ElasticNet



elNet = make_pipeline(RobustScaler(), TransformedTargetRegressor(ElasticNet(alpha=0.0005, l1_ratio=.9),

                                                 func=np.log1p, inverse_func=np.expm1))

scores = cross_validate(elNet, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_elNet = round(scores['test_r2'].mean() , 3)

rmse_elNet = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 2)

rmsle_elNet = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)

print('R2:', r2_elNet)

print('Root Mean Squared Error:', rmse_elNet)

print('Root Mean Squared Log Error:', rmsle_elNet)
# KNN Regression

from sklearn.neighbors import KNeighborsRegressor



knn = TransformedTargetRegressor(KNeighborsRegressor(n_neighbors=15), func=np.log1p, inverse_func=np.expm1)

scores = cross_validate(knn, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_knn = round(scores['test_r2'].mean() , 3)

rmse_knn = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 2)

rmsle_knn = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)

print('R2:', r2_knn)

print('Root Mean Squared Error:', rmse_knn)

print('Root Mean Squared Log Error:', rmsle_knn)
# KernelRidge Regressor

from sklearn.kernel_ridge import KernelRidge



krr = make_pipeline(RobustScaler(), 

                    TransformedTargetRegressor(KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),

                                  func=np.log1p, inverse_func=np.expm1))

scores = cross_validate(krr, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_krr = round(scores['test_r2'].mean() , 3)

rmse_krr = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 2)

rmsle_krr = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)

print('R2:', r2_krr)

print('Root Mean Squared Error:', rmse_krr)

print('Root Mean Squared Log Error:', rmsle_krr)
# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



rf = TransformedTargetRegressor(RandomForestRegressor(n_estimators=400, max_depth=8),

                                func=np.log1p, inverse_func=np.expm1)

scores = cross_validate(rf, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_rf = round(scores['test_r2'].mean() , 3)

rmse_rf = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 4)

rmsle_rf = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)



print('R2:', r2_rf)

print('Root Mean Squared Error:', rmse_rf)

print('Root Mean Squared Log Error:', rmsle_rf)
# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor



gBoost = TransformedTargetRegressor(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                                              max_depth=4, max_features='sqrt', min_samples_leaf=15, 

                                                              min_samples_split=10, loss='huber'),

                                    func=np.log1p, inverse_func=np.expm1)



scores = cross_validate(gBoost, X, y, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error'])

r2_gBoost = round(scores['test_r2'].mean() , 3)

rmse_gBoost = round(np.sqrt(-scores['test_neg_mean_squared_error'].mean()), 2)

rmsle_gBoost = round(np.sqrt(-scores['test_neg_mean_squared_log_error'].mean()), 4)

print('R2:', r2_gBoost)

print('Root Mean Squared Error:', rmse_gBoost)

print('Root Mean Squared Log Error:', rmsle_gBoost)
# Comparison of Models

models = pd.DataFrame({

    'Root Mean Squared Error': [rmse_lr, rmse_lasso, rmse_elNet, rmse_krr, rmse_gBoost, rmse_rf, rmse_knn],

    'R-squared': [r2_lr, r2_lasso, r2_elNet, r2_krr, r2_gBoost, r2_rf, r2_knn],

    'Root Mean Squared Log Error':

        [rmsle_lr, rmsle_lasso, rmsle_elNet, rmsle_krr, rmsle_gBoost, rmsle_rf, rmsle_knn]},

    index=['Linear Regression', 'Lasso', 'Elastic Net', 'Kernel Ridge', 'Gradient Boosting Regressor', 

              'Random Forest', 'KNN'])

models = models.sort_values(by='Root Mean Squared Log Error')



models
gBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                                              max_depth=4, max_features='sqrt', min_samples_leaf=15, 

                                                              min_samples_split=10, loss='huber')



gBoost.fit(X, y)

dfFit = pd.DataFrame(gBoost.feature_importances_, X.columns, 

                     columns=['Coefficient'])

to_keep = dfFit[dfFit.Coefficient > 0.00005].index.values



print(dfFit.sort_values(by='Coefficient', ascending=False).head(15))

print("\n\nWill be maintained:\n", to_keep)
# fit model with train_test_split and verify model in dev data set 

from sklearn.model_selection import train_test_split



X_train, X_dev, y_train, y_dev = train_test_split(X[to_keep], y, random_state=123, test_size=0.2)

gBoost = TransformedTargetRegressor(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                                              max_depth=4, max_features='sqrt', min_samples_leaf=15, 

                                                              min_samples_split=10, loss='huber'),

                                    func=np.log1p, inverse_func=np.expm1)



gBoost.fit(X_train, y_train)

y_pred = gBoost.predict(X_dev)
# scatter plot of real and prediction values

sns.scatterplot(x=y_dev.values.reshape(1,-1)[0], y=y_pred.reshape(1,-1)[0], sizes=4, alpha=0.6)

plt.plot(np.linspace(0,6.5e5), np.linspace(0,6.5e5), linewidth=1.5, c='black')

plt.ylabel('Prediction')

plt.xlabel('Real Value')

plt.title('Comparison of Real Value and Prediction', fontsize=16)

plt.show()
# quantile-quantile with y_dev and y_pred

percs = np.linspace(0,100,21)

qn_dev = np.percentile(y_dev, percs)

qn_pred = np.percentile(y_pred, percs)



plt.plot(qn_dev, qn_pred, ls="", marker="o", alpha=0.7)



x = np.linspace(np.min((qn_dev.min(),qn_pred.min())), np.max((qn_dev.max(),qn_pred.max())))

plt.plot(x,x, color="k", ls="--")

plt.title("QQ of Real Value and Prediction", fontsize=16)



plt.show()
# plot residual

fig, axes= plt.subplots(nrows=1, ncols=2,figsize=(15,4))



plt.subplot(1,2,1)

plt.scatter(y_dev, (y_dev-y_pred)/y_dev, s=2)

plt.axhline(0, c='black')

plt.ylim([-0.6, 0.6])

plt.xlabel('Real Value')

plt.ylabel('Normalized Residuals')

plt.title("Normalized Residuals per Real Value", fontsize=16)



plt.subplot(122)

plt.scatter(y_pred, (y_dev-y_pred)/y_pred, s=3, alpha=0.6)

plt.axhline(0, c='black')

plt.ylim([-0.4, 0.4])

plt.xlabel('Predicted Value')

plt.ylabel('Normalized Residuals')

plt.title("Normalized Residuals per Predicted Value", fontsize=16)



plt.subplots_adjust(wspace=0.3)

plt.show()
sns.distplot(((y_dev-y_pred)/y_dev).values.reshape(1,-1)[0], bins=20, kde = False)

plt.xlim([-1,1])

plt.xlabel('Normalized Residuals')

plt.ylabel('Frequency')

plt.title('Histogram of Normalized Residuals', fontsize=16)

plt.show()
# plot mean squared error per price range (excluding houses with high SalePrice, which error is much bigger)

dfValidation = pd.DataFrame((y_dev-y_pred)**2)

dfValidation = dfValidation.rename(columns={'SalePrice': 'Normalized_Error'})

dfValidation = pd.merge(dfValidation, dfTrain, right_index=True, 

                        left_index=True, how='left')

dfValidation['SalePrice_category'] = dfValidation['SalePrice'] // 1e5

dfValidation['SalePrice_category'] = dfValidation['SalePrice_category'].astype('category')

dfGroupPrice = dfValidation.groupby('SalePrice_category').mean()



plt.figure()

sns.barplot(dfGroupPrice.index.values, dfGroupPrice.Normalized_Error)

plt.xlabel('Price Range (1e5)')

plt.ylabel('Mean Squared Error')

plt.xlim([-0.6,3.6])

plt.ylim([0, 1.5e9])

plt.xticks([0,1,2,3])

plt.title('Mean Squared Error per Price Range', fontsize=15)

plt.show()
gBoost = TransformedTargetRegressor(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                                              max_depth=4, max_features='sqrt', min_samples_leaf=15, 

                                                              min_samples_split=10, loss='huber'),

                                    func=np.log1p, inverse_func=np.expm1)



X_test = pd.get_dummies(dfTest, columns=string, drop_first=True)

to_keep_new = [el for el in to_keep if el in X_test.columns.values]



gBoost.fit(X[to_keep_new], y)

predictions = gBoost.predict(X_test[to_keep_new])



output = pd.DataFrame({'Id': dfTest.index.values, 'SalePrice': predictions.reshape(1,-1)[0]})

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")