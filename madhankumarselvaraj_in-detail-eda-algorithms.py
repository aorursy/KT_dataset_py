import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.impute import KNNImputer
import statsmodels.api as sm

from scipy import stats

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,ElasticNetCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pd.set_option('display.max_columns', 30)

# pd.set_option("max_columns", 2) #Showing only two columns
life_df = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")
life_df.head(5)
life_df.columns
life_df.shape
life_df.info()
life_df.describe()
life_df.isnull().sum().plot(kind='bar')
# profile = ProfileReport(life_df, title = "Life expectancy report")

# profile.to_file("expectancy.html")
life_df = life_df.rename(columns= lambda x: x.strip())


twenty_percent = (life_df.shape[0]/100)*20

for col in life_df.columns:

    life_df[col].isnull().sum()

    if (life_df[col].isnull().sum()) >= twenty_percent:

        print("20%-'",col,"'")

    elif (life_df[col].isnull().sum()) > 0:

        print("> 0 %-'", col,"'")
plt.hist(life_df['Adult Mortality'])
sns.distplot(life_df['Adult Mortality'])
check_expectancy = life_df["Adult Mortality"][~ np.isnan(life_df["Adult Mortality"])]

nan_expectancy = life_df["Adult Mortality"].copy()
plt.boxplot(check_expectancy)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

dummy_simple = imputer.fit_transform(nan_expectancy.values.reshape(-1,1))
plt.boxplot(dummy_simple)
sns.distplot(dummy_simple)
knnimpute = KNNImputer(n_neighbors=4)

dummy_knn = knnimpute.fit_transform(nan_expectancy.values.reshape(-1,1))
sns.distplot(dummy_knn)
sm.qqplot(dummy_knn, fit=True, line="45")
plt.boxplot(dummy_knn)
# sm.qqplot(dummy_knn, fit=True, line="45")
scale_data = preprocessing.scale(dummy_knn)
# sm.qqplot(scale_data, fit=True, line="45")
standard_scale = preprocessing.StandardScaler()
standard_scale_data = standard_scale.fit_transform(dummy_knn)
sm.qqplot(standard_scale_data, fit=True, line="45")
standard_scale_data[:10]
plt.boxplot(standard_scale_data)
plt.hist(standard_scale_data)
sns.distplot(standard_scale_data)
transformer = preprocessing.PowerTransformer(method="box-cox", standardize= True)
dummy_power = transformer.fit_transform(dummy_knn)
sm.qqplot(dummy_power, fit=True, line="45")
sns.distplot(dummy_power)
plt.hist(dummy_power)
life_df.info()
life_df_numeric = life_df.select_dtypes(exclude="object")

life_df_object = life_df.select_dtypes(include="object")
life_df_numeric.columns
# life_df_numeric = life_df_numeric.loc[:,['Measles', 'under-five deaths']]
life_df_numeric.loc[:,'Measles'] = life_df_numeric.loc[:,'Measles'].astype(float)

life_df_numeric.loc[:,'under-five deaths'] = life_df_numeric.loc[:,'under-five deaths'].astype(float)

life_df_numeric.loc[:,'infant deaths'] = life_df_numeric.loc[:,'infant deaths'].astype(float)

life_df_numeric.info()
life_df_numeric.isna().sum()
before_outlier = life_df_numeric.copy()
life_df_numeric.describe()


for col in life_df_numeric.columns:

    q1 = life_df_numeric[col].quantile(0.25)

    q3 = life_df_numeric[col].quantile(0.75)

    iqr = q3-q1

    liqr = q1 - (1.5 * iqr)

    hiqr = q3 + (1.5 * iqr)

#     print(iqr, hiqr)



#     print('\n',col, '\niqr: ',iqr,'\nminimum: ',np.min(life_df_numeric[col]), 'liqr: ',liqr,'\nmaximum: ', np.max(life_df_numeric[col]),'hiqr: ', hiqr)

    life_df_numeric.loc[:][col] = np.where((life_df_numeric.loc[:][col] <= liqr) | (life_df_numeric.loc[:][col] >= hiqr), np.nan , life_df_numeric.loc[:][col])

#     life_df_numeric.loc[:][col] = np.where((life_df_numeric.loc[:][col] >= hiqr), None, life_df_numeric.loc[:][col])
life_df_numeric
after_outlier = life_df_numeric.copy()
life_df_numeric.isna().sum()
life_df_numeric.describe()
for col in life_df_numeric.columns:

#     print(col,"\n\nMinimum\nBefore-",np.min(before_outlier[col]), "After",np.min(after_outlier[col]))

#     print("Maximum\nBefore",np.max(before_outlier[col]), "After",np.max(after_outlier[col]))

    q1b = before_outlier[col].quantile(0.25)

    q3b = before_outlier[col].quantile(0.75)

    iqrb = q3b-q1b

    liqrb = q1b - (1.5 * iqrb)

    hiqrb = q3b + (1.5 * iqrb)

    

    if np.min(after_outlier[col]) > liqrb:

        print

    print(col,'\nminimum- ',np.min(after_outlier[col]), 'liqr- ',liqrb,'\nmaximum- ', np.max(after_outlier[col]),'hiqr- ', hiqrb)

    print('\n')

life_df_final = pd.concat([life_df_numeric, life_df_object], axis=1)
knnimpute = KNNImputer(n_neighbors=3)

for col in life_df_final.columns:

    if (life_df_final[col].isnull().sum()) > 0:

        life_df_final[col] = knnimpute.fit_transform(life_df_final[col].values.reshape(-1,1))
life_df_final.isna().sum()
life_df_final.head(20)
for col in life_df_final.columns:

    if life_df_final[col].dtypes != 'object':

        q1 = life_df_final[col].quantile(0.25)

        q3 = life_df_final[col].quantile(0.75)

        iqr = q3-q1

        liqr = q1 - (1.5 * iqr)

        hiqr = q3 + (1.5 * iqr)

        print('\n',col,'--low---', (life_df_final[col] <= liqr).any())

        print(col,'--high---', (life_df_final[col] >= hiqr).any())

        print('iqr- ',iqr,'\nminimum- ',np.min(life_df_final[col]), 'liqr- ',liqr,'\nmaximum- ', np.max(life_df_final[col]),'hiqr- ', hiqr)
life_df_final.isna().sum()
fig = plt.figure(figsize = (10,10))

ax = fig.gca()

life_df_final.hist(ax=ax)

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)

plt.show()
life_df_final.iloc[:,0:5].boxplot()
life_df_final.iloc[:,5:10].boxplot()
Y = life_df_final["Life expectancy"]
Y
X = life_df_final.drop("Life expectancy", axis=1)
X.head(5)
label = preprocessing.LabelEncoder()

X_numeric = X.apply(label.fit_transform)
X_numeric.head(5)
X_numeric[(X_numeric == 0).any(1)]
# transformer = preprocessing.PowerTransformer(method="box-cox", standardize= True)

transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)

X_transform = transformer.fit_transform(X_numeric)
X_transform = pd.DataFrame(X_transform)
X_transform.columns = X_numeric.columns
X_transform.head()
fig = plt.figure(figsize = (10,10))

ax = fig.gca()

X_transform.hist(ax=ax)

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=1.5)

plt.show()
Y_transform = transformer.fit_transform(Y.values.reshape(-1,1))
plt.hist(Y_transform)
xtrain, xtest, ytrain, ytest = train_test_split(X_transform, Y_transform, test_size = 0.3, random_state=101)
linear_model = LinearRegression()

linear_model.fit(xtrain, ytrain)
linear_model.coef_
ypredict = linear_model.predict(xtest)
linear_model.score(xtest, ytest)
print(mean_absolute_error(ytest, ypredict))
print(mean_squared_error(ytest, ypredict))
print(np.sqrt(mean_squared_error(ypredict, ytest)))
print(r2_score(ytest, ypredict))
ytest_actual = transformer.inverse_transform(ytest)
ypredict_actual = transformer.inverse_transform(ypredict)
print((np.in1d(ytest_actual, ypredict_actual)).shape)
print((np.intersect1d(ytest_actual, ypredict_actual)).shape)
ridge_model = Ridge(alpha=0.01)

ridge_model.fit(xtrain, ytrain)
ypredict_ridge = ridge_model.predict(xtest)



print(ridge_model.score(xtest, ytest))



print(mean_absolute_error(ytest, ypredict_ridge))



print(mean_squared_error(ytest, ypredict_ridge))
ridge_model = Ridge(alpha=50)

ridge_model.fit(xtrain, ytrain)
ypredict_ridge = ridge_model.predict(xtest)



print(ridge_model.score(xtest, ytest))



print(mean_absolute_error(ytest, ypredict_ridge))



print(mean_squared_error(ytest, ypredict_ridge))
lasso_model = Lasso()

lasso_model.fit(xtrain, ytrain)
ypredict_lasso = lasso_model.predict(xtest)



print(lasso_model.score(xtest, ytest))



print("Number of features used:",np.sum(lasso_model.coef_!=0))



print(mean_absolute_error(ytest, ypredict_lasso))



print(mean_squared_error(ytest, ypredict_lasso))
lasso_model = Lasso(alpha=0.01, max_iter=10e5)

lasso_model.fit(xtrain, ytrain)
ypredict_lasso = lasso_model.predict(xtest)



print(lasso_model.score(xtest, ytest))



print("Number of features used:",np.sum(lasso_model.coef_!=0))



print(mean_absolute_error(ytest, ypredict_lasso))



print(mean_squared_error(ytest, ypredict_lasso))
elatic_model = ElasticNet(alpha=0.01)

elatic_model.fit(xtrain, ytrain)
elatic_model.coef_
ypredict_elastic = elatic_model.predict(xtest)



print(elatic_model.score(xtest, ytest))



print("Number of features used:",np.sum(lasso_model.coef_!=0))



print(mean_absolute_error(ytest, ypredict_elastic))



print(mean_squared_error(ytest, ypredict_elastic))
alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

elasticnetcv_model = ElasticNetCV(alphas=alphas, cv=5)

elasticnetcv_model.fit(xtrain, ytrain)
ypredict_elastic = elasticnetcv_model.predict(xtest)



print(elasticnetcv_model.score(xtest, ytest))



print('Best alpha value: ',elasticnetcv_model.alpha_)



print('Intercept: ',elasticnetcv_model.intercept_)



print("Number of features used:",np.sum(lasso_model.coef_!=0))



print("Number of features not used:",np.sum(lasso_model.coef_==0))



print(mean_absolute_error(ytest, ypredict_elastic))



print(mean_squared_error(ytest, ypredict_elastic))
lasso_model.coef_