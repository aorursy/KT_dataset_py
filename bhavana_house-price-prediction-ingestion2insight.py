import pandas as pd 

%matplotlib inline
# Data Ingestion using pandas

contents = pd.read_csv('../input/kc_house_data.csv')
# Data Exploration 

contents.head()
# Features 

len(contents.columns)
contents.info()
contents.get_dtype_counts()
# data for mean house price 

contents.describe()
# get the mean price for the house 

target = contents['price'].tolist()

mean_price = sum(target)/len(target)

print(mean_price)
# Data for the mean,high and low sales price 

meanrange = contents[(contents.price > 540000) & (contents.price <= 550000) ]

lowrange = contents[(contents.price > 70000) & (contents.price <= 75000) ]

highrange = contents[(contents.price > 7000000) & (contents.price <= 7700000 ) ]
low_price = min(target)

print(low_price)

high_price = max(target)

print(high_price)
print("Out of 21613 records")

print("The records in mean range", len(meanrange))

print("The records in high range", len(highrange))

print("The records in low range", len(lowrange))
len(contents)
low_price = min(target)

print(low_price)
#Bar Plots for 'Bedroom' feature in the given dataset

contents.bedrooms.value_counts().plot(kind = 'bar')
contents.boxplot(['lat'])
contents.boxplot(['long'])
contents.boxplot([ 'sqft_lot', 'sqft_living'])
import seaborn as sns

sns.set(color_codes=True)
sns.violinplot(contents['yr_renovated'], color = 'cyan')
sns.violinplot(contents['yr_built'], color = 'cyan')
from scipy import stats

stats.skew(contents.sqft_living, bias=False)
stats.skew(contents.sqft_lot15, bias = False)
stats.kurtosis(contents.sqft_living15, bias=False)
stats.kurtosis(contents.sqft_lot15, bias=False)
lin_cor = contents.corr(method = 'pearson')['price']

lin_cor = lin_cor.sort_values(ascending=False)

print(lin_cor)
import matplotlib.pyplot as plt

plt.scatter(target,contents.sqft_living)
plt.scatter(target,contents.sqft_lot15)
plt.scatter(target,contents.yr_renovated)
plt.scatter(target,contents.grade)
plt.scatter(target, contents.long)
plt.scatter(target, contents.zipcode)
contents.isnull().values.any()
# Convert date to year 

date_posted = pd.DatetimeIndex(contents['date']).year
conv_dates = [1 if values == 2014 else 0 for values in date_posted ]

contents['date'] = conv_dates
contents.date.value_counts().plot(kind = 'bar')
contents = contents.drop('id', axis = 1)
contents.describe()
import numpy as np

from scipy import stats

contents= contents[(np.abs(stats.zscore(contents)) < 3).all(axis=1)]
contents.boxplot([ 'sqft_lot', 'sqft_living'])
contents.boxplot(['long'])
predictors = contents.drop('price', axis = 1)

price = contents['price'].tolist()
#Standardize the data to input to PCA

from sklearn.preprocessing import scale

std_inputs = scale(predictors)

res_inputs = std_inputs.reshape((-1,19))

std_df = pd.DataFrame(data=std_inputs,columns= predictors.columns)
# 1. Principal Component Analysis (PCA)

from sklearn.decomposition import PCA

pca = PCA()   

pca = PCA().fit_transform(std_inputs)
a = list(np.std((pca), axis=0))

summary = pd.DataFrame([a])

summary = summary.transpose()

summary.columns = ['sdev']

summary.index = predictors.columns

kaiser = summary.sdev ** 2

print(kaiser)
y = np.std(pca, axis=0)**2

x = np.arange(len(y)) + 1

plt.plot(x, y, "o-")

plt.show()
import time 

from sklearn.linear_model import RandomizedLasso

rlasso = RandomizedLasso(alpha=0.025)
%time rlasso.fit(predictors, price)
names = predictors.columns

print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 

                 names), reverse=True))
final_predictors = predictors.drop(['yr_renovated', 'waterfront'], axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_predictors, price, test_size=0.33, random_state=42)
#Linear regression 

from sklearn import linear_model

regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)
#r2 score 

regr.score(X_test,y_test)
#GBM model

from sklearn import ensemble

params = {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 2,

          'learning_rate': 0.1, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
#r^2 score

clf.score(X_test,y_test)
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, final_predictors.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()