import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



all_data = pd.read_excel("../input/airbnb.xlsx")

all_data
all_data.info()
all_data.describe()
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending = False)

missing_data = pd.DataFrame(

                            {'Total' : total[total > 0],

                             'Missing Ratio': percent[percent > 0]

                            }

                            )

print("Missing values in the dataset")

print(missing_data)



total = all_data[all_data['Number Of Reviews'] > 0].isnull().sum()

percent = (total / all_data.isnull().count())

missing_data = pd.DataFrame(

                            {'Total' : total[total > 0],

                             'Missing Ratio': percent[percent > 0]

                            }

                            ).sort_values(by='Total', ascending=False)

print("\n Missing values for fields with Number of Reviews > 0")

print(missing_data)
# droping Review Scores Rating (bin) column which gives almost same information like its 'brother'

all_data = all_data.drop('Review Scores Rating (bin)', axis = 1)



# droping 509 observations which have at least 1 review but missing score rating

index_to_drop = all_data[(all_data['Number Of Reviews'] > 0) & (all_data['Review Scores Rating'].isnull())].index

all_data = all_data.drop(index_to_drop)



# droping observations for which Beds, Zipcode, Host Since and Property Types are NA

index_to_drop = all_data[all_data['Beds'].isnull() | all_data['Zipcode'].isnull() | all_data['Host Since'].isnull() | all_data['Property Type'].isnull() ].index

all_data = all_data.drop(index_to_drop)



# fill the rest of missing Review Scores Rating observations with None value

all_data['Review Scores Rating'] = all_data['Review Scores Rating'].fillna("None")
#Check remaining missing values if any 

total = all_data.isnull().sum()

total[total > 0]
from sklearn.model_selection import train_test_split



y = all_data.Price

X_train, X_test, y_train, y_test = train_test_split(all_data,y,test_size=0.2)



X_test.drop(['Price'], axis = 1, inplace = True)



# temporary joining together X_train and X_test for EDA analysis

print(X_train.shape)

print(X_test.shape)



#ntrain = X_train.shape[0]

#ntest = X_test.shape[0]



#all_data = pd.concat((X_train, X_test), sort = False)
import matplotlib.pyplot as plt

import seaborn as sns



# correlation matrix is run only on X_train part which has known Price field.

correlation_matrix = X_train.corr()

plt.subplots(figsize = (8,8))

sns.heatmap(correlation_matrix, annot = True)
# dropping Number of Records from X_train and X_test

X_train = X_train.drop(['Number of Records'], axis = 1)

X_test = X_test.drop(['Number of Records'], axis = 1)



# plotting useful graphs for analytics

sns.pairplot(X_train, height = 2.5)

plt.show();
fig, ax = plt.subplots()

ax.plot_date(x = X_train['Host Since'], y = X_train['Price'])

plt.ylabel('Price')

plt.xlabel('Host Since')

plt.show()
# dropping observation with the Price equal to 10000

X_train = X_train.drop(X_train[X_train['Price'] == 10000].index)

X_train = X_train.drop(X_train[X_train['Zipcode'] > 50000].index)

from scipy import stats

from scipy.stats import norm, skew #for some statistics



sns.distplot(X_train['Price'] , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(X_train['Price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(X_train['Price'], plot=plt)

plt.show()
# log transformation usually works well with skewness

X_train = X_train.drop(X_train[X_train['Price'] > 1000].index)

X_train['Price'] = np.log1p(X_train['Price'])
sns.distplot(X_train['Price'] , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(X_train['Price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(X_train['Price'], plot=plt)

plt.show()
# moving Price column (after previous transformations) to separate variable

y_train = X_train['Price']

X_train = X_train.drop(['Price'], axis = 1)



# transform datatime type of Host Since to integer 

X_train['date_delta'] = (X_train['Host Since'] - X_train['Host Since'].min())  / np.timedelta64(1,'D')

X_train = X_train.drop(["Host Since"], axis = 1)



X_test['date_delta'] = (X_test['Host Since'] - X_test['Host Since'].min())  / np.timedelta64(1,'D')

X_test = X_test.drop(["Host Since"], axis = 1)



# check which columns are categorical

X_train.columns[X_train.dtypes == "object"]



# identify and create distinct values in categorical columns

from sklearn.preprocessing import LabelEncoder

for col in X_train.columns[X_train.dtypes == "object"]:

    X_train[col] = X_train[col].factorize()[0]

    

# identify and create distinct values in categorical columns

from sklearn.preprocessing import LabelEncoder

for col in X_test.columns[X_test.dtypes == "object"]:

    X_test[col] = X_test[col].factorize()[0]
import xgboost as xgb

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_log_error



train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.3, random_state=42)



model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)

model.fit(X_train, y_train, early_stopping_rounds=5, 

             eval_set=[(val_x, val_y)], verbose=False)



# check the accuracy of the full train dataset

accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_log_error")

print("Accuracy of train data:", np.sqrt(-accuracy.mean()))
pd.DataFrame(

                            {'Prediction' : model.predict(train_x),

                             'True values': train_y

                            }

                            )