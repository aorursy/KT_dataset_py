# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print ("Train data shape:", train.shape)

print ("Test data shape:", test.shape)
train.head()
test.head()
train.describe()
test.describe()
df = pd.concat((train, test))

temp_df = df

print("Shape of df: ", df.shape)
df.head(6)
df.tail(6)
df.SalePrice.describe()
plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)

print ("Skew is:", df.SalePrice.skew())

plt.hist(df.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
numeric_features = df.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])
df.OverallQual.unique()
quality_pivot = df.pivot_table(index='OverallQual',

                  values='SalePrice', aggfunc=np.median)

quality_pivot
quality_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls
null_percent = df.isnull().sum()/df.shape[0]*100

null_percent
for i in df.columns:

    print(i + "\t" + str(len(df[i].unique())))
categoricals = df.select_dtypes(exclude=[np.number])

categoricals.describe()
df = train.select_dtypes(include=[np.number]).interpolate().dropna()
one_hot_encoded_training_predictors = pd.get_dummies(train)

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor



def get_mae(X, y):

    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention

    return -1 * cross_val_score(RandomForestRegressor(50), 

                                X, y, 

                                scoring = 'neg_mean_absolute_error').mean()



predictors_without_categoricals = train.select_dtypes(exclude=['object'])



mae_without_categoricals = get_mae(predictors_without_categoricals, target)



mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)



print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))

print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
y = np.log(df.SalePrice)

X = df.drop(['SalePrice', 'Id'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=42, test_size=.33)
from sklearn import linear_model

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import make_scorer, r2_score



def test_model(model, X_train=X_train, y_train=y_train):

    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)

    r2 = make_scorer(r2_score)

    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)

    score = [r2_val_score.mean()]

    return score
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')

test_model(svr_reg)
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state=21)

test_model(dt_reg)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)

test_model(rf_reg)
import xgboost

xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)

test_model(xgb_reg)
xgb2_reg=xgboost.XGBRegressor(n_estimators= 899,

 mon_child_weight= 2,

 max_depth= 4,

 learning_rate= 0.05,

 booster= 'gbtree')



test_model(xgb2_reg)
xgb2_reg.fit(X_train,y_train)

y_pred = np.exp(xgb2_reg.predict(X_test)).round(2)

submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)

submit_test.columns=['Id', 'SalePrice']

submit_test.to_csv('sample_submission.csv', index=False)

submit_test
svr_reg.fit(X_train,y_train)

y_pred = np.exp(svr_reg.predict(X_test)).round(2)

submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)

submit_test.columns=['Id', 'SalePrice']

submit_test.to_csv('sample_submission.csv', index=False)

submit_test
import pickle



pickle.dump(svr_reg, open('model_house_price_prediction.csv', 'wb'))

model_house_price_prediction = pickle.load(open('model_house_price_prediction.csv', 'rb'))

model_house_price_prediction.predict(X_test)
test_model(model_house_price_prediction)