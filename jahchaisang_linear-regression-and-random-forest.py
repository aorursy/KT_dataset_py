import pandas as pd

import numpy as np



data = pd.read_csv("../input/kc_house_data.csv")

data.head()
cats = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']



def category_counts(df, cat_columns):

    for col in cat_columns:

        print("+++ " + col + " +++")

        print(df[col].value_counts())

        

        

category_counts(data, cats)
numes = ['price','sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']



def nume_ranges(df, nume_columns):

    for col in nume_columns:

        print("+++ " + col + " +++")

        print("max: " + str(df[col].max()))

        print("min: " + str(df[col].min()))

        

nume_ranges(data, numes)



# sqft_living15 seems more reasonable?
# FEATURE CONSTRUCTION

# drop variables not used

# also drop price

to_drop = ['id','date','sqft_living','sqft_lot', 'zipcode', 'price']

train = data.drop(to_drop,axis=1)



# convert yr_built and yr_renovated to the age of the house as of 2015

train['yr_built'] = 2015 - train['yr_built']

train['yr_renovated'] = 2015 - train['yr_renovated']



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale

scaler = StandardScaler()

X = scaler.fit_transform(train)

Y = scale(data['price'])
# reconstruct dataframe for plotting



plot_data_dict = {}

i = 0

for col in train.columns[0:3]:

    plot_data_dict[col] = X[:,i]

    i += 1

plot_data_dict['price'] = Y

plot_data = pd.DataFrame(plot_data_dict)



plot_data.head()
import seaborn as sns

sns.set(color_codes=True)

sns.pairplot(data=plot_data)
from sklearn.cross_validation import KFold



def run_cv(X,y,clf_class,**kwargs):

    # Construct a kfolds object

    kf = KFold(len(y),n_folds=5,shuffle=True)

    y_pred = y.copy()

    

    # Iterate through folds

    for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]

        y_train = y[train_index]

        # Initialize a classifier with key word arguments

        clf = clf_class(**kwargs)

        clf.fit(X_train,y_train)

        y_pred[test_index] = clf.predict(X_test)

    return y_pred, clf



from sklearn.linear_model import LinearRegression as LR

from sklearn.ensemble import RandomForestRegressor as RF

from sklearn.metrics import mean_squared_error

from math import sqrt



y_pred, clf = run_cv(X,Y,LR)

result_LR = pd.DataFrame({'Y': Y, 'YPred': y_pred})

rmse_LR = sqrt(mean_squared_error(result_LR['YPred'],result_LR['Y']))

print(train.columns)

print(clf.coef_)



y_pred, clf = run_cv(X,Y,RF)

result_RF = pd.DataFrame({'Y': Y, 'YPred': y_pred})

rmse_RF = sqrt(mean_squared_error(result_RF['YPred'],result_RF['Y']))



print("Linear Regression:")

print("%.3f" % rmse_LR)

print("Random Forest Regression:")

print("%.3f" % rmse_RF)