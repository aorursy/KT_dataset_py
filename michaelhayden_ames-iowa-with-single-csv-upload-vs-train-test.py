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
#Import Statements

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

pd.set_option('display.max_columns', None)

import scipy.stats as st

import math

#SciKit-Learn

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

#test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

#train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#train_2 = train.iloc[:, :-1] may need to trash this line soon

#we pulled the sale price from the train data so we can do wrangling then add it back later before building the model

#train.drop ('SalePrice', axis=1, inplace=True)

#ames = pd.concat([test,train], axis =0)
import pandas as pd

ames = pd.read_csv("../input/amescomplete/ames.csv")

ames.head()
# check the data in the numeric features

numeric_percent_null = ames.isnull().sum()/ames.isnull().count()

numeric_percent_null.sort_values(ascending=False)
#These were removed because of how many values were missing

drop_columns = ['Pool.QC','Fireplace.Qu']

for column in drop_columns:

    ames = ames.drop(column, axis=1)
#fill in the missing garage value since there was no garage

ames['Garage.Yr.Blt'] = ames['Garage.Yr.Blt'].fillna(0)
# re check percentage of missing data

pd.DataFrame(

    ames.isnull().sum()/len(ames),

    columns=['% Missing Values']).transpose()
#fill in the missing garage value since there was no garage

ames['Garage.Yr.Blt'] = ames['Garage.Yr.Blt'].fillna(0)
ames.fillna(ames.mean())
# re check percentage of missing data

pd.DataFrame(

    ames.isnull().sum()/len(ames),

    columns=['% Missing Values']).transpose()
#fill in the na spaces with 0

ames= ames.fillna(0)
# returns percentage of missing values

pd.DataFrame(

    ames.isnull().sum()/len(ames),

    columns=['% Missing Values']).transpose()
#data cleaning

#view what is being worked with

ames.head()
# added columns to tell age of home as well as how long since last remodel. Since we combinded these columns we removed the

#originals to prevent duplicate data

ames['Age.When.Sold'] = ames ['Yr.Sold'] - ames ['Year.Built']

ames['Years.Since.Remodel'] = ames ['Yr.Sold'] - ames ['Year.Remod.Add']

# adding column to tell total living area with each home

ames['Total.Living.Area'] = ames['Total.Bsmt.SF'] + ames['Gr.Liv.Area']
#Lets see how close the features relate to the sales price of the home



ames.corr()["SalePrice"].sort_values(ascending=False)
#dropping columns with a correlation of  <+-.1

ames.drop ('Pool.Area', axis=1, inplace=True)

ames.drop ('Mo.Sold', axis=1, inplace=True)

ames.drop ('X3Ssn.Porch', axis=1, inplace = True)

ames.drop ('BsmtFin.SF.2', axis=1, inplace = True)

ames.drop ('Misc.Val', axis=1, inplace = True)

ames.drop ('Yr.Sold', axis=1, inplace= True)

#ames.drop ('Order', axis=1, inplace = True)

ames.drop ('Bsmt.Half.Bath', axis=1, inplace= True)

ames.drop ('Low.Qual.Fin.SF', axis=1, inplace= True)

ames.drop ('MS.SubClass', axis=1, inplace= True)

ames.drop ('Year.Remod.Add', axis=1, inplace= True)

ames.drop ('Year.Built', axis=1, inplace= True)

ames.drop ('Total.Bsmt.SF', axis=1, inplace= True)

ames.drop ('Gr.Liv.Area', axis=1, inplace= True)

ames.drop ('Lot.Frontage', axis=1, inplace= True)



#ames.drop ('Yr.Sold', axis=1, inplace= True)

#we dropped Total.Bsmt.SF and Gr.Liv.Area because we created the Total.Living.Area

# we removed Year.Remod.Add, Yr.Sold, and Year.Remod.Add because we used them to create 2 other variables.
#percentage of missing values in each column checking again after dropping

pd.DataFrame(

    ames.isnull().sum()/len(ames),

    columns=['% Missing Values']).transpose()

#we have all zeros which means we have no missing values
# Let's remove the outliers based upon square footage

ames = ames.loc[ames['Total.Living.Area']<=4500,:]

ames.shape
# look at the max values

ames.max()
#numerical from categorical info

ames_numerical = ames.select_dtypes(include=['int64','float64'])

ames_categorical = ames.select_dtypes(exclude=['int64','float64'])
corr_2 = ames_numerical.corr()

plt.figure(figsize = (16,12))

sns.heatmap(corr_2,

           xticklabels=corr_2.columns.values,

           yticklabels = corr_2.columns.values)
#checking for skew in the data



#lets visualize the data

y = ames['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)



plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)



plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)

#skew before normalization no good since doesn't fall within +-.5

print("Skew: %f" % ames['SalePrice'].skew())
ames_categorical.head()
ames_numerical += .00001
# the skews are not normailzed so we need to fix this

all_skews = ames_numerical.apply(stats.skew, axis=0)

all_skews
# transform skew to fall within .5 of 0

skews_to_transform = all_skews[all_skews.abs()>.5]

skew_columns = skews_to_transform.index.tolist()
# skewing the columns to fall within the accepted range

skew_transformed_columns = ames_numerical[skew_columns].apply(np.log)

transformed_numeric = pd.concat([ames_numerical[ames_numerical.columns[~ames_numerical.columns.isin(skew_columns)]],

                                skew_transformed_columns],axis=1)



sns.distplot(transformed_numeric.SalePrice)
#the skew falls within the accepted tolerances

print("Skew: %f" % transformed_numeric['SalePrice'].skew())
#converting categorical info to binary info
#creating dummy variables for the categorical columns

categorical_list = ames_categorical.columns.tolist()

housing_dummies=pd.get_dummies(ames_categorical,columns=categorical_list)

housing_dummies.head()
#combines the 2 data frames back together with the dummie variables included

result= pd.concat([transformed_numeric.merge(housing_dummies, left_index=True, right_index=True)])
result.head()
#looking here you can see the 2 data frames are put back together correctly

result.head()

#Check skew again for final results

y = transformed_numeric['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)



plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)



plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
#root mean squared error
housing_X_transformed = result.copy()

housing_y = ames.SalePrice

#housing_X_transformed = housing_X_transformed.drop("SalePrice",axis=1)
# split between train and test set

X_train, X_test, y_train, y_test = train_test_split(

     housing_X_transformed, housing_y, test_size=0.3, random_state=42)
# lets fit the model

linereg = LinearRegression()

linereg.fit(X_train,y_train)
housing_X_transformed = housing_X_transformed.drop("SalePrice",axis=1)

# if fails put this back on line of code above it
# split between train and test set

X_train, X_test, y_train, y_test = train_test_split(

     housing_X_transformed, housing_y, test_size=0.3, random_state=42)
# lets fit the model

linereg = LinearRegression()

linereg.fit(X_train,y_train)

#building the models
# checking the split of the data

X_train.shape, X_test.shape
#Mapping the estimated prices to the predicted price of the project with a 70/30
#build and train regressor

rf_model = RandomForestRegressor(n_estimators=1000, n_jobs=-1,

        random_state=42)

rf_model.fit(X_train, y_train)
rf_test_pred = rf_model.predict(X_test)
# lets score the model .902 is a very good score, perfect is 1.0

rf_model.score(X_test, y_test)
plt.figure(figsize=(10,10))

plt.scatter(rf_test_pred, y_test, alpha=.1, c='blue')

plt.plot(np.linspace(0,600000, 1000), np.linspace(0,600000, 1000), 'r-');

plt.xlabel("Predicted")

plt.ylabel("Actual")
rf_test_pred
#submission = pd.DataFrame(result, columns=['SalePrice'])

#result = pd.concat([result['PID'], submission], axis=1)



#result.columns



sub = pd.DataFrame()

result.to_csv('Submission.csv', index=False)
submission = pd.DataFrame({'Id':test['Id'],'SalePrice':test})
filename = 'AmesIowa.csv'



submission.to_csv(filename,index=False)

print('Saved File: ' + filename)