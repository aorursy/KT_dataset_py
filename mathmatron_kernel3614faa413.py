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
#question 1 code

from scipy.stats import norm

#imports the normal distribution



#1a

N=15

prob_X_i = norm(0,1).cdf(11/17)

print("probability for question 1a is", prob_X_i**N)



#1b

from math import sqrt #importing the square root function

a = 180/sqrt(4335)

prob_sink = 1 - norm.cdf(a)

print("probability of sinking for question ib is", prob_sink)



#1c

inverse_phi = norm.ppf(0.9999)

from numpy import roots #to solve the polynomial

coeff = [-178, -17*inverse_phi, 2850]

print("coefficients of polynomial in root n are", roots(coeff))

#we then square the positive coefficient, as the polynomial was in root n

solution = max(roots(coeff))**2

print("max number of economists allowed on the raft is", solution)



wage_data = pd.read_csv('../input/part-iia-economics-paper-3-s1/wagefull.csv')

wage_data.head()

#this code loads the dataset, and prints the first few rows

#the stata .dta file has been converted into a .csv file

#we want to remove the redundant column

wage_data.drop('Unnamed: 0', axis = 1)
import seaborn as sns

#importing our plotting library...

#Question 2a. Draw a histogram for the whole population. What kind of distribution do you find?

sns.distplot(wage_data.wage, kde = False, color = 'b', bins = 100)

print("We find a log-normal distirbution.")





#now do the same for the log of wages. How do the two distributions compare? Why is that the case?

log_wages = np.log(wage_data.wage)

sns.distplot(log_wages, kde = False, color = 'b', bins = 100)

print("The second distribution appears to be a normal distribution. As wages>0 the original distribution could not be normal")

print('larger values in the data set are brought to a comparable magnitude of other datapoints, as the log function increases very slowly')
#use these data to calculate the mean and standard deviation of log wages for males

wage_data_men = wage_data.loc[wage_data.male == 1]

#this code gets the data for males

wage_data_men.head()

#checking the output
wage_data_men_log = np.log(wage_data_men.wage)

#applies log function to the wage
wage_data_men_log.head()

#checking the output
wage_data_men_log.describe()

#gives a summary of important statistical values
print(wage_data_men.describe())

wage_data_women.describe()
from scipy.stats import ttest_1samp, ttest_ind

#for hypothesis testing

ttest_1samp(wage_data_men_log, 1.7)
#we now test for whether male and female wages are the same
wage_data_women = wage_data.loc[wage_data.male == 0]

wage_data_women_log = np.log(wage_data_women.wage)

wage_data_women_log.describe()
ttest_ind(wage_data_men_log, wage_data_women_log) #if variances are assumed the same, which seems unreasonable
ttest_ind(wage_data_women_log, wage_data_men_log, equal_var =False) #if variances are different. Performs Welch's t test
import scipy.stats
scipy.stats.bartlett(wage_data_women_log, wage_data_men_log)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

#importing decision trees
wage_data.head()
tree_model_log_wages = np.log(wage_data.wage)

tree_model_log_wages.head()

tree_model_attributes = wage_data[['obs', 'exper', 'male', 'school']]

tree_model_attributes.head()
wage_model_tree = DecisionTreeRegressor(random_state=1)
train_X, val_X, train_y, val_y = train_test_split(tree_model_attributes, tree_model_log_wages)
wage_model_tree.fit(train_X, train_y)
test_results_logtree = wage_model_tree.predict(val_X)

test_results_logtree[0:4]
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(test_results_logtree, val_y))

print(train_y.mean())
#okay that was terrible. how about without log wages??
tree_model_attributes = wage_data[['obs', 'exper', 'male', 'school']]

tree_model_wages = wage_data.wage

tree_model_no_log = DecisionTreeRegressor(random_state = 1)

train_nolog_X, val_nolog_X, train_nolog_y, val_nolog_y = train_test_split(tree_model_attributes, tree_model_wages)

tree_model_no_log.fit(train_nolog_X, train_nolog_y)
no_log_predictions = tree_model_no_log.predict(val_nolog_X)

print(mean_absolute_error(no_log_predictions, val_nolog_y))

print(val_nolog_y.mean())
#well that was terrible...
#what about a random forest tree?? only one way to find out...
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

forest_model_for_wage = RandomForestRegressor(random_state = 1)

forest_model_for_wage.fit(train_nolog_X, train_nolog_y)

wage_forest_predictions = forest_model_for_wage.predict(val_nolog_X)

mean_abs_error_wage_forest = mean_absolute_error(wage_forest_predictions, val_nolog_y)
print(mean_abs_error_wage_forest)

print(val_nolog_y.mean())
forest_model_for_logwage = RandomForestRegressor(random_state = 1)

forest_model_for_logwage.fit(train_X, train_y)

logwage_forest_predictions = forest_model_for_wage.predict(val_X)

mean_abs_error_wage_forest = mean_absolute_error(logwage_forest_predictions, val_y)

print(mean_abs_error_wage_forest)

print(val_y.mean())
print(logwage_forest_predictions[0:4])

val_y[0:4]

#even worse results...
#okay so that was a disaster

#let's try and apply our model to the data more intelligently

wage_data.head()

sns.distplot(wage_data.wage, kde=False, bins= 500)

#let's remove datapoints where the data is over a certain quantile

wage_data_log = wage_data

wage_data_log.wage = wage_data_log.wage.apply(np.log)

data_no_extremes = wage_data_log[(wage_data.school <wage_data.school.quantile(.95)) & (wage_data.school > wage_data.school.quantile(0.05))]



data_no_extremes.head()
#okay, we are going to see what happens when we now apply the model!

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

new_attempt_at_forest = RandomForestRegressor(random_state = 1)

attributes = ['exper','school', 'male']

no_extreme_wage = data_no_extremes.wage

no_extreme_attribute_set = data_no_extremes[attributes]

new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(no_extreme_attribute_set, no_extreme_wage)

new_attempt_at_forest.fit(new_train_X, new_train_y)

predictions = new_attempt_at_forest.predict(new_val_X)

print(mean_absolute_error(predictions, new_val_y))

print(new_val_y.mean())

#okay that was a lot better!!



#we now test the multivariate linear regression model

from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(new_train_X, new_train_y)

predictions_2 = reg.predict(new_val_X)

print(mean_absolute_error(predictions_2, new_val_y))

print(new_val_y.mean())
#we now try ridge regression

reg2 = linear_model.Ridge(alpha=0.5)

reg2.fit(new_train_X, new_train_y)

predictions_3 = reg.predict(new_val_X)

print(mean_absolute_error(predictions_3, new_val_y))

print(new_val_y.mean())
#okay, so it's all pretty similar once we remove the extreme values
#we test the regression approach on the data without the extremes removed

train_X, val_X, train_y, val_y

reg_all_data = linear_model.LinearRegression()

reg_all_data.fit(train_X, train_y)

predictions_4 = reg_all_data.predict(val_X)

print(mean_absolute_error(predictions_4, val_y))

print(val_y.mean())
reg_ridge_all_data = linear_model.Ridge(alpha=0.5)

reg_ridge_all_data.fit(train_X, train_y)

predictions_5 = reg_ridge_all_data.predict(val_X)

print(mean_absolute_error(predictions_5, val_y))

print(val_y.mean())