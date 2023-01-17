## import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


## import data
insurance = pd.read_csv('../input/insurance.csv')
## have a peak at the data
print(insurance.head()) # print first 4 rows of data
print(insurance.info()) # get some info on variables
# scatter plot charges ~ age
insurance.plot.scatter(x='age', y='charges') 

# scatter plot charges ~ bmi
insurance.plot.scatter(x='bmi', y='charges')
print(insurance.boxplot(column = 'charges', by = 'smoker'))

print(insurance.boxplot(column = 'charges', by = 'sex'))

print(insurance.boxplot(column = 'charges', by = 'children'))

print(insurance.boxplot(column = 'charges', by = 'region'))
## make scatter plot with age on x and charges on y, 
## also show regression line 
insurance.plot.scatter(x='age', y='charges') 
# simple linear regression using age as a predictor
X = insurance["age"] ## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model1 = sm.OLS(y, X).fit()
predictions = model1.predict(X) # make the predictions by the model

# Print out the statistics
model1.summary()
# multiple linear regression using age and bmi as a predictor
X = insurance[["age", "bmi"]]## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model2 = sm.OLS(y, X).fit()
predictions = model2.predict(X) # make the predictions by the model

# Print out the statistics
model2.summary()
# calculate f-ratio 
anovaResults = anova_lm(model1, model2)
print(anovaResults)
# make dummy variable for categorical variables
insurance = pd.get_dummies(insurance, columns=['smoker'])
print(insurance.head()) # print first 4 rows of data
# multiple linear regression using smoker, age and bmi as a predictor
X = insurance[["smoker_yes", "age", "bmi"]] ## the input variables,
                                            ## only include smoker_yes
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order, first y than X
model3 = sm.OLS(y, X).fit()
predictions = model3.predict(X) # make the predictions by the model

# Print out the statistics
model3.summary()
# make dummy variables
insurance = pd.get_dummies(insurance, columns=['sex', 'region'])
print(insurance.info()) # print first 4 rows of data

# multiple linear regression using smoker, age and bmi as a predictor
X = insurance[["smoker_yes", "age", "bmi", "sex_male", 
               "children", "region_northwest",'region_southeast', 
               'region_southwest']] ## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

X.head()
# Note the difference in argument order, first y than X
model4 = sm.OLS(y, X).fit()
#predictions = model2.predict(X) # make the predictions by the model

# Print out the statistics
model4.summary()
# multiple linear regression using smoker, age and bmi as a predictor
X = insurance[["smoker_yes", "age", "bmi", 
               "children", "region_northwest",'region_southeast', 
               'region_southwest']] ## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

X.head()
# Note the difference in argument order, first y than X
model5 = sm.OLS(y, X).fit()
#predictions = model2.predict(X) # make the predictions by the model

# Print out the statistics
model5.summary()