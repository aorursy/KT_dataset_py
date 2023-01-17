# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import statsmodels.api as sm

from statsmodels.nonparametric.kde import KDEUnivariate

from statsmodels.nonparametric import smoothers_lowess

from pandas import Series, DataFrame

from patsy import dmatrices

from sklearn import datasets, svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv("../input/train.csv") 

df = df.drop(['Ticket','Cabin'], axis=1)

# Remove NaN values

df = df.dropna() 

# model formula

# here the ~ sign is an = sign, and the features of our dataset

# are written as a formula to predict survived. The C() lets our 

# regression know that those variables are categorical.

# Ref: http://patsy.readthedocs.org/en/latest/formulas.html

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' 

# create a results dictionary to hold our regression results for easy analysis later        

results = {} 



# create a regression friendly dataframe using patsy's dmatrices function

y,x = dmatrices(formula, data=df, return_type='dataframe')



# instantiate our model

model = sm.Logit(y,x)



# fit our model to the training data

res = model.fit()



# save the result for outputing predictions later

results['Logit'] = [res, formula]

res.summary()



formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
# import the machine learning library that holds the randomforest

import sklearn.ensemble as ske



# Create the random forest model and fit the model to our training data

y, x = dmatrices(formula_ml, data=df, return_type='dataframe')

# RandomForestClassifier expects a 1 demensional NumPy array, so we convert

y = np.asarray(y).ravel()

#instantiate and fit our model

results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)



# Score the results

score = results_rf.score(x, y)

score