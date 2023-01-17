import pandas as pd

import numpy as np

import math

from sklearn import linear_model # Needed for Linear Regression

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold # K-fold Cross-Validation

from sklearn.model_selection import cross_val_score # Cross-Validation Score

from sklearn.metrics import r2_score # R-Squared

from scipy.stats import pearsonr
Salary_Data = pd.read_csv("../input/random-salary-data-of-employes-age-wise/Salary_Data.csv")

Salary_Data.head()
x1 = Salary_Data.drop('Salary', axis=1).values # Maintain the dataframe by dropping the Salary column

y1 = Salary_Data['Salary'].values              # Create a Pandas Series as the target variable.



kfold = KFold(n_splits=5, shuffle=True, random_state=1)



slr = linear_model.LinearRegression()



results_kfold = cross_val_score(slr, x1, y1, cv=kfold) # Stores the accuracy of each split in a Numpy array.



print("R\N{SUPERSCRIPT TWO}: %.2f%%" % (results_kfold.mean()*100.0))
X = Salary_Data[['YearsExperience']]  # Note the double square brackets.  This must be a DataFrame.

y = Salary_Data.Salary



slr = linear_model.LinearRegression() # Create the linear regression model



kf = KFold(n_splits=5, shuffle=True, random_state=1) # Set the k-fold parameters

kf.split(X)
accuracy_model = []  # The average of this list is the overall R-Squared

slope = []

intercept = []

residuals = []       # These will be used to validate model assumptions



for train_index, test_index in kf.split(X):

    print('Train: %s\nTest: %s\n' % (train_index, test_index))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model = slr.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))

    accuracy_model.append(r2)

    slope.append(slr.coef_)

    intercept.append(slr.intercept_)

    residuals.append(y_test-model.predict(X_test))
# This is the cross-validation score (aka the overall accuracy of the model)

# round((sum(accuracy_model)/len(accuracy_model))*100,2)

print("Average R\N{SUPERSCRIPT TWO}: %.2f%%" % ((sum(accuracy_model)/len(accuracy_model))*100))
plt.bar(range(1,6),accuracy_model)

plt.title("R\N{SUPERSCRIPT TWO} Values by Split")

plt.xlabel("Split")

plt.ylabel("R\N{SUPERSCRIPT TWO} as a Decimal")

caption="Random sampling led to Split 1 having a very low R\N{SUPERSCRIPT TWO} value"

plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12, color="gray")
print('\u03B2\N{SUBSCRIPT ONE}: %.2f' % ((sum(slope)/len(slope))))

print('\u03B2\N{SUBSCRIPT ZERO}: %.2f' % ((sum(intercept)/len(intercept))))
corr_coeff = pearsonr(Salary_Data.YearsExperience, Salary_Data.Salary)

print('Pearson r= %.3f' % corr_coeff[0])

print('p-value= ' + str(corr_coeff[1]))
print("The mean of the residuals is %.2f" % np.mean(residuals))
for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    fold_residuals = y_test-model.predict(X_test)

    plt.scatter(y_test.index, fold_residuals, color='#1f77b4')

#     plt.scatter(y_test.index, fold_residuals)

    plt.xlabel('Index')

    plt.ylabel('Residuals')

    plt.title("Residuals Plot")

    plt.axhline(0, color="#d62728")
for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    fold_residuals = y_test-model.predict(X_test)

#     plt.scatter(y_test.index, fold_residuals, color='#1f77b4') # This line makes all the dots blue

    plt.scatter(y_test.index, fold_residuals) # This line colors the dots according to their assigned test fold

    plt.xlabel('Index')

    plt.ylabel('Residuals')

    plt.title("Residuals Plot")

    plt.axhline(0, color="#d62728")
for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    plt.scatter(y_test, slr.predict(X_test), color='#1f77b4')

#     plt.scatter(y_test, slr.predict(X_test))

plt.xlabel('Actual Salary')

plt.ylabel('Predicted Salary')

plt.title("Actual vs. Predicted Salaries")

y_lim = plt.ylim()

x_lim = plt.xlim()

plt.plot(x_lim, y_lim, 'k-', color ="#d62728")

plt.ylim(y_lim)

plt.xlim(x_lim)
combined_residuals = pd.concat([residuals[0], residuals[1], residuals[2], residuals[3], residuals[4]], ignore_index=True)

plt.hist(combined_residuals, bins=7)

plt.title("Residuals Distribution")

plt.ylabel("Frequency")

plt.xlabel("Residual Amount")
from scipy.stats import shapiro



resid_sum = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    fold_residuals = y_test-model.predict(X_test)

    resid_sum.append(fold_residuals)



all_residuals = pd.concat([resid_sum[0],resid_sum[1],resid_sum[2],resid_sum[3],resid_sum[4]])



# Thanks to Machine Learning Mastery for the Shapiro-Wilk code and tutorial!

# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/



stat, p = shapiro(all_residuals)

print('Test Statistic=%.3f, p-value=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('Distribution of residuals looks Normal (fail to reject H0)')

else:

    print('Distribution of residuals does not look Normal (reject H0)')