# Basic

import numpy as np 

import pandas as pd 



# Plotting

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Scaling

from sklearn.preprocessing import StandardScaler



# train test split

from sklearn.model_selection import train_test_split



# Making Polynomial Features

from sklearn.preprocessing import PolynomialFeatures



# Importing models

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



# Regression Metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# To build optimal model using Backward Elimination

import statsmodels.api as sm



# Cross validation

from sklearn.model_selection import cross_val_score
dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col = False)



dataset.head()
dataset.info()
dataset.describe()
dataset.isna().sum()
dataset.dropna(axis=0, inplace=True)

print(dataset.shape)
dataset.drop(columns = ['status'], axis=1, inplace=True)

dataset.head(2)
sns.distplot(a = dataset['salary'])

plt.title('Salary Distribution')

plt.xlabel('Salary')

plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.minorticks_on()

plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.show()
ax = sns.violinplot(x = 'gender', y = 'salary', data = dataset)



medians = dataset.groupby(['gender'])['salary'].median().values

nobs = dataset['gender'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Gender vs Salary')

plt.grid(b=True, which='major', axis='both', color='#666666', linestyle='-')

plt.minorticks_on()

plt.grid(b=True, which='minor', axis='y', color='#999999', linestyle='-', alpha=0.2)

plt.show()
plt.figure(figsize=(8,6))

sns.regplot(x='ssc_p', y='salary', data = dataset)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('Salary vs SSC Percentage')

plt.show()
ax = sns.violinplot(x = 'ssc_b', y = 'salary', data = dataset)



medians = dataset.groupby(['ssc_b'])['salary'].median().values

nobs = dataset['ssc_b'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Salary vs SSC Board')

plt.grid(b=True, which='major', axis='both', color='#666666', linestyle='-')

plt.minorticks_on()

plt.grid(b=True, which='minor', axis='y', color='#999999', linestyle='-', alpha=0.2)

plt.show()
plt.figure(figsize=(8,6))

sns.regplot(x='hsc_p', y='salary', data = dataset)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('Salary vs HSC Percentage')

plt.show()
ax = sns.violinplot(x = 'hsc_b', y = 'salary', data = dataset)



medians = dataset.groupby(['hsc_b'])['salary'].median().values

nobs = dataset['hsc_b'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Salary vs HSC Board')

plt.grid(b=True, which='major', axis='both', color='#666666', linestyle='-')

plt.minorticks_on()

plt.grid(b=True, which='minor', axis='y', color='#999999', linestyle='-', alpha=0.2)

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'hsc_s', y = 'salary', data = dataset)



medians = dataset.groupby(['hsc_s'])['salary'].median().values

nobs = dataset['hsc_s'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Salary vs HSC Subjects')

plt.show()
plt.figure(figsize=(8,6))

sns.regplot(x='degree_p', y='salary', data = dataset)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('Salary vs Degree Percentage')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'degree_t', y = 'salary', data = dataset)



medians = dataset.groupby(['degree_t'])['salary'].median().values

nobs = dataset['degree_t'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Salary vs Graduate Specialisation')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'degree_t', y = 'salary', data = dataset, hue='gender')

plt.title('Salary vs Graduate Specialisation and gender')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'workex', y = 'salary', data = dataset)



medians = dataset.groupby(['workex'])['salary'].median().values

nobs = dataset['workex'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Salary vs Work Experience')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'workex', y = 'salary', data = dataset, hue='gender')

plt.title('Salary vs Work Experience and Gender')

plt.show()
plt.figure(figsize=(8,6))

sns.regplot(x='etest_p', y='salary', data = dataset)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('Salary vs Employability Test Percentage')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'specialisation', y = 'salary', data = dataset)



medians = dataset.groupby(['specialisation'])['salary'].median().values

nobs = dataset['specialisation'].value_counts().values

nobs = [str(x) for x in nobs.tolist()]

nobs = ['n = ' + i for i in nobs]



pos = range(len(nobs))

for tick,label in zip(pos, ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick]+0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')



plt.title('Salary vs specialisation in MBA')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.violinplot(x = 'specialisation', y = 'salary', data = dataset, hue='gender')

plt.title('Salary vs specialisation in MBA and gender')

plt.show()
plt.figure(figsize=(8,6))

sns.regplot(x='mba_p', y='salary', data = dataset)

plt.minorticks_on()

plt.grid(b=True, which='both', axis='both', alpha=0.1)

plt.title('Salary vs MBA Percentage')

plt.show()
dataset.head(1)
# dropping first column



dataset.drop(columns=['sl_no'], axis=1, inplace=True)

dataset.head(1)
# Gender: F coded as 0 and M as 1

dummy = pd.get_dummies(dataset['gender'])

dummy.rename(columns={'M':'Gender'}, inplace=True)



# drop original column 

dataset.drop("gender", axis = 1, inplace=True)



# merge data frame "dataset" and "dummy_variable_1: Gender column" 

df = pd.concat([dummy['Gender'], dataset], axis=1)



df.head(1)
# ssc_b: Central as 1 and Others as 0

dummy = pd.get_dummies(dataset['ssc_b'])

dummy.rename(columns={'Central':'ssc_b'}, inplace=True)



df.drop("ssc_b", axis = 1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:2], dummy['ssc_b'], df.iloc[:, 2:]], axis=1)



df.head(1)
# hsc_b: Central as 1 and Others as 0

dummy = pd.get_dummies(dataset['hsc_b'])

dummy.rename(columns={'Central':'hsc_b'}, inplace=True)



df.drop("hsc_b", axis = 1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:4], dummy['hsc_b'], df.iloc[:, 4:]], axis=1)



df.head(1)
# Higher Secondary Specialisation: Science: 10 and Commerce: 01 and Arts: 00

dummy = pd.get_dummies(df['hsc_s'])

dummy.rename(columns={'Science': 'HS_Sci', 'Commerce': 'HS_Comm'}, inplace=True)

dummy = pd.concat([dummy['HS_Sci'], dummy['HS_Comm']], axis=1)

dummy.head()



# drop original

df.drop('hsc_s', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:5], dummy, df.iloc[:, 5:]], axis=1)



df.head(1)
# Undergrad specialisation: Sci&Tech: 10 and Comm&Mgmt: 01 and Others: 00

dummy = pd.get_dummies(df['degree_t'])

dummy.rename(columns={'Sci&Tech': 'UG_Sci', 'Comm&Mgmt': 'UG_Comm'}, inplace=True)

dummy = pd.concat([dummy['UG_Sci'], dummy['UG_Comm']], axis=1)

dummy.head()



# drop original

df.drop('degree_t', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:8], dummy, df.iloc[:, 8:]], axis=1)



df.head(1)
# Work experience: Yes as 1 nd No as 0

dummy = pd.get_dummies(df['workex'])

dummy.rename(columns={'Yes': 'workex'}, inplace=True)

# dummy.head()



# drop original

df.drop('workex', axis=1, inplace=True)



# merge data

df = pd.concat([df.iloc[:, 0:10], dummy['workex'], df.iloc[:, 10:]], axis=1)



df.head(1)
# Specialisation: Mkt&Fin as 1 and Mkt&HR as 0

dummy = pd.get_dummies(df['specialisation'])

dummy.rename(columns={'Mkt&Fin': 'specialisation'}, inplace=True)

# dummy.head()



# drop original data

df.drop('specialisation', axis=1, inplace=True)



# merge data

df= pd.concat([df.iloc[:, 0:12], dummy['specialisation'], df.iloc[:, 12:]], axis=1)



df.head(1)
plt.figure(figsize=(14, 12))

sns.heatmap(df.corr(), annot=True)

plt.title('Correlation between all features and salary offered')

plt.show()
# acquiring data for model



X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values



print('X_shape {}'.format(X.shape))

print('y_shape {}'.format(y.shape))
# Splitting



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print('Shape of training set: {} and test set: {}'.format(X_train.shape, X_test.shape))
# Making regressor

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting test values

y_pred = regressor.predict(X_test)



# Model performance through metrics

print('Train Score: ', regressor.score(X_train, y_train))  

print('Test Score: ', regressor.score(X_test, y_test)) 

print()

print('MAE: ', mean_absolute_error(y_test, y_pred))

print('MSE: ', mean_squared_error(y_test, y_pred))

print('R2 score: ', r2_score(y_test, y_pred))
# Creating Polynomial Features

poly_reg = PolynomialFeatures(degree = 3)

X_train_poly = poly_reg.fit_transform(X_train)

X_test_poly = poly_reg.fit_transform(X_test)



# Fitt PolyReg to training set

regressor = LinearRegression()

regressor.fit(X_train_poly, y_train)



# Predicting test values

y_pred = regressor.predict(X_test_poly)



# Model performance through metrics

print('Train Score: ', regressor.score(X_train_poly, y_train))  

print('Test Score: ', regressor.score(X_test_poly, y_test)) 

print()

print('MAE: ', mean_absolute_error(y_test, y_pred))

print('MSE: ', mean_squared_error(y_test, y_pred))

print('R2 score: ', r2_score(y_test, y_pred))
# Applying feature scaling for this



sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_test_sc = sc.fit_transform(X_test)



print('Scaled Successfully')
regressor = SVR(kernel='rbf')

regressor.fit(X_train_sc, y_train)



# Predicting test values

y_pred = regressor.predict(X_test_sc)



# Model performance through metrics

print('Train Score: ', regressor.score(X_train_sc, y_train))  

print('Test Score: ', regressor.score(X_test_sc, y_test)) 

print()

print('MAE: ', mean_absolute_error(y_test, y_pred))

print('MSE: ', mean_squared_error(y_test, y_pred))

print('R2 score: ', r2_score(y_test, y_pred))
regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)



# Predicting test values

y_pred = regressor.predict(X_test)



# Model performance through metrics

print('Train Score: ', regressor.score(X_train, y_train))  

print('Test Score: ', regressor.score(X_test, y_test)) 

print()

print('MAE: ', mean_absolute_error(y_test, y_pred))

print('MSE: ', mean_squared_error(y_test, y_pred))

print('R2 score: ', r2_score(y_test, y_pred))
regressor = RandomForestRegressor(n_estimators = 10)

regressor.fit(X_train, y_train)



# Predicting test values

y_pred = regressor.predict(X_test)



# Model performance through metrics

print('Train Score: ', regressor.score(X_train, y_train))  

print('Test Score: ', regressor.score(X_test, y_test)) 

print()

print('MAE: ', mean_absolute_error(y_test, y_pred))

print('MSE: ', mean_squared_error(y_test, y_pred))

print('R2 score: ', r2_score(y_test, y_pred))
# x_0 has to be given here explicitly because this package does not take in the b_0 constant otherwise.

X_new = df.iloc[:, :-1].values

X_new = np.append(arr = np.ones((148,1)).astype(int), values = X_new, axis = 1)



print(X_new.shape)
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_new[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13, 14]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,5,6,7,8,9,10,11,12,13,14]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,3,4,5,6,8,9,10,11,12]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,4,5,6,7,8,9,10,11]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,3,4,5,6,8,9,10]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,3,4,5,6,7,8,9]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,3,4,5,7,8]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [1,2,3,4,5,6,7]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,4,5,6]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,2,3,4,5]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,1,2,4]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt = X_opt[:, [0,2,3]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
# S0. Create a new set of features that will be our optimal set of features

X_opt_final = X_opt[:, [0,2]]



# S1. SL chosen 0.05



# S2. Taken X_opt. Fit multiple LR

regressor_OLS = sm.OLS(endog = y, exog = X_opt_final).fit()



# S3. predictor with highest p-value. p > SL

regressor_OLS.summary()



# S4. Remove predictor if p > SL and highest p-value. Go to S0.
print('Shape of optimal values or X: ', X_opt.shape)



# Let's visualise the first 3 values to see, which of the features have we selected

X_opt[0:5,:]
df.head(5)
X_final = df.iloc[:, [0,9,13]].values

y = df.iloc[:, 14].values



print('X_shape {}'.format(X_final.shape))

print('y_shape {}'.format(y.shape))
# Splitting

X_final_train, X_final_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=0)



print('Shape of training set: {} and test set: {}'.format(X_final_train.shape, X_final_test.shape))
# Making regressor

regressor = LinearRegression()

regressor.fit(X_final_train, y_train)



# Predicting test values

y_pred = regressor.predict(X_final_test)



# Model performance through metrics

print('Train Score: ', regressor.score(X_final_train, y_train))  

print('Test Score: ', regressor.score(X_final_test, y_test)) 

print()

print('MAE: ', mean_absolute_error(y_test, y_pred))

print('MSE: ', mean_squared_error(y_test, y_pred))

print('R2 score: ', r2_score(y_test, y_pred))
# cross validation

reg_score = cross_val_score(regressor, X_final_train, y_train, cv=10)



print('Cross Validation Scores across all 10 iterations: ', reg_score)

print('Multiple Linear Regression: ', np.mean(reg_score))