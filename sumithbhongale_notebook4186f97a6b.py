import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from IPython.display import display # Allows the use of display() for DataFrames

%matplotlib inline
uni_df = pd.read_csv("IPEDS_data.csv")

col=uni_df.columns

print("uni_ds.columns:\n", uni_df.columns)

list(col)
uni_df.head(3)
uni_df.shape
# Get some summary statistics of our target variable

uni_df['Total  enrollment'].describe()


uni_df = uni_df[np.isfinite(uni_df['Total  enrollment'])]

uni_df.shape
uni_df.isnull().sum()
uni_df = uni_df.drop(uni_df.columns[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,34,35,36,37,38,39,40,41,42,43,44,45,72,73,74,75,76,78,79,80,81,122,123,124,125,126,127,128,129,26,27,28,29,30,31,32,33]], axis=1)
col=uni_df.columns

col
# Produce a scatter matrix for several sample pairs of features in the data

sample_cols = ['Total  enrollment', 'Applicants total', 'Number of students receiving a Bachelor\'s degree', 'Total price for in-state students living on campus 2013-14']

display(uni_df[sample_cols].describe())

pd.scatter_matrix(uni_df[sample_cols], alpha = 0.3, figsize = (14,8), diagonal = 'kde');

# Drop non-numbers, not useful for us

uni_df = uni_df.select_dtypes(include=['float64', 'int64'])



# Force the numeric cells to be proper numbers

uni_df = uni_df.convert_objects(convert_numeric=True)
uni_df.head(5)
# Lets drop the outliers, at least of the enrollment!



# Calculate Q1 (25th percentile of the data) for the given feature

Q1 = np.percentile(uni_df['Total  enrollment'], 25)

    

# Calculate Q3 (75th percentile of the data) for the given feature

Q3 = np.percentile(uni_df['Total  enrollment'], 75)

    

# Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

step = 1.5*(Q3-Q1)

    

# For removing outliers in total_enrollment

uni_df = uni_df[((uni_df['Total  enrollment'] >= Q1 - step) & (uni_df['Total  enrollment'] <= Q3 + step))]
# Handle the missing values

# Impute missing data via interpolation

uni_df = uni_df.interpolate()



# Use mean for remaining missing data

uni_df = uni_df.fillna(uni_df.mean())



display(uni_df[sample_cols].describe())
# Produce a scatter matrix for several sample pairs of features in the data

pd.scatter_matrix(uni_df[sample_cols], alpha = 0.3, figsize = (14,8), diagonal = 'kde');
# We will need these values for normalizing and transforming back from normalizing

enroll_min = uni_df['Total  enrollment'].min()

enroll_max = uni_df['Total  enrollment'].max()

enlarge_factor = 10000



# Use this to transform a scaled enrollment value back to the original scale

def original_scale_transform(value):

    new_data = np.exp(value)

    

    new_data = new_data / enlarge_factor

    

    new_data = new_data * (enroll_max - enroll_min) + enroll_min

    

    return new_data
# min/max scale, and multiply by 10k to avoid tiny numbers.  Then take the log to center.

uni_df = 10000*(uni_df - uni_df.min()) / (uni_df.max() - uni_df.min())

uni_df = np.log(uni_df)



# the log(0) values were turned into -inf, we want those as zero for normalization

uni_df = uni_df.replace(-np.inf, 0)



display(uni_df[sample_cols].describe())
# Produce a scatter matrix for several normalized sample pairs of features in the data

pd.scatter_matrix(uni_df[sample_cols], alpha = 0.3, figsize = (14,8), diagonal = 'kde');


enrollment = uni_df['Total  enrollment']

features = uni_df.drop('Total  enrollment', axis = 1)

from sklearn.cross_validation import train_test_split





X_train, X_test, y_train, y_test = train_test_split(features, enrollment, test_size=0.2, random_state=42)
col_list=X_test.columns

list(col_list)
# Baseline Model

from sklearn.linear_model import LinearRegression



reg = LinearRegression()



# Use admission's count, plus the retention data in a linear regression

reg.fit(X_train[['Admissions total', 'Tuition and fees, 2012-13','Percent of freshmen receiving any financial aid']], y_train)



from sklearn.metrics import mean_squared_error



baseline_pred = reg.predict(X_test[['Admissions total', 'Tuition and fees, 2012-13','Percent of freshmen receiving any financial aid']])



# Turns out this is a terrible way to predict.  Hopefully we can do better than that.

print ("Benchmark Model, Training Mean Squared Error:", mean_squared_error(y_train, reg.predict(X_train[['Admissions total', 'Tuition and fees, 2012-13','Percent of freshmen receiving any financial aid']])))

print ("Benchmark Model, Testing Mean Squared Error:", mean_squared_error(y_test, baseline_pred))
from sklearn.decomposition import FastICA



# n_components = 5, because that's the max number of features with

# the number of observations we have, considering the curse of

# dimensionality

#ica = FastICA(n_components = 10,random_state=42)

ica = FastICA(n_components=5, random_state=42)

ica.fit(X_train)



# Our new 5 features

X_train_ica = ica.transform(X_train)

X_test_ica = ica.transform(X_test)


# Same technique as the baseline, but uses our ICA features instead of the three preset



cm1_reg = LinearRegression()



# Make sure to use the ICA features

cm1_reg.fit(X_train_ica, y_train)



# Generate predictions

cm1_pred = cm1_reg.predict(X_test_ica)



# Report Results

print ("Candidate Model 1, Training Mean Squared Error:", mean_squared_error(y_train, cm1_reg.predict(X_train_ica)))

print ("Candidate Model 1, Testing Mean Squared Error:", mean_squared_error(y_test, cm1_pred))


# Notice: This model runs significantly slower than the baseline and candidate model 1 and 3

from sklearn.svm import SVR



cm2_reg = SVR()



# Make sure to use the ICA features

cm2_reg.fit(X_train_ica, y_train)



# Generate predictions

cm2_pred = cm2_reg.predict(X_test_ica)



# Report Results

print ("Candidate Model 2, Training Mean Squared Error:", mean_squared_error(y_train, cm2_reg.predict(X_train_ica)))

print ("Candidate Model 2, Testing Mean Squared Error:", mean_squared_error(y_test, cm2_pred))


from sklearn.ensemble import RandomForestRegressor



cm3_reg = RandomForestRegressor(random_state=42)



# Make sure to use the ICA features

cm3_reg.fit(X_train_ica, y_train)



# Generate predictions

cm3_pred = cm3_reg.predict(X_test_ica)



# Report Results

print ("Candidate Model 3, Training Mean Squared Error:", mean_squared_error(y_train, cm3_reg.predict(X_train_ica)))

print ("Candidate Model 3, Testing Mean Squared Error:", mean_squared_error(y_test, cm3_pred))
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import make_scorer



# Random Forest Regressor patameters

parameters = {

    'n_estimators': [5, 10, 20, 40,50], # Number of Trees in the Forest

    'max_features': ["auto", "sqrt", "log2"], # Number of features to examine

}



# Initialize the classifier

fin_reg = RandomForestRegressor(random_state=42)



# Make an mean squared error scoring function using 'make_scorer' 

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)



# Perform grid search on the model using the mse_scorer as the scoring method

grid_obj = GridSearchCV(fin_reg, parameters, scoring=mse_scorer)



# Fit the grid search object to the training data and find the optimal parameters

grid_obj = grid_obj.fit(X_train_ica, y_train)



# Get the estimator

fin_reg = grid_obj.best_estimator_



# Report the final mean squared error for training and testing after parameter tuning

print ("Tuned model has a training MSE of: ", mean_squared_error(y_train, fin_reg.predict(X_train_ica)))

print ("Tuned model has a testing MSE score of:", mean_squared_error(y_test, fin_reg.predict(X_test_ica)))
# Find most important feature from our ICA in our Model

print (fin_reg.feature_importances_)



# Also find best parameters of our tuned model

print (grid_obj.best_params_)
# Trying to figure out what exactly ICA(4) is, can be kind of a pain.  Here's a bar chart of the weights

plt.title('Weights of Independent Component Analysis Result #4')

plt.xlabel('Feature Index')

plt.ylabel('Weight')

plt.bar(range(len(ica.mixing_[:,4])), ica.mixing_[:,4], color='cyan')
plt.ylabel('total_enrollment')

plt.xlabel('Independent Component Analysis Result #4')

plt.ylim(ymin=0, ymax=6000)

plt.title('Actual Observations of Enrollment')

plt.scatter(X_test_ica[:,4], original_scale_transform(y_test), s=16)
plt.ylabel('total_enrollment')

plt.xlabel('Independent Component Analysis Result #4')

plt.title('Predicted Enrollments')

plt.ylim(ymin=0, ymax=6000)

plt.scatter(X_test_ica[:,4], original_scale_transform(fin_reg.predict(X_test_ica)), c='g', s=16)