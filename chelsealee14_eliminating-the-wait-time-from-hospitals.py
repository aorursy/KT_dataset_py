import numpy as np # linear algebra
import pandas as pd # data processing, 
import seaborn as sns #for plotting 
import os
df = pd.read_csv("../input/HospInfo.csv")
df.head()
nrows = df.shape[0]
num_unique_hosp = df['Provider ID'].nunique()
print("Is Provider ID a unique identifier?: {0}".format(nrows == num_unique_hosp))
#Replace text with NaN that can be read as true missing value in Python
df = df.replace('Not Available', np.nan )

#Drop all columns whose name contains 'footnote'
cols_to_drop = list(df.filter(regex='footnote'))
df_clean = df[df.columns.drop(cols_to_drop)]

#print to see results
df_clean.head()
#normalize = True gives the percentages of each value instead of frequencies
df_clean['Timeliness of care national comparison'].value_counts(normalize=True)

print("Out of {0} total hospitals, how many have below average wait times?\
 {1} hospitals".format(nrows, round(nrows * 0.255)))
#Calculate percentage of missing data in each column
df_clean.isnull().mean().sort_values(ascending=False)
#store the number of rows before (b) dropping
num_rows_b = df_clean.shape[0]

df_clean = df_clean.loc[df_clean["Timeliness of care national comparison"].notnull(), :]

#check if there is no missnig data in target variable
print("% of missing data in target variable after cleaning: {:.0%}"\
      .format(df_clean["Timeliness of care national comparison"].isnull().mean()))

#store the number of rows after (a) dropping
num_rows_a = df_clean.shape[0]

#Show the change in number of rows
print("# of rows before dropping NAs: {0}\n# of rows after dropping NAs: {1}"\
      .format(num_rows_b, num_rows_a))
#Remove Hospital Name, Address, City, State, Zip Code, County Name, Phone Number, and Location 
#Keep Provider ID for key later on so that we could pull in other information if we want to.
df_clean = df_clean.drop([
    "Hospital Name", "Address", "City", 
    "State", "ZIP Code", "Phone Number",
    "County Name", "Location"
], axis =1)

#See if values that are categorical are truly categorical, bools as truly bool and int as ints
df_clean.dtypes
#Categorical variables are correctly casted as object type
#Emergency Services is bool but Meets criteria for meaningful use of EHR is not. Let's convert this to bool
df_clean['Meets criteria for meaningful use of EHRs'] = \
df_clean['Meets criteria for meaningful use of EHRs'].astype(bool)

#hospital overall rating should be numerical type (int doesn't accept missing values, so conver to float type)
df_clean['Hospital overall rating'] = df_clean['Hospital overall rating'].astype(float)

df_clean.dtypes
#Create dummy variables for Hospital Type and Hospital Ownership and save into dv 
dv = pd.get_dummies(df_clean[['Hospital Type', 'Hospital Ownership']] )
dv.head()

#drop old columns and concatenate new dummy variables
df_clean = df_clean.drop(['Hospital Type', 'Hospital Ownership'], axis=1)
df_clean = pd.concat([df_clean, dv], axis=1)

#print head to check results (they're appended to the end)
df_clean.head()
#Remember that Hospital Type and Hospital Ownership did NOT have missing data from the original data.
#create list of columns to convert to ordinal
    # only modify variables that have "national compmarison" in naming
ordinal_col = list(df_clean.filter(regex="national comparison"))

#Create customized mapper to factorize variables that are ordinal nature
mapper = {
    'Below the national average' : 0,
    'Same as the national average' : 1, 
    'Above the national average' : 2
}
for col in ordinal_col:
    df_clean.loc[:, df_clean.columns == col]= df_clean.loc[:, df_clean.columns == col].replace(mapper)

#print results. 
df_clean.head() 
#Factorize Emergency and Meets criteria for meaningful use of EHRs into booleans
    #true = 1 and False = 0
bool_cols = ['Emergency Services', 'Meets criteria for meaningful use of EHRs']

df_clean[bool_cols] = (df_clean[bool_cols] == True).astype(int)

#print head to see results
df_clean.head()
y = df_clean.pop("Timeliness of care national comparison")
X = df_clean

#randomly split into training and testing data. Let's aside 20% of our data for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Compare dimensions:
#Remember that after we dropped rows from earlier, there were 3,546 rows
print("Original X: {0}, Original y: {1}".format(X.shape, y.shape))
print("X Train: {0}, y train: {1}".format(X_train.shape, y_train.shape))
print("X Train: {0}, y test: {1}".format(X_test.shape, y_test.shape))
#Now remove provider ID after we have split into train/test
X_train_id = X_train.pop("Provider ID")
X_test_id = X_test.pop("Provider ID")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from xgboost import plot_importance

#Instantiate XGB classifier model
xgb = XGBClassifier(seed = 123)

# fit model no training data
xgb.fit(X_train, y_train)

#Predict the lables of the test test
preds = xgb.predict(X_test)

# Compute the accuracy: accuracy

accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: {:.2f}%".format(accuracy * 100))
#plot feature importance graph to see which features contribute to predicting the outcomes
plot_importance(xgb)
plt.show()
#subset features there are "fairly importnat" relate to other features
subset = [
    "Hospital overall rating", "Safety of care national comparison",
    "Efficient use of medical imaging national comparison", 
    "Patient experience national comparison", "Mortality national comparison", 
    "Effectiveness of care national comparison", "Readmission national comparison",
    "Hospital Ownership_Proprietary"
]
X_train = X_train[subset]
X_test = X_test[subset]

xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)

accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: {:.2f}%".format(accuracy * 100))
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 
xgb_param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'n_estimators': np.arange(100, 400, 10),
    'max_depth': np.arange(2, 11)
}

# Perform random search with scoring metric negative MSE. 
#Perform 5 fold CV (this is arbitrary)
randomized_mse = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_grid, 
                                    scoring = "accuracy",n_iter=10, cv=5, verbose=1)


# Fit randomized_mse to the data
randomized_mse.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

#Predict the lables of the test test
preds = randomized_mse.predict(X_test)

# Compute the accuracy: accuracy

accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: {:.2f}%".format(accuracy * 100))
"""
#Import necessary functions
from sklearn.preprocessing import LabelEncoder

#Create a vector of booleans for categorical columns
categorical_mask = (df_clean.dtypes == object)

#Subset list of categorical columns
categorical_columns = df_clean.columns[categorical_mask].tolist()

#Print the head of the categorical columns (should be the same right after we dropped the variables from the first line of code of this chunk)
df_clean[categorical_columns].head()
#Instantiate LabelEncoder Object
le = LabelEncoder()

#Create empty list to store all mappings. This essentially serves as our "data dictionaries"
cat_column_mappings = []

#df_clean = df_clean.fillna("99")

#Loop through each of the categorical columns to convert to discrete numerical values. 
    #At the same time, create the dictionary and append back to cat_column_mappings
for col in categorical_columns:
    df_clean[col] = le.fit_transform(df_clean[col])
    
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    cat_column_mappings.append(pd.DataFrame(sorted(le_name_mapping.items()), columns = [col, "Encoded Labels"]))

#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#le_name_mapping
#cat_column_mappings

#print to see results from LabelEncoder
df_clean[categorical_columns].head()

#Seems the labels of overall ratings have shifted backwards by one. Let's refer to the fourth index of the data dictionary vector to double check
cat_column_mappings[4] # IT is! Quite confusing to read to be honest. Perhaps another mapping would do?
"""
