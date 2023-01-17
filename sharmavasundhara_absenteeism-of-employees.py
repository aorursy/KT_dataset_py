# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_csv_data = pd.read_csv("/kaggle/input/absenteeism_data.csv")



raw_csv_data.head()
df = raw_csv_data.copy()



# Setting maximum number of rows displayed in output

pd.options.display.max_rows = 3
df.info()
df = df.drop(['ID'], axis = 1)



display(df)
display(df['Reason for Absence'].min())

display(df['Reason for Absence'].max())



# Checking all unique values of 'Reason of Absence'

display(pd.unique(df['Reason for Absence']))



# Counting the length of array returned

display(len(pd.unique(df['Reason for Absence'])))
# Sorting array to find out the missing value

sorted(df['Reason for Absence'].unique())
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
# Checking if an employee was absent from work for more than one reason

reason_columns['check'] = reason_columns.sum(axis = 1)



reason_columns
# Calculating sum of check column

display(reason_columns['check'].sum(axis = 0))



# Displaying unique values in check column

display(reason_columns['check'].unique())
# Dropping the check column

reason_columns = reason_columns.drop(['check'], axis = 1)



# Dropping 'Reason for Absence' to avoid multicollinearity

df = df.drop(['Reason for Absence'], axis = 1)
reason_type_1 = reason_columns.iloc[:, 1:14].max(axis = 1)

reason_type_2 = reason_columns.iloc[:, 15:17].max(axis = 1)

reason_type_3 = reason_columns.iloc[:, 18:21].max(axis = 1)

reason_type_4 = reason_columns.iloc[:, 22:].max(axis = 1)
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)



display(df)
# Renaming the columns 0, 1, 2, 4

column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4'] 

df.columns = column_names



# Reordering the columns

column_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours'] 

df = df[column_names_reordered]
df_reasons_mod = df.copy()

type(df_reasons_mod['Date'][0])
# Converting the column values to timestamp

df_reasons_mod['Date'] = pd.to_datetime(df_reasons_mod['Date'], format = '%d-%m-%Y')



df_reasons_mod['Date']
# Extracting the month values from 'Date' column

list_months = []

df_reasons_rows_length = df_reasons_mod.shape[0]

for i in range(df_reasons_rows_length):

    list_months.append(df_reasons_mod['Date'][i].month)

    

# Printing the length of the list_months

display(len(list_months))    
# Adding the list_months to df_reasons_mod

df_reasons_mod['Month Value'] = list_months



# Extracting the day of the week from 'Date'

def date_to_weekday(date_value):

    return date_value.weekday()



# Adding 'Day of the Week' column to the dataset by calling the date_to_weekday function

df_reasons_mod['Day of the Week'] = df_reasons_mod['Date'].apply(date_to_weekday)
# Removing the 'Date' column

df_reasons_mod.drop(['Date'], axis = 1)



# Reordering the columns

column_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours'] 

df_reasons_mod = df_reasons_mod[column_names_reordered]



# Printing df_reasons_mod

display(df_reasons_mod)
df_date_reasons_mod = df_reasons_mod.copy()
# Calculating the number of unique values in the 'Education' column

df_date_reasons_mod['Education'].unique()
# Setting maximum number of rows displayed in output

pd.options.display.max_rows = 4



# Counting the number of each value present in the column

df_date_reasons_mod['Education'].value_counts()
# Grouping values 2, 3, and 4 to one group

# Mapping 1 to 0 & 2, 3, 4 to 1

df_date_reasons_mod['Education'] = df_date_reasons_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
df_preprocessed = df_date_reasons_mod.copy()



df_preprocessed
# Calculating the mean of the 'Absenteeism Time in Hours' column values

df_preprocessed_median = df_preprocessed['Absenteeism Time in Hours'].median()
# Classifying data into two groups, for 'Absenteeism Time in Hours' > median - class 1 otherwise 0

targets = np.where(df_preprocessed['Absenteeism Time in Hours'] > df_preprocessed_median, 1, 0)



# Adding targets to the dataframe 

df_preprocessed['Excessive Absenteeism'] = targets
# Check if the groups have been divded almost equally, proceed if value is between 0.40 to 0.50 for accurate model creation

targets.sum() / targets.shape[0]
# Dropping 'Absenteeism Time in Hours' column, and assigning the returned dataframe

data_with_targets = df_preprocessed.drop(['Absenteeism Time in Hours', 'Distance to Work', 'Daily Work Load Average', 'Age'], axis = 1)
# Selecting first 14 columns as inputs

unscaled_inputs = data_with_targets.iloc[:, :-1]



# Printing the data

display(unscaled_inputs)
# Importing StandardScaler, BaseEstimator, and TransformerMixin from sklean

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin



class CustomScaler(BaseEstimator, TransformerMixin):

    

    def __init__(self, columns, copy = True, with_mean = True, with_std = True):

        self.scaler = StandardScaler(copy, with_mean, with_std)

        self.columns = columns

        self.mean_ = None

        self.var_ = None

    

    def fit(self, X, y = None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.var_ = np.var(X[self.columns])

        return self

    

    def transform(self, X, y = None, copy = None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)

        X_not_scaled = X.iloc[:, ~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]

    

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit] 

        

# Creating an empty scaler object

absenteeism_scaler = CustomScaler(columns_to_scale)
# Calculating mean and standard deviation of each unscaled input

absenteeism_scaler.fit(unscaled_inputs)
# Creating scaled inputs using absenteeism_scaler object

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)



# Printing scaled inputs

display(scaled_inputs)
# Importing train_test_split

from sklearn.model_selection import train_test_split
# Shuffling and dividing the data

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, shuffle = True, random_state = 5)



# Checking shapes train and test data

print(x_train.shape, y_train.shape, "&", x_test.shape, y_test.shape)
# Importing LogisticRegression and matrics from sklearn

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
# Creating object of LogisticRegression

regression = LogisticRegression()



# Creating the model using machine learning

regression.fit(x_train, y_train)
# Calculating accuracy of the model

print("Accuracy: ", regression.score(x_train, y_train))



# Calculationg accuracy manually

model_output = regression.predict(x_train)

correct_predictions = np.sum(model_output == y_train) # Calculating sum of correctly predicted outputs

manual_accuracy = correct_predictions / model_output.shape[0]

print("Accuracy by manual calculation: ", manual_accuracy)
# Finding intercepts and coefficiants

print("Intercepts", regression.intercept_)

print("Coefficiants", regression.coef_)



# Creating table to map coefficiants with indepdent variable values

feature_names = unscaled_inputs.columns.values 



# Creating summary

summary = pd.DataFrame(columns = ['Feature Names'], data = feature_names)

summary['Coefficients'] = np.transpose(regression.coef_)



# Adding intercept to the summary

summary.index = summary.index + 1

summary.loc[0] = ['Intercept', regression.intercept_[0]]



# Sorting my index

summary = summary.sort_index()

display(summary)
# Adding odds ratio column

summary['Odds Ratio'] = np.exp(summary['Coefficients'])



# Setting maximum number of rows displayed in output

pd.options.display.max_rows = None



# Sorting summary table

summary.sort_values('Odds Ratio', ascending = False)
# Testing the accuracy with the testing data

display(regression.score(x_test, y_test))
# Finding estimates for possible outputs with x_test as input

predicted_proba = regression.predict_proba(x_test)



display(predicted_proba[:,1])