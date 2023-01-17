import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_csv_data = pd.read_csv('/kaggle/input/employee-absenteeism-prediction/Absenteeism-data.csv')

raw_csv_data.head()
df = raw_csv_data.copy()

df.head()
pd.options.display.max_columns=None

pd.options.display.max_rows=None
display(df)
df.info()
df = df.drop(['ID'], axis=1)
df.head()
# Maximum value in 'Reason for Absence' column

df['Reason for Absence'].max()
# Minimum value in 'Reason for Absence' column

df['Reason for Absence'].min()
# Unique values in 'Reason for Absence' column

df['Reason for Absence'].unique()
# length of unique values in 'Reason for Absence' column

len(df['Reason for Absence'].unique())
sorted(df['Reason for Absence'].unique())
# Create dummy for 'Reason for Absence' column

reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

reason_columns
# Check whether any missing value is there or not

reason_columns['check'] = reason_columns.sum(axis=1)

reason_columns
reason_columns['check'].sum(axis=0)
# So, we'll be dropping the check column from reason_columns

reason_columns = reason_columns.drop(['check'], axis=1)

reason_columns
df.columns.values
reason_columns.columns.values
# Drop 'Reason for Absence' column to avoid multi-collinearity

df = df.drop(['Reason for Absence'], axis=1)

df.head()
# Group the variables from 'Reason for Absence' column

reason_type_1 = reason_columns.iloc[:, 1:14].max(axis=1)

reason_type_2 = reason_columns.iloc[:, 15:17].max(axis=1)

reason_type_3 = reason_columns.iloc[:, 18:21].max(axis=1)

reason_type_4 = reason_columns.iloc[:, 22:].max(axis=1)
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

df.head()
df.columns.values
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns = column_names
df.head()
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours']
df = df[column_names_reordered]
df.head()
df_reason_mod = df.copy()

df_reason_mod.head()
type(df_reason_mod['Date'][0])
# Converting the Date column to timestamp format

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')

df_reason_mod['Date'].head()
df_reason_mod['Date'][0]
df_reason_mod['Date'][0].month
list_months = []

list_months
len(df_reason_mod)
df_reason_mod.loc[:, 'Date'][0].month
for i in range (len(df_reason_mod)):

    list_months.append(df_reason_mod.loc[:, 'Date'][i].month)
list_months
len(list_months)
df_reason_mod['Month Value'] = list_months

df_reason_mod.head(20)
df_reason_mod.loc[:, 'Date'][0].weekday()
list_days = []
def date_to_weekday(date_value):

    return (date_value.weekday())
df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)

df_reason_mod.head(20)
# Dropping Date column from dataframe to avoid multicollinearity

df_reason_mod = df_reason_mod.drop(['Date'], axis=1)

df_reason_mod.head()
df_reason_mod.columns.values
column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',

                          'Month Value', 'Day of the Week',

                           'Transportation Expense', 'Distance to Work', 'Age',

                           'Daily Work Load Average', 'Body Mass Index', 'Education',

                           'Children', 'Pets', 'Absenteeism Time in Hours']
df_reason_mod = df_reason_mod[column_names_reordered]
df_reason_mod.head(20)
df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod.head()
type(df_reason_date_mod['Transportation Expense'][0])
type(df_reason_date_mod['Distance to Work'][0])
type(df_reason_date_mod['Age'][0])
type(df_reason_date_mod['Daily Work Load Average'][0])
type(df_reason_date_mod['Body Mass Index'][0])
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
df_reason_date_mod['Education'].unique()
df_reason_date_mod['Education'].value_counts()
df_preprocessed = df_reason_date_mod.copy()

df_preprocessed.head()
# Saving the preprocessed CSV file

df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)
data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')
data_preprocessed.head()
data_preprocessed['Absenteeism Time in Hours'].median()
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 

                   data_preprocessed['Absenteeism Time in Hours'].median(),

                   1, 0)
targets
data_preprocessed['Excessive Absenteeism'] = targets
data_preprocessed.head()
targets.sum() / targets.shape[0]
# Drop 'Absenteeism Time in Hours' column

data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 

                                            'Daily Work Load Average', 

                                            'Education', 

                                            'Reason_4', 

                                            'Distance to Work'], 

                                             axis=1)

data_with_targets.head()
# Checking whether the following two dataframes are same or different

data_with_targets is data_preprocessed
data_with_targets.shape
data_with_targets.iloc[:, :14]
data_with_targets.iloc[:, :-1]
unscaled_inputs = data_with_targets.iloc[:, :-1]
# from sklearn.preprocessing import StandardScaler

# absenteeism_scaler = StandardScaler()
# import the libraries needed to create the Custom Scaler

# note that all of them are a part of the sklearn package

# moreover, one of them is actually the StandardScaler module, 

# so you can imagine that the Custom Scaler is build on it



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



# create the Custom Scaler class



class CustomScaler(BaseEstimator,TransformerMixin): 

    

    # init or what information we need to declare a CustomScaler object

    # and what is calculated/declared as we do

    

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):

        

        # scaler is nothing but a Standard Scaler object

        self.scaler = StandardScaler(copy,with_mean,with_std)

        # with some columns 'twist'

        self.columns = columns

        self.mean_ = None

        self.var_ = None

        

    

    # the fit method, which, again based on StandardScale

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.var_ = np.var(X[self.columns])

        return self

    

    # the transform method which does the actual scaling



    def transform(self, X, y=None, copy=None):

        

        # record the initial order of the columns

        init_col_order = X.columns

        

        # scale all features that you chose when creating the instance of the class

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        

        # declare a variable containing all information that was not scaled

        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]

        

        # return a data frame which contains all scaled features and all 'not scaled' features

        # use the original order (that you recorded in the beginning)

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
unscaled_inputs.columns.values
# columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 

#                     'Distance to Work','Age', 'Daily Work Load Average', 'Body Mass Index', 

#                     'Children', 'Pets']



columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs
scaled_inputs.shape
from sklearn.model_selection import train_test_split
# Split

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
reg = LogisticRegression()
reg.fit(x_train, y_train)
reg.score(x_train, y_train)
model_outputs = reg.predict(x_train)
model_outputs
targets
np.sum(model_outputs == y_train)
model_outputs.shape[0]
np.sum(model_outputs == y_train) / model_outputs.shape[0]
# Finding the intercept

reg.intercept_
# Finding the coefficient

reg.coef_
# Finding the column names in unscaled dataframe

unscaled_inputs.columns.values
feature_name = unscaled_inputs.columns.values
# Creating summary table to store different attributes and their corresponding values

summary_table = pd.DataFrame(columns=['Feature Name'], data=feature_name)

summary_table['Coefficients'] = np.transpose(reg.coef_)

summary_table
# Inserting the value of intercept in the summary table

summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
summary_table['Odds_Ratio'] = np.exp(summary_table['Coefficients'])

summary_table
summary_table.sort_values('Odds_Ratio', ascending=False)
reg.score(x_test, y_test)
predicted_proba = reg.predict_proba(x_test)

predicted_proba
predicted_proba[:, 1]
import pickle
# pickle the model file

with open('model', 'wb') as file:

    pickle.dump(reg, file)
# pickle the scaler file

with open('scaler', 'wb') as file:

    pickle.dump(absenteeism_scaler, file)