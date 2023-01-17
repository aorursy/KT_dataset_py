import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
employee_data = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

employee_data.head()
# Radomly replace some of the Monthly Income values with nan for our demonstration on fixing nan values

random_replace = 0.1



item_list = []

for item in employee_data['MonthlyRate']:

    rng = np.random.rand()

    if rng <= random_replace:

        item_list.append(np.nan)

    else:

        item_list.append(item)

    



employee_data['MonthlyRate'] = item_list

employee_data['MonthlyRate'].head()
list(employee_data)
data_types = employee_data.dtypes

data_types
employee_data['MonthlyRate'] = employee_data['MonthlyRate'].astype(int)
len(employee_data)
employee_data.describe().iloc[0]
employee_data['MonthlyRate'][(employee_data['MonthlyRate']).isna()==True].head()
len(employee_data['MonthlyRate'][(employee_data['MonthlyRate']).isna()==True])
len(employee_data['MonthlyRate'][(employee_data['MonthlyRate']).isna()==True])/len(employee_data)*100
employee_data_categorical = employee_data[['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']]

employee_data_categorical.head()
data_types[data_types=='object'].index
employee_data_categorical = employee_data[data_types[data_types=='object'].index]

employee_data_categorical.head()
len(employee_data_categorical['MaritalStatus'][(employee_data_categorical['MaritalStatus']).isna()==True])
len(employee_data_categorical['JobRole'][(employee_data_categorical['JobRole']).isna()==True])
employee_data_categorical_columns = list(employee_data_categorical)

employee_data_categorical_columns
employee_data_categorical_columns = list(employee_data_categorical)



print("Categorical Features - unique value checks")

for col_id in employee_data_categorical_columns:

    column = employee_data_categorical[col_id]

    print("-----------------------------")

    print(col_id)

    print(column.unique())

employee_data['JobRole'][(employee_data['JobRole']).isna()==True].head()
len(employee_data['JobRole'][(employee_data['JobRole']).isna()==True])/len(employee_data)*100
employee_data['MaritalStatus'][(employee_data['MaritalStatus']).isna()==True].head()
len(employee_data['MaritalStatus'][(employee_data['MaritalStatus']).isna()==True])/len(employee_data)*100
employee_data['MonthlyRate'].iloc[0:20]
employee_data_removed_1 = employee_data[employee_data['MonthlyRate'].isna()==False]

print(len(employee_data_removed_1))
employee_data_removed_2 = employee_data.dropna(subset = ['MonthlyRate'])

print(len(employee_data_removed_2))
print(np.round(len(employee_data[employee_data['MonthlyRate'].isna()==True])/len(employee_data),3)*100, "%")
employee_data_removed_3 = employee_data.drop('MonthlyRate', axis=1)

list(employee_data_removed_3)
employee_data[['DailyRate','MonthlyRate']].head(20)
employee_data_monthlyrate_hourlyrate_div = employee_data['MonthlyRate']/employee_data['DailyRate']

employee_data_monthlyrate_hourlyrate_div.head()
plt.hist(employee_data_monthlyrate_hourlyrate_div, bins=20)

plt.title("Histogram of the Multiplication Factor for \n Daily to Monthly Rate")

plt.show()
# fillna() function

employee_data_replaced_1 = employee_data.copy()

employee_data_replaced_1['MonthlyRate'].fillna(employee_data_replaced_1['DailyRate']*30, inplace=True)

employee_data_replaced_1[['DailyRate','MonthlyRate']].head(20)
# np.where() function

employee_data_replaced_1 = employee_data.copy()

employee_data_replaced_1['MonthlyRate'] = np.where(employee_data_replaced_1['MonthlyRate'].isna()==True,

                                                       employee_data_replaced_1['DailyRate']*30,

                                                       employee_data_replaced_1['MonthlyRate'])

employee_data_replaced_1[['DailyRate','MonthlyRate']].head(20)
# fillna() function

employee_data_replaced_2 = employee_data.copy()

employee_data_replaced_2['MonthlyRate'].fillna(0, inplace=True)

employee_data_replaced_2[['DailyRate','MonthlyRate']].head(20)
# np.where() function

employee_data_replaced_2 = employee_data.copy()

employee_data_replaced_2['MonthlyRate'] = np.where(employee_data_replaced_2['MonthlyRate'].isna()==True,

                                                       0,

                                                       employee_data_replaced_2['MonthlyRate'])

employee_data_replaced_2[['DailyRate','MonthlyRate']].head(20)
# fillna() function

employee_data_replaced_3 = employee_data.copy()

employee_data_replaced_3['MonthlyRate'].fillna(employee_data_replaced_3['MonthlyRate'].mean(), inplace=True)

employee_data_replaced_3[['DailyRate','MonthlyRate']].head(20)
# np.where() function

employee_data_replaced_3 = employee_data.copy()

employee_data_replaced_3['MonthlyRate'] = np.where(employee_data_replaced_3['MonthlyRate'].isna()==True,

                                                       employee_data_replaced_3['MonthlyRate'].mean(),

                                                       employee_data_replaced_3['MonthlyRate'])

employee_data_replaced_3[['DailyRate','MonthlyRate']].head(20)
# fillna() function

employee_data_replaced_4 = employee_data.copy()

employee_data_replaced_4['MonthlyRate'].fillna(employee_data_replaced_4['MonthlyRate'].median(), inplace=True)

employee_data_replaced_4[['DailyRate','MonthlyRate']].head(20)
# np.where() function

employee_data_replaced_4 = employee_data.copy()

employee_data_replaced_4['MonthlyRate'] = np.where(employee_data_replaced_4['MonthlyRate'].isna()==True,

                                                       employee_data_replaced_4['MonthlyRate'].median(),

                                                       employee_data_replaced_4['MonthlyRate'])

employee_data_replaced_4[['DailyRate','MonthlyRate']].head(20)
employee_data['Department'].head()
employee_data['Department'].unique()
employee_data_department_monthylratemean = employee_data.groupby('Department').mean()['MonthlyRate']

employee_data_department_monthylratemean
employee_data_department_monthylratemean['Sales']
employee_data_replaced_5 = employee_data.copy()

employee_data_replaced_5['MonthlyRate'] = np.where((employee_data_replaced_5['MonthlyRate'].isna()==True)&(employee_data_replaced_5['Department']=='Sales'),

                                                    employee_data_department_monthylratemean['Sales'],

                                            np.where((employee_data_replaced_5['MonthlyRate'].isna()==True)&(employee_data_replaced_5['Department']=='Research & Development'),

                                                      employee_data_department_monthylratemean['Research & Development'],

                                                np.where((employee_data_replaced_5['MonthlyRate'].isna()==True)&(employee_data_replaced_5['Department']=='Human Resources'),

                                                         employee_data_department_monthylratemean['Human Resources'],employee_data_replaced_5['MonthlyRate'])))





employee_data_replaced_5[['Department','MonthlyRate']].head(20)
# REPLACE EACH ITEM IN THE MONTHLYRATE COLUMN WITH THE AVERAGE OF THE DEPARTMENT CLASS USING A FOR LOOP

#-----------------------------------------------------------------------------------------------------

# Create a copy of the employee dataset

# Compute the mean value of the MonthlyRate for each Department class

#

# MAIN LOOP:

#

# Initialise an empty list for logging results

# for each index,row in our employee data:

#     if the MonthlyRate value is not null --> dont change

#     else:

#         find the Department class for the row, extract the mean from the previously and replace with this value

#

#     store each value into the initialised list with the ".append()" function

#

# replace the MonthlyRate column with this updated list

#-----------------------------------------------------------------------------------------------------



employee_data_replaced_6 = employee_data.copy()



# Find the mean results of the MonthlyRate column for each Department class for all known values

employee_data_department_monthylratemean = employee_data.groupby('Department').mean()['MonthlyRate']



MonthlyRate_replacement = []

for n,row in employee_data_replaced_6.iterrows():

    if pd.notnull(row['MonthlyRate']):

        MonthlyRate_value = row['MonthlyRate']

    else:

        row_dep_class = row['Department']

        row_dep_class_mean = employee_data_department_monthylratemean[row_dep_class]

        MonthlyRate_value = row_dep_class_mean

    

    MonthlyRate_replacement.append(MonthlyRate_value)





employee_data_replaced_6['MonthlyRate'] = MonthlyRate_replacement

employee_data_replaced_6[['Department','MonthlyRate']].head(20)
list(employee_data)
plt.scatter(employee_data['TotalWorkingYears'], employee_data['MonthlyRate'])

plt.title("Comparison of Total Working Years against \n Monthly Rate for all Employees")

plt.xlabel("Total Working Years")

plt.ylabel("Monthly Rate")

plt.show()
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

from sklearn import linear_model
employee_data_replaced_7 = employee_data.copy()



employee_data_7_NAN = employee_data_replaced_7[employee_data_replaced_7['MonthlyRate'].isna()==True]

employee_data_7 = employee_data_replaced_7[employee_data_replaced_7['MonthlyRate'].isna()==False]



X = employee_data_7[['TotalWorkingYears']]

y = employee_data_7[['MonthlyRate']]



# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X, y)



# The correlation coefficient

print('Coefficients: \n', regr.coef_)



# The intercept

print('Intercept: \n', regr.intercept_)

plt.scatter(employee_data['TotalWorkingYears'], employee_data['MonthlyRate'])

plt.plot(X, regr.predict(X),color='r')

plt.title("Comparison of Total Working Years against \n Monthly Rate for all Employees")

plt.xlabel("Total Working Years")

plt.ylabel("Monthly Rate")

plt.show()
# Make predictions using the testing set

nan_pred = regr.predict(employee_data_7_NAN[['TotalWorkingYears']])

nan_pred[0:10]
employee_data_replaced_7 = employee_data.copy()



MonthlyRate_replacement = []

for n,row in employee_data_replaced_7.iterrows():

    if pd.notnull(row['MonthlyRate']):

        MonthlyRate_value = row['MonthlyRate']

    else:

        regr_pred = regr.predict([row[['TotalWorkingYears']]])

        MonthlyRate_value = regr_pred[0][0]

    

    MonthlyRate_replacement.append(MonthlyRate_value)





employee_data_replaced_7['MonthlyRate'] = MonthlyRate_replacement



employee_data_replaced_7[['TotalWorkingYears','MonthlyRate']].head(20)
print(np.round(len(employee_data['JobRole'][(employee_data['JobRole']).isna()==True])/len(employee_data)*100,2),"%")
print(np.round(len(employee_data['MaritalStatus'][(employee_data['MaritalStatus']).isna()==True])/len(employee_data)*100,2),"%")
employee_data_removed_8 = employee_data.copy()

employee_data_removed_8 = employee_data_removed_8.drop('MaritalStatus',axis=1)

list(employee_data_removed_8)
employee_data_removed_8['JobRole'].mode()[0]
# fillna() function

employee_data_removed_9 = employee_data.copy()

employee_data_removed_9['JobRole'].fillna(employee_data_removed_9['JobRole'].mode()[0], inplace=True)

employee_data_removed_9[['JobRole']].head(20)
# np.where() function

employee_data_removed_9 = employee_data.copy()

employee_data_removed_9['JobRole'] = np.where(employee_data_removed_9['JobRole'].isna()==True,

                                                       employee_data_removed_9['JobRole'].mode(),

                                                       employee_data_removed_9['JobRole'])

employee_data_removed_9[['JobRole']].head(20)
employee_data_clean = employee_data.copy()



employee_data_clean['MonthlyRate'].fillna(employee_data_clean['MonthlyRate'].mean(), inplace=True)

employee_data_clean = employee_data_clean.drop('MaritalStatus',axis=1)

employee_data_clean['JobRole'].fillna(employee_data_clean['JobRole'].mode()[0], inplace=True)



#Final check for any null values, if false then there are none :)

employee_data_clean.isnull().values.any()
employee_data_clean.head(10)
employee_data_clean['MonthlyRate'] = employee_data_clean['MonthlyRate'].astype(np.int64)

data_types_clean = employee_data_clean.dtypes

data_types_clean
import statistics 



x = employee_data_clean['MonthlyIncome']

# mean and stdev

mu = employee_data_clean['MonthlyIncome'].mean()

sigma = statistics.stdev(employee_data_clean['MonthlyIncome'])



num_bins = 50



fig, ax = plt.subplots()



# the histogram of the data

n, bins, patches = ax.hist(x, num_bins, density=1)



# add a 'best fit' line

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, 'r--')

ax.set_xlabel('MonthlyRate')

ax.set_ylabel('Probability density')

ax.set_title(r'Histogram of MonthlyRate for 50 bins')



# Tweak spacing to prevent clipping of ylabel

fig.tight_layout()

plt.show()
# compute value for outlier cutoff points

mu + 3*sigma
print(np.round(len(employee_data_clean[employee_data_clean['MonthlyIncome']>=(mu + 3*sigma)])/len(employee_data_clean)*100,5), "%")
print(np.round(len(employee_data_clean[employee_data_clean['MonthlyIncome']<=(mu - 3*sigma)])/len(employee_data_clean)*100,5), "%")

ax = sns.boxplot(x=employee_data_clean['MonthlyIncome'])

plt.title("Simple Box Plot of MonthlyIncome to find Outliers")

plt.show()
employee_data_clean_outliers = employee_data_clean[employee_data_clean['MonthlyIncome']<17000]

print("Mean Monthly Income BEFORE removing outliers = ", employee_data_clean['MonthlyIncome'].mean())

print("Mean Monthly Income AFTER removing outliers = ", employee_data_clean_outliers['MonthlyIncome'].mean())
plt.scatter(employee_data_clean_outliers['MonthlyIncome'],employee_data_clean_outliers['MonthlyRate'])

plt.title("Comparison of Monthly Income against Monthly Rate")

plt.xlabel("Monthly Income")

plt.ylabel("Monthly Rate")

plt.show()
plt.scatter(employee_data_clean_outliers['MonthlyIncome'],employee_data_clean_outliers['MonthlyRate'])

plt.title("Comparison of Monthly Income against Monthly Rate (log scale)")

plt.xlabel("Monthly Income")

plt.ylabel("Monthly Rate (log scale)")



plt.yscale("log")



plt.show()
plt.scatter(employee_data_clean_outliers['MonthlyIncome'],employee_data_clean_outliers['MonthlyRate'])

plt.title("Comparison of Monthly Income (log scale) against Monthly Rate")

plt.xlabel("Monthly Income (log scale)")

plt.ylabel("Monthly Rate")



plt.xscale("log")



plt.show()
plt.scatter(employee_data_clean_outliers['MonthlyIncome'],employee_data_clean_outliers['MonthlyRate'])

plt.title("Comparison of Monthly Income (log scale) against Monthly Rate (log scale)")

plt.xlabel("Monthly Income (log scale)")

plt.ylabel("Monthly Rate (log scale)")



plt.yscale("log")

plt.xscale("log")



plt.show()
plt.hist(employee_data_clean_outliers['HourlyRate'], alpha=0.5)

plt.hist(employee_data_clean_outliers['MonthlyIncome'], alpha=0.5)

plt.title("Histogram comparison between the \n Monthly Income and Hourly Rate")

plt.show()
employee_data_clean_outliers['HourlyRate_norm'] = ((employee_data_clean_outliers['HourlyRate']-min(employee_data_clean_outliers['HourlyRate']))/          

                                            (max(employee_data_clean_outliers['HourlyRate'])-min(employee_data_clean_outliers['HourlyRate'])))

employee_data_clean_outliers['MonthlyIncome_norm'] = ((employee_data_clean_outliers['MonthlyIncome']-min(employee_data_clean_outliers['MonthlyIncome']))/          

                                            (max(employee_data_clean_outliers['MonthlyIncome'])-min(employee_data_clean_outliers['MonthlyIncome'])))

plt.hist(employee_data_clean_outliers['HourlyRate_norm'], alpha=0.5)

plt.hist(employee_data_clean_outliers['MonthlyIncome_norm'], alpha=0.5)

plt.title("Normalised Histogram comparison between the \n Monthly Income and Hourly Rate")

plt.show()