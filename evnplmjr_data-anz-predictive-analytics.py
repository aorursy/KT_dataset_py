import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn import tree

from matplotlib import pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns

import numpy as np 

import pandas as pd 

print('Environment Setup Complete')
file_path = '../input/insidesherpa-anz-dataset-task2/ANZ synthesised transaction dataset.xlsx'

file = pd.read_excel(file_path)

print('Data Set Ready')
# The dataframe is narrowed down to the pay/salary values in the transaction description column

salary_details = file['txn_description'] == 'PAY/SALARY'

salary_file = file[salary_details]

cleaned_salary_file = salary_file.drop(columns=['card_present_flag','bpay_biller_code',

                                                'merchant_id','merchant_code','merchant_long_lat',

                                                'merchant_suburb','merchant_state'])



# The dataframe is segmented and the amount column is summed up to present the quarterly salary of each customer

df = pd.DataFrame(cleaned_salary_file.groupby(['account','first_name','gender','age','long_lat'], as_index=False)['amount'].agg('sum'))



# The annual salary is added and calculated by multiplying the quarterly salary by 4 to get the annual salary of each customer

df['annual_salary'] = 4 * df['amount']
age_corr, age_pv = spearmanr(df['age'], df['annual_salary'])

print(age_corr)



plt.figure(figsize=(10,10))

sns.scatterplot(x=df['age'], y=df['annual_salary'])

plt.title('Annual Salary and Customer Age')

plt.xlabel('Customer Age')

plt.ylabel('Annual Salary')
male_data = df['gender'] == 'M'

female_data = df['gender'] == 'F'



male_df = df[male_data]

female_df = df[female_data]



male_corr, male_pv = spearmanr(male_df['age'], male_df['annual_salary'])

print("Spearman's correlation coefficient value for male-age and annual salary:")

print(male_corr)



female_corr, female_pv = spearmanr(female_df['long_lat'], female_df['annual_salary'])

print("Spearman's correlation coefficient value for female-age and annual salary:")

print(female_corr)



plt.figure(figsize=(10,10))

sns.scatterplot(x=male_df['age'], y=male_df['annual_salary'])

plt.title('Annual Salary and Age of Male Customers')

plt.xlabel('Male Customer Age')

plt.ylabel('Annual Salary')



plt.figure(figsize=(10,10))

sns.scatterplot(x=female_df['age'], y=female_df['annual_salary'])

plt.title('Annual Salary and Age of Female Customers')

plt.xlabel('Female Customer Age')

plt.ylabel('Annual Salary')
df = pd.get_dummies(df, columns=['gender'])



model_features = ['age','gender_F','gender_M']

X = df[model_features]

y = df['annual_salary']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)
lrm = LinearRegression()

lrm.fit(X_train,y_train)



lrm_salary_predictions = lrm.predict(X_test)

print('Linear Regression Model Prediction Results:')

print(lrm_salary_predictions)



lrm_mae = mean_absolute_error(y_test,lrm_salary_predictions)

print('Linear Regression Model Mean Absolute Error:')

print(lrm_mae)
dtm = tree.DecisionTreeRegressor()

dtm.fit(X_train,y_train)



dtm_salary_predictions = dtm.predict(X_test)

print('Decision Tree Regression Model Prediction Results:')

print(dtm_salary_predictions)



dtm_mae = mean_absolute_error(y_test,dtm_salary_predictions)

print('Decision Tree Regression Model Mean Absolute Error:')

print(dtm_mae)