import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
transactions = pd.read_csv('../input/work-sample/ANZ synthesised transaction dataset.csv')

transactions.head()
transactions['txn_description'].value_counts()
#Examining salary transaction data

salary_transactions = transactions[transactions.txn_description == 'PAY/SALARY'].copy()

salary_transactions.account.value_counts()
#Calculating the annual salary by summing the total credit transactions from the quarter and multiplying by 4

annual_salary = salary_transactions.groupby('account').amount.sum().reset_index()

annual_salary['amount'] = annual_salary['amount'] * 4

annual_salary.head()
#columns for new dataframe, to begin with

columns_to_copy = ['account','long_lat','gender','age']



#Creating the dataframe, getting rid of duplicate entries (so each account has only one entry)

df = transactions[columns_to_copy].copy()

df = df.drop_duplicates()

df.account.value_counts()
#Sorting by 'account' so we can easily add features later

df = df.sort_values('account').reset_index()

df.drop(['index'],axis=1,inplace=True)

df.head()
df['Salary'] = annual_salary['amount']



print('Salary column stats:')

print(df['Salary'].describe())



df.head()
# Extracting the long/lat from the long_lat column

long = df.long_lat.apply(lambda x: float(x.split('-')[0].replace(' ','')))

lat = df.long_lat.apply(lambda x: float(x.split('-')[1].replace(' ','')))



#Converting into radians to use with numpys cos/sin functions

long = (long * np.pi) / 180

lat = (lat * np.pi) / 180



#Removing the now useless long_lat column

df.drop('long_lat',axis=1,inplace=True)
#Converting long and lat into 3d cartesians

df['Xpos'] = np.cos(long) * np.cos(lat)

df['Ypos'] = np.sin(long) * np.cos(lat)

df['Zpos'] = np.sin(lat)



#Re-ording columns so salary is at the end

df = df[df.columns[[0,1,2,4,5,6,3]]].copy()
df['gender'] = df['gender'].apply(lambda x: 1 if x =='F' else 0)
df.head()
#Gathering the data

quarter_spend = transactions[transactions.movement == 'debit'].groupby('account').amount.sum().reset_index()



#Adding to our dataframe for analysis

df['quarter_spend'] = quarter_spend['amount']

df = df[df.columns[[0,1,2,3,4,5,7,6]]].copy()

df
#Observing the different possible values

print(list(transactions.txn_description.unique()))
#Excluding 'PAY/SALARY' as this was used to calculate Salary (and is 'credit' not 'debit')

txn_features = [list(transactions.txn_description.unique())[i] for i in [0,1,2,3,5]]
#Adding each feature to the dataframe

for feature in txn_features:

    feature_spend = transactions[transactions.txn_description == feature].groupby('account').amount.sum().reset_index()

    

    df['{}_total'.format(feature)] = feature_spend['amount']

    

df = df[df.columns[[0,1,2,3,4,5,6,8,9,10,11,12,7]]].copy()

df.head()
average_balance = transactions.groupby('account').balance.mean().reset_index()



df['average_balance'] = average_balance['balance']

df =df[df.columns[[0,1,2,3,4,5,6,7,8,9,10,11,13,12]]].copy()

df.head()
df.drop('account',axis=1,inplace=True)
total_na = df.isnull().sum().sort_values(ascending=False)

percentage = (df.isnull().sum()*100/df.shape[0]).sort_values(ascending=False)



sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x=percentage.index[:5],y=percentage[:5])





plt.xlabel('Features')

plt.ylabel('Percent of missing values')

plt.title('Percent missing data by feature')

plt.show()



percentage = percentage.map('{:,.2f}%'.format)

combined = pd.concat([total_na,percentage],axis=1,keys=['Total NA','Percentage'])

combined.head(5)
#Removing columns

df.drop(['INTER BANK_total','PHONE BANK_total'],axis=1,inplace=True)
#Replacing SALES-POS_total NA values with zeros

df['SALES-POS_total'] = df['SALES-POS_total'].fillna(0)

df.head()
df.hist(edgecolor='black', linewidth=1.3,figsize=(16,13))

plt.show()
df['Salary'] = np.log1p(df['Salary'])
#Plotting a heatmap to visualize correlation between variables (most importantly with salary)

plt.figure(figsize=(16, 6))

heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
df = df.drop('POS_total',axis=1)
#importing the tool

from sklearn.preprocessing import StandardScaler
cols_to_scale = ['age', 'Xpos', 'Ypos', 'Zpos', 'quarter_spend', 

                 'SALES-POS_total', 'PAYMENT_total', 'average_balance']



for column in cols_to_scale:

    scale = StandardScaler().fit(df[[column]])

    df[column] = scale.transform(df[[column]])

    

df
#Splitting data into test and train

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

np.random.seed(5)



y_data = df['Salary']

x_data = df.drop('Salary',axis=1)

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,train_size=0.8)



model_1 = LinearRegression()
#Fitting the model

model_1.fit(x_train,y_train)



#Converting the predictions back to a typical salary figure

predictions_1 = model_1.predict(x_test)

predictions_2 = np.expm1(model_1.predict(x_test))
from sklearn.metrics import mean_absolute_error as mae

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer



print('Mean Absolute Error: {}'.format(mae((y_test),predictions_1)))



print('Mean Squared Log Error: {}'.format(metrics.mean_squared_log_error(y_test,predictions_1)))



cv_score = cross_val_score(model_1,x_data,y_data,cv=10,scoring=make_scorer(mae))

print('Mean cross val score: {}'.format(cv_score.mean()))



print('\nMean Absolute Error (un-transformed): {}'.format(mae(np.expm1(y_test),predictions_2)))

print('\nCross val scores: {}'.format(cv_score))

import xgboost as xgb



model_2 = xgb.XGBRegressor()
model_2.fit(x_train,y_train)



#Converting the predictions back to a typical salary figure

predictions_1 = model_2.predict(x_test)

predictions_2 = np.expm1(model_2.predict(x_test))
print('Mean Absolute Error: {}'.format(mae((y_test),predictions_1)))



print('Mean Squared Log Error: {}'.format(metrics.mean_squared_log_error(y_test,predictions_1)))



cv_score = cross_val_score(model_2,x_data,y_data,cv=10,scoring=make_scorer(mae))

print('Mean cross val score: {}'.format(cv_score.mean()))



print('\nMean Absolute Error (un-transformed): {}'.format(mae(np.expm1(y_test),predictions_2)))

print('\nCross val scores: {}'.format(cv_score))

#xgb model

xgbmodel = xgb.XGBRegressor()

xgbmodel.fit(x_data,y_data)

xgbpredictions = np.expm1(xgbmodel.predict(x_data))



#scikit model

scikitmodel = LinearRegression()

scikitmodel.fit(x_data,y_data)

scikitpredictions = np.expm1(scikitmodel.predict(x_data))



final_predictions = 0.80*xgbpredictions + scikitpredictions * 0.2
results = pd.DataFrame([quarter_spend['account'],xgbpredictions,scikitpredictions,final_predictions,np.expm1(y_data)])

results = results.transpose()

results = results.rename(columns={'Unnamed 1': 'SciKit','Unnamed 0':'XGB','Unnamed 2':'Combined'})
print('R^2 score: {}'.format(metrics.r2_score(np.expm1(y_data),final_predictions)))

print('Mean Squared Log Error: {}'.format(metrics.mean_squared_log_error(np.expm1(y_data),final_predictions)))





print('Mean Absoloute Error: {}'.format(mae(np.expm1(y_data),final_predictions)))

results