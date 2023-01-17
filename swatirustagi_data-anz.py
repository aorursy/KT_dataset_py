import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



%matplotlib inline

warnings.simplefilter("ignore")



print("Libraries load successfully!!!")
data = pd.read_csv("../input/anz-synthesised-transaction-dataset/anz.csv")

data.head(2)
print("Rows and Columns in the given dataset is", data.shape[0], "and", data.shape[1], "respectively.")
# Missing values

def missing_values_table(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]))   

        print("There are " + str(mis_val_table_ren_columns.shape[0])+" columns that have missing values.")

        return mis_val_table_ren_columns
missing_values_table(data).style.background_gradient(cmap='vlag_r')
data.info()
#Converting the date column to pandas timestamp format

data['date'] = pd.to_datetime(data['date'])

print(data['date'].dtype)

print(type(data['date'][0]))

data['date'].head()
#getting the weekday out of date column

data['week_day'] = data['date'].dt.day_name()

data['week_day'].head()
#extracting the month out of date column

data['month'] = data['date'].dt.month_name()

data['month'].head()
# Plotting the correlation heatmap 

sns.heatmap(data.corr() ,vmax=.3 ,annot=True, center=0, cmap="nipy_spectral", square=True, linewidths=.5)
oct_amt = (data['month'] == 'October')

print("Mean transaction amount in the month of October is", data.loc[oct_amt , 'amount'].mean())

print("Maximum transaction amount in the month of October is", data.loc[oct_amt , 'amount'].max())

print("Minimum transaction amount in the month of October is", data.loc[oct_amt , 'amount'].min())
sep_amt = (data['month'] == 'September')

print("Mean transaction amount in the month of September is", data.loc[sep_amt , 'amount'].mean())

print("Maximum transaction amount in the month of September is", data.loc[sep_amt , 'amount'].max())

print("Minimum transaction amount in the month of September is", data.loc[sep_amt , 'amount'].min())
aug_amt = (data['month'] == 'August')

print("Mean transaction amount in the month of August is" ,data.loc[aug_amt , 'amount'].mean())

print("Maximum transaction amount in the month of August is", data.loc[aug_amt , 'amount'].max())

print("Minimum transaction amount in the month of August is", data.loc[aug_amt , 'amount'].min())
#checking the count of month wise transaction 

print(data['month'].value_counts())

explode=(0.1,0.05,0.05)

data['month'].value_counts().plot.pie(autopct='%1.1f%%', startangle=60)

plt.show()
print(data['gender'].value_counts())

print("*"*20)

print(((data['gender'].value_counts() / len(data['gender'])).round(3)*100))

sns.countplot(data['gender'])

plt.show()
print(data['card_present_flag'].value_counts())

print("*"*20)

print(((data['card_present_flag'].value_counts() / len(data['card_present_flag'])).round(3)*100))

sns.countplot(data['card_present_flag'])

plt.show()
plt.figure(figsize=(8,8))

print(data['txn_description'].value_counts())

data['txn_description'].value_counts().plot.pie(autopct='%1.1f%%', startangle=60)
plt.figure(figsize=(7,7))

print(data['week_day'].value_counts())

data['week_day'].value_counts().plot.pie(autopct='%1.1f%%', startangle=60)
plt.figure(figsize=(10,10))

print(data['merchant_state'].value_counts())

explode=(0.1,0.05,0.05)

data['merchant_state'].value_counts().plot.pie(autopct='%1.1f%%', startangle=60)
print(data['movement'].value_counts())

print("*"*20)

print(((data['movement'].value_counts() / len(data['movement'])).round(3)*100))

sns.countplot(x='movement' , data=data)
print(data['gender'].groupby(data['card_present_flag']).value_counts())

print("*"*20)

print(((data['gender'].groupby(data['card_present_flag']).value_counts() / len(data['gender'])).round(3)*100))

plt.figure(figsize=(10,8))

sns.countplot(x='card_present_flag' ,hue='gender', data=data)
print(data['gender'].groupby(data['month']).value_counts())

print("*"*20)

print(((data['gender'].groupby(data['month']).value_counts() / len(data['month'])).round(3)*100))

plt.figure(figsize=(10,8))

sns.countplot(x='month' ,hue='gender', data=data)
plt.figure(figsize=(8,8))

fig = sns.countplot(x = "merchant_state", hue = "gender", data = data)

total = len(data)

for p in fig.patches:

    height = p.get_height()

    fig.text(p.get_x()+p.get_width()/2., height + 3, '{:.1%}'.format(height/total),ha="center")

plt.title("Week Day wise Gender Transaction")

plt.show()
plt.figure(figsize=(8,8))

fig = sns.countplot(x = "week_day", hue = "gender", data = data)

total = len(data)

for p in fig.patches:

    height = p.get_height()

    fig.text(p.get_x()+p.get_width()/2., height + 3, '{:.1%}'.format(height/total),ha="center")

plt.title("Week Day wise Gender Transaction")

plt.show()
print(data['movement'].groupby(data["gender"]).value_counts())

print("*"*20)

print(((data['movement'].groupby(data['gender']).value_counts() / len(data['gender'])).round(3)*100))

plt.figure(figsize=(10,8))

sns.countplot(x='movement' ,hue='gender', data=data)
print(data['movement'].groupby(data["txn_description"]).value_counts())

print("*"*20)

print(((data['movement'].groupby(data['txn_description']).value_counts() / len(data['txn_description'])).round(3)*100))

plt.figure(figsize=(10,8))

sns.countplot(x='movement' ,hue='txn_description', data=data)
print(data['movement'].groupby(data["merchant_state"]).value_counts(sort=True))

print("*"*20)

print(((data['movement'].groupby(data['merchant_state']).value_counts(sort=True) / len(data['merchant_state'])).round(3)*100))

plt.figure(figsize=(10,8))

sns.countplot(x='movement' ,hue='merchant_state', data=data)
plt.figure(figsize=(10,7))

sns.distplot(data['age']);
# Figuring out which age group has more balance.

plt.figure(figsize=(10,7))

sns.lineplot(x='age' , y='balance' , data=data)
# Figuring out which age group has transacted more

plt.figure(figsize=(10,7))

sns.lineplot(x='age' , y='amount' , data=data)
data['date'].value_counts(sort=True).plot(kind='line',linewidth=2.5,linestyle='-',marker='o',figsize=(20, 10))

plt.xlabel('\nDates')

plt.ylabel('\nFrequency')

plt.title('Frequency of Tranaction made per day',fontdict = {'fontsize' : 10})

plt.legend()

plt.grid(True)

plt.show()
data.columns
data['txn_description'].unique()
salaries = data[data["txn_description"] == "PAY/SALARY"].groupby("customer_id").mean()

salaries.head()
print(data['age'].corr(data['balance']))

print(data['age'].corr(data['amount']))
fig, ax = plt.subplots(1, 2)



sns.scatterplot(x=data.age, y=data.balance, ax = ax[0])

sns.scatterplot(x=data.age, y=data.amount, ax= ax[1])



fig.show()
sal =[]

for customer_id in data['customer_id']:

    sal.append(int(salaries.loc[customer_id]['amount'].sum()))

data['annual_salary'] = sal

data.head()
print("Rows and Columns in the given dataset is", data.shape[0], "and", data.shape[1], "respectively.")
salary = data[data['txn_description'] == 'PAY/SALARY']

salary.head()
print("Rows and Columns in the given dataset is", salary.shape[0], "and", salary.shape[1], "respectively.")
missing_values_table(salary).style.background_gradient(cmap='vlag_r')
salary.drop(['card_present_flag','merchant_id', "merchant_suburb","merchant_state", 'merchant_long_lat'], axis = 1, inplace = True)
print(salary.country.unique())

print(salary.currency.unique())

print(salary.movement.unique())

print(salary.bpay_biller_code.unique())

print(salary.status.unique())
salary.drop(['country','currency', "bpay_biller_code","movement", 'status'], axis = 1, inplace = True)
print(salary.account.unique())

print(salary.long_lat.unique())

print(salary.txn_description.unique())

print(salary.merchant_code.unique())

print(salary.first_name.unique())
salary.drop(['account','long_lat', "txn_description","merchant_code", 'first_name'], axis = 1, inplace = True)
salary.columns
print("Rows and Columns in the given dataset is", salary.shape[0], "and", salary.shape[1], "respectively.")
salary.head(2)
salary.drop(['extraction', 'transaction_id'], axis = 1, inplace = True)
salary.head(2)
salary.customer_id.nunique()
ann_sal = salary.groupby(['customer_id','month', 'week_day'])['annual_salary'].nunique()

print(ann_sal[ann_sal>1])
salary.drop(['customer_id','month', 'week_day'], axis = 1, inplace = True)

salary.head(2)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor
salary['gender'] = pd.get_dummies(salary['gender'], drop_first=True)
salary
# Plotting the correlation heatmap 

sns.heatmap(salary.corr() ,vmax=.3 ,annot=True, center=0, cmap="nipy_spectral", square=True, linewidths=.5)
salary.head(2)
X = salary.drop(["date",'annual_salary'],axis=1)

y = salary['annual_salary']
X.shape, y.shape
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3)
X_train
lr = LinearRegression()

lr.fit(X_train, y_train) # Fit the model

y_pred_train_lr = lr.predict(X_train) #train model prediction

print("Model accuracy on Train Data", (lr.score(X_train , y_train)*100)) # Model Score on train data 

y_pred_lr = lr.predict(X_test) # Making predictions

print("Model accuracy on Train Data", lr.score(X_test , y_test)*100) # Model Score on test data 
dtr = DecisionTreeRegressor()

dtr.fit(X_train, y_train) # Fit the model

y_pred_train_dtr = dtr.predict(X_train) #train model prediction

print("Model accuracy on Train Data", (dtr.score(X_train , y_train)*100)) # Model Score on train data 

y_pred_dtr = dtr.predict(X_test) # Making predictions

print("Model accuracy on Train Data", dtr.score(X_test , y_test)*100) # Model Score on test data 