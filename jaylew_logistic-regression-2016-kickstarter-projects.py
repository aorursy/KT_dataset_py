#Exploring and then predicting the outcome of the 2016 dataset using logistic regression

#Importing useful packages, setting seaborn style
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('darkgrid')
#Grab the data, need to update it for kaggle notebook
data = pd.read_csv('../input/ks-projects-201612.csv', encoding='ISO-8859-14')
#Remove unnamed columns which seem to be empty
data.drop(['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
#Remove whitespace at the end of the column names just to make it cleaner
cols = data.columns.tolist()
for entry in np.arange(0,len(cols),1):
    cols[entry] = cols[entry].rstrip()
data.columns = cols
data.columns
data.info()
#Change the numeric columns from objects to floats
data[['goal', 'pledged', 'usd pledged', 'backers']] = data[['goal', 'pledged', 'usd pledged', 'backers']].apply(pd.to_numeric, errors='coerce')
data.info()
#Checking the different strings in 'state' and setting a threshold for legitimacy. Removing any random unhelpful ones. 
data['state'].value_counts() < 100
acc_states = ['failed', 'successful', 'live', 'undefined', 'suspended']
#Making a new dataframe with only rows with legit states
acc_data = data[data['state'].isin(acc_states)]
#Checking the distribution of different states, majority by far is successful/faile
plt.figure(figsize=(12,6))
sns.countplot(x='state', data=acc_data)
#Checking the state split in the different categories. 
plt.figure(figsize=(16,6))
sns.countplot(x='main_category', data=acc_data, hue='state')
plt.legend(loc='upper center')
plt.tight_layout
#Checking the distribution of currencies. Unsurprisingly US dominated.
plt.figure(figsize=(16,6))
sns.countplot('currency', data=acc_data, hue='state')
plt.legend(loc='upper center')
plt.tight_layout
#Creating a df containing only successful or failed data. Want to be able to predict either failure or success.
successfaildf = acc_data[(acc_data['state'] == 'successful') | (acc_data['state'] == 'failed')]
#Confirming these are the only 2 states
sns.countplot(x='state', data=successfaildf)
successfaildf.info()
#Converting the two columns containing dates into datetime objects
successfaildf[['deadline', 'launched']] = successfaildf[['deadline', 'launched']].apply(pd.to_datetime, errors='coerce', infer_datetime_format=True)
#Creating a new column in the df containing a timedelta object with the length of the kickstarter
import datetime
successfaildf['length'] = successfaildf['deadline'] - successfaildf['launched']
#Couldn't figure out how to do this inline for some reason, making a daysfinder fcn to extract the number of days
def daysfinder(timedelta):
    numdays = timedelta.days
    return numdays
#Converting the new length column into just a column of ints pertaining to the number of days the kickstarter went on.
successfaildf['length'] = successfaildf['length'].apply(lambda x: daysfinder(x))
successfaildf.info()
#Removing all the kickstarters with a half million dollar or more goal. ~2000 out of ~300,000.
successfaildf = successfaildf[successfaildf['goal'] < 5000000]
plt.figure(figsize=(16,6))
sns.countplot(x='main_category', data=successfaildf, hue='state')
plt.legend(loc='upper center')
plt.tight_layout
#Creating dummies for successful or not, dropping the column for not since successful yes/no is all we need
dummydf = pd.get_dummies(data=successfaildf['state'], drop_first=True)
#Replacing the state column with the dummy column
successfaildf['state'] = dummydf
#Still need to replace main_category with some kind of placeholders. 'Category' is too wide, name/pledges/country will be ignored
#Looking for the unique entries in main category
successfaildf['main_category'].unique()
category_dict = {
    'Publishing':1,
    'Film & Video':2,
    'Music':3,
    'Food':4,
    'Crafts':5,
    'Games':6,
    'Design':7,
    'Comics':8,
    'Fashion':9,
    'Theater':10,
    'Art':11,
    'Photography':12,
    'Technology':13,
    'Dance':14,
    'Journalism':15
}
#Replacing the strings with their new numeric placeholders.
successfaildf['main_category'] = successfaildf['main_category'].replace(category_dict)
#Defining the features to be used and the target column.
features = ['main_category', 'goal', 'backers', 'length']
target = ['state']
#Attempting to predict the outcome using logistic regression begins here
#Importing train/test split
from sklearn.model_selection import train_test_split
#Splitting the data into train and test sets.
X = successfaildf[features]
y = successfaildf[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.linear_model import LogisticRegression
#Initializing the logistic regressor
regressor = LogisticRegression()
#Fitting the regressor to the training data, kicks out a warning but works fine
regressor.fit(X_train, y_train)
#Creating the predictions
predictions = regressor.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
#Checking the metrics
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
#Not bad? Could look at the 2018 data as well and see how it fares.