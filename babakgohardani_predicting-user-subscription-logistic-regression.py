import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from dateutil import parser # for parsing date, time values

import time 



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression



import os

print(os.listdir("../input"))
dataset = pd.read_csv("../input/subscription-train/train.csv")
dataset.head()
dataset.describe()   # we have 50K rows in our dataset.
dataset.hour.head() # this is when the user first opened the app



# turn the hour column into int from string

dataset.hour = dataset.hour.str.slice(1,3).astype(int)
dataset.hour.head(3)
# we create a second dataset for visualization and drop id and string data

ds = dataset.copy().drop(columns=['user','screen_list', 'enrolled_date', 'first_open', 'enrolled'])
ds.head()
sns.pairplot(ds)
sns.countplot(ds['numscreens'])
plt.hist(ds['age'])
sns.countplot(ds['hour'])
sns.scatterplot(x = 'age', y = 'used_premium_feature',data = ds)
# figuring out the correlation between our numerical features and the enrollment value

ds.corrwith(dataset.enrolled).plot.bar(figsize= (20, 10), fontsize=15, grid=True)
# Correlation Matrix



plt.figure(figsize=(20, 10))

sns.heatmap(ds.corr(), annot= True)
dataset.dtypes
# change data type of first_open from string to date type

dataset['first_open'] = [parser.parse(row_data) for row_data in dataset['first_open']]
dataset['enrolled_date'] = [parser.parse(row_data) if isinstance(row_data, str) else row_data for row_data in dataset['enrolled_date']]
# astype('timedelta64[h]') : this will make the difference into hours

dataset["difference"] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[m]')
print(dataset["difference"].dropna().mean())

print(dataset["difference"].dropna().median())
# anyone who enrolled more than 6 hours after enrolling, wolud be considered not enrolled 

dataset.loc[dataset.difference > 360, 'enrolled'] = 0
# we are not gonna use this dates anymore so we drop them

dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])
dataset = dataset.drop(columns = ['user'])
dataset.head()
top_screens = pd.read_csv("../input/top-screens/top_screens.csv")
top_screens.head(10)
dataset['screen_list'] = dataset.screen_list.astype(str) + ', '
top_screens = top_screens.values  # turn it into a nympy array from a dataframe

ls = top_screens[:,1]

for sc in ls:

    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)

    dataset["screen_list"] = dataset.screen_list.str.replace(sc+',', "")
dataset.head(3)
dataset['other'] = dataset.screen_list.str.count(",")
dataset.head(3)
# now we can drop the screen_list column

dataset = dataset.drop(columns = ['screen_list'])
plt.figure(figsize=(20, 10))

ds_2 = dataset[["Saving1", "Saving2", "Saving2Amount", "Saving1","Saving4","Saving5","Saving6", 

                  "Saving7","Saving8","Saving9","Saving10"]]

sns.heatmap(ds_2.corr(), annot= True)
# these columns are all highly corelated so we can add them all up into one column

savings_screen = ["Saving1", "Saving2", "Saving2Amount", "Saving1","Saving4","Saving5","Saving6", 

                  "Saving7","Saving8","Saving9","Saving10"]
dataset.head()
# add up the values from all columns from savings_screen and then put them in SavingsCount  

dataset['SavingsCount'] = dataset[savings_screen].sum(axis=1)

# next drop all columns of list savings_screen

dataset = dataset.drop(columns=savings_screen)
cm_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container","Credit3Dashboard"]



dataset['CMCount'] = dataset[cm_screens].sum(axis=1)

dataset = dataset.drop(columns=cm_screens)
cc_screens = ["CC1", "CC1Category", "CC3"]



dataset['CCCount'] = dataset[cc_screens].sum(axis=1)

dataset = dataset.drop(columns=cc_screens)
loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]



dataset['LoanCount'] = dataset[loan_screens].sum(axis=1)

dataset = dataset.drop(columns=loan_screens)
dataset.head()
Y = dataset['enrolled']

X = dataset.drop(columns="enrolled")
SC = StandardScaler()

X = SC.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(len(X_train))

print(len(X_test))
# here we have a binary classification (0/1) so we can use a logistic regression classifer to predict the results

# we apply L1 penalty for regularization of our model to prevent overfitting

classifier = LogisticRegression(random_state=0, penalty='l1')

classifier.fit(X_train, Y_train)



y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)

print(cm)
print(classification_report(Y_test, y_pred))
param_grid = {'C':[0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}





grid = GridSearchCV(LogisticRegression(), param_grid, verbose= 4, cv=3, refit=True)

grid.fit(X_train, Y_train)
grid.best_params_
y_predict = grid.predict(X_test)

print(classification_report(Y_test, y_predict))
cm = confusion_matrix(Y_test, y_predict)

print(cm)