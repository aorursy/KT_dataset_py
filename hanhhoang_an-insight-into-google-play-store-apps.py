# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt

import sklearn



from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeRegressor 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/googleplaystore.csv')

reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')

data.head()

data.info()
data["Category"].unique()

# found out there is one line which has category of 1.9, the line is distorted, hence removed

data = data[data.Category != "1.9"]

# Installs need to be removed + to be able to be summed up.

def clean(x):

    x = str(x).replace("+", "").replace(",", "")

    return int(x)

data["Installs"] = data["Installs"].apply(clean)
# get the year

data['Last Updated'] = data['Last Updated'].str[-4:]

data['Last Updated'].apply(lambda x: int(x))

data['Last Updated'].describe()
# convert Size to MB, fill varies with devices by mean of the column.

def convert_to_MB(x):

    if "k" in x:

        x = round(float(x.replace("k", ""))/1000, 2)

    elif "M" in x:

        x = round(float(x.replace("M", "")), 2)

    elif x == "Varies with device":

        x = None # "varies with devices" set default to 10MB

    #return round(x, 1)

    return x

data["Size_MB"] = data["Size"].apply(convert_to_MB)

data["Size_MB"].fillna((data["Size_MB"].mean()), inplace=True)
# clean Price

data['Price'] = data['Price'].apply(lambda x: float(str(x).replace('$','')))

# convert reviews to float

data['Reviews'] = data['Reviews'].apply(lambda x: float(x))

data['Reviews'].head()
plt.figure(figsize = (15, 6))

plt.title('Number of installs per category')

install_per_category = data.groupby("Category")['Installs'].sum().reset_index()

ax = sns.barplot(x="Category", y="Installs", data=install_per_category)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

plt.show()
plt.figure(figsize=(15,6))

plt.title('Number of apps per category')

ax = sns.countplot(x='Category',data = data)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

plt.show()
plt.figure(figsize = (15, 6))

plt.title('Average Size per Category')

size_per_category = data.groupby("Category")['Size_MB'].mean().reset_index()

ax = sns.barplot(x="Category", y="Size_MB", data=size_per_category)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, ha = 'right')

plt.show()
data[data['Category']=="FAMILY"].head(10)
plt.figure(figsize=(15,6))

plt.title('Free Apps vs Paid Apps')

ax = sns.countplot(x='Type',data = data)

plt.show()
print("Percentage of free apps: ", round(len(data[data['Type'] == "Free"])/len(data)*100, 2))
paid_apps = data[data['Price']!=0]

plt.figure(figsize=(15,6))

plt.title('Distribution of app prices')

ax = sns.distplot(paid_apps['Price'])

plt.show()
print("Apps with price less than 10: ", round(len(paid_apps[paid_apps['Price'] <= 10])/len(paid_apps)*100,2))

print("Apps with price more than 100: ", round(len(paid_apps[paid_apps['Price'] >= 100])/len(paid_apps)*100,2))
paid_apps[paid_apps['Price'] >= 300]
plt.figure(figsize=(15,6))

plt.title('Last Updated')

ax = sns.countplot(x='Last Updated', data=data)

plt.show()
print("% Last update in 2018 covers: ", round(len(data[data["Last Updated"]=="2018"])/len(data["Last Updated"])*100))
data["Android Ver"].unique()
data['Current Ver'].unique()
print(data.count(axis=0))
df = data.dropna(axis=0, how='any')[["Rating", "Reviews", "Size_MB", "Installs", "Type", "Price", "Last Updated"]]

df.head()
plt.figure(figsize=(15,6))

ax = sns.pairplot(df, hue="Type")

plt.show()
df = df.loc[:, df.columns != "Type"]

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Installs'], df.Installs, test_size=0.33, random_state=42)

X_train.head()


clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print(confusion_matrix(y_test, y_pred))

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

# print(classification_report(y_test, y_pred))

# Compute and plot the RMSE

RMSE_0 = np.sqrt(np.sum(((y_test-y_pred)**2)/len(y_test)))

print("Root Mean Square Error is: ", RMSE_0)


# Fit regression model and predict

regr_1 = DecisionTreeRegressor(criterion="mse", max_depth=2)

regr_2 = DecisionTreeRegressor(criterion="mse", max_depth=5)

regr_1.fit(X_train, y_train)

regr_2.fit(X_train, y_train)

y_pred_1 = regr_1.predict(X_test)

y_pred_2 = regr_2.predict(X_test)

y_pred_1

# Compute and plot the RMSE

RMSE_1 = np.sqrt(np.sum(((y_test-y_pred_1)**2)/len(y_test)))

RMSE_2 = np.sqrt(np.sum(((y_test-y_pred_2)**2)/len(y_test)))

print("Root Mean Square Error 1 is: ", RMSE_1)

print("Root Mean Square Error 2 is: ", RMSE_2)
plt.figure(figsize=(15,6))

plt.title('Rating Distribution')

ax = sns.countplot(x='Rating', data=data)

plt.show()
data_review = data.merge(reviews, how = 'inner', on = 'App')

data_review.head()