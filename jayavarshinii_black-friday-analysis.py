import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

%matplotlib inline

from matplotlib.colors import ListedColormap

import seaborn as sns



from sklearn.model_selection import train_test_split

from math import sqrt

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import Ridge  

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error as mae

df = pd.read_csv('../input/BlackFriday.csv')

df.dtypes[df.dtypes=='object']

df = df.drop_duplicates()

print( df.shape )

df.Product_Category_2.unique()
df.Product_Category_2.fillna(0, inplace=True)
df.Product_Category_2.unique()

df.Product_Category_3.unique()

df.Product_Category_3.fillna(0, inplace=True)
df.select_dtypes(exclude=['object']).isnull().sum()

df.hist(figsize=(25,10)) 

plt.show()
df.describe(include=['object'])

#There are 3623 unique products.

#Most purchases have occured from the age group of 26 to 35. 

plt.figure(figsize=(10,7))

sns.countplot(y='Age', data=df)
plt.figure(figsize=(10,7))

sns.countplot(y='Gender', data=df)
plt.figure(figsize=(10,7))

sns.countplot(y='City_Category', data=df)

plt.figure(figsize=(10,7))

sns.countplot(y='Stay_In_Current_City_Years', data=df)
#Correlation 

df.corr()
plt.figure(figsize=(10,10))

sns.heatmap(df.corr())
gender_gb = df[['Gender', 'Purchase']].groupby('Gender', as_index=False).agg('mean')

sns.barplot(x='Gender', y='Purchase', data=gender_gb)

plt.ylabel('')

plt.xlabel('')

for spine in plt.gca().spines.values():

    spine.set_visible(False)

plt.title('Mean purchase amount by gender', size=14)

plt.show()
plt.figure(figsize=(16, 8))

plt.subplot(121)

sns.countplot(y='Age', data=df, order=sorted(df.Age.unique()))

plt.title('Number of transactions by age group', size=14)

plt.xlabel('')

plt.ylabel('Age Group', size=13)

plt.show()
men = df[df.Gender == 'M']['Occupation'].value_counts(sort=False)

women = df[df.Gender == 'F']['Occupation'].value_counts(sort=False)

pd.DataFrame({'M': men, 'F': women}, index=range(0,21)).plot.bar(stacked=True)

plt.gcf().set_size_inches(10, 4)

plt.title("Count of different occupations in dataset (Separated by gender)", size=14)

plt.legend(loc="upper right")

plt.xlabel('Occupation label', size=13)

plt.ylabel('Count', size=13)

plt.show()
sample_df = df.sample(n=50000,random_state=100)

y = sample_df.Purchase

# Create separate object for input features

X = sample_df.drop('Purchase', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

train_mean = X_train.mean()

train_std = X_train.std()
X_train.describe()
X_test.describe()

#1.Baseline model 

y_train_pred = np.ones(y_train.shape[0])*y_train.mean()

y_pred = np.ones(y_test.shape[0])*y_train.mean()

print("Train Results for Baseline Model:")

print("----------------------------------")

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))

print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("Results for Baseline Model:")

print("----------------------------")

print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))

print("R-squared: ", r2_score(y_test, y_pred))

print("Mean Absolute Error: ", mae(y_test, y_pred))