# We shall see some examples

import pandas as pd

titanic_data = pd.read_csv('../input/titanic/train.csv')

print("Example of some data objects: Each row represents a data object:")

titanic_data.head(3)

# The attributes or features in the dataset describing each data object or tuple

print(list(titanic_data.columns))
# You may have already guessed nominal attributes in the titanic dataset by seeing the rows above

# Here are the columns which represent nominal attributes

print(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin','Embarked'])

print("You may have noticed that Pclass contains numeric values but is still nominal.")
# The 'Survived' attribute is a binary attribute

print(titanic_data['Survived'].value_counts())

print("Clearly only two classes, 0 and 1, exist for 'Survived' attribute. Again it contains numeric values but is actually binary nominal.")
# In our Titanic dataset, we have following numeric attributes

print(['Age','Fare'],": Numeric attributes")

print("Some values of 'Age' are:")

print(titanic_data['Age'][0:3])

print("Some values of 'Fare' are:")

print(titanic_data['Fare'][0:3])
# In the titanic dataset, 'Pclass' is an ordinal attribute.

print("It is obvious that class 1, 2 and 3 have an order among themselves.")

print(titanic_data['Pclass'].value_counts())

print("Here 3, 2, 1 are Pclass values while numbers in front of them are their frequency in the dataset")
# In our Titanic dataset 'Fare' is continuous and 'Pclass', 'SibSp', 'Survived', etc. are discrete.

print("An example of continous attribute:")

print(titanic_data['Fare'][0:3])

print("An example of discrete attribute:")

print(titanic_data['Pclass'][0:3])
# We shall use predefined mean function to calculate the mean, which employs same mathematical procedure

print("Mean of Fare: ",titanic_data['Fare'].mean())

print("Note:Outliers and distribution are not dealt with for sake of understanding.")
# Let's compute the median for 'Pclass'(ordinal) and 'Age' (numeric)

print("Median for Pclass:", titanic_data['Pclass'].median())

print("Median for Age:", titanic_data['Age'].median())
# We shall see the mode of 'Pclass' and 'Survived'

print("Mode of Pclass:", (titanic_data['Pclass'].mode())[0])

print("Mode of Survived:", (titanic_data['Survived'].mode())[0])
# We will calculate range of 'Age' in Titanic dataset

print("Range of Age:", titanic_data['Age'].max()-titanic_data['Age'].min())
# Lets see the distribution of 'Age' in titanic dataset

import matplotlib.pyplot as plt

import seaborn as sns # seaborn is a popular visualization library

sns.distplot(titanic_data['Age'])

plt.show()

print("The distribution is slightly positivly skewed.")
# Let's plot box plot

# Box plot can also be used for outlier detection. It also shows Interquartile range, median, Q1, Q3, etc.

sns.boxplot(x=titanic_data['Survived'], y=titanic_data['Age'])

plt.show()

print("Box plots show outliers, IQR, Q1, Q3, median and min and max in 1.5xIQR on both sides")
# Kurtosis can be calculated using inbuilt methods

print("Kurtosis of Age:", titanic_data['Age'].kurtosis())

print("Note:- This inbuilt method considers kurtosis of normal distribuiton as 0.0 (Fisher's method)")
# We can easily compute variance and standard deviation of data using inbuilt methods

# We shall use a new dataset which contains several different types of attributes.

hp_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
# This dataset has an attribute 'SalePrice' which contains prices of houses

# We shall calculate the variance and standard deviation of this attribute

# This is custom formulae to calculate variance and standard deviation

import numpy as np

print("Custom formulae to calculate variance and standard deviation.")

n=(hp_data['SalePrice'].shape)[0]

mean = (hp_data['SalePrice'].sum())/n

v = (abs(hp_data['SalePrice']-mean))**2 # This is called broadcasting in pyhton. Read more in documentation

var = sum(v)/n

print("Variance:", var)

print("Standard deviation:" , np.sqrt(var))
# Inbuilt functions can also be used for same purpose.

print("Variance and standard deviation using inbuilt numpy functions.")

print("Variance of SalePrice:",np.var(hp_data['SalePrice']))

print("Standard Deviation of SalePrice:",np.std(hp_data['SalePrice']))
# Distribution plot of 'SalePrice' attribute in house price data

plt.figure(figsize=(8,6))

plt.xlabel("Sale Price")

sns.distplot(hp_data['SalePrice'])

plt.show()

print("It is clear from the figure that distribution is positively skewed.")
# Following is an example of histogram of attribute 'MSZoning' is house-price data

plt.figure(figsize=(8,6))

sns.countplot(hp_data['MSZoning'])

plt.show()
# Here is a scatter plot of 'close/last' and 'open' of macdonald-stock-price dataset

mcd_stock = pd.read_csv('../input/eda-and-cleaning-mcdonald-s-stock-price-data/final.csv')

sns.scatterplot(x='Close/Last', y='Open', data=mcd_stock)

plt.show()

print("The plot shows high positive correlation and there a no possible outliers")
# Let's create a box plot for some attributes

sns.boxplot(y='Age',x='Sex',data=titanic_data)

plt.show()

print("The points above 'male' can be possible outliers and the line in box is the median")