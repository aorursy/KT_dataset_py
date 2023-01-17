# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')

df.head()
# Investigate the number of observations and columns.



df.shape
# Are there any columns that contain missing information?



df.isnull().sum()
# Check the data types of each column.



df.dtypes
# Date should be a Datetime type, not object.



df['Date'] = pd.to_datetime(df['Date'])
df.dtypes
# How many different ports are there? Which are the most popular? Least popular?



print(df['Port Name'].nunique())

print(df['Port Name'].value_counts())
# What's the popular method of getting inside the US?



print("There are " + str(df['Measure'].nunique()) + " different ways to get inside the US.")

print(df['Measure'].value_counts())
# How many people entered the US since January 1996?



df['Value'].sum()
# Extracting year and month data.



df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df.head()
import matplotlib.pyplot as plt

import seaborn as sns
# Which border is exposed more?



df.groupby('Border')['Value'].sum().plot(kind = 'bar')

plt.title('Total Number of Border Crossings')
# Let's visualize the trend of people coming into the US as the years go by.



plt.figure(figsize = (20,10))

yearly_values = df.groupby('Year')['Value'].sum().reset_index()

sns.barplot(x = 'Year', y = 'Value', data = yearly_values)

plt.title('Yearly Trend')
# Let's visualize the average number of people entering US based on years.



df.groupby('Year')['Value'].mean().plot(kind = 'bar')

plt.title('Average Count per Year')
# Which method of transportation is the highest?



df.groupby('Measure')['Value'].sum().sort_values(ascending = False).plot(kind = 'bar')

plt.title('Popular Methods of Transportation')
# Investigate if months affect migration rate.



ax = plt.subplot()

df.groupby('Month')['Value'].sum().plot(kind = 'bar')

ax.set_xticks(range(0,12))

ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.title('People Entering the US by Month')
# Which state is most exposed when entering the US?



df.groupby('State')['Value'].sum().sort_values(ascending = False).plot(kind = 'bar')

plt.title('States Where People Enter the US')
# Is there a difference of method of transportation when people travel from Canada or Mexico?



plt.figure(figsize = (20,14))

can_mex = df.groupby(['Border', 'Measure'])['Value'].sum().reset_index()

sns.barplot(x = 'Measure', y = 'Value', hue = 'Border', data = can_mex)
# Let's convert categorical columns into numeric columns with One Hot Encoding.



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



for col in df:

    if df[col].dtypes == 'object':

        df[col] = le.fit_transform(df[col])

        

df.head()
# Split data into train and test sets.



from sklearn.model_selection import train_test_split



X = df[['Port Name', 'State', 'Border', 'Measure', 'Year']]

y = df['Value']

X_numeric = pd.get_dummies(X, columns = ['Port Name', 'State', 'Border', 'Measure'])

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size = 0.2, random_state = 42)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Implement linear regression model.



from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
# Investigate R-Squared and MSE.



from sklearn.metrics import r2_score, mean_squared_error



r_square = r2_score(y_test, y_pred)

MSE = mean_squared_error(y_test, y_pred)

r_square, MSE