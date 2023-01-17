import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
google_data = pd.read_csv('../input/googleplaystore.csv')
google_data.head()         #Inspecting the first 5 rows
google_data.shape
google_data.describe()       # Summary Statistics
google_data.boxplot()
google_data.hist()
google_data.info()
google_data.isnull()
# Count the number of missing values in each column

google_data.isnull().sum()
google_data[google_data.Rating > 5]
google_data.drop([10472],inplace=True)
google_data[10470:10475]
google_data.boxplot()
google_data.hist()
threshold = len(google_data)* 0.1

threshold
google_data.dropna(thresh=threshold, axis=1, inplace=True)
print(google_data.isnull().sum())
#Define a function impute_median

def impute_median(series):

    return series.fillna(series.median())
google_data.Rating = google_data['Rating'].transform(impute_median) 
#count the number of null values in each column

google_data.isnull().sum()
# modes of categorical values

print(google_data['Type'].mode())

print(google_data['Current Ver'].mode())

print(google_data['Android Ver'].mode())
# Fill the missing categorical values with mode

google_data['Type'].fillna(str(google_data['Type'].mode().values[0]), inplace=True)

google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]), inplace=True)

google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]), inplace=True)
#count the number of null values in each column

google_data.isnull().sum()
### Let's convert Price, Reviews and Ratings into Numerical Values

google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

google_data['Price'] = google_data['Price'].apply(lambda x: float(x))

google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors='coerce')
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))

google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))

google_data['Installs'] = google_data['Installs'].apply(lambda x: float(x))
google_data.head(10) 
google_data.describe()
grp = google_data.groupby('Category')

x = grp['Rating'].agg(np.mean)

y = grp['Price'].agg(np.sum)

z = grp['Reviews'].agg(np.mean)

print(x)

print(y)

print(z)
plt.figure(figsize=(12,5))

plt.plot(x, "ro", color='g')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(16,5))

plt.plot(x,'ro', color='r')

plt.xticks(rotation=90)

plt.title('Category wise Rating')

plt.xlabel('Categories-->')

plt.ylabel('Rating-->')

plt.show()
plt.figure(figsize=(16,5))

plt.plot(y,'r--', color='b')

plt.xticks(rotation=90)

plt.title('Category wise Pricing')

plt.xlabel('Categories-->')

plt.ylabel('Prices-->')

plt.show()
plt.figure(figsize=(16,5))

plt.plot(z,'bs', color='g')

plt.xticks(rotation=90)

plt.title('Category wise Reviews')

plt.xlabel('Categories-->')

plt.ylabel('Reviews-->')

plt.show()