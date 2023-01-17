import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
google_data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
type(google_data)
google_data.head(10)
google_data.tail(1)
google_data.shape
google_data.describe()    #Rating is shown as it is the only numerical column in the dataset
google_data.boxplot()
google_data.hist()
google_data.info()
google_data.isnull()
google_data.isnull().sum()
google_data[google_data.Rating > 5]
google_data.drop([10472],inplace=True)     #The Rating value is overrated and is false
google_data[10470:10474]
google_data.boxplot()
google_data.hist()
#writing function to fill in NA with median value as we have numerical data
def assign_median(series):
    return series.fillna(series.median())  #pandas function running on series
google_data.Rating = google_data['Rating'].transform(assign_median)    #Passing each value from Rating to Transform function
google_data.isnull().sum()
#Using mode to assign for categorical values. Examine from above list Type, Current Ver & Android Ver have nulls
#Check if any column has bimodal value 

print(google_data['Type'].mode())
print(google_data['Current Ver'].mode())
print(google_data['Android Ver'].mode())
google_data['Type'].mode().values[0]
google_data['Type'].fillna(str(google_data['Type'].mode().values[0]),inplace=True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]),inplace=True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]),inplace=True)

google_data.isnull().sum()
google_data.dtypes
#Let's convert price, reviews and installs into numerical values

google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$','') if '$' in str(x) else str(x)) #Removing $
google_data['Price'] = google_data['Price'].apply(lambda x: float(x))
google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors='coerce') #coerce means to ignore
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x)) #removing +
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',','') if ',' in str(x) else str(x)) 
google_data['Installs'] = google_data['Installs'].apply(lambda x:float(x))
google_data.head(10)
google_data.describe()
grp = google_data.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)
plt.figure(figsize=(16,5))
plt.plot(x,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Category Wise Rating')
plt.xlabel('Categories -->')
plt.ylabel('Rating -->')
plt.show()
plt.figure(figsize=(16,5))
plt.plot(y,'r--',color='b')
plt.xticks(rotation=90)
plt.title('Category Wise Pricing')
plt.xlabel('Categories -->')
plt.ylabel('Prices -->')
plt.show()
plt.figure(figsize=(16,5))
plt.plot(z,'bs',color='g')
plt.xticks(rotation=90)
plt.title('Category Wise Reviews')
plt.xlabel('Categories -->')
plt.ylabel('Reviews -->')
plt.show()