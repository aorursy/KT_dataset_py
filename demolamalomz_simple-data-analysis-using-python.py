%matplotlib notebook
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

#The code below loads the dataset
df = pd.read_csv("../input/Lemonade2016.csv", sep =",", header= 0)
df.head() #pulls out first 5 sample of the data
df.dtypes #check data types
df.Date = pd.to_datetime(df.Date, format= "%m/%d/%Y") 
#Rerun dtype to see if changes on the dataframe
df.dtypes
#Since the data is relatively small, we can view the table and see missing value in each column as "True" 
df.isnull()
#Filling in the Date Column Missing value
av_date = df.Date[9]
df.Date = df.Date.fillna(av_date)


#Filling in the Leaflets Column Missing value
lf = df.Leaflets.mean()
df.Leaflets = df.Leaflets.fillna(lf)
#Check effected changes and round the leaflet column decimal point to 1
df.Leaflets = df.Leaflets.round(1)
df
#Sales Column
df["Sales"] = df.Lemon + df.Orange

#Revenue Column
df["Revenue"] = df.Sales * df.Price
#Check effected changes
df
df.Revenue.describe()
df.Revenue.sum()
df.Temperature.nlargest(10)
#This checks the correlation between Temperature of the day and Price
df['Temperature'].corr(df.Price)
#This checks the correlation between leaflets shared and Price
df['Leaflets'].corr(df.Price)
#This checks the correlation between Temperature of the day and Sales
df['Temperature'].corr(df.Sales)
#This checks the correlation between Leaflets shared and Sales
df['Leaflets'].corr(df.Sales)
plt.figure(1)
x = df.Date
y = df.Revenue
plt.plot(x,y)
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Shuperu Cocktails Revenue')
plt.show()
plt.figure(2)
x = df.Leaflets
y = df.Sales
plt.scatter(x, y)
plt.xlabel('Leaflets')
plt.ylabel('Sales')
plt.title('Correlation')
plt.show()

#Treadline
td = np.polyfit(x,y, 1)
treadline = np.poly1d(td)
plt.plot(x, treadline(x))
plt.figure(3)
plt.hist(df.Revenue, bins= 10)
plt.title('Revenue Distribution')
plt.show()
