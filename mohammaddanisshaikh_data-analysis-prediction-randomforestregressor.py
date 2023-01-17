import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
#reading data
data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
print(data.shape)
data.head()
#drop un-necessary columns
data=data.drop(columns=['Unnamed: 0','vin','lot',  'country' , 'condition'],axis=1)
data.describe()
# check for null value
data.isnull().sum()
#check for dublicate entres
duplicate = data[data.duplicated()]
duplicate.shape
data.year.value_counts()
# some row have price and mileage zero, lets filter them 
print(f'Total {data[data["price"]==0]["price"].count()} row have price zero and {data[data["mileage"]==0].mileage.count()} row have mileage zero')

# print('Those row are have price zero\n\n', data[data['price']==0])
# print('\n Those row are have mileage zero\n\n', data[data['mileage']==0])

#remove all row with price zero
data = data[data['price']!=0]
data = data[data['mileage']!=0]

#drop entry year less then 2000
data = data[data['year']>2000]
# check for brands
print(f'All brands in Data are: \n')
for brand in data["brand"].unique():
    print(brand, end=',\t')
# Plotting a Histogram for brands and number of cars
data['brand'].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Number of cars by brand")
plt.ylabel("Number of cars")
plt.xlabel("Brand");
# Plotting a scatter plot for brand and price
fig, ax = plt.subplots(figsize=(30,6))
ax.scatter(data['brand'], data['price'])
ax.set_xlabel('brand')
ax.set_ylabel('Price')
plt.show()
print('Higest price entry \n',data[data.price == data.price.max()])
#ploat bar chart for number of car in each year
data['year'].value_counts().nlargest(10).plot(kind='bar', figsize=(20,5))
plt.title("Number of cars vs year")
plt.ylabel("Number of cars")
plt.xlabel("Year")
#ploat bar chart for number of car vs color
data['color'].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Number of cars vs color")
plt.ylabel("Number of cars")
plt.xlabel("Color");
#Look at prise of model Brand wise 
temp =data.groupby(['brand'])#['model'].unique()
for brand_data in temp:
    brand = brand_data[0]
    print('\n','Brand Name: ',brand)
    data_brand = data[data['brand']==brand]
    for model in data_brand['model'].unique():
        data_brand_model = data_brand[data_brand['model']==model]
        print('\t',brand,'Model:' ,model,'\n\t\t\tMax price: ',data_brand_model.price.max(), '\n\t\t\tmin price: ',data_brand_model.price.min())
##pre prosess data
X = data.iloc[:, [1,2,3,4,5,6,7]].values   # input columns are ['brand', 'model', 'year', 'title_status', 'mileage', 'color', 'state']
y = data.iloc[:, 0].values                 # output columns are ['price']

# Encoding text columns 
print('Input row befor encoding : ',X[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1]=le.fit_transform(X[:,1]) 
X[:,3]=le.fit_transform(X[:,3])
X[:,5]=le.fit_transform(X[:,5])
X[:,6]=le.fit_transform(X[:,6])
X[:,0]=le.fit_transform(X[:,0])

print('Input row after encoding : ',X[0])


# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0   )
regressor.fit( X_train, y_train)

y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
r2_score(y_test,y_pred) 
