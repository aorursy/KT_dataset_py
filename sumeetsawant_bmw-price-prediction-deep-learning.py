
import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')
import os
%matplotlib inline 

#Importing Deep Learning Modules 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#import the csv file as a pandas dataframe 
data = pd.read_csv('/kaggle/input/bmw-pricing-challenge/bmw_pricing_challenge.csv')
data.head()
data.shape

# We have 4843 rows and 18 columns 
# We have no missing values in any of the columns 
data.isnull().sum()
data.describe()

# We see that the range of price variable is pretty large but most of the cars sold lie between 10,000 to 19000 price range 
#The minimum value  of mileage does not makes sense .Inter quartile for mileage is 100,000 to 175,000
# In engine we have sold cars having no engine to have 423HP cars 

#Checking the Data types for each column 
data.dtypes
# All data types correct apart from sold_at and registration_data they need to be datatime 

data['registration_date']=pd.to_datetime(data['registration_date'])
data['sold_at']=pd.to_datetime(data['sold_at'])

data[data['mileage']<0]
#correcting the mileage value 

data.set_value(2938, 'mileage', 64)

data[data['mileage']==64]
plt.figure(figsize=(8,8))
base_color=sns.color_palette()[8]
sns.countplot(data=data, x='fuel',color=base_color)
plt.title('Type of Fuel',fontsize=30)
plt.ylabel('Count',fontsize=30)
#plt.xlabel(fontsize=40)

locs, labels = plt.xticks()


# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = data['fuel'].value_counts()[label.get_text()]
    pct_string = '{:}'.format(count)

    # print the annotation just below the top of the bar
    plt.text(loc, count+10, pct_string, ha = 'center', color = 'black',fontsize=20)
plt.figure(figsize=(10,10))
sns.boxplot(data=data,x='fuel',y='price',color=base_color,order=data.fuel.value_counts().index);
plt.title('Fuel',fontsize=30)
plt.ylabel('Price',fontsize=30)
plt.xticks(fontsize=20,rotation=90)
plt.figure(figsize=(8,8))
base_color=sns.color_palette()[8]
sns.countplot(data=data, x='paint_color',color=base_color,order=data.paint_color.value_counts().index)
plt.title('Paint Color',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.xticks(rotation=90,fontsize=20)

locs, labels = plt.xticks()


# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = data['paint_color'].value_counts()[label.get_text()]
    pct_string = '{:}'.format(count)

    # print the annotation just below the top of the bar
    plt.text(loc, count+10, pct_string, ha = 'center', color = 'black',fontsize=20)
#Let see how the color is related to price in the resale market . THe data is has more points for black , grey and blue

plt.figure(figsize=(10,10));
sns.violinplot(data=data,x='paint_color',y='price',inner='quartile',color=base_color);
plt.ylabel('Price',fontsize=30);
#plt.xlabel('Paint Color',fontsize=30);
plt.title("Paint Color",fontsize=30);
plt.xticks(rotation=90,fontsize=20);

plt.figure(figsize=(8,8))
base_color=sns.color_palette()[8]
sns.countplot(data=data, x='car_type',color=base_color,order=data.car_type.value_counts().index)
plt.title('Car Type',fontsize=30)
plt.ylabel('Count',fontsize=30)
plt.xticks(rotation=90,fontsize=20)

locs, labels = plt.xticks()


# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = data['car_type'].value_counts()[label.get_text()]
    pct_string = '{:}'.format(count)

    # print the annotation just below the top of the bar
    plt.text(loc, count+10, pct_string, ha = 'center', color = 'black',fontsize=20)
plt.figure(figsize=(10,10))
sns.boxplot(data=data,x='car_type',y='price',color=base_color,order=data.car_type.value_counts().index);
plt.title('Car Type',fontsize=30)
plt.ylabel('Price',fontsize=30)
plt.xticks(fontsize=20,rotation=90)
plt.figure(figsize=(10,10))
sns.scatterplot(data=data,x='car_type',y='fuel',alpha=0.2,x_jitter=0.2);


#Creating a colum called as registration year 
data['registration_year']=data['registration_date'].dt.year
plt.figure(figsize=(10,10));
sns.countplot(data=data,x='registration_year',color=base_color);
plt.xticks(rotation=90);
plt.xlabel('Registration Year',fontsize=20);
plt.ylabel('Count',fontsize=20);

data['sold_at'].dt.year.value_counts()
plt.figure(figsize=(10,10))
sns.scatterplot(data=data.sample(2000),x='registration_year',y='price',hue='car_type');
data.dtypes

plt.figure(figsize=(10,15));
#sns.regplot(x="car_type",y="registration_date",data=data)
sns.scatterplot(data=data,x='car_type',y='registration_year',alpha=0.3);
plt.xticks(rotation=90,fontsize=20);
plt.xlabel('Car Type',fontsize=20);
plt.xlabel('Registration Year',fontsize=20);
plt.title('Cart_type vs Registration Year',fontsize=20);
 
data['vechile_days']=data['sold_at']-data['registration_date']
data['year_diff']=data['sold_at'].dt.year-data['registration_year']
plt.figure(figsize=(20,10))
color=sns.color_palette()[0]
sns.countplot(data=data,x='model_key',color=color,order=data.model_key.value_counts().index);
plt.xticks(rotation=90);
plt.title('Count BMW models');

locs, labels = plt.xticks() # get the current tick locations and labels

# add annotations


# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = data.model_key.value_counts()[label.get_text()]
    pct_string = '{:}'.format(count)

    # print the annotation just below the top of the bar
    plt.text(loc, count+10, pct_string, ha = 'center', color = 'black')
plt.figure(figsize=(10,10))
color=sns.color_palette()[0]
sns.countplot(data=data,x='car_type',color=color,order=data.car_type.value_counts().index);
plt.xticks(rotation=90,fontsize=20);
plt.title('Type BMW models',fontsize=20);
plt.ylabel('Count',fontsize=20);
plt.xlabel("CarType")
plt.hist(data=data,x='engine_power',bins=5);
plt.xlabel('Engine Capacity',fontsize=20);
plt.title('Histogram of Engine Capacity');
plt.ylabel('Number of Engines')
plt.figure(figsize=(10,10))
bin_edges = 10 ** np.arange(np.log10(data.price.min()), np.log10(data.price.max())+0.1, 0.1);
plt.hist(data=data,x='price',bins=bin_edges);
plt.xscale('log');
tick_locs = [10, 30, 100, 300, 1000, 3000,10000,130000];
ticks=np.arange(100,1300,100);
plt.xticks(tick_locs, tick_locs);
plt.yticks(ticks,ticks)
plt.xlabel('Price cars sold at');
plt.title('Price Distribution');
plt.ylabel('Count');
data.head()
# Here I am only plotting a subsample of the data ( 1500 points ) to prevent over crowding . But we can see a some what negative trend in price as mileage increase 

plt.figure(figsize=(10,10));
sns.scatterplot(data=data.sample(1500),x='mileage',y='price',alpha=0.8,hue='car_type');
plt.xlim(0,500000)
plt.ylim(0,100000)
plt.xlabel('Mileage',fontsize=20);
plt.ylabel('Price',fontsize=20);
plt.title('Mileage vs Price vs Car_type',fontsize=20);
plt.figure(figsize=(10,5))
plt.scatter(data = data.sample(1500), x = 'mileage', y = 'price', c = 'registration_year',cmap='viridis',alpha=0.5)
plt.colorbar();
plt.xlim(0,500000)
plt.ylim(0,100000)
plt.xlabel('Mileage',fontsize=20);
plt.ylabel('Price',fontsize=20);
plt.title('Mileage vs Price vs Year',fontsize=20);
plt.figure(figsize=(10,10))
var=['mileage','engine_power','registration_year','price']
sns.heatmap(data[var].corr(),annot=True,cmap='viridis');
plt.xticks(fontsize=20,rotation=90);
plt.yticks(fontsize=20,rotation=45);

# Fitting a neural network to only 8 boolean features 

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import r2_score


baseline=data[['feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','price']]


#Casting True/Fasle to 1/0 
baseline['feature_1']=baseline['feature_1'].astype('int')
baseline['feature_2']=baseline['feature_2'].astype('int')
baseline['feature_3']=baseline['feature_3'].astype('int')
baseline['feature_4']=baseline['feature_4'].astype('int')
baseline['feature_5']=baseline['feature_5'].astype('int')
baseline['feature_6']=baseline['feature_6'].astype('int')
baseline['feature_7']=baseline['feature_7'].astype('int')
baseline['feature_8']=baseline['feature_8'].astype('int')

columns_names=baseline.columns


#Scaling the dataframe 

sc = StandardScaler()
baseline = sc.fit_transform(baseline[columns_names])


#converting it back to dataframe 
baseline_scaled=pd.DataFrame(baseline,columns=columns_names)



X=baseline_scaled.drop(['price'],axis=1).values
y=baseline_scaled['price'].values.reshape(len(X),1)


print(X.shape,y.shape)



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)

#Baseline Model1

# Initialising the ANN
model_baseline = Sequential()

# Adding the input layer and the first hidden layer
model_baseline.add(Dense(units = 5, kernel_initializer = 'he_normal', activation = 'relu', input_dim = X.shape[1]))

# Adding the second hidden layer
model_baseline.add(Dense(units = 5, kernel_initializer = 'he_normal', activation = 'relu'))


# Adding the second hidden layer
#model_baseline.add(Dense(units = 8, kernel_initializer = 'he_normal', activation = 'relu'))


# Adding the output layer
model_baseline.add(Dense(units = 1, kernel_initializer = 'he_normal', activation = 'linear'))

# Compiling the ANN

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_baseline.compile(optimizer =sgd, loss = 'mean_squared_error',metrics=['MAE'])

# Fitting the ANN to the Training set
model_baseline.fit(X_train, y_train,validation_data=(X_test,y_test) , batch_size = 128, epochs = 500,verbose=0)



predictions=model_baseline.predict(X_test)


model_baseline.summary()
print('Explained_varianve_score={}'.format( explained_variance_score(y_test,predictions)))
print('R-squared={}'.format( r2_score(y_test,predictions)))
losses = pd.DataFrame(model_baseline.history.history)
losses[['loss','val_loss']].plot()

# Model 2
# I am selecting the features below for fitting the price model .
# Dropping the fuel type as it is screwed towards the diseal fuel 
# Also dropping maker_key as its all BMW  and model_key as that is captured in car_type 

data=data[data.price<100000] # Removing price outliers 
data=data[data.mileage<400000] #Removing mileage outliers . In total 8 rows removed 

features=data[['mileage','engine_power','paint_color','car_type','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','price','vechile_days']]

#Creating Dummy Variables 
features=pd.get_dummies(features,drop_first=True)


features['vechile_days']=features['vechile_days'].astype('int')
features['feature_1']=features['feature_1'].astype('int')
features['feature_2']=features['feature_2'].astype('int')
features['feature_3']=features['feature_3'].astype('int')
features['feature_4']=features['feature_4'].astype('int')
features['feature_5']=features['feature_5'].astype('int')
features['feature_6']=features['feature_6'].astype('int')
features['feature_7']=features['feature_7'].astype('int')
features['feature_8']=features['feature_8'].astype('int')

columns_names=features.columns

sc = StandardScaler()
features = sc.fit_transform(features[columns_names])

#converting it back to dataframe 
features_scales=pd.DataFrame(features,columns=columns_names)


X=features_scales.drop(['price'],axis=1).values
y=features_scales['price'].values.reshape(len(X),1)



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =42)

print(X.shape,y.shape)


# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(units = 12, kernel_initializer = 'he_normal', activation = 'relu', input_dim = X.shape[1]))

# Adding the second hidden layer
model.add(Dense(units = 12, kernel_initializer = 'he_normal', activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'he_normal', activation = 'linear'))

# Compiling the ANN

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer =sgd, loss = 'mean_squared_error',metrics=['MAE'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train,validation_data=(X_test,y_test) , batch_size = 128, epochs = 350,verbose=0)
model.summary()

predictions=model.predict(X_test)
print('Explained_varianve_score={}'.format( explained_variance_score(y_test,predictions)))
print('R-squared={}'.format( r2_score(y_test,predictions)))

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
#Saving weights for this model 

weights = model.get_weights()
errors = y_test - predictions
sns.distplot(errors)
# Our predictions
plt.figure(figsize=(8,8));
plt.scatter(y_test,predictions,alpha=0.3);

# Perfect predictions
plt.plot(y_test,y_test,'r');
