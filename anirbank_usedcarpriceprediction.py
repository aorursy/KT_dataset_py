import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from collections import OrderedDict

import os
print(os.listdir("../input"))
carSales = pd.read_csv('../input/data.csv',sep=",")
carSales.head()
#Lets import the dataset and have a look at the first few records.
carSales.info()
carSales['Vehicle Style'].value_counts()
carSales['Vehicle Size'].value_counts()
#As we can see above, there are a total of 16 features. The popularity of car is a number obtained from twitter assigned to a particular 
#Make and Model. Since, the users will not be aware of the popularity of a particular make, we should choose to ignore this feature.
#dropping Popularity
carSales=carSales.drop('Popularity',axis=1)

#We create derived feature,the Age of the car, derived from the Year of Mfr., which is certainly 
#a very important characteristic that a buyer looks into while looking for a used car.
#This dataset is from 2017. So, using that as a reference to calculate the age of the car .
print(carSales['Year'].max())
carSales['Age']=2017-carSales['Year']

carSales.describe()

ax=plt.hist(carSales['MSRP'],bins=50)
plt.xlim(0,150000)
plt.show()
%pylab inline
Make=carSales.groupby(['Make'])['MSRP'].median()
Make.plot(kind='bar',stacked=True)
pylab.ylabel('Median MSRP')
pylab.title('Chart: Median MSRP by Make')
show()



#From the above plot, we see that the Make of a car has a significant impact on price. This is obvious as different manufacturers produce cars in different price ranges. It might be difficult for a single model to fit well to all the data. So, it might be wise to divide the data w.r.t price and then we can have individual models for a single price range.
# Any results you write to the current directory are saved as output.
carSales=carSales.join(carSales.groupby('Make')['MSRP'].median(), on='Make', rsuffix='_Median')
make = carSales.groupby('Make')['MSRP'].median().reset_index()
pd.options.display.float_format = '{:.4f}'.format
make.sort_values('MSRP', ascending=False)

def map_MSRP_to_group(x):
    if x<30000:
        return 'ordinary'
    elif x<60000 :
        return 'deluxe'
    elif x<90000:
        return 'super-deluxe'
    elif x<350000:
        return 'luxury'
    else:
        return 'super-luxury'
#function to convert a series    
def convert_MSRP_series_to_MSRP_group(MSRP):
    return MSRP.apply(map_MSRP_to_group)

MSRP_group=convert_MSRP_series_to_MSRP_group(carSales['MSRP_Median'])
carSales['MSRP_group'] = MSRP_group
carSales.head()
carSales[carSales['MSRP_group']=='ordinary']['Make'].unique()
carSales[carSales['MSRP_group']=='deluxe']['Make'].unique()
carSales[carSales['MSRP_group']=='super-deluxe']['Make'].unique()
carSales[carSales['MSRP_group']=='luxury']['Make'].unique()
carSales[carSales['MSRP_group']=='super-luxury']['Make'].unique()
import pickle
ordinary='./ord.pkl'
deluxe='./del.pkl'
supdel='./supdel.pkl'
luxury='./luxury.pkl'
suplux='./suplux.pkl'

with open(ordinary, "wb") as f:
    w = pickle.dump(carSales[carSales['MSRP_group']=='ordinary'],f)
with open(deluxe, "wb") as f:
    w = pickle.dump(carSales[carSales['MSRP_group']=='deluxe'],f)
with open(supdel, "wb") as f:
    w = pickle.dump(carSales[carSales['MSRP_group']=='super-deluxe'],f)
with open(luxury, "wb") as f:
    w = pickle.dump(carSales[carSales['MSRP_group']=='luxury'],f)
with open(suplux, "wb") as f:
    w = pickle.dump(carSales[carSales['MSRP_group']=='super-luxury'],f)
ordinary=pd.read_pickle('./ord.pkl')
ordinary.info()
ordinary = ordinary.reset_index(drop=True)
ordinary=ordinary.drop(['MSRP_group','MSRP_Median'],axis=1)
corr=ordinary.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#Also showcasing the table below to lookup exact values for reference.
corr
from numpy import cov
cov(carSales['highway MPG'],carSales['city mpg'])
from scipy.stats import pearsonr
corr, _ = pearsonr(carSales['highway MPG'],carSales['city mpg'])
print('Pearsons correlation betweeen highway MPG and city MPG: %.3f' % corr)
#dropping the features as discussed above
ordinary=ordinary.drop(['Year','highway MPG'],axis=1)
plt.scatter(ordinary['MSRP'], ordinary['Engine HP'])
plt.show()
#do some data cleansing to make sure the correlation analysis runs without any error
m=ordinary["MSRP"].isnull().any()
print(m[m])
m=ordinary["Engine HP"].isnull().any()
print(m[m])
ordinary["Engine HP"].fillna(ordinary["Engine HP"].mean())
ordinary['Engine HP'] = ordinary['Engine HP'].apply(lambda x: x if not pd.isnull(x) else ordinary["Engine HP"].mean())
m=ordinary["Engine HP"].isnull().any()
print(m[m])
from scipy.stats import spearmanr

corr_p, _ = pearsonr((ordinary['MSRP']), ordinary['Engine HP'])
print('Pearson correlation between Price and Engine HP : %.3f' % corr_p)

corr_p, _ = pearsonr((ordinary['MSRP']), np.square(ordinary['Engine HP']))
print('Pearson correlation between Price and square of Engine HP : %.3f' % corr_p)

corr_p, _ = pearsonr((ordinary['MSRP']), np.power(ordinary['Engine HP'],3))
print('Pearson correlation between Price and cube of Engine HP : %.3f' % corr_p)

corr_p, _ = pearsonr((ordinary['MSRP']), np.log(1+ordinary['Engine HP']))
print('Pearson correlation between Price and log of Engine HP : %.3f' % corr_p)
corr, _ = spearmanr(ordinary['MSRP'], ordinary['Engine HP'])
print('Spearman correlation between Price and Engine HP: %.3f' % corr)
corr, _ = spearmanr(ordinary['MSRP'], np.log(1+ordinary['Engine HP']))
print('Spearman correlation between Price and log of Engine HP : %.3f' % corr)
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(ordinary,hue='Make')
ordinary[ordinary['Engine Fuel Type'].isnull()]
ordinary[(ordinary['Model']=='Verona')&(ordinary['Make']=='Suzuki')]
#For Suzuki Verona,  Fuel Type is regular unleaded
ordinary.loc[(ordinary['Engine Fuel Type'].isnull())&(ordinary['Model']=='Verona')&(ordinary['Make']=='Suzuki'),'Engine Fuel Type']='regular unleaded'
ordinary.loc[(ordinary['Make']=='Mazda')&(ordinary['Model']=='RX-8'),'Engine Cylinders']=0
ordinary.loc[(ordinary['Make']=='Mazda')&(ordinary['Model']=='RX-7'),'Engine Cylinders']=0
ordinary.loc[(ordinary['Make']=='Mitsubishi')&(ordinary['Model']=='i-MiEV'),'Engine Cylinders']=0
ordinary.loc[(ordinary['Make']=='Chevrolet')&(ordinary['Model']=='Bolt EV'),'Engine Cylinders']=0
ordinary.loc[(ordinary['Make']=='Volkswagen')&(ordinary['Model']=='e-Golf'),'Engine Cylinders']=0
ordinary=ordinary.drop('Market Category',axis=1)
plt.hist(ordinary['Age'])
plt.title('Age Histogram')
plt.show()
plt.hist(ordinary['MSRP'])
plt.title('Price Histogram')
plt.show()
plt.hist(ordinary['city mpg'])
plt.title('City mpg Histogram')
plt.show()

sns.boxplot(x=ordinary['MSRP'])
q75, q25 = np.percentile(ordinary['MSRP'], [75 ,25])
iqr = q75 - q25
q75+1.5*q75
ordinary[ordinary['MSRP']>140000]
ordinary=ordinary[ordinary['MSRP']<140000]
ordinary['log_MSRP']=ordinary['MSRP'].apply(lambda x:np.log(1+x))
plt.hist(ordinary['log_MSRP'])
plt.show()
sns.boxplot(x=ordinary['log_MSRP'])
ordinary['log_Age']=ordinary['Age'].apply(lambda x: np.log(x+1))
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,3))
ax1.set_title('Log Age Histogram')
ax1.hist(ordinary['log_Age'])
ax2.boxplot(ordinary['log_Age'])
ax2.set_title('Log Age Box Plot')
plt.show()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,3))
ax1.scatter(ordinary['Age'],ordinary['log_MSRP'],alpha=.3)
ax1.set_title('Age vs Log MSRP')
ax2.scatter(ordinary['log_Age'],ordinary['log_MSRP'],alpha=.3)
ax2.set_title('Log Age vs Log MSRP')
ordinary['sqrt_Age']=np.sqrt(ordinary['Age'])
plt.scatter(ordinary['sqrt_Age'],ordinary['log_MSRP'],alpha=.3)
plt.title('sqrt Age vs Log MSRP')
plt.show()
sns.lmplot( x="Age", y="log_MSRP", data=ordinary, fit_reg=False, hue='Driven_Wheels', legend=False)
plt.legend(loc='upper right')
ordinary['log_city mpg']=ordinary['city mpg'].apply(lambda x: np.log(x+1))
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,3))
ax1.hist(ordinary['log_city mpg'])
ax1.set_title('log city mpg histogram')
ax2.boxplot(ordinary['log_city mpg'])
ax2.set_title('log city mpg boxplot')
plt.show()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,3))
ax1.scatter(ordinary['city mpg'],ordinary['log_MSRP'],alpha=.1)
ax1.set_title('city mpg vs Log MSRP')
ax2.scatter(ordinary['log_city mpg'],ordinary['log_MSRP'],alpha=.1)
ax2.set_title('Log city mpg vs Log MSRP')
sns.lmplot( x="city mpg", y="log_MSRP", data=ordinary, fit_reg=False, hue='Transmission Type', legend=False)
plt.legend(loc='upper right')
corr=ordinary.corr()
corr['log_MSRP'].sort_values(ascending=True)
ordinary=ordinary.drop(['log_Age','city mpg'],axis=1)
ordinary.head()
#Since we have decided to take log MSRP as a target variable, we decide to check relationship between Engine HP, 
#with powers and Log MSRP
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,3))
ax1.scatter(np.power(ordinary['Engine HP'],1/2), ordinary['log_MSRP'],alpha=.3)
ax1.set_title('HP^2 vs Log MSRP')
ax2.scatter(np.power(ordinary['Engine HP'],1/3), ordinary['log_MSRP'],alpha=.3)
ax2.set_title('HP^3 vs Log MSRP')
ax3.scatter(log(1+ordinary['Engine HP']), ordinary['log_MSRP'],alpha=.3)
ax3.set_title('log Engine HP vs Log MSRP')
plt.show()
corr_p, _ = pearsonr((ordinary['log_MSRP']), ordinary['Engine HP'])
print('Pearson correlation between log Price and Engine HP : %.3f' % corr_p)

corr_p, _ = pearsonr((ordinary['log_MSRP']), np.square(ordinary['Engine HP']))
print('Pearson correlation between log Price and square of Engine HP : %.3f' % corr_p)

corr_p, _ = pearsonr((ordinary['log_MSRP']), np.power(ordinary['Engine HP'],3))
print('Pearson correlation between log Price and cube of Engine HP : %.3f' % corr_p)

corr_p, _ = pearsonr((ordinary['log_MSRP']), np.log(1+ordinary['Engine HP']))
print('Pearson correlation between log Price and log of Engine HP : %.3f' % corr_p)
ordinary['log_Engine HP']=ordinary['Engine HP'].apply(lambda x: np.log(x+1))
fig, (ax1) = plt.subplots(1,1,figsize=(15,5))
Make=ordinary.groupby(['Make'])['MSRP'].mean()
Model=ordinary.groupby(['Model'])['MSRP'].mean()
FuelType=ordinary.groupby(['Engine Fuel Type'])['MSRP'].mean()
Transmission=ordinary.groupby(['Transmission Type'])['MSRP'].mean()
DrivenWheels=ordinary.groupby(['Driven_Wheels'])['MSRP'].mean()
VehicleSize=ordinary.groupby(['Vehicle Size'])['MSRP'].mean()
VehicleStyle=ordinary.groupby(['Vehicle Style'])['MSRP'].mean()
ax1.bar(Make.index,Make.values)
ax1.set_title('Mean MSRP by Make')
plt.sca(ax1)
plt.xticks(rotation=90)
plt.bar(FuelType.index,FuelType.values)
plt.title('Mean MSRP by Engine Fuel Type')
plt.xticks(rotation=90)
plt.show()
plt.bar(Transmission.index,Transmission.values)
plt.title('Mean MSRP by Transmission Type')
plt.xticks(rotation=90)
plt.show()
plt.bar(DrivenWheels.index,DrivenWheels.values)
plt.title('Mean MSRP by Driven_Wheels')
plt.xticks(rotation=90)
plt.show()
plt.bar(VehicleSize.index,VehicleSize.values)
plt.title('Mean MSRP by Vehicle Size')
plt.xticks(rotation=90)
plt.show()
plt.bar(VehicleStyle.index,VehicleStyle.values)
plt.title('Mean MSRP by Vehicle Style')
plt.xticks(rotation=90)
plt.show()
#We take few specific models. This is to restric the number of features.
Makes=['Ford','Chevrolet','Chrysler','Pontiac','Subaru','Hyundai','Honda','Mazda', 'Nissan','Suzuki']
ordinary=ordinary[ordinary.Make.isin(Makes)]
ordinary.Make.value_counts()
ordinary_trans='./ordinarydfUSJap.pkl'
with open(ordinary_trans, "wb") as f:
    w = pickle.dump(ordinary,f)
#Import all necessary libraries
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
df_ordinary=pd.read_pickle('./ordinarydfUSJap.pkl')
df_ordinary.head()
df_ordinary["Make"].value_counts()
print(len(df_ordinary))
#We do some data cleansing, as needed
df_ordinary["Number of Doors"] = df_ordinary["Number of Doors"].replace("?",0)
df_ordinary["Number of Doors"] = df_ordinary["Number of Doors"].astype('float32')
df_ordinary["MSRP"] = df_ordinary["MSRP"].replace("?",0)
df_ordinary["MSRP"] = df_ordinary["MSRP"].astype("float32")
df_ordinary["log_Engine HP"] = df_ordinary["log_Engine HP"].astype("float32")
df_ordinary["Age"].value_counts()
#create a field Age-cat to divide the data into 5 Age categories, based on the Age of the car
df_ordinary["Age-cat"] = np.ceil(df_ordinary["Age"] / 5)
df_ordinary["Age-cat"].where(df_ordinary["Age-cat"] < 5, 5.0, inplace=True)
#check distribution of Age Cat in the original data
df_ordinary["Age-cat"].value_counts() / len(df_ordinary)
car_eng_cyl = df_ordinary["Engine Cylinders"]
encoder_cyl = LabelBinarizer()
encoder_cyl.fit(car_eng_cyl)
print(encoder_cyl.classes_)

car_eng_fuel_type = df_ordinary["Engine Fuel Type"]
encoder_fuel = LabelBinarizer()
encoder_fuel.fit(car_eng_fuel_type)
print(encoder_fuel.classes_)

car_trans_type = df_ordinary["Transmission Type"]
encoder_trans = LabelBinarizer()
encoder_trans.fit(car_trans_type)
print(encoder_trans.classes_)

car_driven_wheels = df_ordinary["Driven_Wheels"]
encoder_wheels = LabelBinarizer()
encoder_wheels.fit(car_driven_wheels)
print(encoder_wheels.classes_)

car_vehicle_size = df_ordinary["Vehicle Size"]
encoder_size = LabelBinarizer()
encoder_size.fit(car_vehicle_size)
print(encoder_size.classes_)

car_make =df_ordinary["Make"]
encoder_make = LabelBinarizer()
encoder_make.fit(car_make)
print(encoder_make.classes_)


car_style =df_ordinary["Vehicle Style"]
encoder_style = LabelBinarizer()
encoder_style.fit(car_style)
print(encoder_style.classes_)

#We use StratifiedShuffleSplit based on Age-cat, to make sure both train and test data have same distribution of New and Old cars
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(df_ordinary,df_ordinary["Age-cat"]):
    strat_train_set = df_ordinary.iloc[train_index]
    strat_test_set = df_ordinary.iloc[test_index]
#check distribution of Age Cat in the train data
strat_train_set["Age-cat"].value_counts() / len(strat_train_set)
#check distribution of Age Cat in the test data
strat_test_set["Age-cat"].value_counts() / len(strat_test_set)
carSales_X = strat_train_set.copy()
carSales_X = strat_train_set.drop("MSRP", axis=1) # drop labels for training set
carSales_X = strat_train_set.drop("log_MSRP", axis=1) # drop labels for training set
carSales_Y = strat_train_set["log_MSRP"].copy() # use log MSRP as labels for training set, based on data Exploration
carSales_Y_orig = strat_train_set["MSRP"].copy() # use MSRP as labels also for training set, to compare fit based on Log and original Price

carSales_test_X = strat_test_set.copy()
carSales_test_X = strat_test_set.drop("MSRP", axis=1) # drop labels for test set
carSales_test_X = strat_test_set.drop("log_MSRP", axis=1) # drop labels for test set
carSales_test_Y = strat_test_set["log_MSRP"].copy()# use log MSRP as labels for test set, based on data Exploration
carSales_test_Y_orig = strat_test_set["MSRP"].copy()# use MSRP as labels also for test set, to compare fit based on Log and original Price
carSales_Y = carSales_Y.values.reshape(carSales_Y.shape[0],1)
carSales_test_Y = carSales_test_Y.values.reshape(carSales_test_Y.shape[0],1)
carSales_Y_orig = carSales_Y_orig.values.reshape(carSales_Y_orig.shape[0],1)
carSales_test_Y_orig = carSales_test_Y_orig.values.reshape(carSales_test_Y.shape[0],1)
print(carSales_X.shape)
print(carSales_Y.shape)
print(carSales_Y_orig.shape)
print(carSales_test_X.shape)
print(carSales_test_Y.shape)
print(carSales_test_Y_orig.shape)
carSales_X.head()
carSales_X_num = carSales_X
carSales_X_num  = carSales_X_num.drop("Make",axis=1) # to be treated as categorical var
carSales_X_num  = carSales_X_num.drop("Model",axis=1)
carSales_X_num  = carSales_X_num.drop("Engine Cylinders",axis=1) # to be treated as categorical var
carSales_X_num  = carSales_X_num.drop("Engine Fuel Type",axis=1) # to be treated as categorical var
carSales_X_num  = carSales_X_num.drop("Transmission Type",axis=1) # to be treated as categorical var 
carSales_X_num  = carSales_X_num.drop("Driven_Wheels",axis=1) # to be treated as categorical var
carSales_X_num = carSales_X_num.drop("Number of Doors",axis=1) # to be treated as categorical var
carSales_X_num  = carSales_X_num.drop("Vehicle Style",axis=1) # to be treated as categorical var
carSales_X_num  = carSales_X_num.drop("Engine HP",axis=1)  # since we are taking log of Engine HP,based on Analysis
carSales_X_num = carSales_X_num.drop("Vehicle Size",axis=1) # to be treated as categorical var
carSales_X_num = carSales_X_num.drop("Age-cat",axis=1) # derived column
carSales_X_num = carSales_X_num.drop("sqrt_Age",axis=1) # derived column
carSales_X_num = carSales_X_num.drop("MSRP",axis=1) #Target / label
carSales_X_num.head()
#Apply the same transformation on Test data
carSales_test_X_num = carSales_test_X
carSales_test_X_num  = carSales_test_X_num.drop("Make",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Model",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Engine Cylinders",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Engine Fuel Type",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Transmission Type",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Driven_Wheels",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Number of Doors",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Vehicle Style",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Vehicle Size",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Engine HP",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Age-cat",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("sqrt_Age",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("MSRP",axis=1)
carSales_test_X_num.head()
#We do some data cleansing as needed
carSales_X_num["log_Engine HP"] = carSales_X_num["log_Engine HP"].astype("float32")
carSales_X_num["Age"] = carSales_X_num["Age"].astype("float32")
carSales_X_num.replace('null',np.NaN,inplace=True)
carSales_X_num = pd.DataFrame(carSales_X_num)
carSales_X_num = carSales_X_num.replace('?',0)
carSales_X_num = carSales_X_num.replace('NaN',0)
carSales_X_num = carSales_X_num.replace(np.NaN,0)

carSales_test_X_num["log_Engine HP"] = carSales_test_X_num["log_Engine HP"].astype("float32")
carSales_test_X_num["Age"] = carSales_test_X_num["Age"].astype("float32")
carSales_test_X_num.replace('null',np.NaN,inplace=True)
carSales_test_X_num = pd.DataFrame(carSales_test_X_num)
carSales_test_X_num = carSales_test_X_num.replace('?',0)
carSales_test_X_num = carSales_test_X_num.replace('NaN',0)
carSales_test_X_num = carSales_test_X_num.replace(np.NaN,0)
m=carSales_X_num.isnull().any()
print(m[m])
m=np.isfinite(carSales_X_num.select_dtypes(include=['float64'])).any()
print(m[m])
m=carSales_test_X_num.isnull().any()
print(m[m])
m=np.isfinite(carSales_test_X_num.select_dtypes(include=['float64'])).any()
print(m[m])
imputer = Imputer(missing_values=0,strategy="mean")
imputer.fit(carSales_X_num)
imputer.fit(carSales_test_X_num)
#Standardize the data using sklearn StandardScaler
scaler = StandardScaler()
train_X = scaler.fit_transform(carSales_X_num)
test_X = scaler.transform(carSales_test_X_num)
print(train_X.shape)
car_eng_cyl = carSales_X["Engine Cylinders"]
car_eng_1hot = encoder_cyl.transform(car_eng_cyl)
print(car_eng_1hot.shape)

train_X = np.concatenate((train_X,car_eng_1hot),axis=1)

car_eng_fuel_type = carSales_X["Engine Fuel Type"]
car_fuel_1hot = encoder_fuel.transform(car_eng_fuel_type)
print(car_fuel_1hot.shape)

train_X = np.concatenate((train_X,car_fuel_1hot),axis=1)

car_trans_type = carSales_X["Transmission Type"]
car_trans_1hot = encoder_trans.transform(car_trans_type)
print(car_trans_1hot.shape)

train_X = np.concatenate((train_X,car_trans_1hot),axis=1)

car_driven_wheels = carSales_X["Driven_Wheels"]
car_drive_1hot = encoder_wheels.transform(car_driven_wheels)
print(car_drive_1hot.shape)

train_X = np.concatenate((train_X,car_drive_1hot),axis=1)

car_vehicle_size = carSales_X["Vehicle Size"]
car_size_1hot = encoder_size.transform(car_vehicle_size)
print(car_size_1hot.shape)

train_X = np.concatenate((train_X,car_size_1hot),axis=1)

car_vehicle_style = carSales_X["Vehicle Style"]
car_style_1hot = encoder_style.transform(car_vehicle_style)
print(car_style_1hot.shape)

train_X = np.concatenate((train_X,car_style_1hot),axis=1)

car_make = carSales_X["Make"]
car_make_1hot = encoder_make.transform(car_make)
print(car_make_1hot.shape)

train_X_make = np.concatenate((train_X,car_make_1hot),axis=1)

#We prepare two sets of train X features, with Make and without Make and compare the performance of both
print(train_X.shape)
print(train_X_make.shape)
car_eng_cyl = carSales_test_X["Engine Cylinders"]
car_eng_1hot = encoder_cyl.transform(car_eng_cyl)
print(car_eng_1hot.shape)

test_X = np.concatenate((test_X,car_eng_1hot),axis=1)

car_eng_fuel_type = carSales_test_X["Engine Fuel Type"]
car_fuel_1hot = encoder_fuel.transform(car_eng_fuel_type)
print(car_fuel_1hot.shape)

test_X = np.concatenate((test_X,car_fuel_1hot),axis=1)

car_trans_type_test = carSales_test_X["Transmission Type"]
car_trans_1hot_test = encoder_trans.transform(car_trans_type_test)
print(car_trans_1hot_test.shape)

test_X = np.concatenate((test_X,car_trans_1hot_test),axis=1)

car_driven_wheels_test = carSales_test_X["Driven_Wheels"]
car_drive_1hot_test = encoder_wheels.transform(car_driven_wheels_test)
print(car_drive_1hot_test.shape)

test_X = np.concatenate((test_X,car_drive_1hot_test),axis=1)

car_vehicle_size_test = carSales_test_X["Vehicle Size"]
car_size_1hot_test = encoder_size.transform(car_vehicle_size_test)
print(car_size_1hot_test.shape)

test_X = np.concatenate((test_X,car_size_1hot_test),axis=1)

car_vehicle_style_test = carSales_test_X["Vehicle Style"]
car_style_1hot_test = encoder_style.transform(car_vehicle_style_test)
print(car_style_1hot_test.shape)

test_X = np.concatenate((test_X,car_style_1hot_test),axis=1)

car_make_test = carSales_test_X["Make"]
car_make_1hot_test = encoder_make.transform(car_make_test)
print(car_make_1hot_test.shape)

test_X_make = np.concatenate((test_X,car_make_1hot_test),axis=1)

print(test_X.shape)
print(test_X_make.shape)
train_Y = pd.DataFrame(carSales_Y)
m=train_Y.isnull().any()
print(m[m])
m=np.isfinite(train_Y.select_dtypes(include=['float64'])).any()
print(m[m])

train_Y_orig = pd.DataFrame(carSales_Y_orig)
m=train_Y_orig.isnull().any()
print(m[m])
m=np.isfinite(train_Y_orig.select_dtypes(include=['float64'])).any()
print(m[m])

test_Y = pd.DataFrame(carSales_test_Y)
m=test_Y.isnull().any()
print(m[m])
m=np.isfinite(test_Y.select_dtypes(include=['float64'])).any()
print(m[m])

test_Y_orig = pd.DataFrame(carSales_test_Y_orig)
m=test_Y_orig.isnull().any()
print(m[m])
m=np.isfinite(test_Y_orig.select_dtypes(include=['float64'])).any()
print(m[m])
train_X_ordinary='./train_X_ordUSJap.pkl'
test_X_ordinary='./test_X_ordUSJap.pkl'
train_Y_ordinary='./train_Y_ordUSJap.pkl'
test_Y_ordinary='./test_Y_ordUSJap.pkl'
train_Y_ordinary_orig='./train_Y_ord_origUSJap.pkl'
test_Y_ordinary_orig='./test_Y_ord_origUSJap.pkl'

with open(train_X_ordinary, "wb") as f:
    w = pickle.dump(train_X,f)
with open(test_X_ordinary, "wb") as f:
    w = pickle.dump(test_X,f)
with open(train_Y_ordinary, "wb") as f:
    w = pickle.dump(train_Y,f)
with open(test_Y_ordinary, "wb") as f:
    w = pickle.dump(test_Y,f)
with open(train_Y_ordinary_orig, "wb") as f:
    w = pickle.dump(train_Y_orig,f)
with open(test_Y_ordinary_orig, "wb") as f:
    w = pickle.dump(test_Y_orig,f)
    
train_X_ord_make='./train_X_ord_makeUSJap.pkl'
test_X_ord_make='./test_X_ord_makeUSJap.pkl'

with open(train_X_ord_make, "wb") as f:
    w = pickle.dump(train_X_make,f)
with open(test_X_ord_make, "wb") as f:
    w = pickle.dump(test_X_make,f)
#Import all necessary libraries
import pickle
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from keras.layers import Dense   
from keras import optimizers

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
train_X=pd.read_pickle('./train_X_ordUSJap.pkl')
test_X=pd.read_pickle('./test_X_ordUSJap.pkl')
train_Y=pd.read_pickle('./train_Y_ordUSJap.pkl') # train Y with log(MSRP)
test_Y=pd.read_pickle('./test_Y_ordUSJap.pkl')
train_Y_orig=pd.read_pickle('./train_Y_ord_origUSJap.pkl') # train Y with MSRP unmodified
test_Y_orig=pd.read_pickle('./test_Y_ord_origUSJap.pkl')

train_X_make=pd.read_pickle('./train_X_ord_makeUSJap.pkl')
test_X_make=pd.read_pickle('./test_X_ord_makeUSJap.pkl')

print(train_X.shape)
print(train_X_make.shape)
#fit train data without make info, and log MSRP
lin_reg = LinearRegression()
lin_reg.fit(train_X, train_Y)

#fit train data without make and MSRP, as it
lin_reg_1 = LinearRegression()
lin_reg_1.fit(train_X, train_Y_orig)

#fit train data with make and log MSRP
lin_reg_make = LinearRegression()
lin_reg_make.fit(train_X_make, train_Y)

#fit train data with make and MSRP, as it
lin_reg_make1 = LinearRegression()
lin_reg_make1.fit(train_X_make, train_Y_orig)
carSales_predictions = lin_reg.predict(test_X)
lin_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
lin_rmse = np.sqrt(lin_mse)
print("rmse without make, log MSRP:"+str(lin_rmse))

carSales_predictions = lin_reg_1.predict(test_X)
lin_mse = mean_squared_error(test_Y, carSales_predictions)
lin_rmse = np.sqrt(lin_mse)
print("rmse without make, MSRP, as is:"+str(lin_rmse))

carSales_predictions = lin_reg_make.predict(test_X_make)
lin_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
lin_rmse = np.sqrt(lin_mse)
print("rmse with make, log MSRP:"+str(lin_rmse))

carSales_predictions = lin_reg_make1.predict(test_X_make)
lin_mse = mean_squared_error(test_Y, carSales_predictions)
lin_rmse = np.sqrt(lin_mse)
print("rmse with make, MSRP, as is:"+str(lin_rmse))
sgd_reg_make = SGDRegressor(max_iter=500,penalty=None,eta0=0.01)
sgd_reg_make.fit(train_X_make, train_Y.values.ravel())
carSales_predictions_make = sgd_reg_make.predict(test_X_make)
sgd_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions_make))
sgd_rmse = np.sqrt(sgd_mse)
print("SGD RMSE:"+str(sgd_rmse))
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_X,train_Y)

tree_reg_make = DecisionTreeRegressor()
tree_reg_make.fit(train_X_make,train_Y)
carSales_predictions = tree_reg.predict(test_X)
tree_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree RMSE, without make:"+str(tree_rmse))
carSales_predictions_make = tree_reg_make.predict(test_X_make)
tree_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions_make))
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree RMSE, with make:"+str(tree_rmse))
#Lets print few predicted prices and actual prices
print("predicted prices")
print(np.around(np.exp(carSales_predictions_make[0:5])))
print("actual prices")
print(np.exp(test_Y[0:5]))
scores = cross_val_score(tree_reg_make,train_X_make,train_Y,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("scores:",tree_rmse_scores)
print("mean:",tree_rmse_scores.mean())
print("std dev:",tree_rmse_scores.std())
forest_reg_make = RandomForestRegressor()
forest_reg_make.fit(train_X_make,train_Y.values.ravel())
carSales_predictions = forest_reg_make.predict(test_X_make)
forest_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
forest_rmse = np.sqrt(forest_mse)
print("Random Forest Regressor RMSE, with make:"+str(forest_rmse))
print("predicted prices")
print(np.around(np.exp(carSales_predictions[0:5])))
print("actual prices")
print(np.exp(test_Y[0:5]))
forest_scores = cross_val_score(forest_reg_make,train_X_make,train_Y.values.ravel(),scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("scores:",forest_rmse_scores)
print("mean:",forest_rmse_scores.mean())
print("std dev:",forest_rmse_scores.std())
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(train_X_make, train_Y.values.ravel())
print("BEST PARAMETERS FOR RANDOM FOREST REGRESSOR IS:")
grid_search.best_params_
#Fit using best parameters and check
forest_reg_make = RandomForestRegressor(max_features=8,n_estimators=30)
forest_reg_make.fit(train_X_make,train_Y.values.ravel())
carSales_predictions = forest_reg_make.predict(test_X_make)
forest_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
forest_rmse = np.sqrt(forest_mse)
print("Random Forest Regressor RMSE, with make:"+str(forest_rmse))
print("predicted prices")
print(np.around(np.exp(carSales_predictions_make[0:5])))
print("actual prices")
print(np.exp(test_Y[0:5]))
feature_importances = grid_search.best_estimator_.feature_importances_
num_attribs = ["Age","City MPG","Engine HP"]
categorical_attribs = [  '0.' , ' 3.',   '4.' ,  '5.' ,  '6.',   '8.' , '10.', '12.'] + ['diesel', 'electric' ,'flex-fuel (unleaded/E85)',
 'flex-fuel (unleaded/natural gas)' ,'natural gas', 'premium unleaded (recommended)', 'premium unleaded (required)',
 'regular unleaded'] + ['AUTOMATED_MANUAL' ,'AUTOMATIC' ,'DIRECT_DRIVE' ,'MANUAL', 'UNKNOWN'] + ['all wheel drive','four wheel drive', 'front wheel drive', 'rear wheel drive'] + ['Compact' ,'Large', 'Midsize']+['2dr Hatchback', '2dr SUV' ,'4dr Hatchback', '4dr SUV', 'Cargo Minivan',
 'Cargo Van', 'Convertible', 'Convertible SUV' ,'Coupe', 'Crew Cab Pickup','Extended Cab Pickup' ,'Passenger Minivan' ,'Passenger Van','Regular Cab Pickup', 'Sedan' ,'Wagon']+['Ford','Chevrolet','Chrysler','Pontiac','Subaru','Hyundai','Honda','Mazda', 'Nissan','Suzuki']
attributes = num_attribs+categorical_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(test_X_make)
final_mse = mean_squared_error(np.exp(test_Y), np.exp(final_predictions))
final_rmse = np.sqrt(final_mse)
print("Random Forest Regressor Final RMSE:"+str(final_rmse))
print("predicted prices")
print(np.around(np.exp(final_predictions[0:5])))
print("actual prices")
print(np.exp(test_Y[0:5]))


final_model_scores = cross_val_score(final_model,test_X_make,test_Y.values.ravel(),scoring="neg_mean_squared_error",cv=10)
final_model_scores = np.sqrt(-final_model_scores)

print("scores:",final_model_scores)
print("mean:",final_model_scores.mean())
print("std dev:",final_model_scores.std())
def plot_learning_curves(model, X, y):
    
    train_errors, val_errors = [], []
    for m in range(1, len(X)):
        model.fit(X[:m], y[:m].values.ravel())
        y_train_predict = model.predict(X[:m])
        y_val_predict = model.predict(test_X_make)
        train_errors.append(mean_squared_error(y[:m], y_train_predict))
        val_errors.append(mean_squared_error(test_Y, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14)   

plot_learning_curves(final_model, train_X_make, train_Y)
plt.show()

#We try ElasticNet
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(train_X_make, train_Y)
carSales_predictions = elastic_net.predict(test_X_make)
elastic_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
elastic_rmse = np.sqrt(elastic_mse)
print("Elastic Net RMSE:"+str(elastic_rmse))
#We try Ridge Regression
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(train_X_make, train_Y)
carSales_predictions = ridge_reg.predict(test_X_make)
ridge_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
ridge_rmse = np.sqrt(ridge_mse)
print("Ridge RMSE:"+str(ridge_rmse))
#We try Lasso Regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(train_X_make, train_Y)
carSales_predictions = lasso_reg.predict(test_X_make)
lasso_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
lasso_rmse = np.sqrt(lasso_mse)
print("Lasso RMSE:"+str(lasso_rmse))
gbrt = GradientBoostingRegressor(max_depth=8, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(train_X_make, train_Y.values.ravel())
carSales_predictions = gbrt.predict(test_X_make)
gbrt_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
gbrt_rmse = np.sqrt(gbrt_mse)
print("Gradient Boosting Regressor RMSE:"+str(gbrt_rmse))
gbrt_slow = GradientBoostingRegressor(max_depth=30, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_slow.fit(train_X_make, train_Y.values.ravel())
carSales_predictions = gbrt_slow.predict(test_X_make)
gbrt_slow_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
gbrt_slow_rmse = np.sqrt(gbrt_slow_mse)
print("Gradient Boosting Regressor SLOW RMSE:"+str(gbrt_slow_rmse))
param_grid = [
    # try 2 (2×2) combinations of hyperparameters
    {'n_estimators': [100,200], 'max_depth': [20, 30]},
    # then try 6 (2×3) combinations with bootstrap set as False
    #{'bootstrap': [False], 'n_estimators': [100,200], 'max_depth': [20, 30, 40]},
  ]

gbrt_reg = GradientBoostingRegressor(random_state=42, learning_rate=0.1)
# train across 5 folds, that's a total of (4)*5=20 rounds of training 
grid_search_gbrt = GridSearchCV(gbrt_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search_gbrt.fit(train_X_make, train_Y.values.ravel())
print("BEST PARAMETERS FOR GRADIENT BOOSTING REGRESSOR IS:")
grid_search_gbrt.best_params_
gbrt_slow = GradientBoostingRegressor(max_depth=20, n_estimators=100, learning_rate=0.1, random_state=42)
gbrt_slow.fit(train_X_make, train_Y.values.ravel())
carSales_predictions = gbrt_slow.predict(test_X_make)
gbrt_slow_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
gbrt_slow_rmse = np.sqrt(gbrt_slow_mse)
print("Gradient Boosting Regressor BEST RMSE:"+str(gbrt_slow_rmse))
model = Sequential()

#We use two hidden layers with 50 and 30 units with Relu activation, and no activation in the output layer, since 
#we want to predict the car price.
model.add(Dense(50,input_dim=(train_X_make.shape[1]),activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1))
model.summary()
myOptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer, metrics=['mse'])
history = model.fit(train_X_make, train_Y, epochs=200,  validation_data=(test_X_make,test_Y), batch_size=5, verbose=0)
plt.plot(history.history['loss'], color = 'blue')
plt.plot(history.history['val_loss'], color=  'red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
carSales_predictions = model.predict(test_X_make)
dl_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
dl_rmse = np.sqrt(dl_mse)
print("Deep Learning RMSE with two hidden layers:"+str(dl_rmse))
model = Sequential()
model.add(Dense(50,input_dim=(train_X_make.shape[1]),activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary()
myOptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer, metrics=['mse'])
history = model.fit(train_X_make, train_Y, epochs=300,  validation_data=(test_X_make,test_Y), batch_size=10, verbose=0)
carSales_predictions = model.predict(test_X_make)
dl_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
dl_rmse = np.sqrt(dl_mse)
print("Deep Learning RMSE with three hidden layers:"+str(dl_rmse))
xgb_model = XGBRegressor() 
xgb_model.fit(train_X_make, train_Y)
carSales_predictions = xgb_model.predict(test_X_make)
xgb_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
xgb_rmse = np.sqrt(xgb_mse)
print("XGB RMSE:"+str(xgb_rmse))
xgb_model = XGBRegressor(n_estimators=350, max_depth=5) 
xgb_model.fit(train_X_make, train_Y)
carSales_predictions = xgb_model.predict(test_X_make)
xgb_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
xgb_rmse = np.sqrt(xgb_mse)
print("XGB RMSE, with n_estmators=350,max_depth=5:"+str(xgb_rmse))
xgb_model = XGBRegressor(n_estimators=350, max_depth=10) 
xgb_model.fit(train_X_make, train_Y)
carSales_predictions = xgb_model.predict(test_X_make)
xgb_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
xgb_rmse = np.sqrt(xgb_mse)
print("XGB RMSE, with n_estmators=350,max_depth=10:"+str(xgb_rmse))
xgb_model = XGBRegressor(n_estimators=350, max_depth=5) 
xgb_model.fit(train_X_make, train_Y)
print(xgb_model.feature_importances_)
# plot
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.show()
print(train_X_make.shape[1])
train_X_make_upd = np.delete(train_X_make,[4,5,6,7,8,9,10,11],1) # we drop the 8 columns after the first 6 numeric ones
print(train_X_make_upd.shape[1])
test_X_make_upd = np.delete(test_X_make,[4,5,6,7,8,9,10,11],1)
xgb_model = XGBRegressor(n_estimators=350, max_depth=5) 
xgb_model.fit(train_X_make_upd, train_Y)
carSales_predictions = xgb_model.predict(test_X_make_upd)
xgb_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
xgb_rmse = np.sqrt(xgb_mse)
print("XGB RMSE, with n_estmators=350,max_depth=5:"+str(xgb_rmse))
print(train_X_make.shape[1])
train_X_make_upd1 = np.delete(train_X_make,[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],1) # we drop the 8 columns after the first 3 numeric ones
print(train_X_make_upd1.shape[1])
test_X_make_upd1 = np.delete(test_X_make,[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],1)
xgb_model = XGBRegressor(n_estimators=350, max_depth=5) 
xgb_model.fit(train_X_make_upd1, train_Y)
carSales_predictions = xgb_model.predict(test_X_make_upd1)
xgb_mse = mean_squared_error(np.exp(test_Y), np.exp(carSales_predictions))
xgb_rmse = np.sqrt(xgb_mse)
print("XGB RMSE, with n_estmators=350,max_depth=5:"+str(xgb_rmse))
# save model to file
pickle.dump(xgb_model, open("./carsales_xgb.pickle.dat", "wb"))
