import numpy as np 

import pandas as pd 

from sklearn.impute import SimpleImputer

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder

from sklearn import preprocessing

from sklearn.feature_selection import SelectPercentile 

from sklearn.feature_selection import f_regression

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import mean_squared_error

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split
file=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")  ## opening data file

data=file

print(data.isnull().sum()) ## Checking for missing values in data
imputer=SimpleImputer(missing_values=np.NaN ,strategy='most_frequent')  # replacing the null values with the most frequent value

imputer.fit(data)  # same as above

data=imputer.transform(data)   # same as above

data=pd.DataFrame(data=data,columns=file.columns)
print(data["Rating"].value_counts())
data["Rating"]=data["Rating"].astype(int)

data=data.drop(["App","Last Updated"] ,axis=1)
Y=data["Rating"]   ## our target variable

data=data.loc[:,data.columns!="Rating"]   ## data features
encoder=OrdinalEncoder()

data[["Category"]]=encoder.fit_transform(data[["Category"]])   ## encoding the Category feature as it is categorical
print(data["Size"].value_counts())  ## There are different things in the size and we need to convert them into numerical data
data['Size']=data.Size.apply(lambda x: x.replace('M',''))

data['Size']=data.Size.apply(lambda x: x.replace('k',''))

data['Size']=data.Size.apply(lambda x: x.replace('Varies with device','0'))

data['Size']=data.Size.apply(lambda x: x.replace('1,000+','0'))

data['Size']=data['Size'].astype(float)
print(data["Installs"].value_counts())  ## Checking types of data in Installs feature
data['Installs']=data.Installs.apply(lambda x: x.replace('+',''))

data['Installs']=data.Installs.apply(lambda x: x.replace(',',''))

data['Installs']=data.Installs.apply(lambda x: x.replace('Free','0'))

data['Installs']=data['Installs'].astype(int)
print(data["Type"].value_counts()) ## Checking types of data in Type feature
data['Type']=data.Type.apply(lambda x: x.replace('0','Free'))

data[["Type"]]=encoder.fit_transform(data[["Type"]])
print(data["Price"].value_counts()) ## Checking types of data in Type feature
data['Price']=data.Price.apply(lambda x: x.replace('$',''))

data['Price']=data.Price.apply(lambda x: x.replace('Everyone','0'))

data['Price']=data['Price'].astype(float)
data[["Content Rating","Genres"]]=encoder.fit_transform(data[["Content Rating","Genres"]])
print(data["Current Ver"].value_counts()) ## Checking types of data in Current Ver feature
print(data["Android Ver"].value_counts()) ## Checking types of data in Current Ver feature
data['Current Ver']=data['Current Ver'].apply(lambda x: x.replace('Varies with device','0.0.0'))

data['Android Ver']=data["Android Ver"].apply(lambda x: x.replace('and up',' '))

data['Android Ver']=data["Android Ver"].apply(lambda x: x.replace('Varies with device','4.0'))

data['Android Ver']=data["Android Ver"].apply(lambda x: x.replace('W','4.0'))
print(data["Reviews"].value_counts())
data['Reviews']=data["Reviews"].apply(lambda x: x.replace('M','0'))
data[['Android Ver']]=encoder.fit_transform(data[['Android Ver']])

data[['Current Ver']]=encoder.fit_transform(data[['Current Ver']])
standardize=preprocessing.StandardScaler() ## Standardizing data 

standard_data=standardize.fit_transform(data)  

print(standard_data)

standard_data=pd.DataFrame(data=standard_data,columns=data.columns)

print(standard_data)
plt.figure(figsize=(10,10))

sns.boxplot(data=pd.DataFrame(data))

plt.plot()
X=standard_data

model = RandomForestClassifier()

model_select_scores=model_selection.cross_val_score(model,X,Y,cv=15)

print(model_select_scores)

print("Accuracy of Random Forrest Classifier:",model_select_scores.mean()*100,"%")