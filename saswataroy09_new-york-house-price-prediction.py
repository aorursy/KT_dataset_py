# Import the modules

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set(style='white', context='notebook', palette='deep') 
import matplotlib.style as style
style.use('fivethirtyeight')

from sklearn.model_selection import KFold
#from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# Data Scaler
from sklearn.preprocessing import StandardScaler

# Regression
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Metrics
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
#Importing the Dataset
data=pd.read_csv('../input/nyc-rolling-sales.csv')
#Drop unnecessary columns
data.drop('Unnamed: 0',axis=1,inplace=True)
data.drop('EASE-MENT',axis=1,inplace=True)
# Change the settings so that you can see all columns of the dataframe when calling df.head()
pd.set_option('display.max_columns',999)
data.head()
# Renaming BOROUGHS according instructions in Kaggle
data['BOROUGH'][data['BOROUGH'] == 1] = 'Manhattan'
data['BOROUGH'][data['BOROUGH'] == 2] = 'Bronx'
data['BOROUGH'][data['BOROUGH'] == 3] = 'Brooklyn'
data['BOROUGH'][data['BOROUGH'] == 4] = 'Queens'
data['BOROUGH'][data['BOROUGH'] == 5] = 'Staten Island'
data.info()
data.isnull().sum()
#No zero values

#SALE PRICE is object but should be numeric
data['SALE PRICE'] = pd.to_numeric(data['SALE PRICE'], errors='coerce')
data['YEAR BUILT'] = pd.to_numeric(data['YEAR BUILT'], errors='coerce')

#LAND and GROSS SQUARE FEET is object but should be numeric
data['LAND SQUARE FEET'] = pd.to_numeric(data['LAND SQUARE FEET'], errors='coerce')
data['GROSS SQUARE FEET']= pd.to_numeric(data['GROSS SQUARE FEET'], errors='coerce')

#SALE DATE is object but should be datetime
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')

#Both TAX CLASS attributes should be categorical
data['TAX CLASS AT TIME OF SALE'] = data['TAX CLASS AT TIME OF SALE'].astype('category')
data['TAX CLASS AT PRESENT'] = data['TAX CLASS AT PRESENT'].astype('category')
data['ZIP CODE'] = data['ZIP CODE'].astype('category')
#Check for possible duplicates
sum(data.duplicated(data.columns))
#Drop duplicates
data = data.drop_duplicates(data.columns, keep='last')
#Check that the duplicates have been removed
sum(data.duplicated(data.columns))
data.shape
#Capture necessary columns
variables=data.columns
count=[]
for variable in variables:
    length=data[variable].count()
    count.append(length)
#Plot number of available data per variable
plt.figure(figsize=(30,6))
sns.barplot(x=variables, y=count)
plt.title('Available data in percent', fontsize=15)
plt.show()
#20% of sales price data is null
# Remove observations with missing SALE PRICE
data = data[data['SALE PRICE'].notnull()]
len(data)
data.describe(include='all')
plt.figure(figsize=(12,4))
#Plot the data and configure the settings
#CountPlot -->  histogram over a categorical, rather than quantitative, variable.
sns.countplot(x='BOROUGH',data=data)
#Remove outliers for graph fitting in separate df
data=data[data['SALE PRICE']<200000000]
data=data[data['YEAR BUILT']>1875]
data=data[data['LAND SQUARE FEET']<20000]
data=data[data['GROSS SQUARE FEET'] < 20000]
print('After removing outliers {}'.format(data.shape))
# Only a handful of properties with 0 total units are remaining and they will now be deleted
data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] < 50)] 
#Remove data where commercial + residential doesn't equal total units
data = data[data['TOTAL UNITS'] == data['COMMERCIAL UNITS'] + data['RESIDENTIAL UNITS']]
print('After removing total units courrupted data {}'.format(data.shape))
# Removes all NULL values
data = data[data['LAND SQUARE FEET'].notnull()] 
data = data[data['GROSS SQUARE FEET'].notnull()] 
# Only a handful of properties with 0 total units are remaining and they will now be deleted
data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] < 50)] 
print('After removing nulls and total unites<50 is {}'.format(data.shape))
#Too many zeroes whhich is affecting visualisation,Impute and re-execute
#DistPlot -- > plot a univariate distribution of observations.
#plt.figure(figsize=(12,4))
sns.distplot(a=data['YEAR BUILT'],kde=True,hist=True)
#For categorical features 
plt.figure(figsize=(20,10))
fig, axs = plt.subplots(ncols=2,nrows=1)
fig.tight_layout(pad=0, h_pad=0, w_pad=0)
fig.subplots_adjust(wspace=0.7)
sns.catplot(x="TAX CLASS AT PRESENT", y="SALE PRICE",color='blue', data=data, ax=axs[0],kind='bar',legend=False);
sns.catplot(x="TAX CLASS AT TIME OF SALE", y="SALE PRICE",color='red', data=data, ax=axs[1],kind='bar',legend=False);
#Generate a column season
def get_season(x):
    if x==1:
        return 'Summer'
    elif x==2:
        return 'Fall'
    elif x==3:
        return 'Winter'
    elif x==4:
        return 'Spring'
    else:
        return ''
data['seasons']=data['SALE DATE'].apply(lambda x:x.month)
data['seasons']=data['seasons'].apply(lambda x:(x%12+3)//3)
data['seasons']=data['seasons'].apply(get_season)
plt.figure(figsize=(18,8))
df_wo_manhattan=data.loc[data['BOROUGH']!='Manhattan']
#Remove Manhattan to get a good graph
sns.relplot(x="BOROUGH", y="SALE PRICE",hue='seasons' ,kind="line", data=df_wo_manhattan,legend='full');
#Regression plotting deweighing outliers
sns.regplot(x="LAND SQUARE FEET", y="SALE PRICE", data=data,ci=100, robust= False)
#Regression plotting deweighing outliers
sns.regplot(x="GROSS SQUARE FEET", y="SALE PRICE", data=data,ci=100, robust= False)
sns.scatterplot(x='RESIDENTIAL UNITS',y='SALE PRICE',data=data)
sns.pairplot(data=data, hue='BOROUGH')
data['Building Age During Sale']=data['SALE DATE'].apply(lambda x:x.year)
data['Building Age During Sale']=data['Building Age During Sale']-data['YEAR BUILT']
data.tail()
plt.figure(figsize=(10,6))
sns.boxplot(x='BOROUGH', y='SALE PRICE', data=data)
plt.title('Sale Price Distribution by Borough')
plt.show()
plt.figure(figsize=(10,6))
order = sorted(data['BUILDING CLASS CATEGORY'].unique())
sns.boxplot(x='BUILDING CLASS CATEGORY', y='SALE PRICE', data=data, order=order)
plt.xticks(rotation=90)
#plt.ylim(0,2)
plt.title('Sale Price Distribution by Bulding Class Category')
plt.show()
# Correlation Matrix

# Compute the correlation matrix
d= data[['TOTAL UNITS','GROSS SQUARE FEET','SALE PRICE', 'Building Age During Sale', 'LAND SQUARE FEET', 'RESIDENTIAL UNITS', 
         'COMMERCIAL UNITS']]
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, 
            square=True, linewidths=.5, annot=True, cmap=cmap)
plt.yticks(rotation=0)
plt.title('Heatmap for all Numerical Variables')
plt.show()
column_model=['BOROUGH','BUILDING CLASS CATEGORY','COMMERCIAL UNITS','GROSS SQUARE FEET',
               'SALE PRICE','Building Age During Sale','LAND SQUARE FEET','RESIDENTIAL UNITS','seasons']
data_model=data.loc[:,column_model]
#OneHotEncoded features
one_hot_features=['BOROUGH','BUILDING CLASS CATEGORY','seasons']
#Check how many columns will be created
longest_str=max(one_hot_features,key=len)
total_num_of_unique_cat=0
for feature in one_hot_features:
    num_unique=len(data_model[feature].unique())
    print('{} : {} unique categorical values '.format(feature,num_unique))
    total_num_of_unique_cat+=num_unique
print('Total {} will be added with one hot encoding'.format(total_num_of_unique_cat))
# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(data_model[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)
#Explore sales price column
data[data['SALE PRICE']==0.0].sum().count()
#We need to impute the zero values columns,otherwise the data will be lost
#Drop original categorical features and keep one hot encoded feature
data_model.drop(one_hot_features,axis=1,inplace=True)
data_model=pd.concat([data_model,one_hot_encoded],axis=1)
data_model.head()
plt.figure(figsize=(12,8))
g=sns.distplot(data_model['SALE PRICE'],bins=2)
plt.title('Histogram of SALE PRICE')
plt.show()
data_model.head()
data_model['SALE PRICE']
#Remove Sale price == 0 to fit the normalised curve
data_model=data_model[data_model['SALE PRICE']!=0]
#data_model_SALE_ZEROES=data_model
#data_model.head()
data_model
# Take the log and normalise
#data_model=data_model[data_model['SALE PRICE']!=0]
#data_model=data_model[data_model['SALE PRICE'].isnull()]
data_model['SALE PRICE'] = StandardScaler().fit_transform(np.log(data_model['SALE PRICE']).values.reshape(-1,1))
plt.figure(figsize=(10,6))
sns.distplot(data_model['SALE PRICE'])
plt.title('Histogram of Normalised SALE PRICE')
plt.show()
# Add 1 to Units to prevent having log0 which tends to infinity
data_model['COMMERCIAL UNITS'] = data_model['COMMERCIAL UNITS'] + 1
data_model['RESIDENTIAL UNITS'] = data_model['RESIDENTIAL UNITS'] + 1

# Take the log and standardise
data_model['COMMERCIAL UNITS'] = StandardScaler().fit_transform(np.log(data_model['COMMERCIAL UNITS']).values.reshape(-1,1))
data_model['RESIDENTIAL UNITS'] = StandardScaler().fit_transform(np.log(data_model['RESIDENTIAL UNITS']).values.reshape(-1,1))
# Add 1 to Units
data_model['GROSS SQUARE FEET'] = data_model['GROSS SQUARE FEET'] + 1
data_model['LAND SQUARE FEET'] = data_model['LAND SQUARE FEET'] + 1

# Take the log and standardise
data_model['GROSS SQUARE FEET'] = StandardScaler().fit_transform(np.log(data_model['GROSS SQUARE FEET']).values.reshape(-1,1))
data_model['LAND SQUARE FEET'] = StandardScaler().fit_transform(np.log(data_model['LAND SQUARE FEET']).values.reshape(-1,1))
# Add 1 to BUILDING AGE
data_model['Building Age During Sale'] = data_model['Building Age During Sale'] + 1

# Take the log and standardise
data_model['Building Age During Sale'] = StandardScaler().fit_transform(np.log(data_model['Building Age During Sale']).values.reshape(-1,1))
data_model.describe()
#Split data into training and testing set with 80% of the data going into training
y=data_model['SALE PRICE']
X=data_model.drop('SALE PRICE',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print('Size of Training data: {} \n Size of test data: {}'.format(X_train.shape[0],X_test.shape[0]))
data_model.shape[0]
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)
y_pred=linear_reg.predict(X_test)

cv_scores_linreg = cross_val_score(linear_reg, X_train, y_train, cv=5)
r2=linear_reg.score(X_test, y_test)
print("R^2: {}".format(r2))
adj_r2 = 1 - (1 - r2 ** 2) * ((X_train.shape[1] - 1) / (X_train.shape[0] - X_train.shape[1] - 1))
print("Adjusted R^2: {}".format(adj_r2))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_linreg)))
# Print the 5-fold cross-validation scores
print(cv_scores_linreg)
sns.distplot(y_test)
plt.show()
sns.distplot(y_pred)
plt.show()
