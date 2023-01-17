# Import necessary modules for data analysis and data visualization. 

import pandas as pd

import numpy as np



# Some visualization libraries

from matplotlib import pyplot as plt

import seaborn as sns



## Some other snippit of codes to get the setting right 

%matplotlib inline 

import warnings ## importing warnings library. 

warnings.filterwarnings('ignore') ## Ignore warning
#Loading the data

car = pd.read_csv("../input/CarPrice_Assignment.csv")

car.head()
#checking basic details

car.shape
car.info()
#checking for duplicated rows

car[car.duplicated()]
#removing car_ID column as it is insignificant 

car.drop('car_ID', axis=1, inplace=True)
#Extracting car company name from CarName column

car['CarName'] = car['CarName'].apply(lambda x: x.split()[0])
#checking the slpit and data quality issues

car['CarName'].value_counts().index.sort_values()
#replacing car names with correct ones

car['CarName'] = car['CarName'].replace('maxda','mazda')

car['CarName'] = car['CarName'].replace('Nissan','nissan')

car['CarName'] = car['CarName'].replace('porcshce','porsche')

car['CarName'] = car['CarName'].replace('toyouta','toyota')

car['CarName'] = car['CarName'].replace('vokswagen','volkswagen')

car['CarName'] = car['CarName'].replace('vw','volkswagen')
#checking the car names again

car['CarName'].value_counts()
#renaming Carname column to companyname

car = car.rename(columns={'CarName':'CompanyName'})
#checking distribution of price column

sns.distplot(car['price'], bins=50)

plt.show()
#creating symbol function and applying it of symboling column

def symbol(x):

    if x >= -3 & x <= -1:

        return 'No Risk'

    elif x>=0 and x <= 1:

        return 'Low Risk'

    else:

        return 'High Risk'

car['symboling'] = car['symboling'].apply(symbol)
car.symboling.value_counts()
#creating fuel economy metric

car['fueleconomy'] = (0.55 * car['citympg']) + (0.45 * car['highwaympg'])
#removing citympg and highwaympg cols as their effect is considered in fueleconomy

car.drop(['citympg','highwaympg'],axis=1, inplace=True)
car.head()
#recognising categorical and numerical features

cat_features = car.dtypes[car.dtypes == 'object'].index

print('No of categorical fetures:',len(cat_features),'\n')

print(cat_features)

print('*'*100)



num_features = car.dtypes[car.dtypes != 'object'].index

print('No of numerical fetures:',len(num_features),'\n')

print(num_features)
car[cat_features].head()
car[num_features].head()
#checking stats of numerical features

car[num_features].describe()
#checking correlation of all numerical variables

plt.figure(figsize=(10,8),dpi=100)

sns.heatmap(car[num_features].corr(), annot=True)

plt.show()
#checking the distribution of highly correlated numerical features with price variable

cols = ['wheelbase','carlength', 'carwidth', 'curbweight', 'enginesize','horsepower']

plt.figure(figsize=(20,4), dpi=100)

i = 1

for col in cols:

    plt.subplot(1,6,i)

    #sns.distplot(car['price'])

    sns.distplot(car[col])

    i = i+1

plt.tight_layout()

plt.show()
num_features
#visualising all the numerical features against price column



nr_rows = 5

nr_cols = 3

from scipy import stats

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3),dpi=200)



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(num_features):

            sns.regplot(car[num_features[i]], car['price'], ax = axs[r][c])

            stp = stats.pearsonr(car[num_features[i]], car['price'])

            str_title = "r = " + "{0:.3f}".format(stp[0]) + "      " "p = " + "{0:.3f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()   
#removing these columns

car.drop(['carheight','stroke','compressionratio','peakrpm'],axis=1, inplace=True)
#checking unique value counts of categorical features

car[cat_features].nunique().sort_values()
#eda on categorical columns

cols = ['fueltype','aspiration', 'doornumber','enginelocation','drivewheel']

i = 1

plt.figure(figsize=(17,8),dpi=100)

for col in cols:

    plt.subplot(1,len(cols),i)

    car[col].value_counts().plot.pie(autopct='%1.0f%%', startangle=90, shadow = True,colors = sns.color_palette('Paired'))

    i = i+1

plt.tight_layout()

plt.show()
#dropping engine location as it is highly imbalanced

car.drop('enginelocation',axis=1,inplace=True)
#making countplot for all below categorical variables

cols = ['symboling','carbody', 'enginetype', 'fuelsystem']

nr_rows = 2

nr_cols = 2

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*6.5,nr_rows*3),dpi=100)



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(cols):

            sns.countplot(car[cols[i]], ax = axs[r][c])

            

plt.tight_layout()    

plt.show()   
#visualising carname feature

plt.figure(figsize=(10,4),dpi=100)

sns.countplot(car['CompanyName'])

plt.xticks(rotation=90)

plt.show()
li_cat_feats = list(car.dtypes[car.dtypes=='object'].index)

nr_rows = 5

nr_cols = 2

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*6,nr_rows*3),dpi=200)

for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y='price', data=car, ax = axs[r][c])

plt.tight_layout()    

plt.show() 
#removing doornumber from the dataset

car.drop('doornumber',axis=1, inplace=True)
cat_features = car.dtypes[car.dtypes == 'object'].index
car[cat_features].nunique().sort_values()
#creating function for targe encoding

#credits : https://maxhalford.github.io/blog/target-encoding-done-the-right-way/

def calc_smooth_mean(df, by, on, m):

    # Compute the global mean

    mean = df[on].mean()



    # Compute the number of values and the mean of each group

    agg = df.groupby(by)[on].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']



    # Compute the "smoothed" means

    smooth = (counts * means + m * mean) / (counts + m)



    # Replace each value by the according smoothed mean

    return df[by].map(smooth)
#performing target encoding with weight of 100

for col in cat_features:

    car[col] = calc_smooth_mean(car,by=col, on='price', m=100)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(car, test_size=0.3, random_state=42)

print(df_train.shape)

print(df_test.shape)
cols = df_train.columns
#importing minmax scaler from sklearn.preprocessing and scaling the training dataframe

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_train[cols] = scaler.fit_transform(df_train[cols])
#transforming the test data set

df_test[cols] = scaler.transform(df_test[cols])
#checking minmax scaling

df_train.describe()
#checking correlation of train dataframe 

plt.figure(figsize=(15,15),dpi=100)

sns.heatmap(df_train.corr(), cmap='RdYlBu')

plt.show()
#creating function for VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(X_train):

    vif = pd.DataFrame()

    vif['Features'] = X_train.columns

    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif
#creating X and y variables

y_train = df_train.pop('price')

X_train = df_train
print(X_train.shape)
#feature selection using RFE

#In this case we are have 57 features , lets select 20 features from the data using RFE and then we will 

# remove statistical insignificant variables one by one

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train,y_train)



rfe = RFE(lr,10)

rfe.fit(X_train,y_train)



print(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))

print('*'*100)

cols_rfe = X_train.columns[rfe.support_]

print('Features with RFE support:')

print(cols_rfe)

print('*'*100)

print('Features without RFE support:')

cols_not_rfe = X_train.columns[~rfe.support_]

print(cols_not_rfe)
#taking cols with RFE support

X_train = X_train[cols_rfe]
#checking VIF

vif(X_train).head(10)
#removing carlength as it is having VIF

X_train.drop('carlength', axis=1, inplace=True)

vif(X_train).head()
#removing curbweight as it is having high VIF

X_train.drop('curbweight', axis=1, inplace=True)

vif(X_train).head()
#removing carwidth as it is having high VIF

X_train.drop('carwidth', axis=1, inplace=True)

vif(X_train).head()
#removing enginesize as it is having high VIF

X_train.drop('enginesize', axis=1, inplace=True)

vif(X_train).head()
#importing statsmodel

import statsmodels.api as sm
#Building the first model

X_train_lr = sm.add_constant(X_train)

lr_1 = sm.OLS(y_train,X_train_lr).fit()

print(lr_1.summary())

print(vif(X_train))
#removing enginetype as it is having p-value  and building 2nd model

X_train.drop('enginetype', axis=1, inplace=True)

X_train_lr = sm.add_constant(X_train)

lr_2 = sm.OLS(y_train,X_train_lr).fit()

print(lr_2.summary())

print(vif(X_train))
#removing wheelbase as it is having high VIF building 3rd model

X_train.drop('wheelbase', axis=1, inplace=True)

X_train_lr = sm.add_constant(X_train)

lr_3 = sm.OLS(y_train,X_train_lr).fit()

print(lr_3.summary())

print(vif(X_train))
#calculating residuals

y_train_pred = lr_3.predict(X_train_lr)

residuals = y_train-y_train_pred
#plotting residuals

plt.figure(dpi=100)

sns.distplot(residuals)

plt.xlabel('Residuals')

plt.show()
#checking mean of residuals

np.mean(residuals)
#scatterplot of resuduals v/s fitted values

plt.figure(figsize=(16,5),dpi=100)

plt.subplot(121)

plt.scatter(y_train_pred,residuals)

plt.xlabel('Fitted Values')

plt.ylabel('Residuals')



plt.subplot(122)

plt.scatter(y_train,residuals)

plt.xlabel('Training Values')

plt.ylabel('Residuals')

plt.show()
plt.figure(dpi=100)

sns.regplot(y_train_pred,residuals)

plt.xlabel('Fitted Values')

plt.ylabel('Residuals')
#checking the test data

df_test.describe()
#creating X and y for test dataframe

y_test = df_test.pop('price')

X_test = df_test

X_test.head()
X_train.columns
#predicting test values

X_test = X_test[X_train.columns]

X_test = sm.add_constant(X_test)

y_test_pred = lr_3.predict(X_test)
#scatterplot of y_test and y_test_pred

plt.scatter(y_test_pred,y_test)
#importing necessary libraries and methods

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

#calculating r2_score 

r2_score(y_test,y_test_pred)
#calculating mean squared error for test set

mean_squared_error(y_test,y_test_pred)
#calculating mean squared error for traning set

mean_squared_error(y_train,y_train_pred)