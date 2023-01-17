# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import csv as csv
import pandas as pd
import seaborn as sns
from sklearn import svm
import matplotlib
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train_path ='../input/house-prices-advanced-regression-techniques/train.csv'


df_train = pd.read_csv(train_path)
df_train.drop(['Id'], axis =1)
#Here we want to analyze the target variable('SalesPrice')
df_train['SalePrice'].describe()
# Plot Histogram(we can see that the distribution is not entirely uniform)
sns.distplot(df_train['SalePrice'] , fit=norm);
#we can also see here that it is not very uniform(we will need to apply a technique later on to make a gaussian Distribution)
res = stats.probplot(df_train['SalePrice'], plot=plt)
#wehen calculating skewness and kurtosis, they are also very high
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#getting all the numerical data points in one set
train_numerical = df_train.select_dtypes(exclude=['object'])

#in the numerical data set we see that all the 0 values are null values in this case, so we can just replace it with 'None'
train_numerical.fillna(0, inplace = True)

#getting all the categorical columns in one dataset(include all the columns in the dataset)
train_categoric = df_train.select_dtypes(include=['object'])
train_categoric.fillna('NONE', inplace = True)#fill all the values in the categorical  with 'None', bc all the 

print(train_categoric)
#Gets the top 10 most correlated variables in the heatmap
corrmat = df_train.corr()

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index #stors the top 10 columns that have the highest correlation values
cm = np.corrcoef(df_train[cols].values.T) #stores the correlation values of the top 10 most correlated columns
sns.set(font_scale=1.25) #sns is the seaborn library object
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#make a new data frame with only the 10 most correlated columns columns 
most_corr = pd.DataFrame(cols)
#now we wanna plot n analyze each of the op 10 variables vs SalesPrice and remove outliers
#Remove any data points that do not show a correlation
#overall quality vs PRice
overallqual = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1) #get a new df with just those two variables
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x= 'OverallQual', y="SalePrice", data=overallqual)
fig.axis(ymin=0, ymax=800000);
#since there is no outliers we can move
# Living Area vs Sale Price
sns.jointplot(x=df_train['GrLivArea'], y= df_train['SalePrice'], kind='reg')
#removing the outliers in the bottom right corner
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) 
                         & (df_train['SalePrice']<300000)].index).reset_index(drop=True)

#plotting again to see what the new graph looks like
sns.jointplot(x=df_train['GrLivArea'], y=df_train['SalePrice'], kind='reg')
#plotting number of cars for garave vs Sales Price
sns.boxplot(x=df_train['GarageCars'], y=df_train['SalePrice'])
#based off the graph it doesnt really make sense that the 4 car garages would be on average cheaper than a 3 car carage, so we can get rid of the to better fit our model
df_train = df_train.drop(df_train[(df_train['GarageCars']>3) 
                         & (df_train['SalePrice']<300000)].index).reset_index(drop=True)
sns.boxplot(x=df_train['GarageCars'], y=df_train['SalePrice'])
#Garage Area vs Sales Price
sns.jointplot(x=df_train['GarageArea'], y=df_train['SalePrice'], kind='reg')#this looks fine
#Removing the outlier in the bottom righ corner
df_train = df_train.drop(df_train[(df_train['GarageArea']>1000) 
                         & (df_train['SalePrice']<100000)].index).reset_index(drop=True)

sns.jointplot(x = df_train['GarageArea'], y = df_train['SalePrice'], kind='reg')#this looks fine
#Basment Square Footage vs SalesPRice

sns.jointplot(x = df_train['TotalBsmtSF'], y = df_train['SalePrice'], kind='reg')#this looks fine
sns.jointplot(x=df_train['1stFlrSF'], y=df_train['SalePrice'], kind='reg')
# Total Rooms vs Sale Price
sns.boxplot(x=df_train['TotRmsAbvGrd'], y=df_train['SalePrice'])
#the data after 11 total rooms is not very well correlated so we will delete the rooms above 11
df_train = df_train.drop(df_train[(df_train['TotRmsAbvGrd']>11) 
                         & (df_train['SalePrice']>0)].index).reset_index(drop=True)
# Re-graph Total Rooms vs Sale Price
sns.boxplot(x=df_train['TotRmsAbvGrd'], y=df_train['SalePrice'])
# Year Built vs Sale Price

dataYB = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=dataYB)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#plot to see percentage of NA values for each column

print(df_train)
#the lot frontage varies by the neighborhood so well group that by neighborhood
df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform


df_train['MasVnrType']=df_train['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])
df_train['MasVnrArea']=df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0])


df_train.drop(['Alley'],axis=1,inplace=True) #sropping this because it contains mostly NA values

df_train['GarageYrBlt']= df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].mean()) #decided not to drop this and to replace with mean

df_train['GarageFinish'] = df_train['GarageFinish'].fillna(df_trian['GarageFinish'].mode()[0])
df_train['GarageQual'] = df_train['GarageQual'].fillna(df_train['GarageQual'].mode()[0])
df_train['GarageCond'] = df_train['GarageCond'].fillna(df_train['GarageCond'].mode()[0])
df_train['BsmtCond'] = df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])
df_train['BsmtQual'] = df_train['BsmtQual'].fillna(df_train['BsmtQual'].mode()[0])
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna(df_train['FireplaceQu'].mode()[0])

df_train['GarageType']=df_train['GarageType'].fillna(df_train['GarageType'].mode()[0])



df_train.drop(['PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)


df_train['BsmtCond'].isnull().values.any() #checking to make sure null vlaues were replaced


df_train.drop(['PoolArea', 'ScreenPorch','MiscVal'],axis=1,inplace=True)#dropping these columns because they have >50% null values
df_train['WoodDeckSF']= df_train['WoodDeckSF'].fillna(df_train['WoodDeckSF'].mean())
df_train['OpenPorchSF']= df_train['OpenPorchSF'].fillna(df_train['OpenPorchSF'].mean())




df_train.isnull().sum()
#Graphing a heatmap of the null values we have left, if null values are still present, continue cleaning data
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# if there is a very small number of null values left drop all the remainding rows with null values, 
#df_train.dropna(inplace = True)


df_train['Total_SF'] = df_train['1stFlrSF']+ df_train['2ndFlrSF']+ df_train['TotalBsmtSF']+ df_train['GarageArea']+ df_train['GrLivArea']+ df_train['PoolArea']+ df_train['WoodDeckSF']
#helps us to see the data objects in the data 
df_train.info()
#Gettinf all the categorical data columns
cat_columns = df_train.select_dtypes(include=['object']).columns

#getting all the numerical data columns
num_columns = df_train.select_dtypes(include=['int64', 'float64']).columns

#read in the data
test_df = pd.read_csv('test.csv')
test_df.shape #Test Data has no SalesPrice column
test_df.head()

test_df['MSZoning'].isnull().sum()
test_df['MSZoning']= test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0]) #MSZoning didnt have null vlaues for train but has it for test

test_df['BsmtCond'] = test_df['BsmtCond'].fillna(test_df['BsmtCond'].mode()[0])
test_df['BsmtQual'] = test_df['BsmtQual'].fillna(test_df['BsmtQual'].mode()[0])

test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna(test_df['FireplaceQu'].mode()[0])
test_df['GarageType']= test_df['GarageType'].fillna(test_df['GarageType'].mode()[0])


df_train['GarageYrBlt']= df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].mean())


test_df.shape

test_df['GarageFinish'] = test_df['GarageFinish'].fillna(test_df['GarageFinish'].mode()[0])
test_df['GarageQual'] = test_df['GarageQual'].fillna(test_df['GarageQual'].mode()[0])
test_df['GarageCond'] = test_df['GarageCond'].fillna(test_df['GarageCond'].mode()[0])


test_df.drop(['PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True) #drop these variables becuase >50% NA values

test_df.drop(['Id'], axis =1, inplace = True)

test_df['MasVnrType']= test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])
test_df['MasVnrArea']= test_df['MasVnrArea'].fillna(test_Df['MasVnrArea'].mode()[0])


#check to see if there are any null values
sns.heatmap(test_df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna(test_df['BsmtExposure'].mode()[0])

sns.heatmap(test_df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna(test_df['BsmtFinType2'].mode()[0])

#display the columns which still contain any null vlaues and handle them
test_df.loc[:, test_df.isnull().any()].head()

##Replace the rest
test_df['Utilities'] = test_df['Utilities'].fillna(test_df['Utilities'].mode()[0])
test_df['Exterior1st'] = test_df['Exterior1st'].fillna(test_df['Exterior1st'].mode()[0])
test_df['Exterior2nd'] = test_df['Exterior2nd'].fillna(test_df['Exterior2nd'].mode()[0])
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].fillna(test_df['BsmtFinType1'].mode()[0])
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean()[0])
test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(test_df['BsmtFinSF2'].mean()[0])
test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean()[0])
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean()[0])
test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mode()[0])
test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(test_df['BsmtHalfBath'].mode()[0])
test_df['KitchenQual'] = test_df['KitchenQual'].fillna(test_df['KitchenQual'].mode()[0])
test_df['Functional'] = test_df['Functional'].fillna(test_df['Functional'].mode()[0])
test_df['GarageCars'] = test_df['GarageCars'].fillna(test_df['GarageCars'].mode()[0])
test_df['GarageArea'] = test_df['GarageArea'].fillna(test_df['GarageArea'].mean()[0])
test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])






test_df.shape

#after handling the null values, we want to store into a csv file, so that we can combine test and training data set
test_df.to_csv('formulatedtest.csv', index = False)

test_df.head(5)

#we do not use this function until after we concat the test_df and train_df(We use it for final_df with both )
###
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True) #this gets rid of one of the columns(dummy trap)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
####


#we make a copy here
main_df = train_df.copy()


#we want to normalize all the variable in our dataset so it can be better applicable 
#we pass in the data frame and recieve and a normalizes dataframe with skewness -1 to 1
#WE APPLY THIS FUNCTION TO OUR FINAL_DF
def bcox_transform(df):
    '''
    IN: Original dataframe 
    OUT: Dataframe with box-cox normalized numerical values. 
    Specified skewness threshold 1 and -1 
    '''
    lam = 0.15
    for feat in df._get_numeric_data():
        if df[feat].skew()>1.0 or df[feat].skew()<-1.0:
            df[feat]=boxcox1p(df[feat], lam)
    return df

#perform for the final data set after concatinating test_df and train_df


final_df = pd.concat([train_df,test_df],axis=0) #adding them row-by-row
#Test Data does not have a sales price column so all of those rows should be Nan
final_df['SalePrice']

#perform one hot encoding on the final dataset
final_df=category_onehot_multcols(columns)

#normalizing the variables of the final dataframe
final_df= bcox_transform(final_df)


#Here we want to get rid of duplicate column
final_df =final_df.loc[:,~final_df.columns.duplicated()]
#Getting necessary paramters for splitting
ntrain = df_train.shape[0]
ntest = test_df.shape[0]


#spliiting into train and test
Train = final_df.iloc[:ntrain,:]
Test = final_df.iloc[ntest:,:]
#get parameters needed for modeling
Test.drop(['SalePrice'],axis=1,inplace=True) #we can remove this because it is entirely NA values
#we want all the dependent variabels in the x_Train
X_Train = Train.drop(['SalePrice'],axis=1)
#we want the independetnt vairable(SalesPrice) in the Y-train
y_Train = Train['SalesPrice']



import xgBoost

#create an object of Regressor
classifier=xgboost.XGBRegressor()
classifier.fit(X_train, y_train)


#this gets you the paramters to pass into the classifier object


#We want to save the file so we do not have iterate training the model over and over again
import pickle
pickle.dump(classifier, open('finalized_model.pkl', 'wb'))

#we use classsifier object on our test data set
y_predict = classifier.predict(test_df)
print(y_predict)
#create a sample Ssubmission file and submit

pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('sample_submission.csv')

#we will take the ID column from the sample submission file 
datasets = pd.concat([sub_df['Id'], pred], axis =1)
datasets.columns = ['Id', 'SalesPrice']
datasets.to_csv('sample_submission.csv', index = False)

