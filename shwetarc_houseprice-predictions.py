# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
#from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline


#load data
df= pd.read_csv('D:/Kaggle/House Price/all/train.csv', index_col=False)
#pandas.read_csv(filepath_or_buffer, yahan file ka location ayega
#sep=', ', default value=','
#header='infer', header mein jo column name rakhne hain wo row no. likhenge 
#names=None, yahan colummn k naam ki list pass karna hai 
#index_col=None, row labels ki list
#usecols=None, columns names ki list
#prefix=None, prefix for col names/nos
#mangle_dupe_cols=True, duplicate cols will be named distinctly
#dtype=None, datatype
#engine=None, parser engine
#converters=None, functions for converting certain values

#check the columns
df.columns
len(df.columns)
#object ka attribute

df.describe()
#percentiles- returns 25, 50, 75th percentile
#include- all the datatypes to be included
#exclude - blacklist for datatypes to be excluded

#descriptive statistics summary
df['SalePrice'].describe()

#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#data cleanning

#remove columns with missing values>80% 
df=df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence' ], axis=1)


# check nulls
print('Columns With Nulls')

#missing data
total2 = df.isnull().sum().sort_values(ascending=False)
percent2 = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
missing_data2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data2.head(20)

#data imputation
#separate numerical and categorical values
#impute numberical nulls with mean
#impute categorical nulls with mode

colNames=df.columns


#NAs as string to be removed still
#df['LotFrontage']=np.where(df['LotFrontage'].isnan(), '',df['LotFrontage'] )
df=df.fillna(df.mode())
 
for c in colNames:
    if(df[c].dtype=='object'):
        df[c] = pd.Categorical(df[c]) # convert string / categoric to numeric
        df[c] = df[c].cat.codes
    else:
        df=df.fillna(df.mean())


#correlation matrix
corrmat = pd.DataFrame(df.corr())
corrmat.head()
corrmat.shape


# DataFrame.corr(method='pearson', min_periods=1)[source]
#method: can be person, kendall, spearman
#min_periods: min. no . obs. to be passed

for i in range (0,76):
    for j in range(0,76):
        if (corrmat.iat[i,j] > 0.8 and corrmat.iat[i,j]!=1):
            print('Row: ', i, 'Col: ', j, 'Value: ', corrmat.iat[i,j])
for i in range(0,76):
    for j in range(0,76):
        if (corrmat.iat[i,j] < -0.8 and corrmat.iat[i,j]!=1):
            print('Row: ', i, 'Col: ', j, 'Value: ', corrmat.iat[i,j])

#remove columns with high correlations
df=df.drop(columns=['Exterior1st', 'TotalBsmtSF', 'GrLivArea', 'Fireplaces', 'GarageCars', 'GarageQual']) 

for i in range (0,70):
    for j in range(0,70):
        if (corrmat.iat[i,j] > 0.8 and corrmat.iat[i,j]!=1):
            print('Row: ', i, 'Col: ', j, 'Value: ', corrmat.iat[i,j])
           

#subplots- nrows, ncols- for grid
#sharex, sharey - shared properties for x and y
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, vmax=.8, square=True);


#histogram
sns.distplot(df['SalePrice']);
#a : Series, jo data ka distribution chaiye
#c=color
#vertical(True)= y-axis

#scatter plot grlivarea/saleprice
plt.scatter(x=df['GrLivArea'], y=df['SalePrice']);
#x= x axis ka data
#y= y axis ka data
#c= color
#marker= marker style


#scatter plot totalbsmtsf/saleprice
plt.scatter(x=df['TotalBsmtSF'], y=df['SalePrice']);



plt.bar(df['OverallQual'], df['SalePrice'], align='center')
#x= x co-ordinates
#align= alignment of bars
#width= width of bars, default is 0.8
#height= height of bars


plt.scatter(x=df['YearBuilt'], y=df['SalePrice'], c=df['OverallQual']);
#plt.bar(df['YearBuilt'], df['SalePrice'], align='center')
#plt.rcParams["figure.figsize"] = [16,9]
#rc parameters: allows you to manage figure parameters

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size = 2.5)
#hue : string (variable name), optional
#markers : marker code
plt.show();

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
df['HasBsmt'] = 0 
df.loc[df['TotalBsmtSF']>0,'HasBsmt'] = 1

#basement area/total area
df['basementRatio']=df['TotalBsmtSF']/df['GrLivArea']

plt.scatter(x=df['basementRatio'], y=df['SalePrice']);


#Cluster analysis
df.groupby(['MSZoning'])['SalePrice'].mean()
df.groupby(['HeatingQC'])['SalePrice'].mean()
df.groupby(['CentralAir'])['SalePrice'].mean()
df.groupby(['FullBath'])['SalePrice'].mean()
df.groupby(['GarageType'])['SalePrice'].mean()

#Cluster analysis
#DataFrame.groupby(by=None, axis=0, level=None, squeeze=False, observed=False)
#Attributes
# By:  : mapping, function, label, or list of labels. Used to determine the groups for the groupby
#axis : {0 or ‘index’, 1 or ‘columns’}, default is 0. Split along rows (0) or columns (1).
#level : int, level name, or sequence of such, default None. If the axis is hierarchical, group by particular levels/level.
#observed : bool, default False. This only applies if any of the groupers are Categoricals. If True: only show observed values for categorical groupers. If False: show all values for categorical groupers.
#squeeze : bool, default False. Reduce the dimensionality of the return type if possible, otherwise return a consistent type.


df.groupby(['Utilities'])['SalePrice'].mean()
plt.scatter(df['Utilities']==0, df['SalePrice'], color='#7f6d5f',  label='No Utiities', linewidths=1)
plt.scatter(df['Utilities']==1, df['SalePrice'], color='#557f2d', label='Utilities', linewidths=1)
plt.legend()
plt.show()

# sk linear
from sklearn.linear_model import LinearRegression
# stats model
import statsmodels.api as sm

#X,y
colNames=df.columns.tolist()
y = df['SalePrice'].values
colNames.remove('SalePrice')
X = df[colNames].values

model = LinearRegression()
model.fit(X,y)

# regression summary
Xconst = sm.add_constant(df[colNames])
Oconst = sm.OLS(y, Xconst)
Mconst = Oconst.fit()
print(Mconst.summary())

#Significant columns- MSSubClass, LotFrontage, LotArea, Street, LandContour,  Neighborhood, Condition2, OverallQual, OverallCond, YearBuilt, RoofMatl, MasVnrType,  MasVnrArea, ExterQual, BsmtQual, BsmtCond, BsmtExposure, BsmtFinSF1, BsmtFinType2, BsmtFinSF2, 1stFlrSF, 2ndFlrSF, BsmtFullBath, BedroomAbvGr, KitchenAbvGr, KitchenQual, TotRmsAbvGrd, Functional, GarageArea, WoodDeckSF,  ScreenPorch, SaleCondition
keep_cols=['MSSubClass', 'LotFrontage', 'LotArea', 'Street',  'Neighborhood', 'Condition2', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofMatl', 'MasVnrType',  'MasVnrArea', 'ExterQual', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'BsmtFullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'GarageArea', 'WoodDeckSF',  'ScreenPorch', 'SaleCondition']
df=df.loc[:, keep_cols]

#check regression summary again
#X,y
colNames=df.columns.tolist()
y = df['SalePrice']
colNames=colNames.remove('SalePrice')
X = df[colNames].values

model = LinearRegression()
model.fit(X,y)

# regression summary
Xconst = sm.add_constant(df[colNames])
Oconst = sm.OLS(y, Xconst)
Mconst = Oconst.fit()
print(Mconst.summary())


#read test data set
Xnew= pd.read_csv('D:/Kaggle/House Price/all/test.csv')


test_colNames=Xnew.columns.tolist()


Xnew=Xnew.fillna(Xnew.mode())

for c in test_colNames:
    if(Xnew[c].dtype=='object'or Xnew[c].dtype=='O'):
        Xnew[c] = pd.Categorical(Xnew[c]) 
        Xnew[c] = Xnew[c].cat.codes
    else:
        Xnew[c]=Xnew[c].fillna(Xnew[c].mean())
        
#train_colNames=X.columns
#test_colNames=Xnew.columns

#subset test for similar columns in train dataset
Xnew = Xnew.loc[:,keep_cols]


#prediction


yNew = model.predict(Xnew.as_matrix())


import numpy as np
y=np.delete(y, len(y)-1)
y=np.float32(y)



