import pandas as pd

pd.set_option('display.max_column',None)

pd.set_option('display.max_rows',None)

pd.set_option('display.max_seq_items',None)

pd.set_option('display.max_colwidth', 500)

pd.set_option('expand_frame_repr', True)

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import warnings

warnings.filterwarnings('ignore')



import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from random import sample

import sklearn

from numpy.random import uniform

from math import isnan



from sklearn import metrics

from sklearn.linear_model import LassoCV

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import KFold, cross_val_score



from scipy.special import boxcox1p, inv_boxcox1p

from scipy.stats import skew,norm
def countOutlier(df_in, col_name):

    if df_in[col_name].nunique() > 2:

        orglength = len(df_in[col_name])

        q1 = df_in[col_name].quantile(0.00)

        q3 = df_in[col_name].quantile(0.95)

        iqr = q3-q1 #Interquartile range 

        fence_low  = q1-1.5*iqr 

        fence_high = q3+1.5*iqr 

        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

        newlength = len(df_out[col_name])

        return round(100 - (newlength*100/orglength),2)  

    else:

        return 0



def drop_columns(dataframe, axis =1, percent=0.3):

    '''

    * drop_columns function will remove the rows and columns based on parameters provided.

    * dataframe : Name of the dataframe  

    * axis      : axis = 0 defines drop rows, axis =1(default) defines drop columns    

    * percent   : percent of data where column/rows values are null,default is 0.3(30%)

    '''

    df = dataframe.copy()

    ishape = df.shape

    if axis == 0:

        rownames = df.transpose().isnull().sum()

        rownames = list(rownames[rownames.values > percent*len(df)].index)

        df.drop(df.index[rownames],inplace=True) 

        print("\nNumber of Rows dropped\t: ",len(rownames))

    else:

        colnames = (df.isnull().sum()/len(df))

        colnames = list(colnames[colnames.values>=percent].index)

        df.drop(labels = colnames,axis =1,inplace=True)        

        print("Number of Columns dropped\t: ",len(colnames))

        

    print("\nOld dataset rows,columns",ishape,"\nNew dataset rows,columns",df.shape)



    return df



def rmse_cv(model, X, y, kfolds):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse) 
df_org=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_org=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df=df_org.loc[:, df_org.columns != 'Id']

test=test_org.loc[:, test_org.columns != 'Id']
df.head()
test.head()
df.shape
test.shape
df.info()
df.describe()
df.dtypes.value_counts()
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
df.select_dtypes('int64').apply(pd.Series.nunique, axis = 0)
df.select_dtypes('float64').apply(pd.Series.nunique, axis = 0)
#Checking null values in each cols.

(df.isnull().sum()*100/df.shape[0]).sort_values(ascending=False)
#Checking null values in each cols.

(test.isnull().sum()*100/test.shape[0]).sort_values(ascending=False)
# Plotting null values for first 20 columns

isna_train = df.isnull().sum().sort_values(ascending=False)

isna_train[:20].plot(kind='bar')
df = drop_columns(df, axis =1, percent=0.4)

test = drop_columns(test, axis =1, percent=0.4)
#Checking null values in each cols and printing top 15 only

missing_pc = (df.isnull().sum()*100/df.shape[0]).sort_values(ascending=False)[:15]

missing_pc
#Finding the columns whether they are categorical or numerical

cols = df[missing_pc.index[:15]].columns

num = df[missing_pc.index[:15]]._get_numeric_data().columns

num_cols = list(num)

print("Numerical Columns",num_cols)

cat_cols=list(set(cols) - set(num))

print("Categorical Columns:",cat_cols)
#Checking null values in each cols and printing top 15 only

missing_pc = (test.isnull().sum()*100/test.shape[0]).sort_values(ascending=False)[:30]

missing_pc
#Finding the columns whether they are categorical or numerical

cols = df[missing_pc.index[:30]].columns

num = df[missing_pc.index[:30]]._get_numeric_data().columns

num_cols_test = list(num)

print("Numerical Columns",num_cols_test)

cat_cols_test=list(set(cols) - set(num))

print("Categorical Columns:",cat_cols_test)
i = 1

plt.figure(figsize=[15,15])

for cols in num_cols:

    plt.subplot(3,3,i)

    sns.distplot(df[cols].dropna().values)

    i = i + 1

plt.suptitle("Distribution of Data for Numeric Columns")

plt.show()
# Looking at mean value

df.groupby('Neighborhood')['LotFrontage'].mean()
# Looking at median values

df.groupby('Neighborhood')['GarageYrBlt','MasVnrArea'].median()
# Imputing missing values

df['LotFrontage']=df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

df['GarageYrBlt']=df.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))

df['MasVnrArea']=df.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))





test['LotFrontage']=test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

test['GarageYrBlt']=test.groupby('Neighborhood')['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))

test['MasVnrArea']=test.groupby('Neighborhood')['MasVnrArea'].transform(lambda x: x.fillna(x.median()))

test['BsmtHalfBath']=test.groupby('Neighborhood')['BsmtHalfBath'].transform(lambda x: x.fillna(x.median()))

test['BsmtFullBath']=test.groupby('Neighborhood')['BsmtFullBath'].transform(lambda x: x.fillna(x.median()))

test['TotalBsmtSF']=test.groupby('Neighborhood')['TotalBsmtSF'].transform(lambda x: x.fillna(x.median()))

test['GarageCars']=test.groupby('Neighborhood')['GarageCars'].transform(lambda x: x.fillna(x.median()))

test['BsmtUnfSF']=test.groupby('Neighborhood')['BsmtUnfSF'].transform(lambda x: x.fillna(x.median()))

test['GarageArea']=test.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.median()))

test['BsmtFinSF2']=test.groupby('Neighborhood')['BsmtFinSF2'].transform(lambda x: x.fillna(x.median()))

test['BsmtFinSF1']=test.groupby('Neighborhood')['BsmtFinSF1'].transform(lambda x: x.fillna(x.median()))
# Plotting again after imputing missing values

i = 1

plt.figure(figsize=[15,15])

for cols in num_cols:

    plt.subplot(3,3,i)

    sns.distplot(df[cols].dropna().values)

    i = i + 1

plt.suptitle("Distribution of Data for Numeric Columns after filling NA Values")

plt.show()
i = 1

plt.figure(figsize=[15,15])

for cols in cat_cols:

    plt.subplot(4,3,i)

    sns.countplot(cols,data=df)

    i = i + 1

plt.suptitle("Distribution of Data for Categorical Columns")

plt.show()
plt.figure(figsize=(20, 12))

sns.countplot('GarageCond', hue = 'Neighborhood', data = df)

plt.show()
plt.figure(figsize=(20, 12))

sns.countplot('BsmtExposure', hue = 'Neighborhood', data = df)

plt.show()
plt.figure(figsize=(20, 12))

sns.countplot('BsmtFinType1', hue = 'Neighborhood', data = df)

plt.show()

plt.figure(figsize=(20, 12))

sns.countplot('BsmtCond', hue = 'Neighborhood', data = df)

plt.show()
for column in cat_cols:

    df[column]=df.groupby('Neighborhood')[column].fillna(df[column].mode()[0])

    test[column]=test.groupby('Neighborhood')[column].fillna(test[column].mode()[0])
# Plotting agaain after filling null values

i = 1

plt.figure(figsize=[15,15])

for cols in cat_cols:

    plt.subplot(4,3,i)

    sns.countplot(cols,data=df)

    i = i + 1

plt.suptitle("Distribution of Data for Categorical Columns after filling NA Values")

plt.show()
#Checking null values in each cols.

(df.isnull().sum()*100/df.shape[0]).sort_values(ascending=False)
# filling null with 0 in tests

categorical_V2 = []

for i in test.columns:

    if test[i].dtype == object:

        categorical_V2.append(i)

test.update(test[categorical_V2].fillna('None'))



Quantitative = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

Quantitative_V2 = []

for i in test.columns:

    if test[i].dtype in Quantitative:

        Quantitative_V2.append(i)

test.update(test[Quantitative_V2].fillna(0))
#Checking null values in each cols.

(test.isnull().sum()*100/test.shape[0]).sort_values(ascending=False)
#Checking duplicate row

df.loc[df.duplicated()]
#Checking duplicate row

test.loc[test.duplicated()]
numerical = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical:

    print(col,"=",countOutlier(df,col))
for i in range(0, len(df.columns), 5):

    sns.pairplot(data=df,

                x_vars=df.columns[i:i+5],

                y_vars=['SalePrice'])
df.shape
test.shape
plt.figure(figsize=(20,5))

sns.distplot(df.SalePrice)

plt.title("Target distribution - Sales Price")

plt.ylabel("Density");
df['SalePrice'].skew()
num_cols = df._get_numeric_data().columns

print("Numerical Columns",num_cols)

cat_cols=list(set(df.columns) - set(num_cols))

print("Categorical Columns:",cat_cols)
plt.figure(figsize=(20,5))

df.hist(figsize=(16, 20), bins=30, xlabelsize=8, ylabelsize=8)

plt.show()
dfg=df.groupby(['Neighborhood','YearBuilt'])['SalePrice']

dfg=dfg.describe()['mean'].to_frame()

dfg=dfg.reset_index(level=[0,1])

dfg=dfg.groupby('Neighborhood')

dfg
dfg_index=df['Neighborhood'].unique()

fig = plt.figure(figsize=(20,30))

fig.suptitle('Yearwise Trend of each Neighborhood')

for num in range(1,25):

    temp=dfg.get_group(dfg_index[num])

    ax = fig.add_subplot(8,3,num)

    ax.plot(temp['YearBuilt'], temp['mean'])

    ax.set_title(temp['Neighborhood'].unique())

    
cat_c = ['SaleType','SaleCondition','MSSubClass','MSZoning','OverallQual','OverallCond']

i = 1

plt.figure(figsize=(20, 10))

for cols in cat_c:

    plt.subplot(2,3,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of Sales Data for Categorical Columns")

plt.show()
cat_c = ['Condition1','Condition2','LandContour','LotConfig','LotShape','HouseStyle','Foundation','Street','LandContour'

        ,'LandSlope','BldgType','Functional']

i = 1

plt.figure(figsize=(20, 15))

for cols in cat_c:

    plt.subplot(4,3,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of House Data for Categorical Columns")

plt.show()
cat_c = ['Exterior1st','Exterior2nd','ExterQual','ExterCond']

i = 1

plt.figure(figsize=(20, 10))

for cols in cat_c:

    plt.subplot(2,2,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    plt.xticks(rotation=30)

    i = i + 1

plt.suptitle("Distribution of House Exterior Data for Categorical Columns")

plt.show()
cat_c = ['RoofStyle','RoofMatl']

i = 1

plt.figure(figsize=(20, 5))

for cols in cat_c:

    plt.subplot(1,2,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of House Roof Data for Categorical Columns")

plt.show()
cat_c = ['KitchenQual','Utilities','Electrical','CentralAir']

i = 1

plt.figure(figsize=(20, 10))

for cols in cat_c:

    plt.subplot(2,2,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of House Features Data for Categorical Columns")

plt.show()
cat_c = ['GarageCond','GarageQual','GarageType','GarageFinish','PavedDrive']

i = 1

plt.figure(figsize=(20, 10))

for cols in cat_c:

    plt.subplot(2,3,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of House Garage Data for Categorical Columns")

plt.show()
cat_c = ['BsmtCond','BsmtQual','BsmtFinType2']



i = 1

plt.figure(figsize=(20, 15))

for cols in cat_c:

    plt.subplot(3,3,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of House Basement Data for Categorical Columns")

plt.show()
cat_c = ['Heating','HeatingQC']



i = 1

plt.figure(figsize=(20, 5))

for cols in cat_c:

    plt.subplot(1,2,i)

    sns.boxplot(x = cols, y = 'SalePrice', data = df)

    i = i + 1

plt.suptitle("Distribution of House Heating Data for Categorical Columns")

plt.show()
df['AgeOfHome'] = df['YrSold'] - df['YearBuilt']

df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] 

df['TotalPorchArea'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] 

df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
test['AgeOfHome'] = test['YrSold'] - test['YearBuilt']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF'] 

test['TotalPorchArea'] = test['WoodDeckSF'] + test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch'] 

test['TotalBathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))
# basically string variable represented as numeric in data so converting in string

df['MSSubClass']= df['MSSubClass'].apply(str)

df['OverallCond'] =df['OverallCond'].astype(str)



test['MSSubClass']= test['MSSubClass'].apply(str)

test['OverallCond'] =test['OverallCond'].astype(str)
cat_c = ['AgeOfHome','TotalSF','TotalPorchArea','TotalBathrooms']

i = 1

plt.figure(figsize=(20, 20))

for cols in cat_c:

    plt.subplot(2,2,i)

    plt.scatter(x = 'SalePrice', y = cols, c='SalePrice', data=df)

    plt.xlabel("Sale Price")

    plt.ylabel(cols)

    cbar= plt.colorbar()

    cbar.set_label("elevation (m)", labelpad=+1)

    i = i + 1

plt.suptitle("Distribution of Feature Data Vs Sale Price")

plt.show()
newdf=df.loc[:, df.columns != 'SalePrice']

fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(newdf.corr(),ax=ax,annot= False,linewidth= 0.02,linecolor='black',fmt='.2f',cmap = 'Blues_r')

plt.show()
# Create correlation matrix

corr_matrix = newdf.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.7) | any(upper[column] < -0.7)]

to_drop
fig,ax = plt.subplots(figsize=(10,10))

result_corr=newdf.corr(method='pearson')

sns.heatmap(result_corr[(result_corr >= 0.7) | (result_corr <= -0.7)],ax=ax,annot= False,linewidth= 0.02,linecolor='black',fmt='.2f',cmap = 'Blues_r')

plt.show()
# dropping these two columns as we derived quality percentage and age of homw

df.drop(['YearBuilt','YrSold','MoSold','OverallCond','GarageYrBlt','Utilities','YearRemodAdd'], axis=1, inplace=True)

# dropping these two columns as we derived quality percentage and age of homw

test.drop(['YearBuilt','YrSold','MoSold','OverallCond','GarageYrBlt','Utilities','YearRemodAdd'], axis=1, inplace=True)
#price correlation with features

corr=df.corr()

corr=corr.sort_values(by=["SalePrice"],ascending=False).iloc[0].sort_values(ascending=False)

plt.figure(figsize=(15,20))

sns.barplot(x=corr.values, y=corr.index.values);

plt.title("Correlation Plot")
# Create correlation matrix

print(corr.values)

print(corr.index.values)
#log transform skewed numeric features 

numeric_features = df.dtypes[df.dtypes != "object"].index



skewed_features = df[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness

print ("skew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_features})   

skewness
#handling skewnees

skewness = skewness[abs(skewness) > 0.75]

print ("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))

skewed_features = skewness.index

lam = 0.1

for feat in skewed_features:

    df[feat] = boxcox1p(df[feat], lam)
#log transform skewed numeric features 

numeric_features = test.dtypes[test.dtypes != "object"].index



skewed_features = test[numeric_features].apply(lambda x : skew (x.dropna())).sort_values(ascending=False)

#compute skewness

print ("skew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_features})   

skewness
#handling skewnees

skewness = skewness[abs(skewness) > 0.75]

print ("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))

skewed_features = skewness.index

lam = 0.1

for feat in skewed_features:

    test[feat] = boxcox1p(test[feat], lam)
test.shape
df.shape
print(df.isnull().values.any())

print(test.isnull().values.any())
# Perform One-Hot Encoding on concatenated Dataframe

temp = pd.get_dummies(pd.concat([df,test],keys=[0,1]))



# Split concatenated dataframe back into train and test dataframes

train,test = temp.xs(0),temp.xs(1)



test.drop(["SalePrice"], axis = 1, inplace = True)
X, y = train.loc[:, train.columns != 'SalePrice'], train[["SalePrice"]]
kfolds = KFold(n_splits= 20, shuffle=True, random_state=42) 



lasso = make_pipeline(RobustScaler(), 

                      LassoCV(max_iter=1e7, 

                              random_state=42, cv=kfolds))
score = rmse_cv(lasso,X,y,kfolds)

print("Lasso RMSE:  ", score.mean())
lasso_model = lasso.fit(X, y)
lr_train_score=lasso.score(X,y)

print('Lasso training score: ', lr_train_score)
predicted_prices = lasso.predict(test)

print(inv_boxcox1p(predicted_prices, lam))
my_submission = pd.DataFrame({'Id': test_org.Id, 'SalePrice': inv_boxcox1p(predicted_prices, lam)})

my_submission.to_csv('House_Lasso.csv', index=False)