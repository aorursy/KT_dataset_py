# Loading Libraries



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats

from scipy.stats import norm
# Loading datasets



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Seeing the first 5 rows



train.head()
print ('The train data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))

print ('----------------------------')

print ('The test data has {} rows and {} columns'.format(test.shape[0],test.shape[1]))
# Seeing the columns in dataset



train.columns
# checking what types of features, we have in our dataset



train.dtypes.head()
# Let's check out for the missing values in our dataset



train.isnull().sum().sort_values(ascending=False).head(20)
# Let's check the percentage of missing values in our dataset



total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train['SalePrice'].count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# missing columns in train



missing_column_train= missing_data[missing_data['Total']>0].index

missing_column_train
# check out unique values in Poolqc



train['PoolQC'].unique()
# check out unique values in MiscFeature



train['MiscFeature'].unique()
# check out unique values in Alley



train['Alley'].unique()
# check out unique values in Fence



train['Fence'].unique()
# check out unique values in FireplaceQu



train['FireplaceQu'].unique()
# Let's check out the unique values in all categorical features



for i in missing_data[missing_data['Total']>0].index:

    if (train[i].dtype == 'object'):

        print(' Unique values in {} feature are {}'.format(i,train[i].unique()))
# Let's check out the unique values in all numerical features to check 

#is any ordinal feature in form of numeriacl feature is present or not



for i in missing_data[missing_data['Total']>0].index:

    if (train[i].dtype != 'object' and len(train[i].unique())<=10):

        print(' Unique values in {} feature are {}'.format(i,train[i].unique()))
# One column we can obviously drop is 'Id'



train_ = train.drop(['Id'],axis=1)

test_ = test.drop(['Id'],axis=1)
# Define how to fill the missing value



def missing_values(df,df2):

    

    for column in df2:

        if (df[column].isnull().sum()) >0:

            print(column +" has missing values type : "+ str(df[column].dtype))

            if df[column].dtype in ('int64','float64'):

                df[column] = df[column].fillna(df[column].mean())

            else:

                if column in ['Elecrical','MasVnrType']:

                    df[column] = df[column].fillna(df[column].dropna().mode()[0])

                else:

                    df[column] = df[column].fillna('NAN')
# Filling the missing values in train dataset



missing_values(train_,missing_column_train)
# Filling the missing values in test dataset



missing_values(test_,missing_column_train)
# There may be still some missing values in test dataset. Let's check out.



missing_test=test_.isnull().sum().sort_values(ascending=False)

test_missing_data = pd.concat([missing_test], axis=1, keys=['Total_missing'])

test_missing_data.head(20)
# Let's check out the unique values in all numerical features to check 

#is any ordinal feature in form of numeriacl feature is present or not in test features



for i in test_missing_data[test_missing_data['Total_missing']>0].index:

    if (test[i].dtype != 'object' and len(test[i].unique())<=10):

        print(' Unique values in {} feature are {}'.format(i,test[i].unique()))
# Let's check out the unique values in all categorical features in test dataset



for i in test_missing_data[test_missing_data['Total_missing']>0].index:

    if (test[i].dtype == 'object'):

        print(' Unique values in {} feature are {} \n'.format(i,test[i].unique()))
# now fill rest missing values in test dataset



def missing_values_test(df):

    

    for column in df:

        if (df[column].isnull().sum()) >0:

            print(column +" has missing values type : "+ str(df[column].dtype))

            if ((df[column].dtype in ('int64','float64')) and (column not in['BsmtFullBath','BsmtHalfBath','GarageCars'])):

                df[column] = df[column].fillna(df[column].mean())

            else:

                df[column] = df[column].fillna(df[column].dropna().mode()[0])
missing_values_test(test_)
# Some infromation about target vaiable



train['SalePrice'].describe()
# Check for skewness: Plot a histogram of target feature



sns.distplot(train['SalePrice'])
# calculate skewness of target feature



print ("The skewness of SalePrice is {}".format(train['SalePrice'].skew()))
# Transform the the target feature and check the skewness



target = np.log(train['SalePrice'])

print ('Skewness is', target.skew())

sns.distplot(target)
# seperate out the Categorical and Numerical features



numerical_feature=train_.dtypes[train.dtypes!= 'object'].index

categorical_feature=train_.dtypes[train.dtypes== 'object'].index



print ("There are {} numeric and {} categorical columns in train data".format(numerical_feature.shape[0],categorical_feature.shape[0]))
numerical_feature.shape
# Correlation heatmap



corr=train_[numerical_feature].corr()

sns.heatmap(corr)
# Let's check the first ten features are the most positively correlated with SalePrice and 

# the next ten are the most negatively correlated.



print (corr['SalePrice'].sort_values(ascending=False)[:10], '\n') #top 10 values

print ('----------------------')

print (corr['SalePrice'].sort_values(ascending=False)[-10:]) #last 10 values`
#saleprice correlation matrix



k = 10 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Dropping the twin brothers



train_ = train_.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1)

test_ = test_.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'],axis=1)
# Dropping the twin brothers from numerical_feature too



numerical_feature = numerical_feature.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'])
# check OverallQual feature 



train_['OverallQual'].unique()
#let's check the mean price per quality and plot it.



pivot = train_.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

pivot.sort_values(ascending=False)
#visualize this pivot table more easily, we can create a bar plot



pivot.plot(kind='bar', color='red')
#GrLivArea variable



sns.jointplot(x=train_['GrLivArea'], y=train['SalePrice'])
# TotalBsmtSF variable



sns.jointplot(x=train_['TotalBsmtSF'], y=train['SalePrice'])

print('Correlation between SalePrice and TotalBsmtSF: {}'.format(corr['SalePrice']['TotalBsmtSF']))
# Fullbath variable



pivot2 = train_.pivot_table(index='FullBath', values='SalePrice', aggfunc=np.median)

print(pivot2.sort)
#visualize this pivot table more easily, we can create a bar plot



pivot2.plot(kind='bar', color='red')

print('Correlation between SalePrice and FullBath: {}'.format(corr['SalePrice']['FullBath']))
# info about categorical features



train_[categorical_feature].describe()
cat = [f for f in train_.columns if train_.dtypes[f] == 'object']

def anova(frame):

    anv = pd.DataFrame()

    anv['features'] = cat

    pvals = []

    for c in cat:

           samples = []

           for cls in frame[c].unique():

                  s = frame[frame[c] == cls]['SalePrice'].values

                  samples.append(s)

           pval = stats.f_oneway(*samples)[1]

           pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')



cat_data= train_[categorical_feature]

cat_data['SalePrice'] = train_.SalePrice.values

k = anova(cat_data) 

k['disparity'] = np.log(1./k['pval'].values) 

sns.barplot(data=k, x = 'features', y='disparity') 

plt.xticks(rotation=90) 

plt 
k
#create numeric plots



num = [f for f in train_.columns if train_.dtypes[f] != 'object']

nd = pd.melt(train_, value_vars = num)

n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
# let's create boxplots for visualizing categorical variables.



def boxplot(x,y,**kwargs):

            sns.boxplot(x=x,y=y)

            x = plt.xticks(rotation=90)



cat = [f for f in train_.columns if train_.dtypes[f] == 'object']



p = pd.melt(train_, id_vars='SalePrice', value_vars=cat)

g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, 'value','SalePrice')

g
# we'll convert the categorical variables into ordinal variables



slope_mapping = {'Gtl': 3,'Mod': 2,'Sev': 1}



train_['LandSlope'] = train_['LandSlope'].map(slope_mapping)

test_['LandSlope'] = test_['LandSlope'].map(slope_mapping)



qual_mapping = {'Ex': 5,'Gd': 4,'TA': 3,'Fa':2,'Po':1,'NA':0,'NAN':0}





#train['ExterQual'] = train['ExterQual'].map(qual_mapping)

#test['ExterQual'] = test['ExterQual'].map(qual_mapping)



name=['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond']

for i in name:

    train_[i] = train_[i].map(qual_mapping)

    test_[i] = test_[i].map(qual_mapping)

    

expo_mapping = {'Gd': 4,'Av': 3,'Mn':2,'No':1,'Na':0,'NAN':0}



train_['BsmtExposure'] = train_['BsmtExposure'].map(expo_mapping)

test_['BsmtExposure'] = test_['BsmtExposure'].map(expo_mapping)



bsmt_mapping = {"Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5,'NAN':0}



train_['BsmtFinType1'] = train_['BsmtFinType1'].map(bsmt_mapping)

test_['BsmtFinType1'] = test_['BsmtFinType1'].map(bsmt_mapping)



train_['BsmtFinType2'] = train_['BsmtFinType2'].map(bsmt_mapping)

test_['BsmtFinType2'] = test_['BsmtFinType2'].map(bsmt_mapping)



funct_mapping={"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}



train_['Functional'] = train_['Functional'].map(funct_mapping)

test_['Functional'] = test_['Functional'].map(funct_mapping)



garage_mapping = { "Unf": 1, "RFn": 2, "Fin": 3,'NAN':0}



train_['GarageFinish'] = train_['GarageFinish'].map(garage_mapping)

test_['GarageFinish'] = test_['GarageFinish'].map(garage_mapping)
# To check the missing values,if any



train_.isnull().sum().sort_values(ascending=False).head(5)
# Let's check is there any ordinal feature present in numerical feature



num_cat=[]

for i in numerical_feature:

    if ( len(train_[i].unique())<=10):

        num_cat.append(i)

        print(' Unique values in {} feature are {}'.format(i,train_[i].unique()))
# convert PoolArea feature into ordinal feature



pool_mapping={0:0,480:1,512:2,519:3,555:4,576:5,648:6,738:7}



train_['PoolArea'] = train_['PoolArea'].map(pool_mapping)

test_['PoolArea'] = test_['PoolArea'].map(pool_mapping)
# add categorical features which are present in numerical feature



categorical_feature=categorical_feature.append(pd.Index(num_cat))
numerical_feature.shape
# remove categorical feature from numerical feature



numerical_feature=numerical_feature.drop(num_cat)
# Transform the skewness of features



def log_skew(df,df2,num_feature):

    for column in df[num_feature]:

        if df[column].dtype in ('int64','float64'):

            old_skew = df[column].skew()

            if (old_skew) > 0.75:

                df[column]=df[column].apply(lambda x: np.log(x+1))

                df2[column]=df[column].apply(lambda x: np.log(x+1))

                print('the skewness of '+column+" is reduced from "+str(old_skew) + " to "+str(df[column].skew()))
log_skew(train_,test_,numerical_feature)
 # Let's standardize the numeric features



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_[numerical_feature])

train_[numerical_feature] = scaler.transform(train_[numerical_feature])

test_[numerical_feature] = scaler.transform(test_[numerical_feature])