import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# from plotly.offline import iplot

%matplotlib inline

pd.pandas.set_option('display.max_columns',None)

pd.pandas.set_option('display.max_rows',None)

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
# # a = np.zeros((train.shape[0],))

# # print(a)

# train.shape[0]
# NFOLDS = 5

# ntest = 4

# oof_test_skf = np.empty((NFOLDS, ntest))

# print(oof_test_skf)
train['SaleCondition'].value_counts()
train['Electrical'].value_counts()
train['LowQualFinSF'].value_counts()
train.isnull().sum()
# list of numerical variables

numerical_features = [feature for feature in train.columns if train[feature].dtypes != 'O']



print('Number of numerical variables: ', len(numerical_features))



# visualise the numerical variables

train[numerical_features].head()
# list of variables that contain year information

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]



year_feature
## Numerical variables are usually of 2 type

## 1. Continous variable and Discrete Variables



discrete_feature=[feature for feature in numerical_features if len(train[feature].unique())<25 and feature not in year_feature+['Id']]

print("Discrete Variables Count: {}".format(len(discrete_feature)))

discrete_feature
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]

print("Continuous feature Count {}".format(len(continuous_feature)))

continuous_feature
data = train.copy()

categorical_features=[feature for feature in train.columns if data[feature].dtypes=='O']

len(categorical_features)

categorical_features
#NUMBER OF CAREGOURY IN A CATEGORICAL FEATURE

for feature in categorical_features:

    print('The feature is {} and number of categories are {}'.format(feature,len(train[feature].unique())))
# HOW THE continious neumerical VARIABLES ARE CHANGING WITH TERGATE VARIABLE

# BAR COUNT OF CATEGORICAL VATIABLE

# HISTOGRAM FOR CONTINIOUS NEUMERICAL VARIABLE TO FIND OUT THE DISTRIBUTION

# BOX PLOT FOR FINDING OUTLIERS

# CDFs
for i in continuous_feature:

    

    plt.scatter(train['SalePrice'],train[i],color = 'tomato')

    plt.xlabel("i",size = 15)

    plt.ylabel('SalePrice',size = 15)

    plt.title('SalePrice Vs '+ i,size = 15)

    plt.show()
for i in continuous_feature:

    train[i]

train[continuous_feature[0]].unique()
#LOGNORMAL DISTRIBUTION



for feature in continuous_feature:

    data=train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data['SalePrice']=np.log(data['SalePrice'])

        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalesPrice')

        plt.title(feature)

        plt.show()
features_with_na=[features for features in train.columns if train[features].isnull().sum()>1]

features_with_na

tips = sns.load_dataset("tips")

tips.head()
for i in categorical_features:

    sns.countplot(x=i,data=train)

    plt.show()
categorical_features[2]
for feature in discrete_feature:

    data=train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar(rot=0)

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
for i in continuous_feature:

    sns.set_style('whitegrid')

    plt.figure(figsize=(10,7))

    sns.distplot(train[i].dropna(),bins=20,hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2},kde = False,norm_hist=False,color = 'darkblue',hist=True)

    plt.xlabel(i,size = 15)

    plt.ylabel('percentage of distribution',size = 15)

    plt.title('histogram of '+ i,size = 15)

#     plt.xticks(np.arange(0,10,1))

#     plt.figure(figsize=(12,8))

    plt.show()
for feature in continuous_feature:

    data=train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()


from string import ascii_letters



corr = train.corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 16))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# plt.figure(figsize=(18,16))

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Fig:1',size=15)

plt.show()
# train['MoSold'].plot(kind='bar',stacked = True)
categorical_features
'''for feature in continuous_feature:

    data=train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data['SalePrice']=np.log(data['SalePrice'])

        plt.line(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalesPrice')

        plt.title(feature)

        plt.show()'''
plt.figure(figsize=(12,10))

ax = sns.lineplot(x="GarageArea", y="SalePrice", data=train,color="coral", label="line")

plt.show()
'''from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles = len(train['GarageArea']),output_distribution = 'normal')



## transforming above distributions to Normal distribution ##

X = qt.fit_transform(train['GarageArea'])

# Y = qt.fit_transform(Y)

print('distributions transformed')'''
'''GarageArea1 = train['GarageArea'].values.reshape(-1,1)

GarageArea2 = train['GarageArea'].values.reshape(1,-1)'''
'''from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(n_quantiles = len(GarageArea1),output_distribution = 'normal')



## transforming above distributions to Normal distribution ##

train['GarageArea1'] = qt.fit_transform(GarageArea1)

# Y = qt.fit_transform(Y)

print('distributions transformed')'''
# from sklearn.preprocessing import QuantileTransformer

# qt1 = QuantileTransformer(n_quantiles = len(GarageArea2),output_distribution = 'normal')



# ## transforming above distributions to Normal distribution ##

# Y = qt1.fit_transform(GarageArea2)

# # Y = qt.fit_transform(Y)

# print('distributions transformed')
# plt.figure(figsize = (10,4), dpi = 120)



# #Plotting transformed exponential

# # plt.subplot(121)

# plt.hist(Y, bins = 100)

# plt.title("transformed 2")
'''plt.figure(figsize = (10,4), dpi = 120)



#Plotting transformed exponential

# plt.subplot(121)

plt.hist(train['GarageArea1'], bins = 100)

plt.title("transformed exponential")'''
# plt.figure(figsize=(12,10))

# ax = sns.lineplot(x="GarageArea1", y="SalePrice", data=train,color="coral", label="line")

# plt.show()
for feature in continuous_feature:

    data=train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data['SalePrice']=np.log(data['SalePrice'])

        ax1 = sns.lineplot(x=feature, y="SalePrice", data=data,color="coral", label="line")

        #     plt.line(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalesPrice')

        plt.title(feature)

        plt.show()
continuous_feature
for feature in continuous_feature:

#     data=train.copy()

#     if 0 in data[feature].unique():

#         pass

#     else:

#     data[feature]=np.log(data[feature])

#     data['SalePrice']=np.log(data['SalePrice'])

    ax1 = sns.lineplot(x=feature, y="SalePrice", data=data,color="coral", label="line")

    #     plt.line(data[feature],data['SalePrice'])

    plt.xlabel(feature)

    plt.ylabel('SalesPrice')

    plt.title(feature)

    plt.show()
# def bar_chart(feature):

#     survived = train[train['Survived']==1][feature].value_counts()

#     dead = train[train['Survived']==0][feature].value_counts()

#     df = pd.DataFrame([survived,dead])

#     df.index = ['Survived','Dead']

#     df.plot(kind='bar',stacked=True, figsize=(10,5))
categorical_features
data = train.copy()



data['LotArea']=np.log(data['LotArea'])

data['SalePrice']=np.log(data['SalePrice'])

ax1 = sns.lineplot(x='LotArea', y="SalePrice",hue= 'LotShape', data=data,color="coral", label="LotArea")

#     plt.line(data[feature],data['SalePrice'])

plt.xlabel('LotArea')

plt.ylabel('SalesPrice')

plt.title('LotArea')

plt.legend()

plt.show()







# ax = sns.lineplot(x='LotArea',y='SalePrice',hue= 'LandContour',data = train,color="coral", label="LotArea")

train['LandContour'].value_counts()
def bar_chart(feature):

    Reg = train[train['LotShape']=='Reg'][feature].value_counts()

    IR1 = train[train['LotShape']=='IR1'][feature].value_counts()

    IR2 = train[train['LotShape']=='IR2'][feature].value_counts()

    IR3 = train[train['LotShape']=='IR3'][feature].value_counts()

    

    df = pd.DataFrame([Reg,IR1,IR2,IR3])

    df.index = ['Reg','IR1','IR2','IR3']

    df.plot(kind='bar',stacked=True, figsize=(10,8))
bar_chart('LandContour')

train['MSZoning'].value_counts()
def bar_chart(feature):

    RL = train[train['MSZoning']=='RL'][feature].value_counts()

    RM = train[train['MSZoning']=='RM'][feature].value_counts()

    FV = train[train['MSZoning']=='FV'][feature].value_counts()

    RH = train[train['MSZoning']=='RH'][feature].value_counts()

    C_all = train[train['MSZoning']=='C (all)'][feature].value_counts()

    

    df = pd.DataFrame([RL,RM,FV,RH,C_all])

    df.index = ['RL','RM','FV','RH','C_all']

    df.plot(kind='bar',stacked=True, figsize=(10,8))
bar_chart('Neighborhood')

import scipy.stats as stats
data=train.copy()

dataset_table=pd.crosstab(data['Neighborhood'],data['MSZoning'])

print(dataset_table)
dataset_table.values

#Observed Values

Observed_Values = dataset_table.values 

print("Observed Values :-\n",Observed_Values)
val=stats.chi2_contingency(dataset_table)

Expected_Values=val[3]

no_of_rows=len(dataset_table.iloc[0:26,0])

no_of_columns=len(dataset_table.iloc[0,0:6])

ddof=(no_of_rows-1)*(no_of_columns-1)

print("Degree of Freedom:-",ddof)

alpha = 0.05
from scipy.stats import chi2

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

chi_square_statistic=chi_square[0]+chi_square[1]


critical_value=chi2.ppf(q=1-alpha,df=ddof)

print('critical_value:',critical_value)
#p-value

p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

print('p-value:',p_value)

print('Significance level: ',alpha)

print('Degree of Freedom: ',ddof)

print('p-value:',p_value)
if chi_square_statistic>=critical_value:

    print("Reject H0,There is a relationship between Neighborhood and MSZoning")

else:

    print("Retain H0,There is no relationship between Neighborhood and MSZoning")

    

if p_value<=alpha:

    print("Reject H0,There is a relationship between Neighborhood and MSZoning")

else:

    print("Retain H0,There is no relationship between Neighborhood and MSZoning")
#box plot overallqual/saleprice

var = 'OverallQual'

# data = train.copy

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
var = 'MoSold'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
var = 'YrSold'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
var = 'Neighborhood'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
plt.figure(figsize=(10,8))



from itertools import cycle, islice

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

# missing.plot.bar()

# sns.countplot(missing,color="salmon")

my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, 19))

# my_colors = ['g', 'b']*5

# my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*5

# my_colors = [(x/10.0, x/20.0, 0.75) for x in range(0,1)]



missing.plot(kind='bar',color=my_colors)
corr_new_train=train.corr()

plt.figure(figsize=(10,20))

sns.heatmap(corr_new_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(60),vmin=-1, cmap='seismic', annot=True)
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=categorical_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=discrete_feature)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")
# f = pd.melt(train, id_vars=['SalePrice'], value_vars=continuous_feature)

# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

# g = g.map(sns.lineplot, "value", "SalePrice")
train[continuous_feature].shape
no_of_rows=len(dataset_table.iloc[0:26,0])

no_of_rows
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

import numpy as np

SEED = 42



pd.pandas.set_option('display.max_columns',None)

pd.pandas.set_option('display.max_rows',None)



import string

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)



# df_train = pd.read_csv('../input/train.csv')

# df_test = pd.read_csv('../input/test.csv')

df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]



print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

# print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)
'''

1.HANDEL MISSING VALUE

2.LABEL ENCODING

3.OUTLIER DETECTION AND HANDELING(performing log,sqrt,cbrt)

4.SCALLING(using sklearn ScaleMinMax)





'''
## Now lets check for numerical variables the contains missing values

numerical_with_nan=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>1 and df_train[feature].dtypes!='O']



## We will print the numerical nan variables and percentage of missing values



for feature in numerical_with_nan:

    print("{}: {}% missing value".format(feature,np.around(df_train[feature].isnull().mean(),4)))


for feature in numerical_with_nan:

    ## We will replace by using median since there are outliers

    median_value=df_train[feature].median()

    

    ## create a new feature to capture nan values

#     dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)

    df_train[feature].fillna(median_value,inplace=True)

    

df_train[numerical_with_nan].isnull().sum()
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']



for feature in num_features:

    df_train[feature]=np.log(df_train[feature])

    df_train[feature].hist(bins = 30)

    plt.title('histogram of : {}'.format(feature))

    plt.show()

    

    

#########



df_train.head()
# with_zero_num = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','2ndFlrSF','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch']

# for i in with_zero_num:

#     df_train[i] = np.sqrt(df_train[i])

#     df_train[i].hist(bins=30)

#     plt.xlabel(i)

#     plt.title(i)

#     plt.show()
features_nan=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>1 and df_train[feature].dtypes=='O']



## Replace missing value with a new label

def replace_cat_feature(df_train,features_nan):

    data=df_train.copy()

    data[features_nan]=data[features_nan].fillna('Missing')

    return data



df_train=replace_cat_feature(df_train,features_nan)



df_train[features_nan].isnull().sum()
for feature in categorical_features:

    temp=df_train.groupby(feature)['SalePrice'].count()/len(df_train)

    temp_df=temp[temp>0.01].index

    df_train[feature]=np.where(df_train[feature].isin(temp_df),df_train[feature],'rear_cat')
df_train.head()
cat_fe = [i for i in df_train.columns if df_train[i].dtype=='O']

cat_fe
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



for i in cat_fe:

    df_train[i]=label_encoder.fit_transform(df_train[i])
df_train.head()
feature_scale=[feature for feature in df_train.columns if feature not in ['Id','SalePrice']]



from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(df_train[feature_scale])
scaler.transform(df_train[feature_scale])
# transform the train and test set, and add on the Id and SalePrice variables

data = pd.concat([df_train[['Id', 'SalePrice']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(df_train[feature_scale]), columns=feature_scale)],

                    axis=1)
data.head()
# data.to_csv('Data_After_FE.csv')
data1 = data.copy() 



Y_train = data1['SalePrice']

X_train = data1.drop(['SalePrice'],axis=1)

# Y_train = data1['SalePrice']
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=SEED)
# from sklearn.matrics import mean_squared_error

from sklearn.metrics import mean_squared_error

from math import sqrt



def validation(y_val,y_pred):

    rmse_val = sqrt(mean_squared_error(y_val,y_pred)) 

    return rmse_val

from sklearn.ensemble import RandomForestRegressor



regr = RandomForestRegressor(max_depth=2, random_state=0)

regr.fit(x_train, y_train)

y_pred = regr.predict(x_val)
validation(y_val,y_pred)
df_test.head()
df_test.isnull().sum()
numerical_with_nan_test=[feature for feature in df_test.columns if df_test[feature].isnull().sum()>1 and df_test[feature].dtypes!='O']

for feature in numerical_with_nan_test:

    print("{}: {}% missing value".format(feature,np.around(df_train[feature].isnull().mean(),4)))
for feature in numerical_with_nan_test:

    ## We will replace by using median since there are outliers

    median_value_test=df_test[feature].median()

    

    ## create a new feature to capture nan values

#     dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)

    df_test[feature].fillna(median_value_test,inplace=True)

    

df_test[numerical_with_nan_test].isnull().sum()
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']



for feature in num_features:

    df_test[feature]=np.log(df_test[feature])
features_nan_test=[feature for feature in df_test.columns if df_test[feature].isnull().sum()>1 and df_test[feature].dtypes=='O']



## Replace missing value with a new label

def replace_cat_feature(df_test,features_nan_test):

#     data_test=df_test.copy()

    df_test[features_nan_test]=df_test[features_nan_test].fillna('Missing')

    return df_test



df_test=replace_cat_feature(df_test,features_nan_test)



# df_test[features_nan_test].isnull().sum()
categorical_var_test = [i for i in df_test.columns if df_test[i].dtype == 'O' ]

len(categorical_var_test)
# for feature in categorical_var_test:

#     temp=df_test.groupby(feature)['SalePrice'].count()/len(df_train)

#     temp_df=temp[temp>0.01].index

#     df_train[feature]=np.where(df_train[feature].isin(temp_df),df_train[feature],'rear_cat')
numerical_features_test = [feature for feature in df_test.columns if df_test[feature].dtypes != 'O']

year_feature_test = [feature for feature in numerical_features_test if 'Yr' in feature or 'Year' in feature]

discrete_feature_test=[feature for feature in numerical_features_test if len(df_test[feature].unique())<25 and feature not in year_feature_test+['Id']]

continuous_feature_test=[feature for feature in numerical_features_test if feature not in discrete_feature_test+year_feature_test+['Id']]

len(numerical_features_test)
year_feature_test
len(discrete_feature_test)
continuous_feature_test
df_test.KitchenQual.dtype
df_test.info()
df_test.isnull().sum()
df_test['Exterior1st']=df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])

df_test['Exterior2nd']=df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])

df_test['KitchenQual']=df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])

df_test['SaleType']=df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])





con_feature = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageArea','GarageCars']











for feature in con_feature:

    ## We will replace by using median since there are outliers

    median_value_con=df_test[feature].median()

    

    ## create a new feature to capture nan values

#     dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)

    df_test[feature].fillna(median_value_con,inplace=True)

    

# df_test[numerical_with_nan_test].isnull().sum()
df_test.isnull().sum()
# categorical_var_test=['Exterior1st',

#  'Exterior2nd',

#  'MasVnrType',

#  'ExterQual',

#  'ExterCond',

#  'Foundation',

#  'BsmtQual',

#  'BsmtCond',

#  'BsmtExposure',

#  'BsmtFinType1',

#  'BsmtFinType2',

#  'Heating',

#  'HeatingQC',

#  'CentralAir',

#  'Electrical',

#  'KitchenQual',

#  'Functional',

#  'FireplaceQu',

#  'GarageType',

#  'GarageFinish',

#  'GarageQual',

#  'GarageCond',

#  'PavedDrive',

#  'PoolQC',

#  'Fence',

#  'MiscFeature',

#  'SaleType',

#  'SaleCondition',]
from sklearn.preprocessing import LabelEncoder

label_encod = LabelEncoder()



for i in categorical_var_test:

    df_test[i]=label_encod.fit_transform(df_test[i])
df_test.head()
feature_scale_test=[feature for feature in df_test.columns if feature not in ['Id']]



from sklearn.preprocessing import MinMaxScaler

scaler_test=MinMaxScaler()

scaler_test.fit(df_test[feature_scale_test])
scaler_test.transform(df_test[feature_scale_test])
# transform the train and test set, and add on the Id and SalePrice variables

data_test = pd.concat([df_test[['Id']].reset_index(drop=True),

                    pd.DataFrame(scaler_test.transform(df_test[feature_scale_test]), columns=feature_scale_test)],

                    axis=1)
data_test.head()
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Lasso,LinearRegression



# select_from_model = SelectFromModel(Lasso(alpha = 0.005,random_state = SEED))

select_from_model = SelectFromModel(LinearRegression())



select_from_model.fit(X_train,Y_train)

select_from_model.get_support().sum()
selected_features = X_train.columns[(select_from_model.get_support())]

selected_features
X_train_fs_linear_regression = X_train[selected_features]
X_train_fs_linear_regression.head()
x_train_fs_linear_regression, x_val_fs_linear_regression, y_train_fs_linear_regression, y_val_fs_linear_regression=train_test_split(X_train_fs_linear_regression,Y_train,test_size = 0.33,random_state = SEED)
print(x_train_fs_linear_regression.shape, x_val_fs_linear_regression.shape, y_train_fs_linear_regression.shape, y_val_fs_linear_regression.shape)
x_train_fs_linear_regression.head()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error



ridge=Ridge()

parameters= {'alpha':[x for x in [0.1,0.2,0.4,0.5,0.7,0.8,1]]}



ridge_reg=GridSearchCV(ridge, param_grid=parameters)

ridge_reg.fit(x_train_fs_linear_regression,y_train_fs_linear_regression)

print("The best value of Alpha is: ",ridge_reg.best_params_)
ridge_mod=Ridge(alpha=1)

ridge_mod.fit(x_train_fs_linear_regression,y_train_fs_linear_regression)

y_pred_train=ridge_mod.predict(x_train_fs_linear_regression)

y_pred_val=ridge_mod.predict(x_val_fs_linear_regression)



y_pred_train_series = pd.Series(np.log(y_pred_train))

y_pred_train_series_median = y_pred_train_series.median()

y_pred_train_series_1=y_pred_train_series.fillna(y_pred_train_series_median)



print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(np.log(y_train_fs_linear_regression), y_pred_train_series_1))))

print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(np.log(y_val_fs_linear_regression), np.log(y_pred_val))))) 
# np.sqrt(mean_squared_error(np.log(y_val_fs_linear_regression), np.log(y_pred_val)))
# np.log(y_train_fs_linear_regression)
# y_pred_train_series = pd.Series(np.log(y_pred_train))

# y_pred_train_series.isnull().sum()
# y_pred_train_series_median = y_pred_train_series.median()

# y_pred_train_series_1=y_pred_train_series.fillna(y_pred_train_series_median)
# np.log(y_pred_train_series_1)
# mean_squared_error(np.log(y_train_fs_linear_regression),y_pred_train_series_1)
# y_val_fs_linear_regression
# y_pred_val
# data_test_fs1 = data_test[selected_features]

# data_test_fs1.head()
# y_pred_fs1_test = ridge_mod.predict(data_test)

# y_pred_fs1_test = ridge_mod.predict(data_test_fs1)

# pred_fs1 = pd.DataFrame(y_pred_fs1_test)

# sub_df = pd.read_csv('sample_submission.csv')

# datasets_fs1 = pd.concat([sub_df['Id'],pred_fs1],axis = 1)

# datasets_fs1.columns=['Id','SalePrice']

# datasets_fs1.to_csv('submission_fs1.csv',index=False)

# datasets_fs1.head()
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

params = {"max_depth":[15,20,25], "n_estimators":[27,30,33]}

rf_reg = GridSearchCV(rf, params, cv = 10, n_jobs =10)

rf_reg.fit(x_train, y_train)

print(rf_reg.best_estimator_)

best_estimator=rf_reg.best_estimator_

y_pred_train = best_estimator.predict(x_train)

y_pred_val = best_estimator.predict(x_val)



print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(np.log(y_train), np.log(y_pred_train)))))

print('Root Mean Square Error val = ' + str(np.sqrt(mean_squared_error(np.log(y_val), np.log(y_pred_val)))))
y_pred_val
y_val
# y_pred_rf_test = best_estimator.predict(data_test)

# pred_rf = pd.DataFrame(y_pred_rf_test)

# sub_df = pd.read_csv('sample_submission.csv')

# datasets_rf = pd.concat([sub_df['Id'],pred_rf],axis = 1)

# datasets_rf.columns=['Id','SalePrice']

# datasets_rf.to_csv('submission_rf.csv',index=False)

# datasets_rf.head()
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost 
# params = {

#     'learning_rate'   :[0.05,0.10,0.15,0.20,0.25,0.30],

#     'max_depth'       :[3,5,6,7,8,11,12,15],

#     'min_child_weight':[1,3,5,7],

#     'gamma'           :[0.0,0.1,0.2,0.3,0.4],

#     'colsample_bytree':[0.3,0.4,0.5,0.7]  

# }

params={'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],

'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}
regressor = xgboost.XGBRegressor()
# random_search = RandomizedSearchCV(regressor,param_distributions = params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search = RandomizedSearchCV(regressor, params,n_iter=5, n_jobs=1, cv=5)
random_search.fit(x_train,y_train)

random_search.best_estimator_

xgbo = xgboost.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1.0, gamma=0.5, gpu_id=-1,

             importance_type='gain', interaction_constraints=None,

             learning_rate=0.300000012, max_delta_step=0, max_depth=3,

             min_child_weight=5, monotone_constraints=None,

             n_estimators=100, n_jobs=0, num_parallel_tree=1,

             objective='reg:squarederror', random_state=0, reg_alpha=0,

             reg_lambda=1, scale_pos_weight=1, subsample=0.7, tree_method=None,

             validate_parameters=False, verbosity=None)
xgbo.fit(x_train,y_train)

y_pred = xgbo.predict(x_val)

mse = mean_squared_error(np.log(y_val),np.log(y_pred))

print(np.sqrt(mse))
y_pred
y_val
y_pred_test = xgbo.predict(data_test)
# pred = pd.DataFrame(y_pred_test)

# sub_df = pd.read_csv('sample_submission.csv')

# datasets = pd.concat([sub_df['Id'],pred],axis = 1)

# datasets.columns=['Id','SalePrice']

# datasets.to_csv('submission2.csv',index=False)



# datasets.head()
# regressor=xgboost.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,

#              colsample_bynode=1, colsample_bytree=1.0, gamma=0.5, gpu_id=-1,

#              importance_type='gain', interaction_constraints=None,

#              learning_rate=0.300000012, max_delta_step=0, max_depth=3,

#              min_child_weight=5, monotone_constraints=None,

#              n_estimators=100, num_parallel_tree=1,

#              objective='reg:squarederror', random_state=0, reg_alpha=0,

#              reg_lambda=1, scale_pos_weight=1, subsample=0.7, tree_method=None,

#              validate_parameters=False, verbosity=None)

# #min_child_weight=5, missing=nan,
# best_x = xgbr_reg.best_estimator_

# y_train_pred_x = best_x.predict(X_train)

# y_val_pred_x = best_x.predict(X_test)
# pred = pd.DataFrame(preds)

# sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

# datasets = pd.concat([sub_df['Id'],pred],axis = 1)

# datasets.columns=['Id','SalePrice']

# datasets.to_csv('submission.csv',index=False)



# data_test.to_csv('test_data_after_FE.csv')
# !pip install xgboost
# import xgboost as xgb

# xgb_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xgb_reg.fit(X_train,y_train)

# y_predxgb = xgb_reg.predict(X_val)
# from sklearn.matrics import mean_squared_error

# from sklearn.metrics import mean_squared_error

# from math import sqrt



# def validation(y_val,y_pred):

#root_mean_squared_error = sqrt(mean_squared_error(y_val,y_pred)) 

#     return rmse_val

#validation(y_val,y_predxgb)

# preds = xgb_reg.predict(data_test)
# from keras import backend as K

# def root_mean_squared_error(y_true, y_pred):

#         return K.sqrt(K.mean(K.square(y_pred - y_true)))
# y_train.shape
# import keras

# from keras.models import Sequential

# from keras.layers import Dense

# from keras.layers import LeakyReLU,PReLU,ELU

# from keras.layers import Dropout





# # Initialising the ANN

# classifier = Sequential()



# # Adding the input layer and the first hidden layer

# classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu',input_dim = 80))



# # Adding the second hidden layer

# classifier.add(Dense(output_dim = 25, init = 'he_uniform',activation='relu'))



# # Adding the third hidden layer

# classifier.add(Dense(output_dim = 50, init = 'he_uniform',activation='relu'))

# # Adding the output layer

# classifier.add(Dense(output_dim = 1, init = 'he_uniform'))



# # Compiling the ANN

# classifier.compile(loss=root_mean_squared_error, optimizer='Adamax')



# # Fitting the ANN to the Training set

# model_history=classifier.fit(x_train.values, y_train.values,validation_split=0.20, batch_size = 10, nb_epoch = 1000)
# y_prednn = classifier.predict(X_val)
# validation(y_val,y_prednn)
# main_preds = classifier.predict(data_test)
from sklearn.model_selection import KFold



ntrain = train.shape[0]

ntest = df_test.shape[0]

NFOLDS = 5

kf = KFold(n_splits = NFOLDS, random_state = SEED)





# # Some useful parameters which will come in handy later on

# ntrain = train.shape[0]

# ntest = test.shape[0]

# SEED = 0 # for reproducibility

# NFOLDS = 5 # set folds for out-of-fold prediction

# kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)

    

# Class to extend XGboost classifer



# class SklearnHelper(object):

#     def __init__(self,clf,seed = SEED,params = None):

#         params['random_state'] = seed

# #         self.clf = clf(**params)

#         self.clf = clf(**params)



#     def train(self,x_train,y_train):

#         self.clf.fit(x_train,y_train)

        

#     def predict(self,x):

#         return self.clf.predict(x)

    

#     def fit(self,x,y):

#         return self.clf.fit(self,x,y)

    

#     def feature_importance(self,x,y):

#         print(self.clf.fit(x,y).feature_importances_)

        

    
##ERROR BOOKMARK_1--> BECAUSE`x_test` is not used in `x_te = x_train[test_index]`



def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS,ntest))

    

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        x_te = x_train[test_index]

        y_tr = y_train[train_index]

        

        clf.train(x_tr,y_tr)

        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i,:] = clf.predict(x_test)

        

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

        
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

# from sklearn import svm

from sklearn.svm import SVC

rf_params= {

    'n_jobs':-1,

    'n_estimators':500,

    'warm_start': True,

    'max_depth':6,

#     'min_sample_leaf':2,

    'max_features':'sqrt',

    'verbose':0

}



#Extra Tree

et_params = {

    'n_jobs':-1,

#     'min_sample_leaf':2,

    'n_estimators':500,

    'max_depth':8,

    'verbose':0

}



#Adaboost

ada_params = {

    'n_estimators':500,

    'learning_rate':0.75

}

# Gradient Boosting

gb_params = {

    'n_estimators':500,

    'max_depth':5,

    'verbose':0,

#     'min_sample_leaf':2

}

#SVM

svc_params = {

    'kernel':'linear',

    'C': 0.025

}
#ERROR_BOOKMARK_2-->AT `SVM`

# Create 5 objects that represent our 5 models

rf = SklearnHelper(clf=RandomForestRegressor,seed = SEED,params = rf_params)

et = SklearnHelper(clf = ExtraTreesRegressor,seed = SEED,params = et_params)

ada = SklearnHelper(clf = AdaBoostRegressor,seed = SEED,params = ada_params)

gb = SklearnHelper(clf = GradientBoostingRegressor,seed = SEED,params = gb_params)

# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# y_train --> y_Train

#train -->Train

#x_train-->x_Train

# x_test --> x_Test

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y_Train = data1['SalePrice'].ravel()

Train = data1.drop(['SalePrice'], axis=1)

x_Train = Train.values # Creates an array of the train data

x_Test = data_test.values # Creats an array of the test data
# Create our OOF train and test predictions. These base results will be used as new features

et_oof_train, et_oof_test = get_oof(et, x_Train, y_Train, x_Test) # Extra Trees

rf_oof_train, rf_oof_test = get_oof(rf,x_Train, y_Train, x_Test) # Random Forest

ada_oof_train, ada_oof_test = get_oof(ada, x_Train, y_Train, x_Test) # AdaBoost 

gb_oof_train, gb_oof_test = get_oof(gb,x_Train, y_Train, x_Test) # Gradient Boost

# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier



print("Training is complete")
rf_feature = rf.feature_importances(x_Train,y_Train)

et_feature = et.feature_importances(x_Train, y_Train)

ada_feature = ada.feature_importances(x_Train, y_Train)

gb_feature = gb.feature_importances(x_Train,y_Train)
rf_features = [2.30296094e-03, 3.71595859e-03, 2.23441229e-03, 1.18077881e-02,

 2.14629766e-02, 5.79594518e-05 ,2.42966131e-04, 2.19595316e-03,

 1.79370560e-03, 2.87098485e-06 ,9.39495555e-04, 1.10329735e-03,

 6.92714279e-03, 5.82765289e-04 ,4.20023299e-04, 1.06039693e-03,

 1.98479854e-03, 1.16456835e-01 ,3.29936804e-03, 5.08134961e-02,

 1.92608039e-02, 2.44888510e-03 ,5.91500081e-04, 2.06369371e-03,

 2.72229843e-03, 1.65819118e-03 ,1.57170228e-02, 6.71367635e-02,

 5.19048976e-04, 7.47771702e-03 ,3.83320255e-02, 4.36231047e-04,

 2.25067050e-03, 2.06893656e-03 ,3.35145896e-02, 5.28560135e-04,

 7.33196408e-04, 4.66762851e-03 ,5.89901525e-02, 3.58310371e-04,

 5.50510322e-03, 1.71525827e-03 ,4.80026349e-04, 6.18887565e-02,

 2.81451320e-02, 6.98993146e-05 ,8.58004582e-02, 2.25542246e-03,

 1.37579218e-03, 3.46473350e-02 ,4.47357199e-03, 4.78203278e-03,

 9.72777607e-04, 3.23197710e-02 ,1.89111254e-02, 3.07640668e-04,

 1.46939251e-02, 6.48567122e-03 ,1.79850578e-02, 2.94025676e-02,

 9.53137990e-03,6.43523524e-02 ,4.89337906e-02, 1.28352792e-03,

 1.87820052e-03, 4.25980769e-04 ,5.94989071e-03, 1.10899915e-02,

 6.12271230e-04, 1.32804362e-04 ,1.27053449e-03, 2.14768461e-03,

 8.02178720e-04, 1.02846874e-03 ,6.09719050e-05, 1.07850614e-04,

 2.91628530e-03, 1.16519794e-03, 8.00933028e-04, 2.41098346e-03]

et_features = [1.45229888e-03, 1.67632511e-03 ,3.87459244e-03 ,2.24189006e-03,

 7.95567000e-03, 6.93134743e-05, 4.90997518e-04, 1.38139084e-03,

 2.49669545e-03 ,5.42242845e-07, 1.86639655e-03, 2.57556847e-03,

 4.50714748e-03, 9.76043543e-04, 8.09376832e-04, 1.61731651e-03,

 7.86374122e-04, 2.36690281e-01, 2.78587276e-03, 2.50454525e-02,

 8.38576900e-03, 1.37067894e-03, 7.23863767e-04, 1.60126640e-03,

 1.31707024e-03, 1.02127413e-03, 4.63077999e-03, 1.41927577e-01,

 7.94185647e-04, 8.37447881e-04, 6.39535113e-02, 3.64713386e-04,

 2.39558674e-03, 1.99223520e-03, 1.38778471e-02, 5.08384328e-04,

 6.78202590e-04, 1.77890716e-03, 1.54030395e-02, 3.97982502e-04,

 7.03740557e-04, 7.00866927e-03, 1.27363628e-04, 2.23378572e-02,

 1.79792628e-02, 2.86997027e-04, 8.54433039e-02, 5.04129398e-03,

 1.92705001e-03, 3.13904508e-02, 2.72465240e-03, 5.35973737e-03,

 1.56699314e-03, 4.08769923e-02, 1.02208682e-02, 8.75570962e-04,

 1.61054827e-02, 1.25893926e-03, 1.16915897e-02, 7.39197727e-03,

 2.21129337e-03, 1.28382514e-01, 1.83760128e-02, 7.37727471e-04,

 3.73880589e-04, 5.35761972e-04, 2.67221341e-03, 2.10482391e-03,

 6.46268484e-04, 7.10688302e-04, 1.43294663e-03, 8.74988921e-04,

 4.17676256e-04, 7.41264061e-04, 8.26651713e-05, 6.47875243e-05,

 1.20801957e-03, 8.28163279e-04, 1.45533902e-03, 2.56430468e-03]

ada_features = [1.34955167e-02 ,3.57745815e-04 ,3.23452048e-05, 1.61118387e-02,

 2.70689827e-02, 0.00000000e+00, 0.00000000e+00, 9.34217904e-05,

 7.23903159e-03, 0.00000000e+00, 2.23152107e-03, 0.00000000e+00,

 6.88125746e-02, 3.75473746e-07, 0.00000000e+00, 0.00000000e+00,

 7.79578997e-05, 1.63406072e-01, 1.89622298e-04, 7.64543326e-03,

 1.46269754e-02, 1.64907988e-05, 5.80140201e-05, 1.05471705e-03,

 1.62060521e-03, 3.30966842e-03, 2.33188343e-03, 3.55039607e-03,

 1.07216281e-04, 7.17935502e-06, 1.08345917e-02, 0.00000000e+00,

 6.32647422e-03, 1.70416472e-03, 3.07856656e-02, 4.19102075e-06,

 4.99386635e-05, 1.20047040e-03, 6.47835551e-02, 0.00000000e+00,

 3.73133666e-04, 1.29402048e-05, 0.00000000e+00, 1.22848097e-02,

 1.52721624e-01, 0.00000000e+00, 1.21573312e-01, 1.72231820e-03,

 2.65856024e-07, 5.41076138e-03, 1.37941633e-03, 6.38597015e-03,

 0.00000000e+00, 3.98298344e-02, 3.35637062e-02, 3.58173850e-05,

 2.18966853e-02, 3.19785862e-03, 2.63871711e-03, 1.24629920e-02,

 4.81050003e-04, 6.68867668e-02, 3.46780312e-03, 3.31265473e-06,

 0.00000000e+00, 0.00000000e+00, 1.25874425e-02, 3.12158901e-02,

 0.00000000e+00, 0.00000000e+00, 5.11559769e-03, 3.13467159e-07,

 1.57088773e-05, 2.80802603e-05, 0.00000000e+00, 0.00000000e+00,

 1.27408608e-02, 2.65764875e-03, 1.74253592e-04, 4.72657878e-07]

gb_features = [1.92053033e-03 ,6.20128455e-04  ,4.56858878e-03 ,5.09113186e-03,

 1.57791012e-02, 3.28417872e-08 ,3.27486727e-05 ,1.15699904e-03,

 1.06671465e-03, 0.00000000e+00 ,4.55082956e-04 ,3.47793078e-04,

 1.10634879e-02, 6.89237180e-04 ,7.25308612e-05 ,3.32459163e-05,

 2.74023059e-04, 5.60148757e-01 ,7.26102679e-03 ,1.05363309e-02,

 5.72780665e-03, 7.27924260e-05 ,2.60411345e-05 ,1.22973192e-03,

 9.86032456e-04, 4.28647522e-04 ,3.33789328e-03 ,3.26937929e-02,

 1.27676424e-04, 1.82465823e-04 ,1.18865067e-02 ,1.61277219e-04,

 1.84338866e-03, 1.26554375e-03 ,3.16493663e-02 ,1.91008267e-04,

 5.03276290e-04, 3.26587709e-03 ,3.99727170e-02 ,5.72462134e-06,

 2.25228404e-04, 2.45829047e-03 ,8.95673426e-05 ,1.70123983e-02,

 3.35628187e-02, 4.34042481e-05 ,1.11890972e-01 ,1.04414773e-03,

 5.17931229e-05, 1.38639004e-03 ,3.74802495e-04 ,4.83005478e-04,

 2.03040778e-03, 3.55519495e-03 ,4.45312124e-03 ,6.15758449e-04,

 5.09871128e-03, 2.39691008e-04 ,4.77095762e-03 ,4.89349000e-03,

 2.57854978e-04, 2.49060662e-02 ,1.01154402e-02 ,6.19381772e-04,

 1.14740460e-04, 2.06420274e-04 ,1.87831623e-03 ,3.58140641e-03,

 5.20430663e-04, 2.67712988e-04 ,7.14431312e-04 ,1.11839285e-04,

 4.54101088e-05, 1.17486462e-04 ,5.13800919e-06 ,3.97132745e-05,

 2.14676543e-03, 6.06264082e-04 ,2.33686424e-04 ,2.55628670e-03]
cols = Train.columns.values

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_features,

     'Extra Trees  feature importances': et_features,

      'AdaBoost feature importances': ada_features,

    'Gradient Boost feature importances': gb_features

    })
feature_dataframe.head()
# Create the new column containing the average of values



feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(3)
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),

     'ExtraTrees': et_oof_train.ravel(),

     'AdaBoost': ada_oof_train.ravel(),

      'GradientBoost': gb_oof_train.ravel()

    })

base_predictions_train.head()
x_Train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis=1)

x_Test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
second_level_reg = xgboost.XGBRegressor()

second_level_reg.fit(x_Train,y_Train)

stack_predictions = second_level_reg.predict(x_Test)
# StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

#                             'Survived': predictions })
stack_pred = pd.DataFrame(stack_predictions)

stack_sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

stack_datasets = pd.concat([stack_sub_df['Id'],stack_pred],axis = 1)

stack_datasets.columns=['Id','SalePrice']

stack_datasets.to_csv('stack_submission2.csv',index=False)



stack_datasets.head()
# cols = train.columns.values

# # Create a dataframe with features

# feature_dataframe = pd.DataFrame( {'features': cols,

#      'Random Forest feature importances': rf_features,

#      'Extra Trees  feature importances': et_features,

#       'AdaBoost feature importances': ada_features,

#     'Gradient Boost feature importances': gb_features

#     })
33525.359153248755/0.20745
33525.359153248755/161606.93734995785
32129.766156650927/161606.93734995785
39984.82004253466/161606.93734995785