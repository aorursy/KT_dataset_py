#Import Pandas and numpy libraries

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')



#Read the dataset, list the features and make a copy.

df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df=pd.concat([df_train,df_test])

df_new=df.copy()



# we are concatenating test and train data sets for now because for data preparation purposes like

#imputation  and scaling, it is better to use a whole data set



#DataOverview

df.shape

#DataOverview



df.head
#Null Value counts in the data set

null_value_counts=df.isnull().sum().sort_values(ascending=False)/len(df)*100

print(null_value_counts)

#From the above results, we can safely drop PoolQC,MiscFeature, Alley,Fence>80%



df=df.drop(columns=['PoolQC','MiscFeature','Alley','Fence'])
#- unique values by column 

unique_values_by_col= df.nunique()

print(unique_values_by_col)

#Identifying categorical and numerical for data cleaning

# unique vlaues>20 or 25 could be categorical

cat_cols=[]

for key,value in unique_values_by_col.items():

    if value  <= 20:

        cat_cols.append(key)

        

#Lets look at the unique values in categorical columns

for col in cat_cols:

    print(col, df[col].unique())

    

#Add neighborhood back to cal cols and year build 

    

cat_cols=cat_cols+['Neighborhood','YearBuilt','GarageYrBlt','YearRemodAdd']
#Identify Numerical columns

num_cols=[x for x in df.columns if x not in (cat_cols +['Id'])]

df_num_cols=df[num_cols]

print(df_num_cols.columns)

from pandas import unique

for i in range(df_num_cols.shape[1]):

    num = len(unique(df_num_cols.iloc[:, i]))

    

    percentage = float(num) / df_num_cols.shape[0] * 100

    print('%d, %s, %d, %.1f%%' % (i, df_num_cols.columns[i],num, percentage))
#Find Percent of non zero values    

for i in range(df_num_cols.shape[1]):

    num = df_num_cols.iloc[:, i].astype(bool).sum(axis=0)

    

    percentage = float(num) / df_num_cols.shape[0] * 100

    print('%d, %s, %d, %.1f%%' % (i, df_num_cols.columns[i],num, percentage))
#Find null values

                

null_value_counts_num=df_num_cols.isnull().sum().sort_values(ascending=False)/len(df)*100

print(null_value_counts_num)
#lot Frontage  - replace null with average and MasVnrArea replace with zereo as 49% are zero.

df_num_cols['MasVnrArea']=df_num_cols['MasVnrArea'].replace(np.nan,0)





#Imputation of null values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(df_num_cols.iloc[:,:])

df_num_cols.loc[:,:] = imputer.transform(df_num_cols.loc[:,:])



#Check null value counts to make sure there are no null values

null_value_counts_num=df_num_cols.isnull().sum().sort_values(ascending=False)/len(df)*100

print(null_value_counts_num)


from numpy import arange

from sklearn.feature_selection import VarianceThreshold

from matplotlib import pyplot

#Get Variance

# split data into inputs and outputs

X = df_num_cols.iloc[:, :-1]

y = df_num_cols.iloc[:, -1]

print(X.shape, y.shape)

# define thresholds to check

thresholds = np.arange(0.0, 0.55, 0.05)

# apply transform with each threshold

results = list()

for t in thresholds:

# define the transform

    transform = VarianceThreshold(threshold=t)

# transform the input data

    X_sel = transform.fit_transform(X)

# determine the number of input features

    n_features = X_sel.shape[1]

    print('>Threshold=%.2f, Features=%d' % (t, n_features))

# store the result

    results.append(n_features)

# plot the threshold vs the number of selected features

pyplot.plot(thresholds, results)

pyplot.show()


%matplotlib inline

import matplotlib.pyplot as plt

fig,ax=plt.subplots(3,6,figsize=(23,10))



i=0

j=0

for column in df_num_cols:

    if column=='SalePrice':

        

        continue

    else:

        

        df_num_cols.boxplot([column],ax=ax[i,j])



    if j<5:

        j+=1

    else:

        j=0

        i+=1











 

import seaborn as sns

fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(25,25))



for feature, ax in zip(num_cols, axes.flat):

    sns.kdeplot( df_num_cols[feature], color= 'navy',ax=ax,bw=0.1)

    ax.set_title("KDE for {}".format(feature))

plt.show()
#Correlation Spearman and Pearson. Pearson assumes guassian but data is not guassian



corr=df_num_cols.corr(method='spearman')

plt.figure(figsize=(20, 20))

ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=True,

                 linewidths=.2, cmap="YlGnBu")
#Get top 30 correlation features

correlation1=corr.abs().unstack().sort_values(ascending=False)

correlation1=correlation1[correlation1!=1]

print(correlation1[0:30])
#import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(df_num_cols.values, i) for i in range(df_num_cols.shape[1])]

vif["features"] = df_num_cols.columns

print(vif)
#Based on correlation matrix and vif, remove 1st Flr SF





df_num_cols=df_num_cols.drop(columns='1stFlrSF')
#log transform skewed numeric features:

from scipy.stats import skew

skewed_cols = df_num_cols.apply(lambda x: skew(x)) #compute skewness

skewed_cols = skewed_cols[skewed_cols > 0.9]

print(skewed_cols)

skewed_cols = skewed_cols.index



#Log transformation of Skewed columns

df_num_cols[skewed_cols] = np.log1p(df_num_cols[skewed_cols])



#Standardization for non skewed column

from sklearn.preprocessing import MinMaxScaler

trans = MinMaxScaler()

df_num_cols.iloc[:,10] = trans.fit_transform(np.array(df_num_cols.iloc[:,10]).reshape(-1,1))

#Create df for categroical features and find unique values

df_cat_cols=df[cat_cols]

unique_values_by_cat_col= df_cat_cols.nunique()

print(unique_values_by_cat_col)
#Binning for Years

df_cat_cols['YearBuilt']=pd.cut(df_cat_cols.YearBuilt,bins=[0,1880,1890,1900,1910,1920,1930,1940,1950,

                                      1960,1970,1980,1990,2000,2010,2020],

       labels=['YB_1','YB_2','YB_3','YB_4','YB_5','YB_6','YB_7','YB_8','YB_9',

               'YB_10','YB_11','YB_12','YB_13','YB_14','YB_15'])



df_cat_cols['GarageYrBlt']=pd.cut(df_cat_cols.GarageYrBlt,bins=[0,1880,1890,1900,1910,1920,1930,1940,1950,

                                      1960,1970,1980,1990,2000,2010,2020],

       labels=['YB_1','YB_2','YB_3','YB_4','YB_5','YB_6','YB_7','YB_8','YB_9',

               'YB_10','YB_11','YB_12','YB_13','YB_14','YB_15'])



df_cat_cols['YearRemodAdd']=pd.cut(df_cat_cols.YearRemodAdd,bins=[0,1880,1890,1900,1910,1920,1930,1940,1950,

                                      1960,1970,1980,1990,2000,2010,2020],

       labels=['YB_1','YB_2','YB_3','YB_4','YB_5','YB_6','YB_7','YB_8','YB_9',

               'YB_10','YB_11','YB_12','YB_13','YB_14','YB_15'])
#Barplot with counts of each class of categorical variable

for col in cat_cols:

    ax=sns.countplot(x=col,data=df_cat_cols)

    ax.set_title(' '.format(col))

    ax.set_ylabel('count {}'.format(col))

    ax.set_xlabel('{}'.format(col))

    plt.show()

cat_cols=df_cat_cols.columns   
'''

#Based on the graphs forllowing can be removed - 

to_remove_cat_cols=['PoolArea','GarageCond','GarageQual','Functional','KitchenAbvGr','Heating','CentralAir','BsmtFinType2','RoofMatl',

'Condition2','Condition1','Utilities','Street']

'''

cat_cols=[x for x in cat_cols if x not in ['PoolArea','GarageCond','GarageQual','Functional','KitchenAbvGr','Heating','CentralAir','BsmtFinType2','RoofMatl',

'Condition2','Condition1','Utilities','Street']]
#Print unique values

for col in cat_cols:

    print(col, df_cat_cols[col].unique()) 



#print null values counts

df_cat_cols=df_cat_cols[cat_cols]

null_value_counts_cat=df_cat_cols.isnull().sum().sort_values(ascending=False)/len(df)*100

print(null_value_counts_cat)
#Fill null with most repetetive class

df_cat_cols = df_cat_cols.fillna(df_cat_cols.mode().iloc[0])
#Encoding of categorical variables

df_cat_cols=pd.get_dummies(df_cat_cols.astype(str), drop_first=True)



#Concatenate numerical and categorical dataframes to get new dataframe

df_new1= pd.concat([df_cat_cols, df_num_cols], axis=1, sort=False)

X=df_new1.iloc[0:1460,0:-1]

y=df_new1.iloc[0:1460,-1]

X_submission=df_new1.iloc[1461:,0:-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#Import libraries

from numpy import mean

from numpy import std

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor

from sklearn.pipeline import Pipeline
#Function to define all the models

def get_models():

    models = dict()

    

    rfe = RFE(estimator=LinearRegression(),n_features_to_select=10)

    model = LinearRegression()

    models['linear'] = Pipeline(steps=[('s',rfe),('m',model)])

    

    rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=100)

    model = DecisionTreeRegressor()

    models['DecisionTree'] = Pipeline(steps=[('s',rfe),('m',model)])



    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100), n_features_to_select=100)

    model = RandomForestRegressor()

    models['RandomForest'] = Pipeline(steps=[('s',rfe),('m',model)])



    rfe = RFE(estimator=XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1),

              n_features_to_select=100)

    model = XGBRegressor()

    models['XGB'] = Pipeline(steps=[('s',rfe),('m',model)])

    

    return models



models = get_models()
#Evaluate the model

from sklearn.metrics import mean_absolute_error,mean_squared_error

from math import sqrt

for name,model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rms = sqrt(mse)

# report performance



    print(name+ ' MAE: %.3f' % mae)

    

    print(name +' MSE: %.3f' % mse)

   

    print(name+ ' RMSE: %.3f' % rms)
#Predict on Test data Set

X_submission=df_new1.iloc[1460:,0:-1]

y_submission=model.predict(X_submission)



#Inverse log transform

y_submit = np.expm1(y_submission)



#Convert to dataframe and reset index

df_submit = pd.DataFrame(data=y_submit,columns=['SalePrice'])

df_submit.index = np.arange(1461, 2920)

df_submit.head