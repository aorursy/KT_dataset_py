import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

import matplotlib.pyplot as plt

%matplotlib inline
def transform_features(df,null_cutoff):

    #dropping columns with more then a missing values

    null_values=df.isnull().sum()

    dorp_missing_values=null_values[null_values>(null_cutoff*len(null_values))]

    df=df.drop(dorp_missing_values.index, axis=1)

    

    # counting null values in text columns

    text_cols_nullcount=df.select_dtypes(include=['object']).isnull().sum().sort_values(ascending=False)

    text_cols_nullcols=text_cols_nullcount.index

    for col in text_cols_nullcols:

        mostcounts=df[col].value_counts().index.tolist()

        df[col]=df[col].fillna(mostcounts[0]) #replacing the missing column in a text with the highest number of values

    

    #missing values in numerical columns 

    num_cols=df.select_dtypes(include=['integer','float']).columns #selecting numerical columns

    num_null_counts=df[num_cols].isnull().sum().sort_values(ascending=False) #counting null values in columns

    num_null_cols=num_null_counts[num_null_counts!=0].index #selecting the ones that have missing values

    df=df.fillna(df[num_null_cols].mode().to_dict(orient='records')[0]) #replacing missing with mode

    

    #transfomring year sold and year built into a meaningful feature

    years_sold = df['YrSold'] - df['YearBuilt']

    years_since_remod = df['YrSold'] - df['YearRemodAdd']

    df['Years Before Sale'] = years_sold

    df['Years Since Remod'] = years_since_remod

    #df = df.drop([1702, 2180, 2181], axis=0) #these rows caused negative values for both of these features



    #drop columns that are not meaningful in ML, or they leak information in sale.

    df = df.drop(["Id", "MoSold", "SaleCondition", "SaleType", "YearBuilt", "YearRemodAdd"], axis=1)

    return df
def select_features(df, coeff_threshold=0.4, uniq_threshold=10):

    num_df=df.select_dtypes(include=['integer','float'])

    corrs=num_df.corr()[target].abs()

    #keeping only columns that have correlation with target higher than threshold

    df=df.drop(corrs[corrs<coeff_threshold].index, axis=1)

    

    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", 

                        "Lot Config", "Neighborhood","Condition 1", "Condition 2", "Bldg Type",

                        "House Style", "Roof Style", "Roof Matl", "Exterior 1st","Exterior 2nd",

                        "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type",

                        "Misc Feature", "Sale Type", "Sale Condition"]

    

    #check to see if our current dataset still keeps the nominal features

    transform_cat_cols=[]

    for col in nominal_features:

        if col in df.columns:

            transform_cat_cols.append(col)

    

    #getting rid of nominal columns with too many unique values

    for col in transform_cat_cols:

        len(df[col].unique())>uniq_threshold

        df=df.drop(col, axis=1)

        

    #convert text columns to dummy variables

    text_cols=df.select_dtypes(include=['object'])

    for col in text_cols:

        df[col]=df[col].astype('category')

    

    df=pd.concat([df,pd.get_dummies(df.select_dtypes(include=['category']))],axis=1)

    

    return df
def train_and_test(df,k=0):

    num_df=df.select_dtypes(include=['integer','float'])

    features=num_df.columns.drop(target)

    model=linear_model.LinearRegression()

    

    if k==0:

        cut=int(num_df.shape[0]/2)

        train=num_df.iloc[:cut]

        test=num_df.iloc[cut:]

        model.fit(train[feature],train[target])

        prediction=model.predict(test[target])

        mse = mean_squared_error(test[target], predictions)

        rmse = np.sqrt(mse)



        return rmse

    elif k==1:

        # Randomize *all* rows (frac=1) from `df` and return

        shuffled_df = df.sample(frac=1, )

        train = df[:1460]

        test = df[1460:]

        

        model.fit(train[features], train[target])

        predictions_one = model.predict(test[features])        

        

        mse_one = mean_squared_error(test[target], predictions_one)

        rmse_one = np.sqrt(mse_one)

        

        model.fit(test[features], test[target])

        predictions_two = model.predict(train[features])        

        mse_two = mean_squared_error(train[target], predictions_two)

        rmse_two = np.sqrt(mse_two)

        

        avg_rmse = np.mean([rmse_one, rmse_two])

        print(rmse_one)

        print(rmse_two)

        return avg_rmse

    else:

        kf = KFold(n_splits=k, shuffle=True)

        rmse_values = []

        for train_index, test_index, in kf.split(df):

            train = df.iloc[train_index]

            test = df.iloc[test_index]

            model.fit(train[features], train[target])

            predictions = model.predict(test[features])

            mse = mean_squared_error(test[target], predictions)

            rmse = np.sqrt(mse)

            rmse_values.append(rmse)

        print(rmse_values)

        avg_rmse = np.mean(rmse_values)

        return avg_rmse

    

df = pd.read_csv('../input/train.csv')

target='SalePrice'

transform_df = transform_features(df,null_cutoff=0.05)

filtered_df = select_features(transform_df)

rmse = train_and_test(filtered_df, k=4)



print('RMSE is:', rmse)
import pandas as pd

import numpy as np

import seaborn as sns

import io

import urllib.request

import matplotlib.pyplot as plt

%matplotlib inline



data=pd.read_csv('../input/train.csv')

train=data.iloc[:1120]

test=data.iloc[1120:]

target='SalePrice'

train.shape
corr=train.corr()

fig,ax=plt.subplots(figsize=(8,6))



sns.heatmap(corr)
sns.lmplot(x='GrLivArea',y='SalePrice',data=train)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

lr=LinearRegression()

lr.fit(train[['GrLivArea']],train['SalePrice'])

print('Coeff is:',lr.coef_)

print('Intercept is:',lr.intercept_)



prediction_test=lr.predict(test[['GrLivArea']])

prediction_train=lr.predict(train[['GrLivArea']])



mse_test=mean_squared_error(test['SalePrice'],prediction_test)

mse_train=mean_squared_error(train['SalePrice'],prediction_train)



rmse_test=mse_test**(1/2)

rmse_train=mse_train**(1/2)

print('rmse_test is:',rmse_test)

print('rmse_train is:',rmse_train)
#only selects the integer and float columns.

num_train=train.select_dtypes(include=['int','float'])

num_train.info()
#dropping invalid columns in regression

num_train=num_train.drop(['Id','MoSold','YrSold'],axis=1)



#displaying numerical columns with no missing values

null_series=num_train.isnull().sum()

full_cols_train=null_series[null_series==0]

#full_cols_train
#selecting a subset of train dataset that only contains numerical values and do not have missing values.

train_subset=train[full_cols_train.index]



#find the correlation of these features with target variable.

corrmatrix=train_subset.corr()

sorted_corrs=corrmatrix['SalePrice'].abs().sort_values(ascending=False)

sorted_corrs
#strong correlation cut-off

a=0.25



#cutoff value for features variance(features with lower variance than this number will be dropped)

b=0.015



strong_corrs=sorted_corrs[sorted_corrs>a]

corrmatrix=train_subset[strong_corrs.index].corr()

fig,ax=plt.subplots(figsize=(8,6))

sns.heatmap(corrmatrix,ax=ax)
final_corr_cols=strong_corrs.drop(['GarageCars', 'TotRmsAbvGrd'])



features=final_corr_cols.drop(['SalePrice']).index



clean_test=test[final_corr_cols.index].dropna()
target='SalePrice'

lr=LinearRegression()

lr.fit(train[features],train[target])



train_predictions=lr.predict(train[features])

test_predictions=lr.predict(clean_test[features])



train_rmse=np.sqrt(mean_squared_error(train[target],train_predictions))

test_rmse=np.sqrt(mean_squared_error(clean_test[target],test_predictions))



print('train rmse:',train_rmse)

print('test rmse:',test_rmse)
rescale_train=(train[features]-train[features].min())/(train[features].max()-train[features].min())
sorted_vars=rescale_train.var().sort_values()

sorted_vars
features.drop(sorted_vars[sorted_vars<b].index)

#fetures.drop(sorted_vars<b)

#features.drop(['Lot Area','Open Porch SF'])
#re-fittin the model with the new features list

lr=LinearRegression()

lr.fit(train[features],train[target])



train_predictions=lr.predict(train[features])

test_predictions=lr.predict(clean_test[features])



train_rmse_2=np.sqrt(mean_squared_error(train[target],train_predictions))

test_rmse_2=np.sqrt(mean_squared_error(clean_test[target],test_predictions))



print('train rmse_2:',train_rmse_2)

print('test rmse_2:',test_rmse_2)