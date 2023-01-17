# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#IMPORTING THE DATA 

pd.set_option('display.max_columns',None)

train_full=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv",index_col="Id")

test_full=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv",index_col="Id")

train_full.head()
test_full.head(2)
train_full.columns
Categorical_cols=[cname for cname in train_full.columns if train_full[cname].dtypes=='object']

Numerical_cols=set(train_full.columns)-set(Categorical_cols)
'''factor=4

for col in Numerical_cols:

    upper_limit=train_full[col].mean()+(factor*train_full[col].std())

    lower_limit=train_full[col].mean()-(factor*train_full[col].std())

    train_full = train_full[(train_full[col] < upper_limit) & (train_full[col] > lower_limit)]'''
train_full.describe()
train_full.info()
train_full.Alley.value_counts()
#columns to get null values replced by 0

OK_columns=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

X_full=train_full.copy()

X_full=X_full.drop('SalePrice',axis=1)

Y=train_full['SalePrice']

X_full[OK_columns]=X_full[OK_columns].fillna(0)

X_full.info()
#Getting Numerical columns and Categorical columns

Categorical_cols=[cname for cname in X_full.columns if X_full[cname].dtypes=='object']

Numerical_cols=[cname for cname in X_full.columns if X_full[cname].dtypes in ['float64','int64']]



print('Printing Categorical Columns\n\n',Categorical_cols)

print('\n\n\n')

print('Printing Numerical Columns\n\n',Numerical_cols)

#Lets impute it

from sklearn.impute import SimpleImputer

catimp=SimpleImputer(missing_values=np.nan, strategy='most_frequent')

numimp=SimpleImputer(missing_values=np.nan, strategy='mean')



#train

imputed_categorical_train=pd.DataFrame(catimp.fit_transform(X_full[Categorical_cols].astype(str)))

imputed_numerical_train=pd.DataFrame(numimp.fit_transform(X_full[Numerical_cols]))

imputed_categorical_train.columns=X_full[Categorical_cols].columns

imputed_numerical_train.columns=X_full[Numerical_cols].columns

imputed_X_full=imputed_categorical_train.join(imputed_numerical_train)



#test

imputed_categorical_test=pd.DataFrame(catimp.transform(test_full[Categorical_cols]))

imputed_numerical_test=pd.DataFrame(numimp.transform(test_full[Numerical_cols]))

imputed_categorical_test.columns=test_full[Categorical_cols].columns

imputed_numerical_test.columns=test_full[Numerical_cols].columns

imputed_test_full=imputed_categorical_test.join(imputed_numerical_test)



imputed_X_full.info()

"""

new_vis = train_full.drop(Categorical_cols,axis=1)

plt.figure(figsize=(30,30))



correlation = new_vis.corr()

print(correlation)



sns.heatmap(correlation,annot=True)

"""
for col in Numerical_cols:

    imputed_X_full[col]=(imputed_X_full[col]-imputed_X_full[col].mean())/imputed_X_full[col].std()
#finding bad columns

good_columns=[cname for cname in Categorical_cols if set(imputed_X_full[cname]) == set(imputed_test_full[cname])]

print(good_columns)

bad_columns=list(set(Categorical_cols)-set(good_columns))

print('\n\n')

print(bad_columns)
#OnehotEncoding and label encodings

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



"""#Finding low cardinality columns 

low_cardinality_col=[cname for cname in good_columns if imputed_X_full[cname].nunique()<10]

print(low_cardinality_col)



#Encoding low cardinality columns using one hot encoding

OH=OneHotEncoder(handle_unknown='ignore', sparse=False)

One_Hot_Encoded_columns_train=pd.DataFrame(OH.fit_transform(imputed_X_full[low_cardinality_col]))

One_Hot_Encoded_columns_test=pd.DataFrame(OH.transform(imputed_test_full[low_cardinality_col]))



One_Hot_Encoded_columns_train.index=imputed_X_full.index

One_Hot_Encoded_columns_test.index=imputed_test_full.index



#finding high cardinality columns

high_cardinality_col=list(set(good_columns)-set(low_cardinality_col))

print(high_cardinality_col)



#Encoding high cardinality columns using label encoding

label_encoder=LabelEncoder()

Label_Encoded_columns_train=pd.DataFrame(label_encoder.fit_transform(imputed_X_full[high_cardinality_col]))

Label_Encoded_columns_test=pd.DataFrame(label_encoder.transform(imputed_test_full[high_cardinality_col]))

Label_Encoded_columns_test.columns=imputed_test_full[high_cardinality_col].columns

Label_Encoded_columns_train.columns=imputed_X_full[high_cardinality_col].columns

X_train=Label_Encoded_columns_train.join(One_Hot_Encoded_columns_train)

X_train=X_train.join(imputed_X_full[Numerical_cols])

X_test=Label_Encoded_columns_test.join(One_Hot_Encoded_columns_test)

X_test=X_test.join(imputed_test_full[Numerical_cols])

X_train.describe()

#One_Hot_Encoded_columns_train

#Label_Encoded_columns_train.head()"""



#Encoding every categorical using label encoding

label_encoder=LabelEncoder()

Label_Encoded_columns_train=imputed_X_full[good_columns].apply(label_encoder.fit_transform)

Label_Encoded_columns_test=imputed_test_full[good_columns].apply(label_encoder.fit_transform)

Label_Encoded_columns_test.columns=imputed_test_full[good_columns].columns

Label_Encoded_columns_train.columns=imputed_X_full[good_columns].columns

X_train=Label_Encoded_columns_train.join(imputed_X_full[Numerical_cols])



X_test=Label_Encoded_columns_test.join(imputed_test_full[Numerical_cols])



X_train.describe()

#One_Hot_Encoded_columns_train

#Label_Encoded_columns_train.head()





"""X_train=imputed_X_full.drop(Categorical_cols,axis=1)

X_test=imputed_test_full.drop(Categorical_cols,axis=1)

correlation_correction=['YrSold','MiscVal','PoolArea','BsmtHalfBath','LowQualFinSF','BsmtFinSF2','OverallCond','MSSubClass']

X_train=X_train.drop(correlation_correction,axis=1)

X_test=X_test.drop(correlation_correction,axis=1)"""

'''sns.barplot(x=train_full['CentralAir'],y=train_full["SalePrice"])'''
'''sns.barplot(x=train_full['MSSubClass'],y=train_full["SalePrice"])'''
'''sns.barplot(x=train_full['Street'],y=train_full["SalePrice"])'''
'''sns.barplot(x=train_full['BsmtExposure'],y=train_full["SalePrice"])'''
'''sns.barplot(x=train_full['BsmtFinType1'],y=train_full["SalePrice"])'''
print(train_full.SalePrice.describe())

plt.figure(figsize=(9,8));

sns.distplot(train_full.SalePrice, color='r', bins=100,hist_kws={'alpha':0.2})
set(train_full.dtypes.tolist())

num_data=train_full.select_dtypes(include=['float64','int64'])

#X_train.hist(figsize=(16,20), bins=50, xlabelsize=8, ylabelsize=8);
#corelation of numerical data

num_corr=num_data.corr()['SalePrice'][:-1]#because sale price is the latest column and we dont want self correlation

print(num_corr)

golden_num_features=num_corr[num_corr>0.5].index

print(golden_num_features)

rest_features=golden_num_features=num_corr[num_corr<0.3].index
'''X_train=X_train.drop(rest_features,axis=1)

X_test=X_test.drop(rest_features,axis=1)'''
columnlist=[col for col in X_train.columns if col in Numerical_cols]

X_train.describe()

print(columnlist)
Y.head()
"""#THE LAST MERGE

X_Full=X_train.copy()

X_Full.append(Y)

X_Full.SalePrice"""

"""#REMOVING OUTLIERS

factor=3

for col in columnlist:

    upper_limit=X_train[col].mean()+(factor*X_train[col].std())

    lower_limit=X_train[col].mean()-(factor*X_train[col].std())

    X_Full = X_train[(X_train[col] < upper_limit) & (X_train[col] > lower_limit)]

    

    

#X_FU.describe()"""
'''

new_vis = X_train.drop(low_cardinality_col,axis=1)

plt.figure(figsize=(14,7))



correlation = new_vis.corr()

print(correlation)



sns.heatmap(correlation)

'''






from sklearn.model_selection import train_test_split

# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_train, Y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

# Define the model

my_model_3 = XGBRegressor(n_estimators=2000,learning_rate=0.01)



# Fit the model

my_model_3.fit(X_train_full,y_train) # Your code here



# Get predictions on test

predictions_3 = my_model_3.predict(X_valid_full)



#Get predictions on train

predictions_train=my_model_3.predict(X_train_full)



# Calculate MAE on test

mae_3 = mean_absolute_error(y_valid,predictions_3)



# Calculate MAE on train

mae = mean_absolute_error(y_train,predictions_train)



# Uncomment to print MAE

print("Mean Absolute Error: on test" , mae_3)

# Uncomment to print MAE

print("Mean Absolute Error: on train" , mae)
preds_test=preds_test =my_model_3.predict(X_test)
output = pd.DataFrame({'Id': test_full.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)