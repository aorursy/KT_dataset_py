import pandas as pd

import numpy as np

%matplotlib inline

traindf = pd.read_csv('/kaggle/input/predict-the-housing-price/train.csv')

testdf = pd.read_csv('/kaggle/input/predict-the-housing-price/Test.csv')

samplesubdf = pd.read_csv('/kaggle/input/predict-the-housing-price/Samplesubmission.csv')

samplesubdf.head()
print('number of columns in train dataset:-' + str(len(traindf.columns))) ## 81 columns

print('train dataframe shape' + str(traindf.shape)) ## 81 columns



cols_with_na = traindf.isna().sum().reset_index().rename(columns={'index':'col_name',0:'na_count'})

cols_to_delete = list(cols_with_na[cols_with_na['na_count'] > 100].col_name)

print("removing these columns:- \n",cols_to_delete)



### removing the columns from main data set

traindf.drop(cols_to_delete, axis=1, inplace=True)

testdf.drop(cols_to_delete, axis=1, inplace=True)

print(traindf.shape)

print(testdf.shape)
cols_with_na.set_index('col_name',inplace=True)

cols_with_na.drop(cols_to_delete, inplace=True)

na_cols = cols_with_na[cols_with_na['na_count'] > 0].sort_values(by='na_count', ascending=False).index

print(na_cols)


##Get the max value for each column where theere is na in the record and replace with NA for that columns

for item in na_cols:

    maxcolval = traindf[item].value_counts().index[0]

    traindf[traindf[item].isna()][item] = maxcolval

    testdf[testdf[item].isna()][item] = maxcolval   
#traindf.isna().sum() ## There is no NA in the dataset

traindf['HouseStyle'].value_counts()

traindf.describe()
import matplotlib.pyplot as plt

%matplotlib inline

traindf.hist(bins=50, figsize=(20,15))

plt.show()
traindf.head()
traindf1 = traindf.copy() 

testdf1 = testdf.copy() 

coldatatype = pd.DataFrame(traindf1.dtypes, columns=['datatype']).reset_index()

count = 0

for item in range(0,len(coldatatype)):

    if coldatatype.loc[item,'datatype'] == 'object':

        colnm = coldatatype.loc[item,'index']

        traindf1[colnm] = traindf1[colnm].astype('category')  ### Converting all the object data type into categorical variable

        traindf1[colnm] = traindf1[colnm].cat.codes ### replacing all the categorical values with codes

        testdf1[colnm] = testdf1[colnm].astype('category')  ### Converting all the object data type into categorical variable

        testdf1[colnm] = testdf1[colnm].cat.codes ### replacing all the categorical values with codes
#### Remove columns which does not have much correlation

corr_matrix = traindf1.corr()

cols_corr = pd.DataFrame(corr_matrix['SalePrice']).reset_index().rename(columns={'SalePrice':'corr_val'})

cols_corr['corr_val'].sort_values(ascending=False)

less_corr_cols_del_list  = cols_corr[(cols_corr['corr_val'] < 0.2) & (cols_corr['corr_val'] > -0.2)]['index']

len(less_corr_cols_del_list)  ### Removing 37 columns

traindf1.drop(less_corr_cols_del_list, axis=1,inplace=True)

testdf1.drop(less_corr_cols_del_list, axis=1,inplace=True)

print("train data shape {}  and test data shape {} ".format(traindf1.shape,testdf1.shape))
y= traindf1['SalePrice']

X= traindf1.drop(['SalePrice'], axis=1)

### Removing 2 columns from the X as they still have some NaN values - MasVnrArea     , GarageYrBlt         

X.drop(['MasVnrArea','GarageYrBlt'], axis=1, inplace=True)

X_test_data = testdf1.drop(['MasVnrArea','GarageYrBlt'], axis=1)

print("After removing 2 columns - X data shape shape {} and X_test_data Shape {} ".format(X.shape,X_test_data.shape))
###### Breaking the Train data set into Train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)


from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor



lin_reg = LinearRegression()

des_tree_reg = DecisionTreeRegressor()



lin_reg.fit(X_train, y_train)

des_tree_reg.fit(X_train, y_train)
# let's try few training instances

some_data = X.iloc[:5]

some_labels = y.iloc[:5]



print("lin_reg Predictions:", lin_reg.predict(some_data))

print("des_tree_reg Predictions:", des_tree_reg.predict(some_data))

print("Actual Value:", list(some_labels))
#### Applying linear Regression on test data from train data

y_pred_lin_reg = lin_reg.predict(X_test)

y_pred_des_tre_reg = des_tree_reg.predict(X_test)

y_pred_lin_reg = list(y_pred_lin_reg.round().astype(int))

y_pred_des_tre_reg = list(y_pred_des_tre_reg.round().astype(int))

y_tst = list(y_test)

y_pred_test = pd.DataFrame(list(zip(y_tst,y_pred_lin_reg,y_pred_des_tre_reg)) , columns=['y_test','y_pred_lin_reg','y_pred_des_tre_reg'])

y_pred_test['lin_reg_pred_diff_test'] = y_pred_test['y_pred_lin_reg'] - y_pred_test['y_test']

y_pred_test['des_tree_reg_pred_diff_test'] = y_pred_test['y_pred_des_tre_reg'] - y_pred_test['y_test']

y_pred_test.head() ### looks like linear regression is predicting better
### Predicting on test data

y_test_pred = lin_reg.predict(X_test_data)

y_test_pred =  list(y_test_pred.round())

Y_test_pred_id = list(testdf['Id'])

myfinalSubdf = pd.DataFrame(list(zip(Y_test_pred_id,y_test_pred)), columns=['Id','SalePrice'])

myfinalSubdf.shape

myfinalSubdf.to_csv('submission.csv', index=False)