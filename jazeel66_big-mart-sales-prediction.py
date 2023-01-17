import warnings

warnings.simplefilter('ignore')
# Importing the Standard Libraries here

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest, ExtraTreesRegressor

from sklearn.metrics import mean_squared_error, r2_score
# Importing the datasets



data= pd.read_csv('../input/Train_UWu5bXk.csv')
print(data.shape)



# We will append the train and test for easier preprocessing
print(data.shape)



data.describe()



# we see that these are the numeric columns, we will verify this in a moment and make changes if needed.
# we will see if the numerical columns have some relationship with eachother



# before that we will check if pandas infered all the columns correctly, make changes if not.



data.info()



# we see that all of them to be correct, except year column. we will check the values first.
# before everything else we will check if there are any missing data in the dataset



data.isna().sum()

# we see missing data in Two variables, the third ids due to appending the test set.
data['Item_Weight'].interpolate(inplace= True)



# Let's check the values in there. 



print(data['Item_Weight'].isna().sum())

data['Item_Weight'].sample(10)



# we see that the value has been filled
# We need to impute the values on this variable



print(data['Outlet_Size'].isna().sum())



data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace= True)



print(data['Outlet_Size'].isna().sum())



# we imputed the missing values, now we will proceed
print(f'Number of uniqure values in "Estd.Year" is: {data.Outlet_Establishment_Year.nunique()}')

print(data.Outlet_Establishment_Year.value_counts(dropna= False))



# we will convert this in to an object as there aren't many values
data['Outlet_Establishment_Year']= data.Outlet_Establishment_Year.astype('object')
print(data.Outlet_Establishment_Year.dtype)



# we see that data type is now an object
# we will now see a pairpolot of the numerical columns



num_cols= [*data.select_dtypes(['int64', 'float64']).columns]



sns.pairplot(data[num_cols])



# we don't see a lot of serious relationships betweem the variables and the target variables,

# we will plot a more meaningful plot using seaborn
num_cols.remove('Item_Outlet_Sales')

num_cols
plt.figure(figsize= (24, 9))



count= 1



for col in num_cols:

    

    plt.subplot(3, 2, count)

    

    sns.regplot(x= col, y= 'Item_Outlet_Sales', data= data)

    

    plt.xlabel(col)

    

    count+=1

    

    plt.subplot(3, 2, count)

    

    sns.distplot(data.loc[data[col].notnull(), col])

    

    count+= 1

    

    # We can't see no clear relationship in the data
data.head()
# We'll check the values of categorical columns ('objects')



obj_cols= [*data.select_dtypes('object').columns]



obj_cols
for col in obj_cols:

    

    if data[col].nunique() > 10:

        print(f'Number of unique values in {col} is {data[col].nunique()} so not printing values.')

        print(" ")

    else:

        

        print(f'Values in {col} are: \n {data[col].value_counts()}')

        print(" ")

        

# we see that there are duplicate values in the Item_Fat_Content. We'll work on it
data['Item_Fat_Content'].value_counts(dropna= False)
data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace= True)



# We will check the values again

data['Item_Fat_Content'].value_counts(dropna= False)

# seems we have replace correctly
# We also saw that there are 1559 unique values in Item_Identifier column and the total number of observations are close to 8k

# let's check this out



print(data['Item_Identifier'].sample(10)) # Let's try extracting the first three letters from the variable and check if there's a pattern



print(data['Item_Identifier'].str[:3].value_counts(dropna= False)) # looks like there are 71 values



print(data['Item_Identifier'].str[:2].value_counts(dropna= False)) # looks like there are only 3 values if we extract 2 letters



data['Item_Identifier']= data['Item_Identifier'].str[:2]
# We see that there is a value called NC, non-consumable, we will have to change fat content



data['Item_Fat_Content']= np.where(data['Item_Identifier']== 'NC', 'Non-durable', data['Item_Fat_Content'])
data.sample(10)
plt.figure(figsize= (24, 12))



for idx, col in enumerate(obj_cols):

    

    plt.subplot(3, 3, idx+1)

    

    sns.boxplot(col, 'Item_Outlet_Sales', data= data)
data.boxplot(column= 'Item_Outlet_Sales', by= ['Item_Fat_Content', 'Item_Identifier'], figsize= (12, 4), rot= 45)
data.boxplot(column= 'Item_Outlet_Sales', by= ['Outlet_Location_Type', 'Outlet_Size'], figsize= (12, 4), rot= 45)
# we'll create a dummy dataframe from the preprocessed dataFrame





df= pd.get_dummies(data, drop_first= True)
print(df.shape)

df.head()
for col in num_cols:

    

    print(f'Minimum value in {col} is: {data[col].min()}')

    print(" ")

    print(f'Minimum value in {col} is: {data[col].max()}')

    print(" ")

    

    # seems like there 
df['Non-Visible']= np.where(df['Item_Visibility']==0, 1, 0)



df['Non-Visible'].value_counts(dropna= False)
df.head()
df.isna().sum()
X, y= df.drop('Item_Outlet_Sales', axis= 1), df.Item_Outlet_Sales
X.shape, y.shape
y.head()
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 123)
lr = LinearRegression()



lr.fit(X_train, y_train)



lr_pred= lr.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, lr_pred)))
rf= RandomForestRegressor(max_depth= 5)



rf.fit(X_train, y_train)



rf_pred= rf.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, rf_pred)))



# We get a better score from a RandomForest Model
gbm= GradientBoostingRegressor(max_depth= 2)



gbm.fit(X_train, y_train)



gbm_pred= gbm.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, gbm_pred)))



# we get a slightly better score.

from sklearn.metrics import make_scorer
# creating a custom scoring function for cross validation



def RMSE(y_true, y_pred):

    

    RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))

    

    return RMSE



rmse= make_scorer(RMSE, greater_is_better= False)
score= cross_val_score(estimator= gbm, X= X_train, y= y_train, scoring= rmse, cv= 5,\

                n_jobs= -1, verbose= 1)



score.mean(), score.std()

# we get negative score as the scorer function returns negative score in cross validation
et= ExtraTreesRegressor()



et.fit(X_train, y_train)



et_pred= et.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, et_pred)))



# we get a slightly better score.
iso_forest= IsolationForest(contamination= 'auto', behaviour= 'New')



outliers= iso_forest.fit_predict(X, y)



pd.Series(outliers).value_counts(dropna= False)



# -1 indicate that the values are outliers
# we will remove the outliers from the original predictor(X) and traget(y) variables



out_bool= outliers == 1



X_new, y_new= X[out_bool], y[out_bool]
X_new.shape, y_new.shape



# we'll now create new train and test values
X_train, X_test, y_train, y_test= train_test_split(X_new, y_new, random_state= 123, test_size= 0.2)



# splitting data 80/20
lr= LinearRegression()



lr.fit(X_train, y_train)



lr_pred= lr.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, lr_pred)))
rf= RandomForestRegressor(max_depth= 5)



rf.fit(X_train, y_train)



rf_pred= rf.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, rf_pred)))



# the score improves a little.
gbm= GradientBoostingRegressor(max_depth= 2)



gbm.fit(X_train, y_train)



gbm_pred= gbm.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, gbm_pred)))



# We see a slight improvement in the socre after removing the outliers

# we can perform a GridSearch to see if we can improve the score further



gbm_params= {'max_depth': np.arange(1, 10, 2), "max_features": [.7, .8, .9],

             'max_leaf_nodes': np.arange(2, 10, 2), "min_samples_leaf": np.arange(1, 10, 2),

             'min_samples_split': np.arange(2, 10, 2)}



gbm_grid= GridSearchCV(gbm, gbm_params, scoring= rmse, n_jobs= -1, cv= 3, verbose= 1)



gbm_grid.fit(X_train, y_train)


gbm_grid_pred= gbm_grid.predict(X_test)



print(np.sqrt(mean_squared_error(y_test, gbm_grid_pred)))



# We see almost the same score