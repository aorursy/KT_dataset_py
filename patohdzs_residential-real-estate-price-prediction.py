

# Importing libraries and modules



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder





# Reading training data into Pandas DataFrame object

data = pd.read_csv('../input/home-data-for-ml-course/train.csv')



# Reading test data file into Pandas DataFrame object

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')

# Examining the first five training data rows

data.head()
# Deleting 'Id' column from training dataset

data = data.drop('Id', axis=1)



# Examining the first five test data rows

test_data.head()


# General overview of training dataset

print('')

data.info()

print('')



# General overview of test dataset

print('')

test_data.info()

print('')



# Separating data for numerical features

num_feats = data.select_dtypes(include='number').drop(['SalePrice'], axis=1).copy()



# Plotting distribution of data for numerical features

fig = plt.figure(figsize=(16,22))

for i in range(len(num_feats.columns)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(num_feats.iloc[:,i].dropna(), hist=False, kde_kws={'bw':0.1})

    plt.xlabel(num_feats.columns[i])

plt.tight_layout()

plt.show()



# Separating data for categorical features

cat_feats = data.select_dtypes(exclude='number').copy()



# Plotting distribution of data for categorical features

fig = plt.figure(figsize=(16,30))



for i in range(len(cat_feats.columns)):

    fig.add_subplot(11,4,i+1)

    sns.countplot(x=cat_feats.iloc[:,i].dropna())

    plt.xlabel(cat_feats.columns[i])

    plt.xticks(rotation=90)

    plt.ylabel('Frequency')

plt.tight_layout()

plt.show()



# Plotting scaterrplots of numerical features versus the target variable 'SalePrice'

fig = plt.figure(figsize=(16,22))

for i in range(len(num_feats.columns)):

    fig.add_subplot(9, 4, i+1)

    sns.scatterplot(num_feats.iloc[:, i],data['SalePrice'])

plt.tight_layout()

plt.show()



# Plotting missing value counts for features in training data

print('')



missing = data.isnull().sum()

missing = missing[missing>0]

missing.sort_values(inplace=True)

missing = missing.to_frame()

missing.reset_index(inplace=True)

missing.columns=['Features', 'Number of missing values']



fig_dims = (8, 5)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x='Features', y='Number of missing values', data=missing)

plt.xticks(rotation=90)

plt.title('Missing values in training data')



print('')

print('')



# Plotting missing value counts for features in test data

missing = test_data.isnull().sum()

missing = missing[missing>0]

missing.sort_values(inplace=True)

missing = missing.to_frame()

missing.reset_index(inplace=True)

missing.columns=['Features', 'Number of missing values']



fig_dims = (10, 5)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x='Features', y='Number of missing values', data=missing)

plt.xticks(rotation=90)

plt.title('Missing values in test data')



# Plotting enlarged scatterplots of numerical features with possible outliers versus 'SalePrice', along with fitted regresssion lines

figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2)

figure.set_size_inches(16,28)

_ = sns.regplot(data['LotFrontage'], data['SalePrice'], ax=ax1)

_ = sns.regplot(data['LotArea'], data['SalePrice'], ax=ax2)

_ = sns.regplot(data['MasVnrArea'], data['SalePrice'], ax=ax3)

_ = sns.regplot(data['BsmtFinSF1'], data['SalePrice'], ax=ax4)

_ = sns.regplot(data['TotalBsmtSF'], data['SalePrice'], ax=ax5)

_ = sns.regplot(data['1stFlrSF'], data['SalePrice'], ax=ax6)

_ = sns.regplot(data['LowQualFinSF'], data['SalePrice'], ax=ax7)

_ = sns.regplot(data['GrLivArea'], data['SalePrice'], ax=ax8)

_ = sns.regplot(data['EnclosedPorch'], data['SalePrice'], ax=ax9)

_ = sns.regplot(data['MiscVal'], data['SalePrice'], ax=ax10)



# Deleting outliers from dataset

data = data.drop(data[data['LotFrontage']>200].index)

data = data.drop(data[data['LotArea']>100000].index)

data = data.drop(data[data['MasVnrArea']>1200].index)

data = data.drop(data[data['BsmtFinSF1']>4000].index)

data = data.drop(data[data['TotalBsmtSF']>4000].index)

data = data.drop(data[data['1stFlrSF']>4000].index)

data = data.drop(data[(data['LowQualFinSF']>550) & (data['SalePrice']>400000)].index)

data = data.drop(data[data['GrLivArea']>4000].index)

data = data.drop(data[data['EnclosedPorch']>500].index)

data = data.drop(data[data['MiscVal']>5000].index)



# Printing dataset summary

print('')

data.info(verbose= False)

print('')



# Checking training and testing datasets missing value percentage

null = pd.DataFrame(data={'%Missing Values (training)': data.isnull().mean()[data.isnull().mean() > 0], '%Missing Values (test)': test_data.isnull().mean()[test_data.isnull().mean() > 0]})

null = round(null* 100, 2)

null = null.sort_values(by= '%Missing Values (training)', ascending=False)



null.index.name='Feature'

null



# Deleting PoolQC, MiscFeature, Alley, Fence, and FireplaceQu from training and test datasets

data = data.drop(columns= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])

test_data = test_data.drop(columns= ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])



# Separating numerical feature names and categorical feature names

num_cont_feats = list(data.select_dtypes(include='float').columns)

num_dis_feats = list(data.select_dtypes(include='int').drop(['SalePrice'], axis=1).columns)

cat_feats = list(data.select_dtypes(exclude='number').columns)



# Filling in missing values

for feat in num_cont_feats:

    data[feat] = data[feat].fillna(data[feat].mean())

    test_data[feat] = test_data[feat].fillna(test_data[feat].mean())



for feat in num_dis_feats:

    data[feat] = data[feat].fillna(round(data[feat].mean()))

    test_data[feat] = test_data[feat].fillna(round(test_data[feat].mean()))





for feat in cat_feats:

    data[feat] = data[feat].fillna(data[feat].mode()[0])

    test_data[feat] = test_data[feat].fillna(test_data[feat].mode()[0])



# Printing dataset summary

print('')

print('Training dataset:')

data.info(verbose=False)

print('')

print('Test dataset:')

test_data.info(verbose=False)

print('')



# Getting categorical feature column names

cat_feats = list(data.select_dtypes(exclude='number').columns)



# Transforming categorical features into dummy features

data = pd.get_dummies(data, columns=cat_feats, prefix= cat_feats)

test_data = pd.get_dummies(test_data, columns=cat_feats, prefix= cat_feats)



# Printing dataset summary

print('')

print('Training dataset:')

data.info(verbose=False)

print('')

print('Test dataset:')

test_data.info(verbose=False)

print('')



# Deleting training dataset features that don't appear in test dataset

test_feats = list(test_data.columns) #.append('SalePrice')

test_feats.append('SalePrice')

test_feats.remove('Id')



data = data[test_feats]



# Printing dataset summary

print('')

print('Training dataset:')

data.info(verbose=False)

print('')

print('Test dataset:')

test_data.info(verbose=False)

print('')

 

def score(y,y_predictions):

    

    msle = mean_squared_log_error(y, y_predictions)

    mae = mean_absolute_error(y, y_predictions)

    print('RMSLE: ', round((msle)**0.5 , 5))

    print('MAE: ', round(mae, 5))



# Splitting training data into temporary training-validation samples

y = data['SalePrice']

X = data.drop('SalePrice',axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=0)



# Fitting first RF model

model_1 = RandomForestRegressor(random_state=0)

model_1.fit(X_train, y_train)



# Listing features in order of importance

feature_importance = pd.DataFrame(data={'Feature': X_train.columns,'Gini Importance': model_1.feature_importances_ })

feature_importance = round(feature_importance, 3).set_index('Feature').sort_values(by='Gini Importance', ascending=False)

feature_importance



# Selecting features with Gini Importance > 0

selected_features = list(feature_importance[feature_importance['Gini Importance']>0].index)



# Printing out names of selected features

print('')

print('Selected Features:')

print('')



for feature in selected_features:

    print(feature)

    

print('')



# Construct reduced feature matrices for training and test samples

X_selected_train = X_train[selected_features]

X_selected_val = X_val[selected_features]



# Fitting second RF model using only selected features 

model_2 = RandomForestRegressor(random_state=0)

model_2.fit(X_selected_train, y_train)



# Predicting target variable for test sample using model_1 and model_2

y_preds_1 = model_1.predict(X_val)

y_preds_2 = model_2.predict(X_selected_val)



# Calculating mean squared error for model_1 and model_2

print('Score for model_1 predictions (trained with all features): ')

score(y_val, y_preds_1)

print('')

print('Score for model_2 predictions (trained using only selected features): ')

score(y_val, y_preds_2)



# Tuning max_depth parameter (original range = [1,100])

min_mae = 35000

depth_star = 0



for depth in range(5,30, 1):    

    model_3 = RandomForestRegressor(random_state=0, max_depth=depth)

    model_3.fit(X_selected_train, y_train)

    y_preds_3 = model_3.predict(X_selected_val)

    mae = round(mean_absolute_error(y_val, y_preds_3), 5)

    

    if mae < min_mae:

        depth_star = depth

        min_mae = mae  



print('')

print('Optimal max_depth = ',depth_star )

print('MAE for model 3 (with optimal max_depth): ', min_mae )

print('')



# Tuning min_samples_split parameter (original range = [2,20])

min_mae = 35000

samples_star = 0



for samples in range(2,6, 1):    

    model_4 = RandomForestRegressor(random_state=0, max_depth=16, min_samples_split=samples)

    model_4.fit(X_selected_train, y_train)

    y_preds_4 = model_4.predict(X_selected_val)

    mae = round(mean_absolute_error(y_val, y_preds_4), 5)

    

    if mae < min_mae:

        samples_star = samples

        min_mae = mae



print('')

print('Optimal min_samples_split = ',samples_star )

print('MAE for model 4 (with optimal max_depth): ', min_mae )

print('')



# Tuning min_samples_leaf parameter (original range = [1,30])

min_mae = 35000

samples_star = 0



for samples in range(1,20, 1):    

    model_5 = RandomForestRegressor(random_state=0, max_depth=16, min_samples_split=2, min_samples_leaf=samples)

    model_5.fit(X_selected_train, y_train)

    y_preds_5 = model_5.predict(X_selected_val)

    mae = round(mean_absolute_error(y_val, y_preds_5), 5)

    

    if mae < min_mae:

        samples_star = samples

        min_mae = mae



print('')

print('Optimal min_samples_leaf = ',samples_star )

print('MAE for model 5 (with optimal min_samples_leaf): ', min_mae )

print('')


# Tuning n_estimators parameter (original range = [1,400])

min_mae = 35000

arg_star = 0



for arg in range(150,201,1):    

    model_6 = RandomForestRegressor(random_state=0, max_depth=16, min_samples_split=2, min_samples_leaf=1, n_estimators=arg)

    model_6.fit(X_selected_train, y_train)

    y_preds_6 = model_6.predict(X_selected_val)

    mae = round(mean_absolute_error(y_val, y_preds_6), 5)

    

    if mae < min_mae:

        arg_star = arg

        min_mae = mae



print('')

print('Optimal n_estimators = ',arg_star )

print('MAE for model 6 (with optimal n_estimators): ', min_mae )

print('')

# Separating target variable vector y

y = data['SalePrice']



# Creating feature matrix X

X = data[selected_features]



# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_final_model = RandomForestRegressor(random_state=0, n_estimators=180, max_depth=16 )



# fit rf_model_on_full_data on all data from the training data

rf_final_model.fit(X,y)


# Creating feature matrix for test data

test_X = test_data[selected_features]



# Predicting SalePrice for test data 

test_preds = rf_final_model.predict(test_X )



# Creating submission file

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})

output.to_csv('submission1.csv', index=False)