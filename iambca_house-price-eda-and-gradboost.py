# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt



#KNNImputer

from sklearn.impute import KNNImputer



from numpy import loadtxt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import MinMaxScaler
train_data = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
def object_cols(df):

    return list(df.select_dtypes(include='object').columns)



def numerical_cols(df):

    return list(df.select_dtypes(exclude='object').columns)
print("Number of non-numeric data columns = {0} and number of numeric data columns = {1}".format(len(object_cols(train_data)),len(numerical_cols(train_data)) ))
train_data.head()
train_data.describe()
def missing_values(df):

    total_nans_df = pd.DataFrame(df.isnull().sum(), columns=['Values'])

    total_nans_df = total_nans_df.reset_index()

    total_nans_df.columns = ['Columns', 'Values']

    total_nans_df['% Missing Values'] = 100*total_nans_df['Values']/df.shape[0]

    total_nans_df = total_nans_df[total_nans_df['% Missing Values'] > 0 ]

    total_nans_df = total_nans_df.sort_values(by=['% Missing Values'])



    plt.rcdefaults()

    plt.figure(figsize=(10,5))

    ax = sns.barplot(x="Columns", y="% Missing Values", data=total_nans_df)

    ax.set_ylim(0, 100)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.show()
missing_values(train_data) #checking % emptiness of train data columns
missing_values(test_data) #checking % emptiness of test data columns
# function to drop columns with missing that values based on a predefine cutoff criteria

def drop_columns_with_missing_values(df, cutoff):

    """Drop columns with missing values greater than the specified cut-off %

    

    Parameters:

    -----------

    df     : pandas dataframe

    cutoff : % missing values

    

    Returns:

    ---------

    Returns clean dataframe

    """

    # create a dataframe for missing values by column

    total_nans_df = pd.DataFrame(df.isnull().sum(), columns=['values'])

    total_nans_df = total_nans_df.reset_index()

    total_nans_df.columns = ['cols', 'values']

    

    # calculate % missing values

    total_nans_df['% missing values'] = 100*total_nans_df['values']/df.shape[0]

    

    total_nans_df = total_nans_df[total_nans_df['% missing values'] >= cutoff ]

    

    # get columns to drop

    cols = list(total_nans_df['cols'])

    print('Features with missing values greater than specified cutoff : ', cols)

    print('Shape before dropping: ', df.shape)

    new_df = df.drop(labels=cols, axis=1)

    print('Shape after dropping: ',new_df.shape)

    

    return new_df
train_df = drop_columns_with_missing_values(train_data, 80)

test_df = drop_columns_with_missing_values(test_data, 80)
cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



values = {col: 'None' for col in cols}



new_train_df = train_df.fillna(value=values)

new_test_df = test_df.fillna(value=values)
other_cols = [col for col in object_cols(new_train_df) if col not in cols]

print(other_cols)



values = {col:  new_train_df[col].mode()[0] for col in other_cols}



new_train_df = new_train_df.fillna(value=values)



values = {col:  new_test_df[col].mode()[0] for col in other_cols}



new_test_df = new_test_df.fillna(value=values)
imp = KNNImputer(missing_values=np.nan, n_neighbors=7)

imp.fit(new_train_df[numerical_cols(new_train_df)])

new_train_df[numerical_cols(new_train_df)] = imp.transform(new_train_df[numerical_cols(new_train_df)])
imp = KNNImputer(missing_values=np.nan, n_neighbors=7)

imp.fit(new_test_df[numerical_cols(new_test_df)])

new_test_df[numerical_cols(new_test_df)] = imp.transform(new_test_df[numerical_cols(new_test_df)])
print(new_train_df.isnull().any() if new_train_df.isnull().any == True else "There is no any empty value in train data")
print(new_test_df.isnull().any() if new_test_df.isnull().any == True else "There is no any empty value in test data")
corr = new_train_df[numerical_cols(new_train_df)].corr()



fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(corr[['SalePrice']].sort_values(by=['SalePrice'],ascending=False),ax=ax, vmin = -1, cmap='coolwarm',annot=True)
fig = sns.pairplot(data = new_train_df[numerical_cols(new_train_df)], 

                   x_vars = ["OverallQual","GrLivArea", "GarageCars","GarageArea","TotalBsmtSF", "1stFlrSF"], 

                   y_vars = ["OverallQual","GrLivArea", "GarageCars","GarageArea","TotalBsmtSF", "1stFlrSF"]

                   )
#OverallQual

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="OverallQual", y="SalePrice", data=new_train_df)

ax.set_ylim(0, max(new_train_df["SalePrice"]))

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
#GrLivArea

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = plt.scatter(x="GrLivArea", y="SalePrice", data=new_train_df)

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
#GarageCars

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="GarageCars", y="SalePrice", data=new_train_df)

ax.set_ylim(0, max(new_train_df["SalePrice"]))

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
#KitchenAbvGr

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="KitchenAbvGr", y="SalePrice", data=new_train_df)

ax.set_ylim(0, max(new_train_df["SalePrice"]))

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
#EnclosedPorch

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = plt.scatter(x="EnclosedPorch", y="SalePrice", data=new_train_df)

plt.xlabel("EnclosedPorch")

plt.ylabel("SalePrice")

plt.show()
#MSSubClass

plt.rcdefaults()

plt.figure(figsize=(10,5))

ax = sns.barplot(x="MSSubClass", y="SalePrice", data=new_train_df)

ax.set_ylim(0, max(new_train_df["SalePrice"]))

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.show()
correlation = new_train_df[numerical_cols(new_train_df)].corr()



correlation_df = pd.DataFrame(correlation['SalePrice'].sort_values(ascending=False))



correlation_df = correlation_df.reset_index()



correlation_df.columns = ['Column', 'Correlation']



cols_to_drop = list(correlation_df[correlation_df['Correlation'] < 0]['Column'])

cols_to_drop
new_train_df = new_train_df.drop(labels=cols_to_drop, axis=1)

new_test_df = new_test_df.drop(labels=cols_to_drop, axis=1)
new_train_df['train']  = 1 # if this pointer's value is 1 that means the related data is a train data

new_test_df['train']  = 0



df = pd.concat([new_train_df, new_test_df], axis=0,sort=False)
encoding_columns = [

    'ExterQual',

    'ExterCond',

    'GarageCond',

    'GarageQual',

    'FireplaceQu',

    'KitchenQual',

    'CentralAir',

    'HeatingQC',

    'BsmtFinType2',

    'BsmtFinType1',

    'BsmtExposure',

    'BsmtCond',

    'BsmtQual'

]
encoding_values = {

    'Ex': 5,

    'Gd': 4,

    'TA': 3,

    'Fa': 2,

    'Po': 1,

    'None': 0,

    'Av':   3,

    'Mn':   2,

    'No':   1,

    'GLQ':  6,

    'ALQ':  5,

    'BLQ':  4,

    'Rec':  3,

    'LwQ':  2,

    'Unf':  1,

    'N':    0,

    'Y':    1

}
for col in encoding_columns:

    df[col] = df[col].map(encoding_values)
train_df_to_normalize =  df[df['train'] == 1] # getting train data which we labeled by 1

train_df_to_normalize = train_df_to_normalize.drop(labels=['train'], axis=1) # dropping 'train' column which we added





test_df_to_normalize = df[df['train'] == 0] # getting test data which we labeled by 0

test_df_to_normalize = test_df_to_normalize.drop(labels=['SalePrice'], axis=1) # dropping 'SalePrice' column

test_df_to_normalize = test_df_to_normalize.drop(labels=['train'], axis=1) # dropping 'train' column which we added
list_of_norm_cols = numerical_cols(test_df_to_normalize)

print(list_of_norm_cols)
x_train_norm =  train_df_to_normalize[list_of_norm_cols].values #returns a numpy array

min_max_scaler = MinMaxScaler()

x_scaled_train = min_max_scaler.fit_transform(x_train_norm)

train_df_to_normalize[list_of_norm_cols] = x_scaled_train



x_test_norm =  test_df_to_normalize[list_of_norm_cols].values #returns a numpy array

min_max_scaler = MinMaxScaler()

x_scaled_test = min_max_scaler.fit_transform(x_test_norm)

test_df_to_normalize[list_of_norm_cols] = x_scaled_test
test_df_to_normalize
train_df_to_normalize['train']  = 1 # if this pointer's value is 1 that means the related data is a train data

test_df_to_normalize['train']  = 0





df = pd.concat([train_df_to_normalize, test_df_to_normalize], axis=0,sort=False)
one_hot_enc = [col for col in object_cols(new_train_df) if col not in encoding_columns ] # filtering rest of the categorical columns



one_hot_df = pd.get_dummies(df[one_hot_enc], drop_first=True) # one hot encoding



df_final = pd.concat([df, one_hot_df], axis=1, sort=False) # combining one-hot-encoded columns by our dataframe



df_final = df_final.drop(labels=one_hot_enc, axis=1) # droping columns which we used in one hot encoding
train_df_final =  df_final[df_final['train'] == 1] # getting train data which we labeled by 1

train_df_final = train_df_final.drop(labels=['train'], axis=1) # dropping 'train' column which we added





test_df_final = df_final[df_final['train'] == 0] # getting test data which we labeled by 0

test_df_final = test_df_final.drop(labels=['SalePrice'], axis=1) # dropping 'SalePrice' column

test_df_final = test_df_final.drop(labels=['train'], axis=1) # dropping 'train' column which we added
y= train_df_final['SalePrice']

X = train_df_final.drop(labels=['SalePrice'], axis=1)
# split data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)
# kfold = KFold(n_splits=5, shuffle=True, random_state=7)

# params = {

#     'n_estimators': [250,500,1000],

#     'learning_rate': [0.1, 0.2, 0.25, 0.275 ,0.3],

#     'max_depth' : [2,4,8],

#     'max_leaf_nodes' : [8, 16, 24, 32]

    



    

# }



# grad_boost = GradientBoostingRegressor(criterion = 'mse', max_features = 'auto')

# clf = GridSearchCV(grad_boost,param_grid=params, verbose=0, cv=kfold, n_jobs=-1)

# clf.fit(X, y)

# print(clf.best_score_)

# print(clf.best_params_)
model = GradientBoostingRegressor(criterion = 'mse', max_features = 'auto', max_depth = 2, n_estimators = 1000, learning_rate = 0.25, 

                                 max_leaf_nodes = 24)



model.fit(X_train, y_train)

# make predictions for test data and evaluate

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test.values, predictions))

print("RMSE:", (rmse))

# Fit model using each importance as a threshold

thresholds = np.sort(model.feature_importances_)

best_threshold = X_train.shape[1]

best_score = rmse

thresh_list = list()

n_features_list = list()

RMSE_list = list()

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    

    # train model

    selection_model = GradientBoostingRegressor(criterion = 'mse', max_features = 'auto', max_depth = 2, n_estimators = 1000, 

                                                learning_rate = 0.25, 

                                                max_leaf_nodes = 24)

    selection_model.fit(select_X_train, y_train)

    # eval model

    select_X_test = selection.transform(X_test)

    predictions = selection_model.predict(select_X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))

    if score < best_score:

        best_score = score

        best_threshold = select_X_train.shape[1]

    

    

    thresh_list.append(thresh)

    n_features_list.append(select_X_train.shape[1])

    RMSE_list.append(score)

    print("Thresh={}, n={}, RMSE: {}".format(thresh, select_X_train.shape[1], score))

print('Best RMSE: {}, n={}'.format(best_score, best_threshold))
min_rmse = min(RMSE_list) # findig the best RMSE to draw with another color at our graph

best_n_features = n_features_list[RMSE_list.index(min_rmse)] # and related best n_features value



#deleting those elements from list

RMSE_list.remove(min_rmse)

n_features_list.remove(best_n_features)



plt.plot(best_n_features, min_rmse, marker='o', color='g')

plt.plot(n_features_list, RMSE_list, color='y')



plt.xlabel("Number of Fatures")

plt.ylabel("RMSE")

plt.show()
feature_importance = pd.DataFrame(pd.Series(model.feature_importances_, index=X_train.columns, 

                               name='Feature_Importance').sort_values(ascending=False)).reset_index()

selected_features = feature_importance.iloc[0:best_threshold]['index']

selected_features = list(selected_features)
# now use the selected features  and fit the model on X and Y

new_X = X[selected_features]

new_test = test_df_final[selected_features]



model.fit(new_X, y)
# Now lets make predictions on the test dataset for submission

submission_predictions = model.predict(new_test)
# prepare a csv file for submission

sub_df = pd.DataFrame(submission_predictions)

sub_df['Id'] = test_df['Id']

sub_df.columns = ['SalePrice', 'Id']

sub_df = sub_df[['Id', 'SalePrice']]



sub_df.to_csv('submission.csv', index=False)