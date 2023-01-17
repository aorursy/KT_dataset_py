#For numerical analysis

import numpy as np 



#For dataframe and tabular analysis

import pandas as pd



#To visualise all the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)



#Feature encoding

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import FeatureHasher



#Feature scaling

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# to build the models

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

train= pd.read_csv('train.csv')

test= pd.read_csv('test.csv')
# rows and columns of the data

print(train.shape)



#A sample view of data

train.head()
# make a list of the categorical variables that contain missing values

vars_with_na = [var for var in train.columns if train[var].isnull().sum()>1 and train[var].dtypes=='O']



# print the variable name and the percentage of missing values

for var in vars_with_na:

    print(var, 100*np.round(train[var].isnull().mean(), 3),  ' % missing values')
# function to replace NA in categorical variables

def fill_categorical_na(df, var_list):

    X = df.copy()

    X[var_list] = df[var_list].fillna('Missing')

    return X
# replace missing values with new label: "Missing"

train = fill_categorical_na(train, vars_with_na)

test = fill_categorical_na(test, vars_with_na)





# check that we have no missing information in the engineered variables

test[vars_with_na].isnull().sum()
# make a list of the numerical variables that contain missing values

vars_with_na = [var for var in train.columns if train[var].isnull().sum()>1 and train[var].dtypes!='O']



# print the variable name and the percentage of missing values

for var in vars_with_na:

    print(var, 100*np.round(train[var].isnull().mean(), 3),  ' % missing values')
# replace the missing values

for var in vars_with_na:

    

    # calculate the mode

    mode_val = train[var].mode()[0]

    

    # train

    train[var+'_na'] = np.where(train[var].isnull(), 1, 0)

    train[var].fillna(mode_val, inplace=True)

    

    # test

    test[var+'_na'] = np.where(test[var].isnull(), 1, 0)

    test[var].fillna(mode_val, inplace=True)



# check that we have no more missing values in the engineered variables

test[vars_with_na].isnull().sum()
# let's capture the categorical variables first

cat_vars = [var for var in train.columns if train[var].dtype == 'O']
def find_frequent_labels(df, var, rare_perc):

    # finds the labels that are shared by more than a certain % of the houses in the dataset

    df = df.copy()

    tmp = df.groupby(var)['SalePrice'].count() / len(df)

    return tmp[tmp>rare_perc].index



for var in cat_vars:

    frequent_ls = find_frequent_labels(train, var, 0.01)

    train[var] = np.where(train[var].isin(frequent_ls), train[var], 'Rare')

    test[var] = np.where(test[var].isin(frequent_ls), test[var], 'Rare')
#Separate High cardinal vs Low cardinal features



#high_card_vars= [var for var in cat_vars if len(train[var].unique())>= 5]

#low_card_vars= [var for var in cat_vars if len(train[var].unique())< 5]




#Feature Hasher 

def feature_hasher(data, var):

    X= data.copy()

    

    fh = FeatureHasher(n_features=4, input_type='string')

    fh.fit(X[var])

    return fh



#Take care to fit the feature hasher only on the training set, and then transform both test and train set

#to avoid generalisation errors on test set



def feature_transformer(data, var, fh):

    X= data.copy()

    

    hashed_features= fh.transform(X[var])

    hashed_features= hashed_features.toarray()

    hashed_features= pd.DataFrame(hashed_features, columns=[var+'0', var+'1', var+'2', var+'3'])

    return hashed_features



transformed_hash_features_train= pd.DataFrame()

transformed_hash_features_test= pd.DataFrame()



for var in cat_vars:

    fh= feature_hasher(train, var)

    tmp_train= feature_transformer(train, var, fh)

    tmp_test= feature_transformer(test, var, fh)

    transformed_hash_features_train= pd.concat([transformed_hash_features_train, tmp_train], axis= 1)

    transformed_hash_features_test= pd.concat([transformed_hash_features_test, tmp_test], axis= 1)

    

transformed_hash_features_train.shape
#OneHot Encoder for low cardinality features

# def onehot_encoder(data, var):

#     X= data.copy()

    

#     ohe = OneHotEncoder(categories='auto')

#     ohe.fit(X[[var]])

#     return ohe



# #Take care to fit the feature hasher only on the training set, and then transform both test and train set

# #to avoid generalisation errors on test set



# def onehot_transformer(data, var, ohe):

#     X= data.copy()

    

#     ohe_feature_arr = ohe.transform(X[[var]]).toarray()

#     ohe_feature_labels = [var+str(column_label) for column_label in X[var].unique()]

#     ohe_features = pd.DataFrame(ohe_feature_arr, columns=ohe_feature_labels)

#     return ohe_features



# transformed_onehot_features_train= pd.DataFrame()

# transformed_onehot_features_test= pd.DataFrame()



# for var in low_card_vars:

#     ohe_train= onehot_encoder(train, var)

#     ohe_test= onehot_encoder(test, var)

#     tmp_train= onehot_transformer(train, var, ohe_train)

#     tmp_test= onehot_transformer(test, var, ohe_test)

#     transformed_onehot_features_train= pd.concat([transformed_onehot_features_train, tmp_train], axis= 1)

#     transformed_onehot_features_test= pd.concat([transformed_onehot_features_test, tmp_test], axis= 1)

####################################

#drop the categorical features



train_copy= train.copy()

test_copy= test.copy()

for var in cat_vars:

    train_copy.drop(var, axis= 1, inplace= True)

    test_copy.drop(var, axis= 1, inplace= True)





#Final encoded train and test sets without the categorical features ready for feature scaling



train_encoded= pd.concat([train_copy.reset_index(drop=True), transformed_hash_features_train], axis= 1)

test_encoded= pd.concat([test_copy.reset_index(drop=True), transformed_hash_features_test], axis= 1)
train_encoded.shape
test_encoded.shape
train_encoded.head()
test_encoded.head()
X_train_encoded= train_encoded.drop(['SalePrice', 'Id'], axis=1)

y_train_encoded= train_encoded['SalePrice']



X_test_encoded= test_encoded.drop('Id', axis=1)
#Fit scaler to the training set

scaler = StandardScaler() 

scaler.fit(X_train_encoded) 



#Transform the train and test set

X_train_scaled = pd.DataFrame(scaler.transform(X_train_encoded), columns=X_train_encoded.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns)

X_train_scaled= pd.concat([train_encoded.Id, X_train_scaled], axis= 1)

X_test_scaled= pd.concat([test_encoded.Id, X_test_scaled], axis= 1)



#check absence of missing values in train set

X_test_scaled.dropna(inplace=True)
X_train_scaled.shape
X_test_scaled.shape
#List of possible number of components in the reduced dimensions

# n_components= [5, 10, 15, 20, 30]





# for n in n_components:

#     pca= PCA(n_components= n)

#     X= pca.fit_transform(X_train_encoded)

#     print(n, 'features explain ', 100* pca.explained_variance_ratio_.sum(), '% of variance')
# here I will do the model fitting and feature selection

# altogether in one line of code



# first, I specify the Lasso Regression model, and I

# select a suitable alpha (equivalent of penalty).

# The bigger the alpha the less features that will be selected.



# Then I use the selectFromModel object from sklearn, which

# will select the features which coefficients are non-zero



# sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function

# sel_.fit(X_train_encoded, y_train_encoded)
# this command let's us visualise those features that were kept.

# Kept features have a True indicator

# sel_.get_support()
# let's print the number of total and selected features



# this is how we can make a list of the selected features

#selected_feat = X_train_encoded.columns[(sel_.get_support())]



# let's print some stats

# print('total features: {}'.format((X_train_encoded.shape[1])))

# print('selected features: {}'.format(len(selected_feat)))

# print('features with coefficients shrank to zero: {}'.format(

#     np.sum(sel_.estimator_.coef_ == 0)))
#Linear regression



lin= LinearRegression()

cross_val_score(lin, X_train_scaled, y_train_encoded, cv= 3)
# Random Forest

rf= RandomForestRegressor(n_estimators=100)
cross_val_score(rf, X_train_scaled, y_train_encoded, cv= 3)
# Decision Tree



dt= DecisionTreeRegressor()
cross_val_score(dt, X_train_scaled, y_train_encoded, cv= 3)
rf.fit(X_train_scaled, y_train_encoded)
SalePrice= rf.predict(X_test_scaled)
dic= {'Id': X_test_scaled['Id'], 'SalePrice': SalePrice}
df= pd.DataFrame(dic)
df.to_csv('PricePrediction.csv')