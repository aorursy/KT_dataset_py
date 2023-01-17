import numpy as np
import pandas as pd
train_data=pd.read_csv('application_train.csv')
test=pd.read_csv('application_test.csv')
train_data.shape
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
         # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# Missing values statistics
missing_values = missing_values_table(train_data)
missing_values.head(20)
# Number of each type of column
np.where(train_data.dtypes=='object')
# Number of each type of column
train_data.dtypes.value_counts()
train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

train_data.nunique()
# Create a label encoder object
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in train_data:
    if train_data[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train_data[col].unique())) <= 2:
            # Train on the training data
            le.fit(train_data[col])
            # Transform both training and testing data
            train_data[col] = le.transform(train_data[col])
            test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
for i in train_data.columns:
    if train_data[i].dtypes=='object':
        print(train_data[i].value_counts())
#one hot encoding
train_data= pd.get_dummies(train_data)
test= pd.get_dummies(test)
train_data.shape
test.shape
train_target=train_data['TARGET']
train_data, test = train_data.align(test, join = 'inner', axis = 1)
train_data.shape
test.shape
train_data['TARGET']=train_target
train_data.shape
# Extract the EXT_SOURCE variables and show correlations
ext_data = train_data[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs
poly_features= train_data[['TARGET','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

poly_features_test= test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# to impute missing values we gonna use sklearn.imputer method 

from sklearn.preprocessing import Imputer
imputer= Imputer(strategy='median')
#poly_features['TARGET']=poly_target
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])
# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)


# Create the polynomial object with specified degree
#poly_transformer = PolynomialFeatures(degree = 3)
from sklearn.preprocessing import PolynomialFeatures

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)
poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
# Create a dataframe of the features 
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features.columns
# Add in the target
poly_features['TARGET'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))
poly_features.to_pickle("Poly_features.csv")
# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = train_data['SK_ID_CURR']
train_data_poly = train_data.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
test_poly = test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
train_data_poly, test_poly = train_data_poly.align(test_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training data with polynomial features shape: ', train_data_poly.shape)
print('Testing data with polynomial features shape:  ', test_poly.shape)
train_data_poly.isnull().sum()

test_poly.isnull().sum()
train_data.columns
from sklearn.preprocessing import  Imputer

# Drop the target from the training data
if 'TARGET' in train_data:
    train = train_data.drop(columns = ['TARGET'])
else:
    train = train_data.copy()
    
# Feature names
features = list(train.columns)
# Copy of the testing data
test_test = test.copy() # after imputation test set would be converted to numpy array which in turn would be made difficult 
#for visualization.
test_test.head()
train.columns
#ignore_col= ['SK_ID_CURR']
# Median imputation of missing values
imputer = Imputer(strategy = 'median')


#train["AMT_GOODS_PRICE"].fillna(trainX["AMT_GOODS_PRICE"].mean(), inplace = True)
# Fit on the training data
imputer.fit(train)
# Transform both training and testing data
train = imputer.transform(train)

test = imputer.transform(test)
print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y= train_test_split(train,train_target,test_size=0.25)
# Make the model with the specified regularization parameter
Classifier = LogisticRegression(C = 0.0001)

# Train on the training data
Classifier.fit(train_X,train_y)
ypred= Classifier.predict(test_X) # predicting on validation set(test_X) 
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(test_y,ypred)
accuracy
# k fold cross validation
from sklearn.model_selection import cross_val_score
cv_score= cross_val_score(Classifier, train,train_target,cv=10)
cv_score
cv_score.mean()
# after performing 10 fold cross validation cv_score is 91.9 which is similar to the accuracy_score of the logistic regresion 
#classifier
test_pred= Classifier.predict(test)
test_pred.shape
test_pred
# Make predictions
# Make sure to select the second column only
#Classifier_pred = Classifier.predict_proba(test)[:, 1]
#ypred=  Classifier.predict(test)
# Submission dataframe
submit = test_test[['SK_ID_CURR']]
submit['TARGET'] = test_pred

submit.head()

#pred=np.apply_along_axis((lambda X: 1 if X>0.5 else 0),0,Classifier_pred)
submit.to_csv("submit1")
submit.to_csv("Logreg.csv", index = False)
### Logistic regression performs bad on test set. completely a biased model. shouldn't be deployed.
import xgboost as xgb
from sklearn.model_selection import train_test_split
Xtrain, Xval, ytrain, yval = train_test_split(train_data_poly, train_target, test_size = 0.2,
                                             random_state = 1982)
#total rows if we append test set to the training set. just for assumption no use in later code

train_1=pd.concat([train_data_poly, test_poly], axis= 0)
train_1.shape

# to feed data to xgboost first training set is transformed into Dmatrix. in code below train, validation and test sets are transformed.
xgtrain = xgb.DMatrix(Xtrain, label = ytrain)
xgval = xgb.DMatrix(Xval, label = yval) 
xgtest = xgb.DMatrix(test_poly)
watchlist = [(xgtrain,'train'),(xgval, 'eval')]
params = {}
params["objective"] =  "binary:logistic"
params["booster"] = "gbtree"
params["max_depth"] = 7
params["eval_metric"] = 'auc'
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["silent"] = 1
params["seed"] = 4
params["eta"] = 0.1

plst = list(params.items())
num_rounds = 500
model_cv = xgb.train(plst, xgtrain, num_rounds, evals = watchlist, early_stopping_rounds = 10, verbose_eval = True)
testxg_pred = model_cv.predict(xgtest)
test_id=test_test['SK_ID_CURR'] #this code is to run the above code
preds = pd.DataFrame({"SK_ID_CURR": test_id, "TARGET": testxg_pred})
preds.to_csv("xgb_model22aug.csv", index = False)
testxg_pred
#  to determine the acccuracy or any metric we need test set of target variable which we don't have with us. so I'll submit the 
# xgb_model22aug.csv to check how good the model predicts on unseen data.
feat_imp = pd.Series(model_cv.get_fscore()).sort_values(ascending=False)
import matplotlib.pyplot as plt
feat_imp[:25].plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()


import pickle
pickle.dump(model_cv, open("xgb_model13july.pickle.dat", "wb"))
loaded_model = pickle.load(open("xgb_model13july.pickle.dat", "rb"))
