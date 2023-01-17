import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn import svm,preprocessing
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.feature_selection import SelectFromModel
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
pd.pandas.set_option("display.max_columns",None)
train.shape
test.shape
#test dataset doesnt contain SalePrice column
train.isnull().any().sum()
#digging out the features with nan values and percentage of nan values in train and test dataset
train.isnull().any().sum()
features_nan=[features for features in train.columns if train[features].isnull().sum()>0]
features_nan_test=[features for features in test.columns if test[features].isnull().sum()>0]
print(features_nan)
for features in features_nan:
    print(features+" "+str(np.round(train[features].isnull().mean(),4))+"% of nan values")
print("\n\n")
for features in features_nan_test:
    print(features+" "+str(np.round(test[features].isnull().mean(),4))+"% of nan values")
for feature in features_nan:
    data = train.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
#digging out numerical features
features_num=[feature for feature in train.columns if train[feature].dtype!='O']
features_num_test=[feature for feature in test.columns if test[feature].dtype!='O']
print(len(features_num),len(features_num_test))
train[features_num]
#digging out temporal features
#features based on time
features_temp=[feature for feature in train.columns if 'Yr' in feature or 'Year' in feature]
print(features_temp)
for feature in features_temp:
    train.groupby(feature)['SalePrice'].median().plot()
    plt.xlabel(feature)
    plt.ylabel('Median House Price')
    plt.show()
#we are comparing relation between all year features and saleprice
for feature in features_temp:
    if feature !='YrSold':
        data=train.copy()
        data[feature]=data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('Saleprice')
        plt.show()
#numerical feature is of two types, discrete and continuous

#digging discrete features

dis_features=[feature for feature in features_num if len(train[feature].unique())<25 and feature not in features_temp+['Id']]
dis_features_test=[feature for feature in features_num_test if len(test[feature].unique())<25 and feature not in features_temp+['Id']]
print(len(dis_features),len(dis_features_test))
for feature in dis_features:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.show()
#digging out continuous numerical features
con_features=[feature for feature in features_num if feature not in dis_features+features_temp+['Id']]
con_features_test=[feature for feature in features_num_test if feature not in dis_features_test+features_temp+['Id']]
con_features
for feature in con_features:
    data=train.copy()
    plt.scatter(data[feature],data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalesPrice')
    plt.title(feature)
    plt.show()
for feature in con_features:
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
for feature in con_features:
    data=train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()

cat_features=[feature for feature in train.columns if train[feature].dtypes=='O']
cat_features_test=[feature for feature in test.columns if test[feature].dtypes=='O']
len(cat_features)-len(cat_features_test)
for feature in cat_features:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

#missing values of cat_features
features_nan_cat=[feature for feature in train.columns if train[feature].isnull().sum()>0 and train[feature].dtype=='O']

for feature in features_nan_cat:
    print("{}: {}% missing values".format(feature,np.round(train[feature].isnull().mean(),4)))
def replace_cat_feature(train,features_nan):
    data=train.copy()
    data[features_nan]=data[features_nan].fillna("Missing")
    return data
train=replace_cat_feature(train,features_nan_cat)
test=replace_cat_feature(test,features_nan_cat)
test[features_nan_cat]
## Now lets check for numerical variables the contains missing values
numerical_with_nan=[feature for feature in train.columns if train[feature].isnull().sum()>0 and train[feature].dtype!='O']
numerical_with_nan_test=[feature for feature in test.columns if test[feature].isnull().sum()>0 and test[feature].dtype!='O']
## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(train[feature].isnull().mean(),4)))
#filling nan with median of that column as there are ouliers in continues features
for feature in numerical_with_nan:
    med=train[feature].median()
    #train[feature+'nan']=np.where(train[feature].isnull(),1,0)
    train[feature]=train[feature].fillna(med)
for feature in numerical_with_nan_test:
    med=test[feature].median()
    #test[feature+'nan']=np.where(test[feature].isnull(),1,0)
    test[feature]=test[feature].fillna(med)
train[numerical_with_nan].isnull().sum()
## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    train[feature]=train['YrSold']-train[feature]
    test[feature]=test['YrSold']-test[feature]
train[features_temp]
#normalizaing continuous features
for feature in con_features:
    train[feature]=np.log(train[feature]+1)
for feature in con_features_test:
    test[feature]=np.log(test[feature]+1)
train[con_features]
def handle_non_numerical_data(train,cat_features):
    for column in cat_features:
        target_values={}
        def convert_to_int(val):
            return target_values[val]
  
        x=2
        unique_values=train[column].unique()
        for unique in unique_values:
            if unique not in target_values.keys():
                target_values[unique]=x
                x+=1
        train[column]=list(map(convert_to_int,train[column]))
    return train
train=handle_non_numerical_data(train,cat_features)
test=handle_non_numerical_data(test,cat_features_test)
test[cat_features]

for feature in cat_features:
    data=train.copy()
    data[feature].hist(bins=5)
    plt.title(feature)
    plt.show()
# since the features or not normally distributed we will use log normal distribution
for feature in cat_features:
    train[feature]=np.log(train[feature])
    test[feature]=np.log(test[feature])
test[cat_features]
colnames=[features for features in train.columns if features not in ['SalePrice','Id']]
len(colnames)

y_train=train['SalePrice']
X_train=train.drop(['SalePrice','Id'],axis=1)
X_test=test.drop(['Id'],axis=1)
X_train=preprocessing.scale(X_train)
X_test=preprocessing.scale(X_test)
X_train=pd.DataFrame(X_train)
X_train.columns=colnames
X_train
X_test=pd.DataFrame(X_test)
X_test.columns=colnames
features_na=[feature for feature in X_test.columns if X_test[feature].isnull().sum()>0]
features_na
### Apply Feature Selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)
feature_sel_model.get_support()

# this is how we can make a list of the selected features
selected_feat = X_train.columns[feature_sel_model.get_support()]
# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
X_train[selected_feat]
X_test[selected_feat]
X_train,X_eval,y_train,y_eval=train_test_split(X_train,y_train,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy=clf.score(X_eval,y_eval)
print(accuracy)
predictions=clf.predict(X_test)
predictions=np.exp(predictions)
print(list(predictions))
plt.hist(predictions)
plt.show()
ridge_params={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
clf=Ridge()
ridge_tun=GridSearchCV(clf,ridge_params,scoring='neg_mean_squared_error',cv=5)
ridge_tun.fit(X_train,y_train)
print(ridge_tun.best_params_)
print(ridge_tun.best_score_)
clf=Ridge(alpha=100)
clf.fit(X_train,y_train)
accuracy=clf.score(X_eval,y_eval)
print(accuracy)
predictions=clf.predict(X_test)
predictions=np.exp(predictions)
print(list(predictions))
plt.hist(predictions)
plt.show()
svm_params={'C':[1.0,2.0,3.0,5.0,10.0,7.0],
        'epsilon':[0.1,0.2,0.2,0.25,0.15],
        'cache_size':[200,300,250,350,150]
       }
clf=svm.SVR()
svm_tun=RandomizedSearchCV(clf,param_distributions=svm_params,scoring='neg_mean_squared_error',n_jobs=-1,cv=5)
svm_tun.fit(X_train,y_train)
print(svm_tun.best_params_)
print(svm_tun.best_score_)
clf=svm.SVR(epsilon= 0.1, cache_size= 300, C= 5.0)
clf.fit(X_train,y_train)
accuracy=clf.score(X_eval,y_eval)
print(accuracy)
predictions=clf.predict(X_test)
predictions=np.exp(predictions)
print(list(predictions))
plt.hist(predictions)
plt.show()
clf=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_tun=GridSearchCV(clf,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_tun.fit(X_train,y_train)
print(lasso_tun.best_params_)
print(lasso_tun.best_score_)
clf=Lasso(alpha=0.01)
clf.fit(X_train,y_train)
accuracy=clf.score(X_eval,y_eval)
print(accuracy)
predictions=clf.predict(X_test)
predictions=np.exp(predictions)
print(list(predictions))
plt.hist(predictions)
plt.show()
xgb_params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
clf=xgboost.XGBRegressor()
xgb_tun=RandomizedSearchCV(clf,param_distributions=xgb_params,n_iter=5,scoring='neg_mean_squared_error',n_jobs=-1,cv=5,verbose=3)
xgb_tun.fit(X_train,y_train)
print(xgb_tun.best_params_)
print(xgb_tun.best_score_)
clf=xgboost.XGBRegressor(min_child_weight=7, max_depth= 5, learning_rate= 0.1, gamma =0.1, colsample_bytree= 0.5)
clf.fit(X_train,y_train)
accuracy=clf.score(X_eval,y_eval)
print(accuracy)
predictions=clf.predict(X_test)
predictions=np.exp(predictions)
print(list(predictions))
plt.hist(predictions)
plt.show()
final_preds=pd.DataFrame(predictions)
final_preds.index=range(1461,1461+1459)
final_preds.columns=['SalePrice']
final_preds