import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv("../input/black-friday-sale-prediction/train.csv")
test=pd.read_csv("../input/black-friday-sale-prediction/test.csv")

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestRegressor,BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import sklearn.metrics as metrics
from sklearn.metrics import r2_score,roc_auc_score,classification_report,mean_squared_error,accuracy_score,confusion_matrix
train.head()
train_cpy=train.copy()
test_cpy=test.copy()
train.shape
train.info()
train.Product_ID.nunique()
train.User_ID.nunique()
train_cat=train.select_dtypes(include='object')
train_cat.drop(['Product_ID'],axis=1,inplace=True)
train_cat.columns
for i in train_cat.columns:
    train[i].value_counts().plot.bar()
    plt.title('{0}'.format(i))
    plt.show()
train_numeric=train.select_dtypes(include=['int64','float64'])
train_numeric.drop(['User_ID'],axis=1,inplace=True)
train_numeric.columns
for i in train_numeric.columns:
    plt.hist(train[i])
    plt.title('{0}'.format(i))
    plt.show()
train_numeric.corr()
# bar plot with default statistic=mean
sns.barplot(x='Gender', y='Purchase', data=train)
plt.show()

# though majorly males shop but the purchase amount is quite close
# bar plot with default statistic=mean
sns.barplot(x='Age', y='Purchase', data=train)
plt.show()

# purchase amount is same for almost all age groups
# bar plot with default statistic=mean
sns.barplot(x='City_Category', y='Purchase', data=train)
plt.show()

# purchase amount is higher for C
# bar plot with default statistic=mean
sns.barplot(x='Stay_In_Current_City_Years', y='Purchase', data=train)
plt.show()

# amount nearly same for all groups
# bar plot with default statistic=mean
sns.barplot(x='Marital_Status', y='Purchase', data=train)
plt.show()

# amount nearly same for all groups
train["Product_Category_1_Count"] = train.groupby(['Product_Category_1'])['Product_Category_1'].transform('count')
pc1_count_dict = train.groupby(['Product_Category_1']).size().to_dict()
test['Product_Category_1_Count'] = test['Product_Category_1'].apply(lambda x:pc1_count_dict.get(x,0))

train["Product_Category_2_Count"] = train.groupby(['Product_Category_2'])['Product_Category_2'].transform('count')
pc2_count_dict = train.groupby(['Product_Category_2']).size().to_dict()
test['Product_Category_2_Count'] = test['Product_Category_2'].apply(lambda x:pc2_count_dict.get(x,0))

train["Product_Category_3_Count"] = train.groupby(['Product_Category_3'])['Product_Category_3'].transform('count')
pc3_count_dict = train.groupby(['Product_Category_3']).size().to_dict()
test['Product_Category_3_Count'] = test['Product_Category_3'].apply(lambda x:pc3_count_dict.get(x,0))

train["User_ID_Count"] = train.groupby(['User_ID'])['User_ID'].transform('count')
userID_count_dict = train.groupby(['User_ID']).size().to_dict()
test['User_ID_Count'] = test['User_ID'].apply(lambda x:userID_count_dict.get(x,0))

train["Product_ID_Count"] = train.groupby(['Product_ID'])['Product_ID'].transform('count')
productID_count_dict = train.groupby(['Product_ID']).size().to_dict()
test['Product_ID_Count'] = test['Product_ID'].apply(lambda x:productID_count_dict.get(x,0))
train["User_ID_MinPrice"] = train.groupby(['User_ID'])['Purchase'].transform('min')
userID_min_dict = train.groupby(['User_ID'])['Purchase'].min().to_dict()
test['User_ID_MinPrice'] = test['User_ID'].apply(lambda x:userID_min_dict.get(x,0))

train["User_ID_MaxPrice"] = train.groupby(['User_ID'])['Purchase'].transform('max')
userID_max_dict = train.groupby(['User_ID'])['Purchase'].max().to_dict()
test['User_ID_MaxPrice'] = test['User_ID'].apply(lambda x:userID_max_dict.get(x,0))

train["User_ID_MeanPrice"] = train.groupby(['User_ID'])['Purchase'].transform('mean')
userID_mean_dict = train.groupby(['User_ID'])['Purchase'].mean().to_dict()
test['User_ID_MeanPrice'] = test['User_ID'].apply(lambda x:userID_mean_dict.get(x,0))


train["Product_ID_MinPrice"] = train.groupby(['Product_ID'])['Purchase'].transform('min')
productID_min_dict = train.groupby(['Product_ID'])['Purchase'].min().to_dict()
test['Product_ID_MinPrice'] = test['Product_ID'].apply(lambda x:productID_min_dict.get(x,0))

train["Product_ID_MaxPrice"] = train.groupby(['Product_ID'])['Purchase'].transform('max')
productID_max_dict = train.groupby(['Product_ID'])['Purchase'].max().to_dict()
test['Product_ID_MaxPrice'] = test['Product_ID'].apply(lambda x:productID_max_dict.get(x,0))

train["Product_ID_MeanPrice"] = train.groupby(['Product_ID'])['Purchase'].transform('mean')
productID_mean_dict = train.groupby(['Product_ID'])['Purchase'].mean().to_dict()
test['Product_ID_MeanPrice'] = test['Product_ID'].apply(lambda x:productID_mean_dict.get(x,0))

userID_25p_dict = train.groupby(['User_ID'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()
train['User_ID_25PercPrice'] = train['User_ID'].apply(lambda x:userID_25p_dict.get(x,0))
test['User_ID_25PercPrice'] = test['User_ID'].apply(lambda x:userID_25p_dict.get(x,0))

userID_75p_dict = train.groupby(['User_ID'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()
train['User_ID_75PercPrice'] = train['User_ID'].apply(lambda x:userID_75p_dict.get(x,0))
test['User_ID_75PercPrice'] = test['User_ID'].apply(lambda x:userID_75p_dict.get(x,0))

productID_25p_dict = train.groupby(['Product_ID'])['Purchase'].apply(lambda x:np.percentile(x,25)).to_dict()
train['Product_ID_25PercPrice'] = train['Product_ID'].apply(lambda x:productID_25p_dict.get(x,0))
test['Product_ID_25PercPrice'] = test['Product_ID'].apply(lambda x:productID_25p_dict.get(x,0))

productID_75p_dict = train.groupby(['Product_ID'])['Purchase'].apply(lambda x:np.percentile(x,75)).to_dict()
train['Product_ID_75PercPrice'] = train['Product_ID'].apply(lambda x:productID_75p_dict.get(x,0))
test['Product_ID_75PercPrice'] = test['Product_ID'].apply(lambda x:productID_75p_dict.get(x,0))

round((train.isnull().sum()/len(train.index))*100,2)
round((test.isnull().sum()/len(test.index))*100,2)
train.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

train['Age']=le.fit_transform(train['Age'])
test['Age']=le.fit_transform(test['Age'])

train['City_Category']=le.fit_transform(train['City_Category'])
test['City_Category']=le.fit_transform(test['City_Category'])

train['Stay_In_Current_City_Years']=le.fit_transform(train['Stay_In_Current_City_Years'])
test['Stay_In_Current_City_Years']=le.fit_transform(test['Stay_In_Current_City_Years'])

pd.set_option('display.max_columns', 100)
train.head(10)
train['Gender']=train['Gender'].map({'M':1, 'F':0})
test['Gender']=test['Gender'].map({'M':1, 'F':0})
train.head()
#filling missing values in product categories 2 & 3 by by any constant number say 0
train['Product_Category_2']=train['Product_Category_2'].fillna(0)
test['Product_Category_2']=test['Product_Category_2'].fillna(0)

train['Product_Category_3']=train['Product_Category_3'].fillna(0)
test['Product_Category_3']=test['Product_Category_3'].fillna(0)

train['Product_Category_2_Count']=train['Product_Category_2_Count'].fillna(0)
test['Product_Category_2_Count']=test['Product_Category_2_Count'].fillna(0)

train['Product_Category_3_Count']=train['Product_Category_3_Count'].fillna(0)
test['Product_Category_3_Count']=test['Product_Category_3_Count'].fillna(0)
round((test.isnull().sum()/len(test.index))*100,2)
train=train.drop(['User_ID','Product_ID'],axis=1)
test=test.drop(['User_ID','Product_ID'],axis=1)
train.head()
q1 = train['Purchase'].quantile(0.25)
q3 = train['Purchase'].quantile(0.75)
iqr = q3-q1 #Interquartile range
fence_low  = q1-1.5*iqr
fence_high = q3+1.5*iqr
train = train[(train['Purchase'] > fence_low) & (train['Purchase'] < fence_high)]

X=train.drop('Purchase',1)
y=train['Purchase']

# # Create the parameter grid based on the results of random search 
# param_grid = {
# 'max_depth': [10], 'max_features': [10], 'min_samples_leaf': [100], 
# 'min_samples_split': [200], 'n_estimators': [200]
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data
# grid_search.fit(X, y)
# printing the optimal accuracy score and hyperparameters
# print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# We can get accuracy of 0.6598673565353703 using {'max_depth': 10, 'max_features': 10, 'min_samples_leaf': 100, 'min_samples_split': 200, 'n_estimators': 200}
# # plotting accuracies with max_depth
# plt.figure()
# plt.plot(scores["param_max_depth"], 
#          scores["mean_train_score"], 
#          label="training accuracy")
# plt.plot(scores["param_max_depth"], 
#          scores["mean_test_score"], 
#          label="test accuracy")
# plt.xlabel("max_depth")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# y_test_pred=grid_search.predict(test)
# finalpred=pd.concat([test_cpy['User_ID'],test_cpy['Product_ID'],pd.DataFrame(y_test_pred,columns=['Purchase'])],1)
# finalpred.to_csv("RF.csv",index=False)
import xgboost as xgb 
from xgboost.sklearn import XGBRegressor
params = {}
params["eta"] = 0.03
params["min_child_weight"] = 10
params["subsample"] = 0.8
params["colsample_bytree"] = 0.7
params["max_depth"] = 10
params["seed"] = 0
plst = list(params.items())
num_rounds = 1100
xgb=XGBRegressor()
xgb.fit(X,y)
y_test_pred_x=xgb.predict(test)
finalpred=pd.concat([test_cpy['User_ID'],test_cpy['Product_ID'],pd.DataFrame(y_test_pred_x,columns=['Purchase'])],1)
finalpred.to_csv("xgb.csv",index=False)
import lightgbm as lgb
lgbm=lgb.LGBMRegressor()
# params={'num_leaves':[200], 'objective':['regression'],'max_depth':[15],'learning_rate':[.1],'max_bin':[200]}
# model = GridSearchCV(lgbm,
#                         params,
#                         cv = 3,
#                         n_jobs = 5,
#                         verbose=True)

# model.fit(X,y)
# y_test_pred_l=model.predict(test)
# finalpred=pd.concat([test_cpy['User_ID'],test_cpy['Product_ID'],pd.DataFrame(y_test_pred_l,columns=['Purchase'])],1)
# finalpred.to_csv("lgbm.csv",index=False)
import catboost as cb
model=cb.CatBoostRegressor()
grid = {'learning_rate': [0.1],
        'depth': [10],
        'l2_leaf_reg': [15]}

model = GridSearchCV(model,
                        grid,
                        cv = 3,
                        n_jobs = 5,
                        verbose=True)
model.fit(X,y)
y_test_predict_c=model.predict(test)
finalpred=pd.concat([test_cpy['User_ID'],test_cpy['Product_ID'],pd.DataFrame(y_test_predict_c,columns=['Purchase'])],1)
finalpred.to_csv("catb_1.csv",index=False)
alist = ['Gender',
'Age',
'Occupation',
'City_Category',
'Stay_In_Current_City_Years',
'Marital_Status',
'Product_Category_1',
'Product_Category_2',
'Product_Category_3',
'User_ID_Count',
'Product_ID_Count']
         
blist = ['User_ID_MinPrice',
'User_ID_MaxPrice',
'User_ID_MeanPrice',
'Product_ID_MinPrice',
'Product_ID_MaxPrice',
'Product_ID_MeanPrice']

clist = ['User_ID_25PercPrice',
'User_ID_75PercPrice',
'Product_ID_25PercPrice',
'Product_ID_75PercPrice',
'Product_Category_1_Count',
'Product_Category_2_Count',
'Product_Category_3_Count',]

#XGB model 1 dataframe
train1 = train[alist+blist]
test1 = test[alist+blist]

#XGB model 2 dataframe 
train2 = train[alist+clist]
test2 = test[alist+clist]
mod_1=lgb.LGBMRegressor(learning_rate=[.2],importance_type='gain')
mod_2=lgb.LGBMRegressor(learning_rate=[.4],importance_type='gain')
X_train,X_test,Y_train,Y_test = train_test_split(train1,y,test_size=0.2,random_state=42)
mod_1.fit(X_train,Y_train)
y_test=mod_1.predict(X_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,y_test)
print('Root Mean Square Value: ',np.sqrt(mse))
# feature_important 

feature_important = pd.DataFrame({'Features':X_train.columns,'Importance':mod_1.feature_importances_})

keys = list(X_train.columns)
values = list(mod_1.feature_importances_)
total = sum(values)
new = [value * 100. / total for value in values]
new = np.round(new,2)

feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new


feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
feature_importances.style.set_properties(**{'font-size':'10pt'})
plt.figure(figsize=(20, 8))
sns.barplot(data=feature_importances, x='Importance (%)', y='Features');
plt.title('Feature importance',fontsize=24)
plt.xlabel('Importance (%)',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Features',fontsize=20)
#prediction1
pred_lgbm_m1 = mod_1.predict(test1)
sub=pd.concat([test_cpy['User_ID'],test_cpy['Product_ID'],pd.DataFrame(pred_lgbm_m1,columns=['Purchase'])],1)
sub.to_csv('lgbm_mod1.csv',index=False)
X_train,X_test,Y_train,Y_test = train_test_split(train2,y,test_size=0.2,random_state=42)
mod_2.fit(X_train,Y_train)
y_test=mod_2.predict(X_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,y_test)
print('Root Mean Square Value: ',np.sqrt(mse))
# feature_important 

feature_important = pd.DataFrame({'Features':X_train.columns,'Importance':mod_2.feature_importances_})

keys = list(X_train.columns)
values = list(mod_2.feature_importances_)
total = sum(values)
new = [value * 100. / total for value in values]
new = np.round(new,2)

feature_importances = pd.DataFrame()
feature_importances['Features'] = keys
feature_importances['Importance (%)'] = new


feature_importances = feature_importances.sort_values(['Importance (%)'],ascending=False).reset_index(drop=True)
feature_importances
feature_importances.style.set_properties(**{'font-size':'10pt'})
plt.figure(figsize=(20, 8))
sns.barplot(data=feature_importances, x='Importance (%)', y='Features');
plt.title('Feature importance',fontsize=24)
plt.xlabel('Importance (%)',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('Features',fontsize=20)
#prediction2
pred_lgbm_m2 = mod_2.predict(test2)
sub=pd.concat([test_cpy['User_ID'],test_cpy['Product_ID'],pd.DataFrame(pred_lgbm_m2,columns=['Purchase'])],1)
sub.to_csv('lgbm_mod2.csv',index=False)
## Weighted average of above two models
sub['Purchase'] = 0.5*pred_lgbm_m1 + 0.5*pred_lgbm_m2
sub.to_csv('final.csv',index=False)
### CATBoost is the final model, as it has performed better than the stacked model