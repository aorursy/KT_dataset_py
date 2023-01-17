import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns 
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train['Product_Category_2'].fillna(train['Product_Category_2'].median(), inplace=True)
train['Product_Category_3'].fillna(train['Product_Category_3'].median(), inplace=True)
test['Product_Category_2'].fillna(test['Product_Category_2'].median(), inplace=True)
test['Product_Category_3'].fillna(test['Product_Category_3'].median(), inplace=True)
train.isnull().sum()
test.isnull().sum()
gender_dict = {'F':0, 'M':1}
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
city_dict = {'A':0, 'B':1, 'C':2}
stay_dict = {'0':0, '1':1, '2':2, '3':3, '4+':4}
train["Gender"] = train["Gender"].apply(lambda x: gender_dict[x])
test["Gender"] = test["Gender"].apply(lambda x: gender_dict[x])

train["Age"] = train["Age"].apply(lambda x: age_dict[x])
test["Age"] = test["Age"].apply(lambda x: age_dict[x])

train["City_Category"] = train["City_Category"].apply(lambda x: city_dict[x])
test["City_Category"] = test["City_Category"].apply(lambda x: city_dict[x])

train["Stay_In_Current_City_Years"] = train["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])
test["Stay_In_Current_City_Years"] = test["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])

train.head(3)
train1 = pd.read_csv("../input/train.csv")
test1 = pd.read_csv("../input/test.csv")

targets = train.Purchase

train.drop('Purchase', 1, inplace=True)
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['User_ID', 'Product_ID'], inplace=True, axis=1)
print (combined['Age'].value_counts())
print (combined['Marital_Status'].value_counts())
print (combined['Gender'].value_counts())
print (combined['City_Category'].value_counts())
print (combined['Occupation'].value_counts())
print (combined['Stay_In_Current_City_Years'].value_counts())
combined.describe()
def feature_scaling(dataframe):
    dataframe -= dataframe.min()
    dataframe /= dataframe.max()
    return dataframe
combined['Occupation'] = feature_scaling(combined['Occupation'])
combined['Stay_In_Current_City_Years'] = feature_scaling(combined['Stay_In_Current_City_Years'])
combined['Product_Category_1'] = feature_scaling(combined['Product_Category_1'])
combined['Product_Category_2'] = feature_scaling(combined['Product_Category_3'])
combined['Product_Category_3'] = feature_scaling(combined['Product_Category_3'])
combined['index'] = feature_scaling(combined['index'])
combined['Age'] = feature_scaling(combined['Age'])
combined.tail()
#prediction model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#recovering train test &target
global combined, data_train
targets = train1['Purchase']
train = combined.head(550068)
test = combined.iloc[550068:]
####Prediction model########
#Train-Test split
from sklearn.model_selection import train_test_split
datatrain, datatest, labeltrain, labeltest = train_test_split(train, targets, test_size = 0.2, random_state = 42)
labeltrain.shape
from sklearn import cross_validation, metrics
import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(datatrain,labeltrain)
predictions_xgb = xgb.predict(datatest)
error_xgb = metrics.mean_squared_error(labeltest, predictions_xgb)
print(error_xgb)
#Prediction using test data
y_pred =xgb.predict(datatest)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(labeltest, y_pred))

xgb_score_train = xgb.score(datatrain, labeltrain)
print("Training score: ",xgb_score_train)
xgb_score_test = xgb.score(datatest, labeltest)
print("Testing score: ",xgb_score_test)
#saving output as output.csv of decision tree
output2 = xgb.predict(test).astype(int)
df_output2 = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output2['User_ID'] = aux['User_ID']
df_output2['Product_ID'] = aux['Product_ID']
df_output2['Purchase'] = np.vectorize()(output2)
df_output2[['User_ID','Product_ID','Purchase']].to_csv('output2.csv',index=False)
#random forest
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier()
# rfc.fit(datatrain, labeltrain)
# rfc_score_train = rfc.score(datatrain, labeltrain)
# print("Training score: ",rfc_score_train)
# rfc_score_test = rfc.score(datatest, labeltest)
# print("Testing score: ",rfc_score_test)



# clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
# clf = clf.fit(train, targets)

# #correlation map
# f,ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(combined.corr(), annot=True, linewidths=.5, fmt= '.5f',ax=ax)
# plt.show()
# # Saving id variables to create final submission
# ids_test = test['User_ID'].copy()
# product_ids_test = test['Product_ID'].copy()
# Reducing boundaries to decrease RMSE
# cutoff_purchase = np.percentile(train['Purchase'], 99.9)  # 99.9 percentile
# train.ix[train['Purchase'] > cutoff_purchase, 'Purchase'] = cutoff_purchase
# Label Encoding User_IDs
# le = LabelEncoder()
# train['User_ID'] = le.fit_transform(train['User_ID'])
# test['User_ID'] = le.transform(test['User_ID'])
# Label Encoding Product_IDs
# new_product_ids = list(set(pd.unique(test['Product_ID'])) - set(pd.unique(train['Product_ID'])))
# le = LabelEncoder()
# train['Product_ID'] = le.fit_transform(train['Product_ID'])
# test.ix[test['Product_ID'].isin(new_product_ids), 'Product_ID'] = -1
# new_product_ids.append(-1)
#test.ix[~test['Product_ID'].isin(new_product_ids), 'Product_ID'] = le.transform(test.ix[~test['Product_ID'].isin(new_product_ids), 'Product_ID'])

# y = train['Purchase']
# train.drop(['Purchase', 'Product_Category_2', 'Product_Category_3'], inplace=True, axis=1)
# test.drop(['Product_Category_2', 'Product_Category_3'], inplace=True, axis=1)

# train = pd.get_dummies(train)
# test = pd.get_dummies(test)
# # Modeling
# dtrain = xgb.DMatrix(train.values, label=y, missing=np.nan)

# param = {'objective': 'reg:linear', 'booster': 'gbtree', 'silent': 1,
#          'max_depth': 10, 'eta': 0.1, 'nthread': 4,
#           'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
#          'max_delta_step': 0, 'gamma': 0}
# num_round = 690
# seeds = [1122, 2244, 3366, 4488, 5500]
# test_preds = np.zeros((len(test), len(seeds)))
# for run in range(len(seeds)):
#     #sys.stdout.write("\rXGB RUN:{}/{}".format(run+1, len(seeds)))
#     #sys.stdout.flush()
#     param['seed'] = seeds[run]
#     clf = xgb.train(param, dtrain, num_round)
#     dtest = xgb.DMatrix(test.values, missing=np.nan)
#     test_preds[:, run] = clf.predict(dtest)


# test_preds = np.mean(test_preds, axis=1)
# Submission file
# submit = pd.DataFrame({'User_ID': ids_test, 'Product_ID': product_ids_test, 'Purchase': test_preds})
# submit = submit[['User_ID', 'Product_ID', 'Purchase']]

# submit.ix[submit['Purchase'] < 0, 'Purchase'] = 12  # changing min prediction to min value in train
# submit.to_csv("final_solution.csv", index=False)