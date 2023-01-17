import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

from scipy.stats import norm 

from matplotlib import cm

import seaborn as sns
import os

print(os.listdir("/kaggle/input/forest-cover-type-kernels-only"))
import zipfile

train_zip = zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')

test_zip = zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')



train = pd.read_csv(train_zip.open('train.csv'))

test = pd.read_csv(test_zip.open('test.csv'))



Id = test['Id']
train.head()
test.head()
train.info()
print("The number of traning examples(data points) = %i " % train.shape[0])

print("The number of features we have = %i " % train.shape[1])
print("The number of traning examples(data points) = %i " % test.shape[0])

print("The number of features we have = %i " % test.shape[1])
train.describe()
train.isnull().sum()
f,ax = plt.subplots(figsize=(25, 25))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.show()
train.corr()
#train.drop(['Id'], inplace = True, axis = 1 )

train.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )

test.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
train['HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])

train['Neg_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])

train['HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])

train['Neg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])

train['HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])

train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])



train['Neg_Elevation_Vertical'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']

train['Elevation_Vertical'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']



train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] ) / 3



train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])/2

train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])/2

train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])/2



train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])/2

train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])/2

train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])/2



train['Slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)

train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways']) / 3

train['Mean_Fire_Hyd']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology']) / 2 



train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])



train['Neg_EHyd'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2





test['HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])

test['Neg_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

test['HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

test['Neg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

test['HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])



test['Neg_Elevation_Vertical'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']

test['Elevation_Vertical'] = test['Elevation'] + test['Vertical_Distance_To_Hydrology']



test['mean_hillshade'] = (test['Hillshade_9am']  + test['Hillshade_Noon']  + test['Hillshade_3pm'] ) / 3



test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])/2

test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])/2

test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])/2



test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])/2

test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])/2

test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])/2



test['Slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)

test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways']) / 3 

test['Mean_Fire_Hyd']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology']) / 2





test['Vertical_Distance_To_Hydrology'] = abs(test["Vertical_Distance_To_Hydrology"])



test['Neg_EHyd'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2
train.head()
test.head()
train.shape
test.shape
from sklearn.model_selection import train_test_split

x = train.drop(['Cover_Type'], axis = 1)

y = train['Cover_Type']





x_train, x_val, y_train, y_val = train_test_split( x.values, y.values, test_size=0.2, random_state=42 )

print(x_train.shape)

print(x_val.shape)

print(y_train.shape)

print(y_val.shape)
unique, count= np.unique(y_train, return_counts=True)

print("The number of occurances of each class in the dataset = %s " % dict (zip(unique, count) ), "\n" )
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_val = scaler.transform(x_val)



test = scaler.transform(test)
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
rf_1 = RandomForestClassifier(n_estimators = 200,criterion = 'entropy',random_state = 0)

rf_1.fit(X=x_train, y=y_train)



y_pred_train_rf_1 = rf_1.predict(x_train)

y_pred_val_rf_1 = rf_1.predict(x_val)



y_pred_test_rf_1 = rf_1.predict(test)
rf_2 = RandomForestClassifier(n_estimators = 200,criterion = 'gini',random_state = 0)

rf_2.fit(X=x_train, y=y_train)



y_pred_train_rf_2 = rf_2.predict(x_train)

y_pred_val_rf_2 = rf_2.predict(x_val)



y_pred_test_rf_2 = rf_2.predict(test)
et_1 = ExtraTreesClassifier(n_estimators = 200,criterion = 'entropy',random_state = 0)

et_1.fit(X=x_train, y=y_train)



y_pred_train_et_1 = et_1.predict(x_train)

y_pred_val_et_1 = et_1.predict(x_val)



y_pred_test_et_1 = et_1.predict(test)
et_2 = ExtraTreesClassifier(n_estimators = 200,criterion = 'gini',random_state = 0)

et_2.fit(X=x_train, y=y_train)



y_pred_train_et_2 = et_2.predict(x_train)

y_pred_val_et_2 = et_2.predict(x_val)



y_pred_test_et_2 = et_2.predict(test)
lgb = LGBMClassifier(n_estimators = 200,learning_rate = 0.1)

lgb.fit(X=x_train, y=y_train)



y_pred_train_lgb = lgb.predict(x_train)

y_pred_val_lgb = lgb.predict(x_val)



y_pred_test_lgb = lgb.predict(test)
lr_1 = LogisticRegression(solver = 'liblinear',multi_class = 'ovr',C = 1,random_state = 0)

lr_1.fit(X=x_train, y=y_train)



y_pred_train_lr_1 = lr_1.predict(x_train)

y_pred_val_lr_1 = lr_1.predict(x_val)



y_pred_test_lr_1 = lr_1.predict(test)
xgb_1 = XGBClassifier(seed = 0,colsample_bytree = 0.7, silent = 1, subsample = 0.7, learning_rate = 0.1, objective = 'multi:softprob',

                      num_class = 7,max_depth = 4, min_child_weight = 1, eval_metric = 'mlogloss', nrounds = 200)

xgb_1.fit(X=x_train, y=y_train)



y_pred_train_xgb_1 = xgb_1.predict(x_train)

y_pred_val_xgb_1 = xgb_1.predict(x_val)



y_pred_test_xgb_1 = xgb_1.predict(test)
knn = KNeighborsClassifier(n_neighbors=5)  



knn.fit(x_train, y_train)



y_pred_train_knn = knn.predict(x_train)

y_pred_val_knn = knn.predict(x_val)



y_pred_test_knn = knn.predict(test)
## Creating DF from Predections



stack_train = pd.DataFrame([y_pred_train_rf_1,y_pred_train_rf_2,y_pred_train_et_1,y_pred_train_et_2,y_pred_train_lgb,

                            y_pred_train_lr_1,y_pred_train_xgb_1,y_pred_train_knn])



stack_val = pd.DataFrame([y_pred_val_rf_1,y_pred_val_rf_2,y_pred_val_et_1,y_pred_val_et_2,y_pred_val_lgb,

                            y_pred_val_lr_1,y_pred_val_xgb_1,y_pred_val_knn])



stack_test = pd.DataFrame([y_pred_test_rf_1,y_pred_test_rf_2,y_pred_test_et_1,y_pred_test_et_2,y_pred_test_lgb,

                            y_pred_test_lr_1,y_pred_test_xgb_1,y_pred_test_knn])
print(stack_train.head())

print(stack_val.head())



print(stack_test.head())
## Transpose - it will change row into columns and columns into rows



stack_train = stack_train.T

stack_val = stack_val.T



stack_test = stack_test.T
print(stack_train.head())

print(stack_val.head())

print(stack_test.head())
print(stack_train.shape)

print(stack_val.shape)

print(stack_test.shape)
stack_test.isnull().sum()
lr_2 = LogisticRegression(solver = 'liblinear',multi_class = 'ovr',C = 5,random_state = 0)

lr_2.fit(X=stack_train, y=y_train)



stacked_pred_train = lr_2.predict(stack_train)

stacked_pred_val = lr_2.predict(stack_val)



stacked_pred_test = lr_2.predict(stack_test)
#Id = test['Id']

#test.drop(['Id'], inplace = True, axis = 1 )

#-final_pred = lr_2.predict(test)





submission_1 = pd.DataFrame()

submission_1['Id'] = Id

submission_1['Cover_Type'] = stacked_pred_test

submission_1.to_csv('submission_stack.csv', index=False)

submission_1.head(5)
from catboost import Pool, CatBoostClassifier



cat = CatBoostClassifier()



cat.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(cat.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(cat.score(x_val, y_val) * 100))
cat_predictions = cat.predict(test)
submission_2 = pd.DataFrame()

submission_2['Id'] = Id

submission_2['Cover_Type'] = cat_predictions

submission_2.to_csv('submission.csv', index=False)

submission_2.head(5)
XGB = XGBClassifier()



XGB.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(XGB.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(XGB.score(x_val, y_val) * 100))
XGB_predictions = XGB.predict(test)
submission_3 = pd.DataFrame()

submission_3['Id'] = Id

submission_3['Cover_Type'] = XGB_predictions

submission_3.to_csv('submission_XGB.csv', index=False)

submission_3.head(5)
RFC = RandomForestClassifier()



RFC.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(RFC.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(RFC.score(x_val, y_val) * 100))
RFC_predictions = RFC.predict(test)
submission_4 = pd.DataFrame()

submission_4['Id'] = Id

submission_4['Cover_Type'] = RFC_predictions

submission_4.to_csv('submission_RFC.csv', index=False)

submission_4.head(5)