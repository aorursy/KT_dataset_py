import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from itertools import product

from sklearn.mixture import GaussianMixture



import warnings

import sys



from sklearn import preprocessing



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb

from catboost import CatBoostClassifier



from mlxtend.classifier import StackingCVClassifier



warnings.simplefilter(action='ignore', category=FutureWarning)

import random

random_state = 42

random.seed(random_state)

np.random.seed(random_state)

os.environ['PYTHONHASHSEED'] = str(random_state)

train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')
train.head()
test.head()
print('Train shape: '+ str(train.shape))

print('Test shape: ' + str(test.shape))
soil_cols = [col  for col in train.columns.values if col.startswith('Soil_Type')]

#Checking if there is a sample with not exactly one soil class

np.sum(np.sum(train[soil_cols].values, axis=1)!=1)
train_soil=train.copy()[soil_cols]

train_soil['Soil_Class']=np.argmax(train_soil.values, axis=1)

train['Soil_Class']=train_soil['Soil_Class']

train.drop(soil_cols, axis=1, inplace=True)

train.head()
test_soil=test.copy()[soil_cols]

test_soil['Soil_Class']=np.argmax(test_soil.values, axis=1)

test['Soil_Class']=test_soil['Soil_Class']

test.drop(soil_cols, axis=1, inplace=True)

test.head()
wild_cols = [col  for col in train.columns.values if col.startswith('Wilderness_Area')]

np.sum(np.sum(train[wild_cols].values,axis=1)!=1)
train_wild=train.copy()[wild_cols]

train_wild['Wilderness_Area']=np.argmax(train_wild.values, axis=1)

train['Wilderness_Area']=train_wild['Wilderness_Area']

train.drop(wild_cols, axis=1, inplace=True)

train.head()
test_wild=test.copy()[wild_cols]

test_wild['Wilderness_Area']=np.argmax(test_wild.values, axis=1)

test['Wilderness_Area']=test_wild['Wilderness_Area']

test.drop(wild_cols, axis=1, inplace=True)

test.head()
train.describe(include='all')
test.describe(include='all')
train.hist(figsize=(16, 16), bins=50, xlabelsize=5, ylabelsize=5);
test.hist(figsize=(16, 16), bins=50, xlabelsize=5, ylabelsize=5);
f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True)



sns.distplot(train['Elevation'], color="r", ax=axes[0])

sns.distplot(test['Elevation'], color="b", ax=axes[1])

plt.scatter(train.Elevation, train.Cover_Type,  s=50)

plt.scatter(train.Aspect, train.Cover_Type, s=50)

plt.figure(figsize=(5,10))

plt.subplot(311)

plt.scatter(train.Hillshade_9am, train.Hillshade_Noon, s=50, c=train.Cover_Type)

plt.xlabel("9am")

plt.ylabel("Noon")



plt.subplot(312)

plt.scatter(train.Hillshade_Noon, train.Hillshade_3pm, s=50, c=train.Cover_Type)

plt.xlabel("Noon")

plt.ylabel("3pm")



plt.subplot(313)

plt.scatter(train.Hillshade_9am, train.Hillshade_3pm, s=50, c=train.Cover_Type)

plt.xlabel("9am")

plt.ylabel("3pm")

plt.figure(figsize=(15,5))

plt.subplot(131)

sns.distplot(train['Hillshade_9am'], color="r")

plt.subplot(132)



sns.distplot(train['Hillshade_Noon'], color="b")

plt.subplot(133)



sns.distplot(train['Hillshade_3pm'], color="g")
f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)



sns.distplot(train['Hillshade_9am'], color="r", ax=axes[0])

sns.distplot(train['Hillshade_Noon'], color="b", ax=axes[1])

sns.distplot(train['Hillshade_3pm'], color="g", ax=axes[2])
plt.scatter(train.Horizontal_Distance_To_Hydrology, 

            train.Vertical_Distance_To_Hydrology, s=50, c=train.Cover_Type)

plt.figure(figsize=(5,10))

plt.subplot(311)

plt.scatter(train.Elevation, train.Slope, s=50, c=train.Cover_Type)

plt.xlabel("Elevation")

plt.ylabel("Slope")

 

plt.subplot(312)

plt.scatter(train.Aspect, train.Slope, s=50, c=train.Cover_Type)

plt.xlabel("Aspect")

plt.ylabel("Slope")



plt.subplot(313)

plt.scatter(train.Aspect, train.Elevation, s=50, c=train.Cover_Type)

plt.xlabel("Aspect")

plt.ylabel("Elevation")

plt.figure(figsize=(5,10))

plt.subplot(311)

plt.scatter(train.Horizontal_Distance_To_Fire_Points, train.Soil_Class, s=50, c=train.Cover_Type)

plt.xlabel("H. distance to fire")

plt.ylabel("Soil Class")

 

plt.subplot(312)

plt.scatter(train.Elevation, train.Soil_Class, s=50, c=train.Cover_Type)

plt.xlabel("Elevation")

plt.ylabel("Soil Class")



plt.subplot(313)

plt.scatter(train.Wilderness_Area, train.Soil_Class, s=50, c=train.Cover_Type)

plt.xlabel("Wilderness Area")

plt.ylabel("Soil Class")



plt.scatter( train.Elevation,train.Cover_Type, s=50)

plt.figure(figsize=(5,10))

plt.subplot(211)

plt.scatter( train.Soil_Class,train.Cover_Type, s=50)

plt.xlabel("Soil Class")

plt.ylabel("Cover Type")

 

plt.subplot(212)

plt.scatter(train.Wilderness_Area, train.Cover_Type, s=50)

plt.xlabel("Wilderness Area")

plt.ylabel("Cover Type")







from sklearn import preprocessing

x = train.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

train_scaled = min_max_scaler.fit_transform(x)

df_train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
x = test.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

test_scaled = min_max_scaler.fit_transform(x)

df_test_scaled = pd.DataFrame(test_scaled, columns=test.columns)
plt.figure(figsize=(15, 15))

sns.boxplot(data=df_train_scaled, orient="h", palette="Set2")

plt.show()
plt.figure(figsize=(15, 15))

sns.boxplot(data=df_test_scaled, orient="h", palette="Set2")

plt.show()
plt.figure(figsize=(8, 5))

sns.boxplot(train.Slope)

plt.show()
plt.scatter(train[train.Slope>40].Slope, train[train.Slope>40].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Horizontal_Distance_To_Hydrology)

plt.show()
plt.scatter(train[train.Horizontal_Distance_To_Hydrology>600].Slope, train[train.Horizontal_Distance_To_Hydrology>600].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Horizontal_Distance_To_Roadways)

plt.show()
plt.scatter(train[train.Horizontal_Distance_To_Roadways>4000].Slope, train[train.Horizontal_Distance_To_Roadways>4000].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Horizontal_Distance_To_Fire_Points)

plt.show()
plt.scatter(train[train.Horizontal_Distance_To_Fire_Points>3000].Slope, train[train.Horizontal_Distance_To_Fire_Points>3000].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Hillshade_9am)

plt.show()
plt.scatter(train[train.Hillshade_9am<150].Slope, train[train.Hillshade_9am<150].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Hillshade_Noon)

plt.show()
plt.scatter(train[train.Hillshade_Noon<180].Slope, train[train.Hillshade_Noon<180].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Hillshade_3pm)

plt.show()
plt.scatter(train[train.Hillshade_3pm<50].Slope, train[train.Hillshade_3pm<50].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Vertical_Distance_To_Hydrology)

plt.show()
plt.scatter(train[train.Vertical_Distance_To_Hydrology<0].Slope, train[train.Vertical_Distance_To_Hydrology<0].Cover_Type, s=50)

plt.scatter(train[train.Vertical_Distance_To_Hydrology>150].Slope, train[train.Vertical_Distance_To_Hydrology>150].Cover_Type, s=50)

plt.figure(figsize=(8, 5))

sns.boxplot(train.Elevation)

plt.show()
plt.figure(figsize=(8, 5))

sns.boxplot(test.Elevation)

plt.show()
X=train.drop(['Cover_Type'] , axis =1)

X.shape
test.shape
Y=train['Cover_Type']

Y.head()
X['Distance_To_Hydrology']=np.linalg.norm([X.Horizontal_Distance_To_Hydrology,

                                           X.Vertical_Distance_To_Hydrology], axis=0)
plt.figure(figsize=(15,5))



sns.distplot(X.Distance_To_Hydrology , color="r")
plt.figure(figsize=(10,5))

plt.scatter(X.Distance_To_Hydrology, Y, s=50, c=train.Cover_Type)

plt.xlabel("Distance to Hydrology")

plt.ylabel("Cover Type")
test['Distance_To_Hydrology']=np.linalg.norm([test.Horizontal_Distance_To_Hydrology,

                                              test.Vertical_Distance_To_Hydrology], axis=0)
X['M_Distance']=np.linalg.norm([X.Horizontal_Distance_To_Hydrology,

                                X.Horizontal_Distance_To_Fire_Points,

                               X.Horizontal_Distance_To_Roadways], axis=0)
plt.figure(figsize=(10,5))

plt.scatter(X.M_Distance, Y, s=50, c=train.Cover_Type)

plt.xlabel("M-Distance")

plt.ylabel("Cover Type")
test['M_Distance']=np.linalg.norm([test.Horizontal_Distance_To_Hydrology,

                                test.Horizontal_Distance_To_Fire_Points,

                               test.Horizontal_Distance_To_Roadways], axis=0)
X['N_Distance']=(

    X.Horizontal_Distance_To_Fire_Points+X.Horizontal_Distance_To_Roadways)/(X.Horizontal_Distance_To_Hydrology+0.01)
plt.scatter(X.N_Distance, Y, s=50, c=train.Cover_Type)

plt.xlabel("N-Distance")

plt.ylabel("Cover Type")
test['N_Distance']=(test.Horizontal_Distance_To_Fire_Points+test.Horizontal_Distance_To_Roadways)/(test.Horizontal_Distance_To_Hydrology+0.01)
plt.scatter( X.Elevation,Y, s=30, c=train.Cover_Type)

plt.scatter( X.Elevation**2,Y, s=30, c=train.Cover_Type)

X['Elevation-Squared']=X.Elevation**2

test['Elevation-Squared']=test.Elevation**2
np.sum(X.Elevation<=0)
plt.scatter( np.sqrt(X.Elevation),Y, s=30, c=train.Cover_Type)

X['Elevation-Sqrt']=np.sqrt(X.Elevation)

test['Elevation-Sqrt']=np.sqrt(test.Elevation)
plt.scatter( np.log(X.Elevation),Y, s=30, c=train.Cover_Type)

X['Elevation-Log']=np.log(X.Elevation)

test['Elevation-Log']=np.log(test.Elevation)
plt.scatter( X.Elevation/(np.sin(X.Slope+1)),train.Cover_Type, s=30, c=train.Cover_Type)

X['Elevation-Slope']=X.Elevation/np.sin(X.Slope+0.1)

test['Elevation-Slope']=test.Elevation/np.sin(test.Slope+0.1)
plt.scatter( X.Elevation/np.sin(X.Aspect+1),train.Cover_Type, s=30, c=train.Cover_Type)

X['Elevation-Aspect']=X.Elevation/np.sin(X.Aspect+1)

test['Elevation-Aspect']=test.Elevation/np.sin(test.Aspect+1)
plt.scatter( X.Elevation -X.Distance_To_Hydrology,Y, s=30, c=train.Cover_Type)

X['Elevation-Hydro']=X.Elevation -X.Distance_To_Hydrology

test['Elevation-Hydro']=test.Elevation -test.Distance_To_Hydrology
plt.scatter( X.Elevation -X.Vertical_Distance_To_Hydrology,Y, s=30, c=train.Cover_Type)

X['Elevation_V_Hydro']=X.Elevation -X.Vertical_Distance_To_Hydrology

test['Elevation_V_Hydro']=test.Elevation -test.Vertical_Distance_To_Hydrology
plt.scatter( X.Elevation +X.Vertical_Distance_To_Hydrology,Y, s=30, c=train.Cover_Type)

plt.scatter( X.Elevation -X.Horizontal_Distance_To_Hydrology,Y, s=30, c=train.Cover_Type)

X['Elevation_H_Hydro']=X.Elevation -X.Horizontal_Distance_To_Hydrology

test['Elevation_H_Hydro']=test.Elevation -test.Horizontal_Distance_To_Hydrology
plt.scatter( X.Elevation +X.Horizontal_Distance_To_Hydrology,Y, s=30, c=train.Cover_Type)

X['Elevation_sum_H_Hydro']=X.Elevation +X.Horizontal_Distance_To_Hydrology

test['Elevation_sum_H_Hydro']=test.Elevation + test.Horizontal_Distance_To_Hydrology
plt.scatter( X.Elevation -X.Horizontal_Distance_To_Fire_Points,Y, s=30, c=train.Cover_Type)

X['Elevation_Fire']=X.Elevation -X.Horizontal_Distance_To_Fire_Points

test['Elevation_Fire']=test.Elevation -test.Horizontal_Distance_To_Fire_Points
plt.scatter( X.Elevation + X.Horizontal_Distance_To_Fire_Points,Y, s=30, c=train.Cover_Type)

X['Elevation_sum_Fire']=X.Elevation + X.Horizontal_Distance_To_Fire_Points

test['Elevation_sum_Fire']=test.Elevation + test.Horizontal_Distance_To_Fire_Points
plt.scatter( X.Elevation -X.Horizontal_Distance_To_Roadways,Y, s=30, c=train.Cover_Type)

X['Elevation_Road']=X.Elevation -X.Horizontal_Distance_To_Roadways

test['Elevation_Road']=test.Elevation -test.Horizontal_Distance_To_Roadways
plt.scatter( X.Elevation + X.Horizontal_Distance_To_Roadways,Y, s=30, c=train.Cover_Type)

X['Elevation_sum_Road']=X.Elevation + X.Horizontal_Distance_To_Roadways

test['Elevation_sum_Road']=test.Elevation + test.Horizontal_Distance_To_Roadways
plt.scatter( X.Elevation/(X.Soil_Class+1),train.Cover_Type, s=30, c=train.Cover_Type)

X['Elevation_by_Soil']= X.Elevation/(X.Soil_Class+1)

test['Elevation_by_Soil']= test.Elevation/(test.Soil_Class+1)
plt.scatter( X.Elevation*X.Soil_Class,train.Cover_Type, s=30, c=train.Cover_Type)

X['Elevation_x_Soil']= X.Elevation*X.Soil_Class

test['Elevation_x_Soil']= test.Elevation*test.Soil_Class
plt.scatter( np.sin(X.Slope)/(np.sin(X.Aspect+1)),train.Cover_Type, s=30, c=train.Cover_Type)

plt.scatter( X.Aspect,train.Cover_Type, s=30, c=train.Cover_Type)

plt.scatter( X.Elevation*(X.Soil_Class)**10,train.Cover_Type, s=30, c=train.Cover_Type)

X['ElevationSoil10']= X.Elevation*(X.Soil_Class)**10 /np.max(X.Elevation*(X.Soil_Class)**10)

test['ElevationSoil10']= test.Elevation*(test.Soil_Class)**10 /np.max(X.Elevation*(X.Soil_Class)**10)
plt.scatter( X.Elevation*(X.Wilderness_Area),train.Cover_Type, s=30, c=train.Cover_Type)

X['ElevationWilderness']=X.Elevation*X.Wilderness_Area

test['ElevationWilderness']=test.Elevation*test.Wilderness_Area
plt.scatter( X.Soil_Class/(X.Wilderness_Area+1),train.Cover_Type, s=30, c=train.Cover_Type)

plt.scatter( (X.Wilderness_Area)/(X.Soil_Class+1),train.Cover_Type, s=30, c=train.Cover_Type)

X['Wildernesss_by_Soil']=X.Wilderness_Area/(X.Soil_Class+1)

test['Wildernesss_by_Soil']=test.Wilderness_Area/(test.Soil_Class+1)

plt.scatter( X.Soil_Class**2,train.Cover_Type, s=30, c=train.Cover_Type)

X['Soil_Class2'] = X.Soil_Class**2

test['Soil_Class2'] = test.Soil_Class**2
plt.scatter( np.sqrt(X.Soil_Class),train.Cover_Type, s=30, c=train.Cover_Type)

X['Soil_sqrt'] = np.sqrt(X.Soil_Class)

test['Soil_sqrt']= np.sqrt(test.Soil_Class)
plt.scatter( -X.Hillshade_3pm+X.Hillshade_9am+X.Hillshade_Noon,train.Cover_Type, s=30, c=train.Cover_Type)

X['Shade912_3'] = -X.Hillshade_3pm+X.Hillshade_9am+X.Hillshade_Noon

test['Shade912_3']= -test.Hillshade_3pm+test.Hillshade_9am+test.Hillshade_Noon
X['Shade93_12'] = X.Hillshade_3pm+X.Hillshade_9am-X.Hillshade_Noon

test['Shade93_12']= test.Hillshade_3pm+test.Hillshade_9am-test.Hillshade_Noon
X['Shade123_9'] = X.Hillshade_3pm-X.Hillshade_9am+X.Hillshade_Noon

test['Shade123_9']= test.Hillshade_3pm-test.Hillshade_9am+test.Hillshade_Noon
plt.scatter(np.abs(X.Horizontal_Distance_To_Fire_Points-X.Horizontal_Distance_To_Roadways ),train.Cover_Type, s=30, c=train.Cover_Type)

X['Abs_Fire_Roads'] =np.abs(X.Horizontal_Distance_To_Fire_Points-X.Horizontal_Distance_To_Roadways )

test['Abs_Fire_Roads']= np.abs(test.Horizontal_Distance_To_Fire_Points-test.Horizontal_Distance_To_Roadways )
plt.scatter(np.abs(X.Horizontal_Distance_To_Fire_Points-X.Horizontal_Distance_To_Hydrology ),train.Cover_Type, s=30, c=train.Cover_Type)

X['Abs_Fire_Hydro'] =np.abs(X.Horizontal_Distance_To_Fire_Points-X.Horizontal_Distance_To_Hydrology )

test['Abs_Fire_Hydro']= np.abs(test.Horizontal_Distance_To_Fire_Points-test.Horizontal_Distance_To_Hydrology )
plt.scatter(np.abs(X.Horizontal_Distance_To_Roadways-X.Horizontal_Distance_To_Hydrology ),train.Cover_Type, s=30, c=train.Cover_Type)

X['Abs_Roads_Hydro'] =np.abs(X.Horizontal_Distance_To_Roadways-X.Horizontal_Distance_To_Hydrology )

test['Abs_Roads_Hydro']= np.abs(test.Horizontal_Distance_To_Roadways-test.Horizontal_Distance_To_Hydrology )
gm = GaussianMixture(n_components=8)

gm.fit(X)



X['Gaussian_Mixture'] = gm.predict(X)

test['Gaussian_Mixture'] = gm.predict(test)
X.shape
test.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(X, 

                                                      Y, 

                                                      test_size = 0.20,

                                                      random_state=42)

X_test=test

X_test.shape
rf = RandomForestClassifier(n_estimators=600, 

                            criterion='gini',

                            max_depth=133,

                            max_features='auto',

                            random_state=42)
rf.fit(X_train, Y_train) 

Y_pred = rf.predict(X_valid)

print("Accuracy:",metrics.accuracy_score(Y_valid, Y_pred))
sns.set(font_scale=1.5)



importances = pd.DataFrame({'Features': X_train.columns, 

                             'Importances': rf.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18,10))

sns.barplot(x='Importances', y='Features', data=importances)

plt.xticks(rotation='vertical')

plt.show()
xgb = XGBClassifier(learning_rate=0.1, n_estimators=450, max_depth=25,

                        min_child_weight=3, gamma=0.05, subsample=0.6, colsample_bytree=1.0,

                        objective='multiclass:softmax', nthread=4, scale_pos_weight=1, seed=42)
xgb_model=xgb.fit(X_train,Y_train)
Y_pred = xgb_model.predict(X_valid)



print("Accuracy:",metrics.accuracy_score(Y_valid, Y_pred))
sns.set(font_scale=1.5)



importances = pd.DataFrame({'Features': X_train.columns, 

                             'Importances': xgb_model.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18,15))

sns.barplot(x='Importances', y='Features', data=importances)

plt.xticks(rotation='vertical')

plt.show()
xtc=ExtraTreesClassifier(

           max_depth=350, 

           n_estimators=450, n_jobs=-1,

           oob_score=False, random_state=42, 

           warm_start=True)
xtc.fit(X_train, Y_train) 
Y_pred = xtc.predict(X_valid)



print("Accuracy:",metrics.accuracy_score(Y_valid, Y_pred))
sns.set(font_scale=1.5)



importances = pd.DataFrame({'Features': X_train.columns, 

                             'Importances': xtc.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18,15))

sns.barplot(x='Importances', y='Features', data=importances)

plt.xticks(rotation='vertical')

plt.show()
ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth  = 20,

                                                                        min_samples_split = 2,

                                                                        min_samples_leaf = 1,

                                                                        random_state=42),

                                                                        n_estimators=100,

                                                                        random_state=42)
ada.fit(X_train, Y_train) 
Y_pred = ada.predict(X_valid)



print("Accuracy:",metrics.accuracy_score(Y_valid, Y_pred))
sns.set(font_scale=1.5)



importances = pd.DataFrame({'Features': X_train.columns, 

                             'Importances': ada.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18,15))

sns.barplot(x='Importances', y='Features', data=importances)

plt.xticks(rotation='vertical')

plt.show()
lgb_model=lgb.LGBMClassifier(n_estimators=375,

                        num_leaves=100,

                        verbose=-1,

                        random_state=42,

                        n_jobs=-1)
lgb_model.fit(X_train, Y_train) 
Y_pred = lgb_model.predict(X_valid)



print("Accuracy:",metrics.accuracy_score(Y_valid, Y_pred))
sns.set(font_scale=1.5)



importances = pd.DataFrame({'Features': X_train.columns, 

                             'Importances': lgb_model.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18,15))

sns.barplot(x='Importances', y='Features', data=importances)

plt.xticks(rotation='vertical')

plt.show()
cat= CatBoostClassifier(n_estimators =300, 

                        #loss_function='Logloss',

                        eval_metric='Accuracy',

                        metric_period=200,

                        max_depth = None, 

                        random_state=42)
cat.fit(X_train, Y_train) 
Y_pred = cat.predict(X_valid)



print("Accuracy:",metrics.accuracy_score(Y_valid, Y_pred))
sns.set(font_scale=1.5)



importances = pd.DataFrame({'Features': X_train.columns, 

                             'Importances': cat.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



fig = plt.figure(figsize=(18,15))

sns.barplot(x='Importances', y='Features', data=importances)

plt.xticks(rotation='vertical')

plt.show()
ensemble = [('rf', rf),

            #('xgb', xgb),

            ('ada', ada),

           #('lgbm', lgb_model),

           ('xtc', xtc),

           ('cat', cat)

           ]
#with rf

stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],

                             meta_classifier=rf,

                             cv=4,

                             use_probas=True,

                             use_features_in_secondary=False,

                             verbose=1,

                             random_state=42,

                             n_jobs=-1)
# #with lgb

# stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],

#                              meta_classifier=lgb_model,

#                              cv=5,

#                              use_probas=True,

#                              use_features_in_secondary=False,

#                              verbose=1,

#                              random_state=42,

#                              n_jobs=-1)
stack = stack.fit(X, Y)
prediction_test = stack.predict(X_test.values)
output_dict = {'Id': test.index,

                       'Cover_Type': prediction_test}





output_lr = pd.DataFrame(output_dict, columns = ['Id', 'Cover_Type'])

output_lr.head(10)
output_lr.tail(10)
output_lr.to_csv('submission.csv', index=False)