#importing libraries

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from scipy.stats import norm 

from matplotlib import cm

import seaborn as sns
#reading data

train_data = pd.read_csv('../input/forestdata/train.csv')

test_data=pd.read_csv('../input/forestdata/test.csv')

sample_data = pd.read_csv('../input/forestdata/sample_submission.csv')
#dimension

train_data.shape
train_data.head()
test_data.head()
sample_data.head()
train_data.nunique()
#checking null values

train_data.isnull().sum()
#Forest cover types

sns.countplot(x='Cover_Type',data=train_data)

plt.show()
cor=train_data.corr()

fig, ax = plt.subplots(figsize=(25,25))

sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,fmt= '.3f',annot=True,ax=ax)
train_data.corr()
#dropping columns

Id = test_data['Id']

train_data.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )

test_data.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
train_data['HorizontalHydrology_HorizontalFire'] = (train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Fire_Points'])

train_data['Neg_HorizontalHydrology_HorizontalFire'] = (train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Fire_Points'])

train_data['HorizontalHydrology_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Roadways'])

train_data['Neg_HorizontalHydrology_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Roadways'])

train_data['HorizontalFire_Points_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Fire_Points']+train_data['Horizontal_Distance_To_Roadways'])

train_data['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Fire_Points']-train_data['Horizontal_Distance_To_Roadways'])



train_data['Neg_Elevation_Vertical'] = train_data['Elevation']-train_data['Vertical_Distance_To_Hydrology']

train_data['Elevation_Vertical'] = train_data['Elevation']+train_data['Vertical_Distance_To_Hydrology']



train_data['mean_hillshade'] =  (train_data['Hillshade_9am']  + train_data['Hillshade_Noon'] + train_data['Hillshade_3pm'] ) / 3



train_data['Mean_HorizontalHydrology_HorizontalFire'] = (train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Fire_Points'])/2

train_data['Mean_HorizontalHydrology_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Roadways'])/2

train_data['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Fire_Points']+train_data['Horizontal_Distance_To_Roadways'])/2



train_data['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Fire_Points'])/2

train_data['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Roadways'])/2

train_data['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train_data['Horizontal_Distance_To_Fire_Points']-train_data['Horizontal_Distance_To_Roadways'])/2



train_data['Slope2'] = np.sqrt(train_data['Horizontal_Distance_To_Hydrology']**2+train_data['Vertical_Distance_To_Hydrology']**2)

train_data['Mean_Fire_Hydrology_Roadways']=(train_data['Horizontal_Distance_To_Fire_Points'] + train_data['Horizontal_Distance_To_Hydrology'] + train_data['Horizontal_Distance_To_Roadways']) / 3

train_data['Mean_Fire_Hyd']=(train_data['Horizontal_Distance_To_Fire_Points'] + train_data['Horizontal_Distance_To_Hydrology']) / 2 



train_data["Vertical_Distance_To_Hydrology"] = abs(train_data['Vertical_Distance_To_Hydrology'])



train_data['Neg_EHyd'] = train_data.Elevation-train_data.Horizontal_Distance_To_Hydrology*0.2





test_data['HorizontalHydrology_HorizontalFire'] = (test_data['Horizontal_Distance_To_Hydrology']+test_data['Horizontal_Distance_To_Fire_Points'])

test_data['Neg_HorizontalHydrology_HorizontalFire'] = (test_data['Horizontal_Distance_To_Hydrology']-test_data['Horizontal_Distance_To_Fire_Points'])

test_data['HorizontalHydrology_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Hydrology']+test_data['Horizontal_Distance_To_Roadways'])

test_data['Neg_HorizontalHydrology_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Hydrology']-test_data['Horizontal_Distance_To_Roadways'])

test_data['HorizontalFire_Points_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Fire_Points']+test_data['Horizontal_Distance_To_Roadways'])

test_data['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Fire_Points']-test_data['Horizontal_Distance_To_Roadways'])



test_data['Neg_Elevation_Vertical'] = test_data['Elevation']-test_data['Vertical_Distance_To_Hydrology']

test_data['Elevation_Vertical'] = test_data['Elevation'] + test_data['Vertical_Distance_To_Hydrology']



test_data['mean_hillshade'] = (test_data['Hillshade_9am']  + test_data['Hillshade_Noon']  + test_data['Hillshade_3pm'] ) / 3



test_data['Mean_HorizontalHydrology_HorizontalFire'] = (test_data['Horizontal_Distance_To_Hydrology']+test_data['Horizontal_Distance_To_Fire_Points'])/2

test_data['Mean_HorizontalHydrology_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Hydrology']+test_data['Horizontal_Distance_To_Roadways'])/2

test_data['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Fire_Points']+test_data['Horizontal_Distance_To_Roadways'])/2



test_data['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test_data['Horizontal_Distance_To_Hydrology']-test_data['Horizontal_Distance_To_Fire_Points'])/2

test_data['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Hydrology']-test_data['Horizontal_Distance_To_Roadways'])/2

test_data['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test_data['Horizontal_Distance_To_Fire_Points']-test_data['Horizontal_Distance_To_Roadways'])/2



test_data['Slope2'] = np.sqrt(test_data['Horizontal_Distance_To_Hydrology']**2+test_data['Vertical_Distance_To_Hydrology']**2)

test_data['Mean_Fire_Hydrology_Roadways']=(test_data['Horizontal_Distance_To_Fire_Points'] + test_data['Horizontal_Distance_To_Hydrology'] + test_data['Horizontal_Distance_To_Roadways']) / 3 

test_data['Mean_Fire_Hyd']=(test_data['Horizontal_Distance_To_Fire_Points'] + test_data['Horizontal_Distance_To_Hydrology']) / 2





test_data['Vertical_Distance_To_Hydrology'] = abs(test_data["Vertical_Distance_To_Hydrology"])



test_data['Neg_EHyd'] = test_data.Elevation-test_data.Horizontal_Distance_To_Hydrology*0.2

train_data.head()
test_data.head()
X = train_data.drop('Cover_Type',1)

y = train_data['Cover_Type']
from sklearn.model_selection import train_test_split
#splitting data into train and test

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=369)

print(X_train.shape)

print(X_val.shape)

print(y_train.shape)

print(y_val.shape)
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
xgb = XGBClassifier(n_jobs=-1) 

 

# Use a grid over parameters of interest

param_grid = {

                  'n_estimators' :[100,150,200,250,300],

                  "learning_rate" : [0.001,0.01,0.0001,0.05, 0.10 ],

                  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3 ],

                  "colsample_bytree" : [0.5,0.7],

                  'max_depth': [3,4,6,8]

              }
xgb_randomgrid = RandomizedSearchCV(xgb, param_distributions=param_grid, cv=5)
%%time

xgb_randomgrid.fit(X_train,y_train)
best_est = xgb_randomgrid.best_estimator_
print('Accuracy of classifier on training set: {:.2f}'.format(xgb_randomgrid.score(X_train, y_train) * 100))

print('Accuracy of classifier on validation set: {:.2f}'.format(xgb_randomgrid.score(X_val, y_val) * 100))
prediction_XGB = xgb_randomgrid.predict(test_data)
sample_data["Cover_Type"]=prediction_XGB

sample_data.head()
sample_data.to_csv("submission.csv",index=False)
from sklearn.linear_model import LogisticRegression  

LG= LogisticRegression(random_state=0)  

LG.fit(X_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(LG.score(X_train, y_train) * 100))

print('Accuracy of classifier on validation set: {:.2f}'.format(LG.score(X_val, y_val) * 100))
from sklearn.tree import DecisionTreeClassifier  

DCT= DecisionTreeClassifier(criterion='entropy', random_state=0)  

DCT.fit(X_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(DCT.score(X_train, y_train) * 100))

print('Accuracy of classifier on validation set: {:.2f}'.format(DCT.score(X_val, y_val) * 100))
from sklearn.ensemble import RandomForestClassifier

RDF= RandomForestClassifier(n_estimators= 10, criterion="entropy")  

RDF.fit(X_train, y_train) 
print('Accuracy of classifier on training set: {:.2f}'.format(RDF.score(X_train, y_train) * 100))

print('Accuracy of classifier on validation set: {:.2f}'.format(RDF.score(X_val, y_val) * 100))
prediction_RDF = RDF.predict(test_data)
sample_data["Cover_Type"]=prediction_RDF

sample_data.head()
sample_data.to_csv("submission1.csv",index=False)