import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from scipy.stats import norm 

from matplotlib import cm

import seaborn as sns
df_train = pd.read_csv('../input/given-data/train.csv')
df_test = pd.read_csv('../input/given-data/test.csv')
df_sample = pd.read_csv('../input/given-data/sample_submission.csv')
df_train.head()
df_test.head()
df_sample.head()
df_sample.nunique()
df_train.nunique()
df_train.isnull().sum()
f,ax = plt.subplots(figsize=(25, 25))

sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.show()
df_train.corr()
Id = df_test['Id']

df_train.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )

df_test.drop(['Id','Soil_Type15' , "Soil_Type7"], inplace = True, axis = 1 )
df_train['HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Fire_Points'])

df_train['Neg_HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Fire_Points'])

df_train['HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Roadways'])

df_train['Neg_HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Roadways'])

df_train['HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']+df_train['Horizontal_Distance_To_Roadways'])

df_train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']-df_train['Horizontal_Distance_To_Roadways'])



df_train['Neg_Elevation_Vertical'] = df_train['Elevation']-df_train['Vertical_Distance_To_Hydrology']

df_train['Elevation_Vertical'] = df_train['Elevation']+df_train['Vertical_Distance_To_Hydrology']



df_train['mean_hillshade'] =  (df_train['Hillshade_9am']  + df_train['Hillshade_Noon'] + df_train['Hillshade_3pm'] ) / 3



df_train['Mean_HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Fire_Points'])/2

df_train['Mean_HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']+df_train['Horizontal_Distance_To_Roadways'])/2

df_train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']+df_train['Horizontal_Distance_To_Roadways'])/2



df_train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Fire_Points'])/2

df_train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Hydrology']-df_train['Horizontal_Distance_To_Roadways'])/2

df_train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df_train['Horizontal_Distance_To_Fire_Points']-df_train['Horizontal_Distance_To_Roadways'])/2



df_train['Slope2'] = np.sqrt(df_train['Horizontal_Distance_To_Hydrology']**2+df_train['Vertical_Distance_To_Hydrology']**2)

df_train['Mean_Fire_Hydrology_Roadways']=(df_train['Horizontal_Distance_To_Fire_Points'] + df_train['Horizontal_Distance_To_Hydrology'] + df_train['Horizontal_Distance_To_Roadways']) / 3

df_train['Mean_Fire_Hyd']=(df_train['Horizontal_Distance_To_Fire_Points'] + df_train['Horizontal_Distance_To_Hydrology']) / 2 



df_train["Vertical_Distance_To_Hydrology"] = abs(df_train['Vertical_Distance_To_Hydrology'])



df_train['Neg_EHyd'] = df_train.Elevation-df_train.Horizontal_Distance_To_Hydrology*0.2





df_test['HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Fire_Points'])

df_test['Neg_HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Fire_Points'])

df_test['HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Roadways'])

df_test['Neg_HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Roadways'])

df_test['HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']+df_test['Horizontal_Distance_To_Roadways'])

df_test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']-df_test['Horizontal_Distance_To_Roadways'])



df_test['Neg_Elevation_Vertical'] = df_test['Elevation']-df_test['Vertical_Distance_To_Hydrology']

df_test['Elevation_Vertical'] = df_test['Elevation'] + df_test['Vertical_Distance_To_Hydrology']



df_test['mean_hillshade'] = (df_test['Hillshade_9am']  + df_test['Hillshade_Noon']  + df_test['Hillshade_3pm'] ) / 3



df_test['Mean_HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Fire_Points'])/2

df_test['Mean_HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']+df_test['Horizontal_Distance_To_Roadways'])/2

df_test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']+df_test['Horizontal_Distance_To_Roadways'])/2



df_test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Fire_Points'])/2

df_test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Hydrology']-df_test['Horizontal_Distance_To_Roadways'])/2

df_test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df_test['Horizontal_Distance_To_Fire_Points']-df_test['Horizontal_Distance_To_Roadways'])/2



df_test['Slope2'] = np.sqrt(df_test['Horizontal_Distance_To_Hydrology']**2+df_test['Vertical_Distance_To_Hydrology']**2)

df_test['Mean_Fire_Hydrology_Roadways']=(df_test['Horizontal_Distance_To_Fire_Points'] + df_test['Horizontal_Distance_To_Hydrology'] + df_test['Horizontal_Distance_To_Roadways']) / 3 

df_test['Mean_Fire_Hyd']=(df_test['Horizontal_Distance_To_Fire_Points'] + df_test['Horizontal_Distance_To_Hydrology']) / 2





df_test['Vertical_Distance_To_Hydrology'] = abs(df_test["Vertical_Distance_To_Hydrology"])



df_test['Neg_EHyd'] = df_test.Elevation-df_test.Horizontal_Distance_To_Hydrology*0.2
df_train.head()
df_test.head()
from sklearn.model_selection import train_test_split

x = df_train.drop(['Cover_Type'], axis = 1)

y = df_train['Cover_Type']



x_train, x_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.2, random_state=116214 )
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
df_test.head()
df_test.head()

df_test = scaler.transform(df_test)
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(DT.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(DT.score(x_test, y_test) * 100))
prediction_DT = DT.predict(df_test)
submission_DT = pd.DataFrame()

submission_DT['Id'] = Id

submission_DT['Cover_Type'] = prediction_DT

submission_DT.to_csv('submission_DT.csv', index=False)

submission_DT.head(5)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

RF = RandomForestClassifier()
param_grid = {'n_estimators' :[100,150,200,250,300]}
RF_R = RandomizedSearchCV(estimator = RF, param_distributions = param_grid, n_jobs = -1, cv = 10)
RF_R.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(RF_R.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(RF_R.score(x_test, y_test) * 100))
prediction_RF = RF_R.predict(df_test)
submission_RF = pd.DataFrame()

submission_RF['Id'] = Id

submission_RF['Cover_Type'] = prediction_RF

submission_RF.to_csv('submission_RF.csv', index=False)

submission_RF.head(5)
XGB = XGBClassifier()
XGB.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(XGB.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(XGB.score(x_test, y_test) * 100))
prediction_XGB = XGB.predict(df_test)
submission_XGB = pd.DataFrame()

submission_XGB['Id'] = Id

submission_XGB['Cover_Type'] = prediction_XGB

submission_XGB.to_csv('submission_XGB.csv', index=False)

submission_XGB.head(5)
from catboost import Pool, CatBoostClassifier
CBR = CatBoostClassifier(iterations=100)
CBR.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(CBR.score(x_train, y_train) * 100))

print('Accuracy of classifier on test set: {:.2f}'.format(CBR.score(x_test, y_test) * 100))
prediction_CBR = CBR.predict(df_test)
submission_CBR = pd.DataFrame()

submission_CBR['Id'] = Id

submission_CBR['Cover_Type'] = prediction_CBR

submission_CBR.to_csv('submission_CBR.csv', index=False)

submission_CBR.head(5)
submission_RF.to_csv('submission.csv', index=None)