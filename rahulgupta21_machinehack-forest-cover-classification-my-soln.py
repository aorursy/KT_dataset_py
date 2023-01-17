import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import numpy as np

import pandas as pd

pd.set_option('display.max_columns',None)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import *

from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold, StratifiedShuffleSplit

import xgboost as xgb

import lightgbm as lgb

import warnings

warnings.filterwarnings(action='ignore')
train = pd.read_csv(r'/kaggle/input/machinehack-forest-cover-classification/train.csv')

test = pd.read_csv(r'/kaggle/input/machinehack-forest-cover-classification/test.csv')

sample = pd.read_csv(r'/kaggle/input/machinehack-forest-cover-classification/sample_submission.csv')
train.head()
test.head()
train.describe()
test.describe()
train.columns
train.rename(columns={'Elevation(meters)':'Elevation' ,'Aspect(degrees)':'Aspect','Slope(degrees)':'Slope','Horizontal_Distance_To_Hydrology(meters)':'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology(meters)':'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways(meters)':'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points(meters)':'Horizontal_Distance_To_Fire_Points'},inplace=True)

test.rename(columns={'Elevation(meters)':'Elevation' ,'Aspect(degrees)':'Aspect','Slope(degrees)':'Slope','Horizontal_Distance_To_Hydrology(meters)':'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology(meters)':'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways(meters)':'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points(meters)':'Horizontal_Distance_To_Fire_Points'},inplace=True)
sns.distplot(train.Elevation)

t1 = train[train.Elevation<2200].index

train.drop(index=t1,inplace=True)

train.reset_index(drop=True,inplace=True)

t2 = train[train.Elevation>=3600].index

train.drop(index=t2,inplace=True)

train.reset_index(drop=True,inplace=True)
sns.distplot(test.Elevation)
sns.boxplot(train.Cover_Type,train.Slope)
train.drop(index=train[train.Slope>56].index,inplace=True)

train.reset_index(drop=True,inplace=True)
sns.boxplot(train.Cover_Type,train.Aspect)
sns.distplot(train.Aspect)
sns.boxplot(train.Cover_Type,train.Horizontal_Distance_To_Hydrology)
sns.boxplot(train.Cover_Type,train.Horizontal_Distance_To_Roadways)
sns.boxplot(train.Cover_Type,train.Horizontal_Distance_To_Fire_Points)
train.Vertical_Distance_To_Hydrology=train.Vertical_Distance_To_Hydrology.apply(lambda x: x if x>0 else 0)

test.Vertical_Distance_To_Hydrology=test.Vertical_Distance_To_Hydrology.apply(lambda x: x if x>0 else 0)
t1 = train[train.Vertical_Distance_To_Hydrology>550].index

train.drop(index=t1,inplace=True)

train.reset_index(drop=True,inplace=True)

sns.boxplot(train.Cover_Type,train.Vertical_Distance_To_Hydrology)
sns.boxplot(train.Cover_Type,train.Hillshade_3pm)
t1=train[train.Hillshade_Noon<180].index

train.drop(index=t1,inplace=True)

train.reset_index(drop=True,inplace=True)
t1=train[train.Hillshade_9am<160].index

train.drop(index=t1,inplace=True)

train.reset_index(drop=True,inplace=True)
train.drop(index=train[train.Hillshade_3pm<50].index,inplace=True)

train.reset_index(drop=True,inplace=True)
sns.distplot(train.drop(index=train[train.Hillshade_9am<160].index).Hillshade_9am)
## 3,7,5,8,9,14,15,,18,21,25,27,28,34,36,37........16,17,26,35,4,3,18...dropped irrelevant attributes
train.drop(columns=['Soil_Type_3','Soil_Type_5','Soil_Type_7','Soil_Type_8','Soil_Type_9','Soil_Type_14','Soil_Type_15','Soil_Type_18','Soil_Type_21','Soil_Type_25','Soil_Type_27','Soil_Type_28','Soil_Type_34','Soil_Type_36','Soil_Type_37','Soil_Type_16','Soil_Type_17','Soil_Type_26','Soil_Type_35','Soil_Type_6','Soil_Type_1','Soil_Type_19'],inplace=True)

test.drop(columns=['Soil_Type_3','Soil_Type_5','Soil_Type_7','Soil_Type_8','Soil_Type_9','Soil_Type_14','Soil_Type_15','Soil_Type_18','Soil_Type_21','Soil_Type_25','Soil_Type_27','Soil_Type_28','Soil_Type_34','Soil_Type_36','Soil_Type_37','Soil_Type_16','Soil_Type_17','Soil_Type_26','Soil_Type_35','Soil_Type_6','Soil_Type_1','Soil_Type_19'],inplace=True)
df = pd.concat([train,test],axis=0)

df.reset_index(drop=True,inplace=True)
df['straight_dist'] = np.sqrt((df.Horizontal_Distance_To_Hydrology**2)+(df.Vertical_Distance_To_Hydrology**2))

df['Sum'] = df.Horizontal_Distance_To_Hydrology + df.Vertical_Distance_To_Hydrology +df.Horizontal_Distance_To_Roadways

df['Diff'] = np.sqrt((df.Elevation**2) - (df.straight_dist**2))
df['dist_road'] = np.sqrt((df.Horizontal_Distance_To_Roadways**2) + (df.Elevation**2))

df['dist_fire'] = np.sqrt((df.Horizontal_Distance_To_Fire_Points**2) + (df.Elevation**2))

df['dist_hydro'] = np.sqrt((df.Horizontal_Distance_To_Hydrology**2) + (df.Elevation**2))

df['dist_vert'] = np.sqrt((df.Vertical_Distance_To_Hydrology**2)+(df.Elevation**2))

df['rem_angle'] = 180-(abs(180-df.Aspect) - df.Slope)

df['road_fire_dist'] = abs(df.dist_road-df.dist_fire)

df['road_fire_hydro'] = abs(df.dist_hydro-df.dist_fire)

df['road_fire_rd_hydro'] = abs(df.dist_road-df.dist_hydro)
df.head()
df['area_eval_mean']=df.groupby(['Wilderness_Area_1','Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4'])[['Elevation']].transform('mean')

df['area_eval_max']=df.groupby(['Wilderness_Area_1','Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4'])[['Elevation']].transform('max')

df['area_eval_min']=df.groupby(['Wilderness_Area_1','Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4'])[['Elevation']].transform('min')

df['area_eval_nunique']=df.groupby(['Wilderness_Area_1','Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4'])[['Elevation']].transform('nunique')
df['elev_soil_mean']=df.groupby(['Soil_Type_2', 'Soil_Type_4',

       'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13',

        'Soil_Type_20',

       'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 

       'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',

       'Soil_Type_33',  'Soil_Type_38', 'Soil_Type_39',

       'Soil_Type_40'])[['Elevation']].transform('mean')

df['eval_soil_std']=df.groupby(['Soil_Type_2', 'Soil_Type_4',

       'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13',

        'Soil_Type_20',

       'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 

       'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',

       'Soil_Type_33',  'Soil_Type_38', 'Soil_Type_39',

       'Soil_Type_40'])[['Elevation']].transform('std')

df['eval_soil_max']=df.groupby(['Soil_Type_2', 'Soil_Type_4',

       'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13',

        'Soil_Type_20',

       'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 

       'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',

       'Soil_Type_33',  'Soil_Type_38', 'Soil_Type_39',

       'Soil_Type_40'])[['Elevation']].transform('max')

df['eval_soil_min']=df.groupby(['Soil_Type_2', 'Soil_Type_4',

       'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13',

        'Soil_Type_20',

       'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 

       'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',

       'Soil_Type_33',  'Soil_Type_38', 'Soil_Type_39',

       'Soil_Type_40'])[['Elevation']].transform('min')

df['eval_soil_nunique']=df.groupby(['Soil_Type_2', 'Soil_Type_4',

       'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13',

       'Soil_Type_20',

       'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 

       'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',

       'Soil_Type_33',  'Soil_Type_38', 'Soil_Type_39',

       'Soil_Type_40'])[['Elevation']].transform('nunique')
train = df.iloc[:train.shape[0],:]

test = df.iloc[train.shape[0]:,:]

train.drop(columns=['Hillshade_3pm','Hillshade_9am','Hillshade_Noon','Wilderness_Area_1','Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4'],inplace=True)

test.drop(columns=['Hillshade_3pm','Hillshade_9am','Hillshade_Noon','Wilderness_Area_1','Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4','Cover_Type'],inplace=True)

test.reset_index(drop=True,inplace=True)
label = train.Cover_Type

train.drop(columns=['Cover_Type'],inplace=True)

#test.drop(columns=['Soil_Type_15'],inplace=True)
def kfold(m,train,label,test,splits,rnd_st):

    score1,score2=[],[]

    p = pd.DataFrame(np.zeros(shape=(test.shape[0],splits*7)),index=test.index)

    pred=pd.DataFrame(np.zeros(shape=(test.shape[0],7)),index=test.index)

    splitter=StratifiedShuffleSplit(n_splits=splits)

    i=0

    l=[]

    for tr_index,te_index in splitter.split(train,label):

        x_train,x_test = train.loc[tr_index,:], train.loc[te_index,:]

        y_train,y_test = label.loc[tr_index],label.loc[te_index]

        

        m.fit(x_train,y_train)

        tr_pred = m.predict_proba(x_train)

        te_pred = m.predict_proba(x_test)

        

        score1.append(log_loss(pd.get_dummies(y_train).values,tr_pred))

        score2.append(log_loss(pd.get_dummies(y_test).values,te_pred))

        

        p.iloc[:,i:i+7] = m.predict_proba(test)

        i=i+7

        print('Training loss: {} \t\t\t\t Validation Loss: {}'.format(log_loss(pd.get_dummies(y_train).values,tr_pred),log_loss(pd.get_dummies(y_test).values,te_pred)))

    

    pd.Series(m.feature_importances_,index = train.columns).sort_values(ascending=True).plot.barh()

    print(np.mean(score1),np.mean(score2))   

    

    pred.iloc[:,0] = (p.iloc[:,0]+p.iloc[:,7]+p.iloc[:,14]+p.iloc[:,21]+p.iloc[:,28])/5

    pred.iloc[:,1] = (p.iloc[:,1]+p.iloc[:,8]+p.iloc[:,15]+p.iloc[:,22]+p.iloc[:,29])/5

    pred.iloc[:,2] = (p.iloc[:,2]+p.iloc[:,9]+p.iloc[:,16]+p.iloc[:,23]+p.iloc[:,30])/5

    pred.iloc[:,3] = (p.iloc[:,3]+p.iloc[:,10]+p.iloc[:,17]+p.iloc[:,24]+p.iloc[:,31])/5

    pred.iloc[:,4] = (p.iloc[:,4]+p.iloc[:,11]+p.iloc[:,18]+p.iloc[:,25]+p.iloc[:,32])/5

    pred.iloc[:,5] = (p.iloc[:,5]+p.iloc[:,12]+p.iloc[:,19]+p.iloc[:,26]+p.iloc[:,33])/5

    pred.iloc[:,6] = (p.iloc[:,6]+p.iloc[:,13]+p.iloc[:,20]+p.iloc[:,27]+p.iloc[:,34])/5

    

    return pred#   return(pd.DataFrame(m.predict_proba(test)))
seed=9291872
plt.figure(figsize=(15,9))

pred1 = kfold(ExtraTreesClassifier(n_estimators=500,random_state=seed),train,label,test,5,seed)
pred1.columns=sample.columns
pred1.to_csv('extratree_sol.csv',index=False)