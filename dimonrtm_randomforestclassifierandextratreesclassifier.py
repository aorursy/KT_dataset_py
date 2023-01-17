import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
train_data.info()
test_data.info()
train_data.describe()
train_data.Cover_Type.value_counts()
train_data['HF1'] = train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Fire_Points']
train_data['HF2'] = abs(train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Fire_Points'])
train_data['HR1'] = abs(train_data['Horizontal_Distance_To_Hydrology']+train_data['Horizontal_Distance_To_Roadways'])
train_data['HR2'] = abs(train_data['Horizontal_Distance_To_Hydrology']-train_data['Horizontal_Distance_To_Roadways'])
train_data['FR1'] = abs(train_data['Horizontal_Distance_To_Fire_Points']+train_data['Horizontal_Distance_To_Roadways'])
train_data['FR2'] = abs(train_data['Horizontal_Distance_To_Fire_Points']-train_data['Horizontal_Distance_To_Roadways'])
train_data['ele_vert'] = train_data.Elevation-train_data.Vertical_Distance_To_Hydrology

train_data['slope_hyd'] = (train_data['Horizontal_Distance_To_Hydrology']**2+train_data['Vertical_Distance_To_Hydrology']**2)**0.5
train_data.slope_hyd=train_data.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)
train_data['Mean_Amenities']=(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Hydrology + train_data.Horizontal_Distance_To_Roadways) / 3 
train_data['Mean_Fire_Hyd']=(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Hydrology) / 2 
test_data['HF1'] = test_data['Horizontal_Distance_To_Hydrology']+test_data['Horizontal_Distance_To_Fire_Points']
test_data['HF2'] = abs(test_data['Horizontal_Distance_To_Hydrology']-test_data['Horizontal_Distance_To_Fire_Points'])
test_data['HR1'] = abs(test_data['Horizontal_Distance_To_Hydrology']+test_data['Horizontal_Distance_To_Roadways'])
test_data['HR2'] = abs(test_data['Horizontal_Distance_To_Hydrology']-test_data['Horizontal_Distance_To_Roadways'])
test_data['FR1'] = abs(test_data['Horizontal_Distance_To_Fire_Points']+test_data['Horizontal_Distance_To_Roadways'])
test_data['FR2'] = abs(test_data['Horizontal_Distance_To_Fire_Points']-test_data['Horizontal_Distance_To_Roadways'])
test_data['ele_vert'] = test_data.Elevation-test_data.Vertical_Distance_To_Hydrology

test_data['slope_hyd'] = (test_data['Horizontal_Distance_To_Hydrology']**2+test_data['Vertical_Distance_To_Hydrology']**2)**0.5
test_data.slope_hyd=test_data.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)
test_data['Mean_Amenities']=(test_data.Horizontal_Distance_To_Fire_Points + test_data.Horizontal_Distance_To_Hydrology + test_data.Horizontal_Distance_To_Roadways) / 3 
test_data['Mean_Fire_Hyd']=(test_data.Horizontal_Distance_To_Fire_Points + test_data.Horizontal_Distance_To_Hydrology) / 2
real_data_columns=["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",
          "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Horizontal_Distance_To_Roadways",
                  "HF1","HF2","HR1","HR2","FR1","FR2","ele_vert","slope_hyd","Mean_Amenities","Mean_Fire_Hyd"]
train_data.Soil_Type40.value_counts()
train_data=train_data.drop(['Soil_Type25'],axis=1)
test_data=test_data.drop(['Soil_Type25'],axis=1)
train_data=train_data.drop(['Soil_Type7'],axis=1)
test_data=test_data.drop(['Soil_Type7'],axis=1)
print(train_data.shape)
print(test_data.shape)
test_id=test_data["Id"].values
test_data=test_data.drop(["Id"],axis=1)
train_target=train_data["Cover_Type"].values
train_data=train_data.drop(["Id","Cover_Type"],axis=1)
train_data.head()
test_data.head()
train_data_real=train_data[real_data_columns]
test_data_real=test_data[real_data_columns]
train_data_bynary=train_data.drop(real_data_columns,axis=1)
test_data_bynary=test_data.drop(real_data_columns,axis=1)
mean=train_data_real.mean(axis=0)
std=train_data_real.std(axis=0)
train_data_real-=mean
train_data_real/=std
test_data_real-=mean
test_data_real/=std
X_train=np.hstack((train_data_real,train_data_bynary))
X_test=np.hstack((test_data_real,test_data_bynary))
mapping={1:0,2:1,3:2,4:3,5:4,6:5,7:6}
Y_train=[mapping[y] for y in train_target]
from sklearn.model_selection import train_test_split
training_data,val_data,training_target,val_target=train_test_split(X_train,
                                                                   Y_train,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
%%time
cv=5
clf=RandomForestClassifier(n_estimators=475,criterion="entropy",max_depth=25,random_state=49)
scores=cross_val_score(clf,training_data,training_target,scoring="accuracy",cv=cv)
print(scores.mean())
%%time
cv=5
clf=ExtraTreesClassifier(n_estimators=303,random_state=49)
scores=cross_val_score(clf,training_data,training_target,scoring="accuracy",cv=cv)
print(scores.mean())
%%time
from sklearn.metrics import accuracy_score
clf.fit(training_data,training_target)
prediction=clf.predict(val_data)
print(accuracy_score(val_target,prediction))
%%time
clf=ExtraTreesClassifier(n_estimators=303,random_state=49)
clf.fit(X_train,Y_train)
%%time
prediction=clf.predict(X_test)
mapping={0:1,1:2,2:3,3:4,4:5,5:6,6:7}
prediction=[mapping[y] for y in prediction]
data_submission=pd.DataFrame()
data_submission['Id']=test_id
data_submission["Cover_Type"]=prediction
data_submission.to_csv("my_submission.csv",index=False)













