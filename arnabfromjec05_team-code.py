import numpy as np
import pandas as pd
import seaborn as sns 
%matplotlib inline
import os
print(os.listdir("../input"))
train_data = pd.read_csv("../input/train.csv")
train_data.info()
train_data.head(5)
train_data['area_assesed'].unique()
train_data['damage_grade'].unique()
train_data['has_geotechnical_risk'].unique()
len(train_data['vdcmun_id'].unique())
len(train_data['district_id'].unique())
## filling missing values ##
train_data['has_repair_started'].fillna(0,inplace = True)
train_data.info()
## dropping columns which are not useful ##
def drop_features(data,features):
    data.drop(features,inplace = True, axis = 1)
def convert_categorical_to_numerical(data,feature):
    return pd.get_dummies(data[feature], drop_first = True, prefix = feature)
new_col = convert_categorical_to_numerical(train_data,'area_assesed')
train_data = pd.concat([train_data,new_col],axis=1)
drop_features(train_data,'area_assesed')
## deep analysis ##
building_structure = pd.read_csv('../input/Building_Structure.csv')
building_structure.info()
building_structure.dropna(inplace = True)
categorical_values = [col for col in building_structure.columns if building_structure[col].dtype == 'O']
print(categorical_values)
for col in categorical_values[1:]:
    new_col = convert_categorical_to_numerical(building_structure,col)
    building_structure = pd.concat([building_structure,new_col],axis=1)
common_cols = sorted(list(set(train_data.columns).intersection(set(building_structure.columns))))
common_cols
common_cols.pop(0)
common_cols
drop_features(building_structure,categorical_values[1:] + common_cols)
## merging with train data ##
[col for col in building_structure.columns if building_structure[col].dtype == 'O']
train_data = pd.merge(train_data,building_structure,how="left",on=['building_id'])
[col for col in train_data.columns if train_data[col].dtype == 'O']
train_data.info()
## deep analysis on building ownership ##
building_ownership = pd.read_csv('../input/Building_Ownership_Use.csv')
building_ownership.info()
building_ownership['legal_ownership_status'].unique()
new_col = convert_categorical_to_numerical(building_ownership,'legal_ownership_status')
building_ownership = pd.concat([building_ownership,new_col],axis=1)
common_cols = sorted(list(set(train_data.columns).intersection(set(building_ownership.columns))))
common_cols
common_cols.pop(0)
common_cols
drop_features(building_ownership,['legal_ownership_status'] + common_cols)
building_ownership.info()
## merging building_ownership with train data ##
train_data = pd.merge(train_data,building_ownership,how="left",on="building_id")
train_data.info()
## we have one null value in count_families ##
train_data['count_families'].fillna(train_data['count_families'].mode()[0],inplace=True)
## fitting the model with the train data ##
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_data.drop(['building_id','damage_grade'],axis = 1),train_data['damage_grade'])
## reading test data ##
test_data = pd.read_csv('../input/test.csv')
test_data.info()
## filling missing values ##
test_data['has_repair_started'].fillna(test_data['has_repair_started'].mode()[0], inplace = True)
area_assesed = convert_categorical_to_numerical(test_data,'area_assesed')
test_data = pd.concat([test_data,area_assesed], axis = 1)
drop_features(test_data,['area_assesed'])
common_cols = list(set(test_data.columns).intersection(set(building_structure.columns)))
common_cols
## merging building structure data with test data ##
test_data = pd.merge(test_data,building_structure,how='left',on='building_id')
test_data.info()
common_cols = list(set(test_data.columns).intersection(set(building_ownership.columns)))
common_cols
## merge building ownership data with test data ##
test_data = pd.merge(test_data,building_ownership,how='left',on='building_id')
test_data.info()
building_id = test_data['building_id']
drop_features(test_data,['building_id'])
test_data
predictions = model.predict(test_data)
print(predictions)
final_result = pd.DataFrame({ 'building_id' : building_id , 'damage_grade' : predictions})
final_result.to_csv('output.csv', index = False)
print(final_result)
print(test_data.info())
test_data1=pd.read_csv('../input/test.csv')
print(test_data1.info())
x= pd.DataFrame()
x['building_id']=test_data1['building_id']
x['district_id']=test_data1['district_id']
#print(x)
x= pd.merge(x,building_structure,how='left',on='building_id')
x= x.drop(x.columns[5:], axis=1)
x['floors_diff']=x['count_floors_pre_eq']-x['count_floors_post_eq']
x['mun_id']=test_data1['vdcmun_id']
x['grade']=final_result['damage_grade']
#print(x)
x
y = x.sort_values(by=['grade','floors_diff'],ascending=False)
y
z = y.sort_values(by=['district_id'],kind='mergesort')
z[:1000]
files,f=[],[]
for i in range(31):
    files.append("file"+str(i+1)+".csv")
print(files)
grp=z.groupby('district_id')
counter=0
for name,group in grp:
    print(name)
    print(group)
    group.to_csv(files[counter], index = False)
    counter+=1
    if counter==31:
        break
        
grading=z.drop(z.columns[1:-1],axis=1)
grading= pd.merge(grading,building_structure,how='left',on='building_id')
grading
grp1=grading.groupby('grade')
counter=0
for name,group in grp1:
    if name=='Grade 5':
        group.to_csv('grade_5.csv', index = False)
    if name=='Grade 1':
        group.to_csv('grade_1.csv', index = False)
    counter+=1
    if counter==5:
        break
test = pd.read_csv('grade_5.csv')
print(test)
