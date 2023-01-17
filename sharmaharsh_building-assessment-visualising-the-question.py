# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import cross_val_score,train_test_split
from lightgbm import LGBMClassifier
import xgboost as xgb
import os
import csv
print(os.listdir("../input"))
import re
# Any results you write to the current directory are saved as output.
TestFile="../input/test.csv"
TrainFile="../input/train.csv"
Building_structure_File="../input/Building_Structure.csv"
Building_ownership_File="../input/Building_Ownership_Use.csv"
train=pd.read_csv(TrainFile)
ownership=pd.read_csv(Building_ownership_File)
structure=pd.read_csv(Building_structure_File)
test=pd.read_csv(TestFile)
train
test
train.columns
ownership.columns
structure.columns
train.info()
trainy=pd.DataFrame(train[['building_id','damage_grade']])
train.drop('damage_grade',inplace=True,axis=1)
intersect=list(set(train.columns).intersection(set(structure.columns),set(ownership.columns)))
intersect
sizetrain=train.shape[0]
sizetrain
mix=pd.concat([train,test])
mix.info()
mix.has_repair_started[np.isnan(mix.has_repair_started)].shape[0] #number of rows having nan values
mix.has_repair_started.fillna(value=0,inplace=True) # fill nan values with 0
mix.has_repair_started[np.isnan(mix.has_repair_started)].shape[0] 
mix.info()
mix.area_assesed.value_counts()
merged=pd.merge(mix,ownership,on=intersect)
merged=pd.merge(merged,structure,on=intersect)
merged #let's check if all columns have values
merged.info()
for value in merged.columns.values:
    if((merged[value][pd.isnull(merged[value])].shape[0])!=0):
        print(value,merged[value][pd.isnull(merged[value])].shape[0])
        merged[value].fillna(value=merged[value].mode()[0],inplace=True)
for value in merged.columns:
    print(value,merged[value].unique(),merged[value].unique().shape[0],merged.dtypes[value])
encodable_features=[]
for value in merged.columns:
    if(merged[value].unique().shape[0]<=15 and merged[value].unique().shape[0]>2):
        print(value,merged[value].unique().shape[0])
        encodable_features.append(value)
encodable_features
new_merged=pd.get_dummies(merged,columns=encodable_features)
new_merged.head()
train_mod=new_merged[0:sizetrain]
test_mod=new_merged[sizetrain:]
train_mod.head()
## back to original dataset
trainy.damage_grade=trainy.damage_grade.apply(lambda x:re.sub('Grade ','',x)).astype(int) 
# converted damage_grade from form "Grade 1" to 1  i.e to type int also
trainy.info()
train_modx,test_modx,train_mody,test_mody=train_test_split(train_mod,trainy['damage_grade'],test_size=0.33,random_state=0) 
#33% data as test data
from sklearn.tree import DecisionTreeClassifier
Dtree=DecisionTreeClassifier()
Dtree.fit(X=train_modx.select_dtypes(include=['int','float']),y=train_mody)
print(Dtree.score(X=test_modx.select_dtypes(include=['int','float']),y=test_mody))
print(Dtree.tree_.max_depth) 
#tree depth also printed
a=[]
for i in range(1,40):
    Dtree=DecisionTreeClassifier(max_depth=i)
    Dtree.fit(X=train_modx.select_dtypes(include=['int','float']),y=train_mody)
    a.append([i,Dtree.score(X=test_modx.select_dtypes(include=['int','float']),y=test_mody)])
plt.plot([a[i][1] for i in range(len(a))])
a[17] #tree length is 18
np.asarray(a)[:,1].max() ## maximum obtained accuracy 
Dtree.max_depth=18
Dtree.fit(X=train_mod.select_dtypes(include=['int','float']),y=trainy.damage_grade)
predicted=Dtree.predict(X=test_mod.select_dtypes(['int','float']))
with open("predict1_DT.csv","w") as outfile:
    writer=csv.writer(outfile,delimiter=",")
    writer.writerow(("building_id","damage_grade"))
    for i in range(test_mod.shape[0]):
        writer.writerow([test_mod.building_id.values[i],"Grade {}".format(predicted[i])])
os.listdir("../working/")

