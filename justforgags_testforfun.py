# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def ensemble_average(x_train,y_train,x_test,y_test):
    model1=XGBRegressor(n_estimators=100,random_state=0)
    model2=RandomForestRegressor(max_depth=10, random_state=0)
    model3 = DecisionTreeClassifier(random_state=1)
    model1_1=model1.fit(x_train,y_train)
    
    model2_1=model2.fit(x_train,y_train)
    
   
    model3_1=model3.fit(x_train,y_train)
    
    
    modelall=[model1_1,model2_1,model3_1]
    return modelall
def ensemble_MaxVoting_Tech(x_train,y_train,x_test,y_test):
    model1 = LogisticRegression(random_state=1)
    model2 = DecisionTreeClassifier(random_state=1)
    model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
    model.fit(x_train,y_train)
    #return model.score(x_test,y_test)
    return model
path_train="../input/train.csv"
path_test="../input/test.csv"

data_train=pd.read_csv(path_train)
data_test=pd.read_csv(path_test)
data_test_cpy=data_test.copy()
    
k_fold = KFold(10, True, 2)
feature_lis=[]
feature_lis_no_target=[]
for cols in data_train:
    if data_train[cols].isnull().sum().sum() <1000:
        feature_lis.append(cols)
        if cols != 'Target':
            feature_lis_no_target.append(cols)
data_train=data_train[feature_lis]   
data_test=data_test[feature_lis_no_target]
data_train.isnull().sum().sum()        

data_train=data_train.drop(['meaneduc','SQBmeaned'],axis=1).select_dtypes(exclude=['object'])
data_test=data_test.drop(['meaneduc','SQBmeaned'],axis=1).select_dtypes(exclude=['object'])
        

i=0
for train_index, holdout_index  in k_fold.split(data_train):
    data_train_train_set=data_train.iloc[train_index]  #train on this 
    data_train_test_set=data_train.iloc[holdout_index]
    # print(data_train_train_set.Target)
    data_train_train_set_X=data_train_train_set.drop(['Target'],axis=1).select_dtypes(exclude=['object'])
    data_train_train_set_y=data_train_train_set.Target
    data_train_test_set_X=data_train_test_set.drop(['Target'],axis=1).select_dtypes(exclude=['object'])
    data_train_test_set_y=data_train_test_set.Target
    data_train_train_set_X.fillna(data_train_train_set_X.mean())
    print(data_train_test_set_X.isnull().sum().sum())
    
    models=ensemble_average(data_train_train_set_X,data_train_train_set_y,data_train_test_set_X,data_train_test_set_y)
    #prediction=model.predict(data_test)
    
    predicterr = np.full((data_test.shape[0], 1), 0)
    for model in models:
        pred=model.predict(data_test)
        print(pred)
        print("lk")
        predicterr=predicterr+pred
        #print(predicterr)
    
    predicterr/=3
    
    break
    #break
print(predicterr.shape)    
df_ = pd.DataFrame(columns=['Id','Target'])
predicterr=[int(np.around(im,0)) for im in predicterr[0]]
print(predicterr)
data_test.columns
df_['Id']=data_test_cpy['Id']
df_['Target']=predicterr
df_    


df_.to_csv('submission4.csv', index=False)


