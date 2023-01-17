# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/atividade-3-pmr3508"))
print(os.listdir("../input/distance-to-coast"))
print(os.listdir("../input/califdata"))
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/atividade-3-pmr3508/train.csv")
test_data = pd.read_csv("../input/atividade-3-pmr3508/test.csv")
#Saving the id's, just in case they are needed in the submission
train_Id = train_data.loc[:,'Id']
test_Id = test_data.loc[:,'Id']
test_data = test_data.drop('Id',axis = 'columns')
train_data = train_data.drop('Id',axis = 'columns')
train_data


test_Id


def adding_new_features(df):
    df.loc[:,'mean_rooms'] = df.loc[:,'total_rooms']/df.loc[:,'households']
    df.loc[:,'rooms_per_person'] = df.loc[:,'total_rooms']/df.loc[:,'population']
    df.loc[:,'mean_bedrooms'] = df.loc[:,'total_bedrooms']/df.loc[:,'households']
    df.loc[:,'bedrooms_per_person'] = df.loc[:,'total_bedrooms']/df.loc[:,'households']
    df.loc[:,'persons_per_household'] = df.loc[:,'population']/df.loc[:,'households']
    df.loc[:, 'median_income_per_person'] = df.loc[:,'median_income']/df.loc[:,'persons_per_household']
adding_new_features(train_data)
adding_new_features(test_data)
train_data



train_data['longitude'].plot(kind='hist')
test_data['longitude'].plot(kind='hist')

train_data['latitude'].plot(kind='hist')
test_data['latitude'].plot(kind='hist')
dist2coast = pd.read_csv("../input/distance-to-coast/dist2coast.txt",delim_whitespace = True)

dist2coast = pd.DataFrame(dist2coast)
dist2coast = dist2coast.rename({'-179.98': 'longitude', '89.98': 'latitude', '712.935': 'dist2coast'}, axis='columns')
dist2coast = dist2coast.query('-114.00 > longitude > -126.00')
dist2coast = dist2coast.query ('28.00 < latitude < 42.00')
#dist2coast.set_index('longitude', inplace = True)
dist2coast
#for i in range (6192):
    
def saving(name,y_predict): # "saving" some time with a compressed writing code
    df = pd.DataFrame()
    df['Id'] = test_Id
    df.set_index('Id', inplace=True)
    df['median_house_value'] =y_predict
    print(df)
    return df.to_csv(name)
from sklearn import tree
x_train_data = train_data.drop('median_house_value', axis = 'columns')
y_train_data = train_data.loc[:,'median_house_value']
reg1 = tree.DecisionTreeRegressor(max_depth = 1)
reg1 = reg1.fit(x_train_data, y_train_data)
DTR_y = reg1.predict(test_data)
saving("DecisionTreeRegression_1.csv",DTR_y)

from sklearn.model_selection import cross_val_score
Scores1 = []
for i in range (1,10):
    reg1 = tree.DecisionTreeRegressor(max_depth = i)
    scores1 = cross_val_score(reg1, x_train_data, y_train_data, cv=10)
    Scores1.append(scores1.mean())
Scores1_df = pd.DataFrame()
Scores1_df['Regression Tree Scoring']= Scores1
Scores1_df
Scores1_df.plot(title = "R2 score Vs Max Regression Tree Depth")
from sklearn.neighbors import KNeighborsRegressor
reg2 = KNeighborsRegressor(n_neighbors=50) #50nn = 0.37494 at 70% of the test database, 1000nn = 0.38598
reg2 = reg2.fit(x_train_data, y_train_data) 
knnR_y = reg2.predict(test_data)
saving("50nn Regressor.csv", knnR_y)
from sklearn.model_selection import cross_val_score
Scores2 =[]
Scores3= []
for i in range (1,80,4):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn2 = KNeighborsRegressor(n_neighbors = i, weights = 'distance')
    scores2 = cross_val_score(knn, x_train_data, y_train_data, cv=10)
    Scores2.append(scores2.mean())
    scores3 = cross_val_score(knn2, x_train_data, y_train_data, cv=10)
    Scores3.append(scores3.mean())
Scores2_df = pd.DataFrame()
Scores2_df['Knn Scores with Uniform Weight']= Scores2
Scores3_df = pd.DataFrame()
Scores3_df['Knn Scores w/ inversely proportional to dist. Weight']= Scores3
Scores2_df
ax = Scores3_df.plot()
Scores2_df.plot(ax=ax, title = "R2 score Vs N-Nearest Neighbors") #0.37494
from sklearn import linear_model
reg3 = linear_model.LassoLars(alpha=.1, positive = True)
reg3.fit(x_train_data, y_train_data)  
print (reg3.coef_)
LASSO_y = reg3.predict(test_data)
saving("LASSO LARS.csv", LASSO_y) # score = 0.38951

def saving2(p,q,r,Reg,Line_Title,Title):
    Scores =[]
    for i in range (p,q,r):
        regressor = Reg(i)
        scores = cross_val_score(regressor, x_train_data, y_train_data, cv=10)
        Scores.append(scores.mean())
    
    Scores_df = pd.DataFrame()
    Scores_df[Line_Title]= Scores
    Scores_df.plot( title = Title)
from sklearn.neural_network import MLPRegressor
reg4 = MLPRegressor()
reg4.fit(x_train_data, y_train_data)                         
MLP_y = reg4.predict(test_data)
saving("MultiLayerPerceptrons Regressor.csv", MLP_y) #0.373
reg5 = linear_model.BayesianRidge()
reg5.fit(x_train_data, y_train_data)
BRR_y = reg5.predict(test_data)
scores5 = cross_val_score(reg5, x_train_data, y_train_data, cv = 10)
saving("Bayesian Ridge Regressor.csv", BRR_y)
scores5
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
reg6 = RandomForestRegressor()
scores6 = cross_val_score(reg6, x_train_data, y_train_data, cv=10)
scores6 = scores6.mean()
print(scores6)
reg6.fit(x_train_data, y_train_data)
RF_y = reg6.predict(test_data)
saving("RandomForestsRegressor20.csv", RF_y) # scores = 0.23713
print(scores6)
reg7 = ExtraTreesRegressor()
reg7.fit(x_train_data, y_train_data)
ET_y = reg7.predict(test_data)
saving("ExtraTreesRegressor.csv", ET_y) # scores = 0.24383

from sklearn.ensemble import AdaBoostRegressor
reg8 = AdaBoostRegressor(base_estimator = RandomForestRegressor(max_depth=20), n_estimators=50)
reg8.fit(x_train_data, y_train_data)
ADA_y = reg8.predict(test_data)
saving("ADA Boost Regression.csv",ADA_y) # 0.22756
dist2coast_reg = KNeighborsRegressor(n_neighbors=1)
scores_dist = cross_val_score(dist2coast_reg,dist2coast.loc[:,:'latitude'],dist2coast.loc[:,'dist2coast':],cv=10) 
scores_dist = scores_dist.mean()
dist2coast_reg.fit(dist2coast.loc[:,:'latitude'],dist2coast.loc[:,'dist2coast':])
train_pred_dist= dist2coast_reg.predict(train_data.loc[:,:'latitude'])
test_pred_dist= dist2coast_reg.predict(test_data.loc[:,:'latitude'])
train_data['dist2coast'] = train_pred_dist
test_data['dist2coast']= test_pred_dist
print(scores_dist)
x_train_data = train_data.drop('median_house_value', axis = 'columns')
y_train_data = train_data.loc[:,'median_house_value']
train_data
reg_F = AdaBoostRegressor(base_estimator = RandomForestRegressor(max_depth=20), n_estimators=50)
reg_F.fit(x_train_data, y_train_data)
final = reg_F.predict(test_data)
saving("Final ADA Boost Regression.csv",final) # 0.22756