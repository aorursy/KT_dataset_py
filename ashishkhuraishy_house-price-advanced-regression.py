# This Python 3 environment comes with many helpful analytics libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
train_path = "../input/train.csv"
house_data = pd.read_csv(train_path)
house_data.dropna(axis=1)
house_data.columns
y = house_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = house_data[features]
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import explained_variance_score as score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

models = [DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor]
def check_models(n):
    model = n()
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    Score = score(pred, val_y)
    Mae = mae(pred, val_y)
    
    print(str(n), "Mean Absolute Error : ", Mae, "Score : ", round(Score*100),"%")
for i in models:
    check_models(i)
nodes = [ 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000, 5000]

def leaf_nodes(n, a=10):
    model = RandomForestRegressor(max_leaf_nodes= n, n_estimators = a)
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    Score = score(val_y, pred)
    Mae = mae(val_y, pred)
    
    #print("Nodes :{}   MAE : {} n_est: {} Score: {}%".format(n, Mae, a, round(Score*100)))
    return Mae
def details(n, a=10):
    model = RandomForestRegressor(max_leaf_nodes = n, n_estimators = a)
    model.fit(train_X, train_y)
    pred = model.predict(val_X)
    Score = score(val_y, pred)
    s = round(Score, 3)*100
    Mae = mae(val_y, pred)
    
    print("Nodes :{} MAE :{} N_EST :{} Score :{}%".format(n, Mae, a, s))

def complete():
    print("Process Finished")
for i in nodes:
    leaf_nodes(i)
    details(i)
nodes_final = [ i for i in range(50, 650)]
final_nodes = [i for i in range(0,1000)]
for i in nodes_final:
    final_nodes[i] = leaf_nodes(i)
complete()
Nodes = final_nodes[50: 650]
c = final_nodes.index(min(Nodes))
details(c)
n_est = [i for i in range(10, 1000, 10)]
i_est = []
for i in n_est:
    i_est.append(leaf_nodes(c, a=i))

d = n_est[i_est.index(min(i_est))]
details(c, d)
final_model = RandomForestRegressor(max_leaf_nodes = c, n_estimators = d, random_state =1)
final_model.fit(X, y)
complete()
test_data = "../input/test.csv"
test = pd.read_csv(test_data)
test_features = test[features]
final_prediction = final_model.predict(test_features)
complete()
output = pd.DataFrame({'Id':test.Id , 'SalePrice' : final_prediction})
output.to_csv('my_submission.csv', index = False)
complete()