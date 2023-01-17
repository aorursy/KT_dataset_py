import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainpath='../input/house-prices-dataset/train.csv'

testpath='../input/house-prices-dataset/test.csv'

test_values='../input/house-prices-dataset/sample_submission.csv'



traindata=pd.read_csv(trainpath)

testdata=pd.read_csv(testpath)

valuestest=pd.read_csv(test_values)
traindata.columns
traindata.info()


train_data=traindata[['LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'SalePrice']]

pd.plotting.scatter_matrix(train_data, figsize=(14,6), diagonal='kde')
import seaborn as sns

sns.heatmap(train_data.corr(), cmap='RdYlGn', annot=True)
features=['LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# features=['LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'CentralAir', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



X=traindata[features]

y=traindata.SalePrice



X_test=testdata[features]

y_test=valuestest.SalePrice
from sklearn.model_selection import train_test_split



(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.1, random_state=23)
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

candidate_max_leaf_nodes = [167,168, 169,170, 171, 172, 173, 174]

values=[]

for value in candidate_max_leaf_nodes:

    values.append(get_mae(value,X_train, X_valid, y_train, y_valid))

best_tree_size = candidate_max_leaf_nodes[values.index(min(values))]

print(best_tree_size)

print(min(values))
import matplotlib.pylab as plt



plt.plot(candidate_max_leaf_nodes, values)
from sklearn.metrics import mean_absolute_error



final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

final_model.fit(X, y)

preds=final_model.predict(X_test)

print(mean_absolute_error(y_test, preds))

print(final_model.score(X_test, y_test))
from sklearn.ensemble import RandomForestRegressor



def get_mae_forest(max_depth, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(max_depth=max_depth, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



max_depth= [25, 26, 27, 28, 29, 30]



values=[]

for value in max_depth:

    values.append(get_mae_forest(value,X_train, X_valid, y_train, y_valid))

    

best_max_depth = max_depth[values.index(min(values))]

print(best_max_depth)

print(values)

plt.plot(max_depth, values)
from sklearn.ensemble import RandomForestRegressor



forest_model = RandomForestRegressor(random_state=0)

forest_model.fit(X, y)

preds = forest_model.predict(X_test)

print(mean_absolute_error(y_test, preds))
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



linreg = LinearRegression()

linreg.fit(X,y)

preds=linreg.predict(X_test)

print(mean_absolute_error(y_test, preds))

print(linreg.score(X_test, y_test))

from sklearn.linear_model import Ridge



def get_mae_ridge(alpha, train_X, val_X, train_y, val_y):

    model = Ridge(alpha=alpha, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# alphas=[0, 0.11, 1, 10, 100, 1000, 10000]

# alphas=[0, 0.11, 1, 10, 100, 1000, 2000]

# alphas=[100, 300, 600, 1000, 1100]

# alphas=[300, 400, 500, 600, 700, 800, 900, 1000]

# alphas=[400, 450, 500, 550, 600]

# alphas= [450, 475, 500, 550, 600]

# alphas= [450, 460, 465, 470, 475, 480, 485, 490, 500]

# alphas= [475, 480, 482, 485, 487, 490]

alphas= [480, 481, 482, 483, 484, 485]



values=[]

for value in alphas:

    values.append(get_mae_ridge(value,X_train, X_valid, y_train, y_valid))

    

best_alpha = alphas[values.index(min(values))]

print(best_alpha)

print(values)

plt.plot(alphas, values)
rdg = Ridge(alpha=best_alpha, normalize=True, random_state=0)

rdg.fit(X,y)

preds=rdg.predict(X_test)

print(mean_absolute_error(y_test, preds))

print(rdg.score(X_test, y_test))
from sklearn.linear_model import Lasso



def get_mae_lasso(alpha, train_X, val_X, train_y, val_y):

    model = Lasso(alpha=alpha, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# alphas=[0, 0.11, 1, 10, 100, 1000, 10000]

# alphas=[1000, 2000, 5000]

# alphas=[1000, 1500, 2000, 3000, 5000]

# alphas=[1500, 1700, 1900, 2000,2200, 2500, 3000]

# alphas=[1700, 1800, 1850, 1900, 1920, 1960, 2000]

# alphas=[1900, 1910, 1915, 1920, 1930, 1940, 1960]

# alphas=[1930,1935,1940,1945,1950, 1960]

# alphas=[1930,1933, 1935,1937,1940]

alphas=[1935,1936,1937,1938, 1939,1940]



values=[]

for value in alphas:

    values.append(get_mae_lasso(value,X_train, X_valid, y_train, y_valid))

    

best_alpha = alphas[values.index(min(values))]

print(best_alpha)

print(min(values))



plt.plot(alphas, values)
rdg = Ridge(alpha=best_alpha, normalize=True, random_state=0)

rdg.fit(X,y)

preds=rdg.predict(X_test)

print(mean_absolute_error(y_test, preds))

print(rdg.score(X_test, y_test))
from sklearn.linear_model import ElasticNet



def get_mae_elnet(alpha, l1_ratio, train_X, val_X, train_y, val_y):

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1]

# l1_ratio=[0.4, 0.5, 0.6, 0.7, 0.8]

# l1_ratio=[0.5, 0.55, 0.6, 0.65, 0.7]

# l1_ratio=[0.6, 0.63, 0.65, 0.67, 0.7]



# l1_ratio=[0.6, 0.61, 0.62, 0.63, 0.64, 0.65]

# values=[]

# for value in l1_ratio:

#     values.append(get_mae_elnet(value, X_train, X_valid, y_train, y_valid))

    

# best_l1_ratio = l1_ratio[values.index(min(values))]

# print(best_l1_ratio)

# print(min(values))



# plt.plot(l1_ratio, values)



best_l1_ratio=0.63

alphas=[0.5,0.7, 1,1.5,2, 5]

values=[]

for value in alphas:

    values.append(get_mae_elnet(value, best_l1_ratio, X_train, X_valid, y_train, y_valid))

    

best_alpha = alphas[values.index(min(values))]

print(best_alpha)

print(min(values))



plt.plot(alphas, values)
en = ElasticNet(alpha=best_alpha,l1_ratio=best_l1_ratio, normalize=True, random_state=0)

en.fit(X,y)

preds=en.predict(X_test)

print(mean_absolute_error(y_test, preds))

print(en.score(X_test, y_test))
#Saving predictions for competitions



# output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds})

# output.to_csv('submission.csv', index=False)