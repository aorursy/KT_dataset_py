%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from sklearn import ensemble
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

#import seaborn as sns
#import matplotlib
import matplotlib.pyplot as plt
#from scipy.stats import skew
#from scipy.stats.stats import pearsonr

#DATA_PATH = "./data/"
DATA_PATH = "../input"

data = pd.read_csv(DATA_PATH + "/train.csv")
validation = pd.read_csv(DATA_PATH + "/test.csv")

data = data.reindex(np.random.permutation(data.index))
#heat map
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

data["HouseAge"] =  (data["YrSold"] - data["YearBuilt"])
corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

data.fillna(value=0, inplace=True)
data = pd.get_dummies(data)

corrmat = data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
def preprocess_features(data):
    data['TotalBsmtSF'].fillna(value=0, inplace=True)
    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    selected_features = data[
    ["GrLivArea",
     "LotArea",
     "OverallQual",
     "OverallCond",
     "Fireplaces",
     "TotalBsmtSF",
     "KitchenAbvGr",
     "FullBath"
    ]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["HouseAge"] = (data["YrSold"] - data["YearBuilt"])
    # one hot code
    dummies_BsmtQual = pd.get_dummies( data.loc[:, "BsmtQual"], prefix="BsmtQual" ) 
    dummies_CentralAir = pd.get_dummies( data.loc[:, "CentralAir"], prefix="CentralAir" )
    processed_features = pd.concat( [processed_features, dummies_BsmtQual, dummies_CentralAir], axis = 1 )
    return processed_features
def select_features(data):
    corrmat = data.corr()
    colsNum = len(corrmat)
    c = data.corr().abs()
    s = c.unstack()
    so = s.sort_values(kind="quicksort")
    top = 100
    cols = so["SalePrice"][colsNum-top:colsNum].index
    print(  len(so["SalePrice"]))
    selected_cols = []
    for col in cols:
        if col!='SalePrice':
            selected_cols.append(col)
    #heat map
    '''
    features = preprocess_features(data)
    data["HouseAge"] = features["HouseAge"]
    corrmat = data.corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    '''
    return selected_cols
selected_cols = select_features(data)
print(selected_cols)
def preprocess_features2(data):
    selected_features = data[selected_cols]
    processed_features = selected_features.copy()
    return processed_features
def preprocess_targets(data):
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["SalePrice"] = np.log1p(data["SalePrice"])
    return output_targets

def train(model, x_train, y_train):
    model = model.fit(x_train, y_train)
    melb_preds = model.predict(x_test)
    print(mean_absolute_error(y_test, melb_preds))
    print(model.score(x_test, y_test))
    return model
def train_tree_base(model, x_train, y_train):
    forest_model = model.fit(x_train, y_train.values.ravel())
    melb_preds = forest_model.predict(x_test)
    print(mean_absolute_error(y_test, melb_preds))
    print(forest_model.score(x_test, y_test))
    return forest_model
def train_nn(model, x_train, y_train):
    nn = model.fit(x_train, y_train.values.ravel())
    melb_preds = nn.predict(x_test)
    print(mean_absolute_error(y_test, melb_preds))
    print(nn.score(x_test, y_test))
    return nn
x = preprocess_features2(data)
y = preprocess_targets(data)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
#linear
model_ridge = Ridge(alpha = 0.1)
model_ridge = train(model_ridge, x_train, y_train)
#tree
forest = ensemble.RandomForestRegressor()
model_forest = train_tree_base(forest, x_train, y_train)
#nn
#nn = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(1,2,3), random_state=1)
#model_nn = train_nn(nn, x_train, y_train)
validation.fillna(value=0, inplace=True)
validation = pd.get_dummies(validation)
validation["HouseAge"] =  (validation["YrSold"] - validation["YearBuilt"])
validation_reindex = validation.reindex(columns=selected_cols)
x_validation = preprocess_features2(validation_reindex)

result_linear = model_ridge.predict(x_validation)
result_tree = model_forest.predict(x_validation)
#result_nn = model_nn.predict(x_validation)
result_linear = np.expm1(result_linear)
result_tree = np.expm1(result_tree)
#result_nn = np.expm1(result_nn)
 
sub = pd.DataFrame()
sub['Id'] = validation['Id']
sub['SalePrice'] = result_linear
sub.to_csv('submission_linear.csv',index=False)
sub['SalePrice'] = result_tree
sub.to_csv('submission_tree.csv',index=False)
#sub['SalePrice'] = result_nn
#sub.to_csv('submission_nn.csv',index=False)

#print(result)



