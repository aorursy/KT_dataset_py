import pandas as pd
melbourne_file_path = '../input/melb_data.csv'
data = pd.read_csv(melbourne_file_path)
data.columns


#data = data.dropna(axis = 0)
y = data.Price
y.head()
features = ['Rooms','Bathroom','Landsize','Lattitude', 'Longtitude']
X = data[features]
X.describe()

X.head()
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=4)
model.fit(X,y)
print(X.head())
print(model.predict(X.head()))
from sklearn.metrics import mean_absolute_error
php = model.predict(X)
mean_absolute_error(y,php)
from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)
model = DecisionTreeRegressor()
model.fit(train_X,train_y)
val_predict = model.predict(val_X)
print(mean_absolute_error(val_y,val_predict))

def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    model.fit(train_X,train_y)
    pred_y = model.predict(val_X)
    mae = mean_absolute_error(pred_y,val_y)
    return mae
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d"%(max_leaf_nodes,my_mae))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
mvcbc = (data.isnull().sum())
print(mvcbc[mvcbc>0])
#Simple method
cols_w_missing = (col for col in train_X.columns if data[col].isnull().any())
redTrain_X = train_X.drop(cols_w_missing,axis=1)
redTest_X = val_X.drop(cols_w_missing,axis=1)
d = score_dataset(redTrain_X,redTest_X,train_y,val_y)
d2 = score_dataset(train_X,val_X,train_y,val_y)
print(d,d2)

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
#ndata = my_imputer.fit_transform(train_X)

ndata = train_X.copy()
cols_missing =[cols for cols in ndata.columns if train_X[cols].isnull().any()]
for col in cols_missing:
    ndata[col+"was_missing"] =  ndata[col].isnull()
ndata = pd.DataFrame(my_imputer.fit_transform(ndata))
ndata.columns = train_X.columns
#print(ndata)
imputTrain_X = train_X.copy()
imputTest_X = val_X.copy()
cols_with_missing = (cols for cols in train_X.columns if train_X[cols].isnull().any())
for cols in cols_with_missing:
    imputTrain_X[cols+'was_missing'] = imputTrain_X[cols].isnull()
    imputTest_X[cols+'was_missing'] = imputTest_X[cols].isnull()
imputTrain_X = my_imputer.fit_transform(imputTrain_X)
imputTest_X = my_imputer.transform(imputTest_X)
print(score_dataset(imputTrain_X,imputTest_X,train_y,val_y))

