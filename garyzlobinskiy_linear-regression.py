import pandas as pd

filepath = '../input/melbourne-housing-snapshot/melb_data.csv'
data = pd.read_csv(filepath)
data.describe()
#data.columns()
data = data.dropna(axis=0)
data
y = data.Price
y
X = data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Regionname']]
X.head(50)
X.head()
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)
print(model.predict(X.head()))
y.head()
from sklearn.metrics import mean_absolute_error
predictedY = model.predict(X)
mean_absolute_error(y, predictedY)
from sklearn.model_selection import train_test_split
a, b = [1,2]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
model2 = DecisionTreeRegressor(random_state=0)
model2.fit(train_X, train_y)
predictionsTestX = model2.predict(test_X)
mean_absolute_error(test_y, predictionsTestX)
def getMAE(nodes, train_X, train_y, test_X, test_y):

    model = DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=0)

    model.fit(train_X, train_y)

    predictions = model.predict(test_X)

    return(mean_absolute_error(test_y, predictions))
def mae(a,b):

    return(mean_absolute_error(a,b))
for num in [750,760,770,780,790,800,810,820,830,840,850]:

    val = getMAE(num, train_X, train_y, test_X, test_y)

    print("Nodes: {}; MAE: {}".format(num, val))
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0)
model.fit(train_X, train_y)
predictions = model.predict(test_X)
mean_absolute_error(test_y, predictions)
from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col="Id")

y = data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'Neighborhood']

X = data[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)
missingDataColumns = [col for col in train_X.columns if train_X[col].isnull().any()]
missingDataColumns
sX_train = train_X.drop(missingDataColumns, axis=1)

sX_test = test_X.drop(missingDataColumns, axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))

imputed_X_test = pd.DataFrame(imputer.transform(X_test))
imputed_X_train.columns = X_train.columns

imputed_X_test.columns = X_test.columns
def getMAE(nodes, train_X, train_y, test_X, test_y):

    model = DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=0)

    model.fit(train_X, train_y)

    predictions = model.predict(test_X)

    return(mean_absolute_error(test_y, predictions))
getMAE(790, X_train, y_train, X_test, y_test)
getMAE(790, imputed_X_train, y_train, imputed_X_test, y_test)
X_train_copy = X_train.copy()

X_test_copy = X_test.copy()
for col in missingDataColumns:

    X_train_copy[col + '_was_missing'] = X_train_copy[col].isnull()

    X_test_copy[col + '_was_missing'] = X_test_copy[col].isnull()
imputer2 = SimpleImputer()

imputed_X_train_copy = pd.DataFrame(imputer2.fit_transform(X_train_copy))

imputed_X_test_copy = pd.DataFrame(imputer2.transform(X_test_copy))

imputed_X_train_copy.columns = X_train_copy.columns

imputed_X_test_copy.columns = X_test_copy.columns
X_train.shape
missingValCount = (X_train.isnull().sum())

print(missingValCount[missingValCount > 0])
tempTuple = (X_train.dtypes == 'object')
tempTuple
dropped_X_train = X_train.select_dtypes(exclude=['object'])

dropped_X_test = X_test.select_dtypes(exclude=['object'])
getMAE(790, dropped_X_train, y_train, dropped_X_test, y_test)