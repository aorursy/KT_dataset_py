import pandas as pd
data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
data.describe()
data.columns
data = data.dropna(axis=0)
data.rename(columns={'Longtitude':'Longitude'}, inplace=True)

data
y = data.Price
X = data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longitude']]
X.head()
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)
model.predict(X.head())
y.head()
X.shape
from sklearn.metrics import mean_absolute_error
predictedY = model.predict(X)
mean_absolute_error(y, predictedY)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
model2 = DecisionTreeRegressor(random_state=0)
model2.fit(X_train, y_train)
predictionsXTest = model.predict(X_test)
mean_absolute_error(y_test, predictionsXTest)
def getMAE(nodes, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    model = DecisionTreeRegressor(max_leaf_nodes=nodes, random_state=0)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return mean_absolute_error(y_test, predictions)
for num in [787,789,790,791,792,793,794]:

    mae = getMAE(num)

    print("Nodes: {}, MAE: {}".format(num, mae))
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1)

model.fit(X_train, y_train)
p = model.predict(X_test)

mean_absolute_error(y_test, p)
data2 = pd.read_csv("../input/home-data-for-ml-course/train.csv", index_col='Id')
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = data2[features].copy()
y = data2.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
def getMAE(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return mean_absolute_error(y_test, predictions)
# models = [model0, model1, model2]



# for i in range(0, len(models)):

#     mae = getMAE(models[i])

#     print("Model: {}, MAE: {}".format((i+1), mae))
m1 = RandomForestRegressor(n_estimators=50, random_state=1)

m2 = RandomForestRegressor(n_estimators=100, random_state=1)

m3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=1)

m4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=1)

m5 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=1)

m6 = RandomForestRegressor(n_estimators=200, max_depth=11, random_state=1)
models = [m1,m2,m3,m4,m5,m6]



for i in range(0, len(models)):

    mae = getMAE(models[i], X_train, X_test, y_train, y_test)

    print("Model: {}, MAE: {}".format((i+1), mae))
m5.fit(X_train,y_train)

predictions = m5.predict(X_test)
output = pd.DataFrame({'Id':X_test.index, 'SalePrice':predictions})

output.to_csv('submission.csv', index=False)