import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
bike_rentals = pd.read_csv("../input/bike-rental-hour/bike_rental_hour.csv", index_col="instant")
bike_rentals.head()
%matplotlib inline


plt.hist(bike_rentals["cnt"])
bike_rentals.corr()
sns.heatmap(bike_rentals.corr())
def hour_car(hour):
    if hour >= 6 & hour < 12:
        return 1
    if hour >= 12 & hour < 18:
        return 2
    if hour >= 18 & hour < 24:
        return 3
    if hour >= 0 & hour < 6:
        return 4
bike_rentals["time_label"] = bike_rentals["hr"].apply(hour_car)
from sklearn.model_selection import train_test_split

train, test = train_test_split(bike_rentals, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
features = ["yr", "hr", "temp", "hum"]
X = train[features]
y = train ["cnt"]

X_test = test[features]
y_test = test["cnt"]
model_full = LinearRegression()
model_full.fit(X,y)
train_predictions = model_full.predict(X)
test_predictions = model_full.predict(test[features])
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(train["cnt"], train_predictions)
train_score = model_full.score(X,y)
test_error = mean_squared_error(test["cnt"], test_predictions)
test_score = model_full.score(X_test,y_test)

print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(X,y)
train_tree_predictions = tree_model.predict(X)
test_tree_predictions =  tree_model.predict(test[features])
train_error = mean_squared_error(train["cnt"], train_tree_predictions)
train_score = tree_model.score(X,y)
test_error = mean_squared_error(test["cnt"], test_tree_predictions)
test_score = tree_model.score(X_test,y_test)

print("Evaluating our decision tree model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))
tree_model = DecisionTreeRegressor(random_state=1, min_samples_leaf=2)
tree_model.fit(X,y)
train_tree_predictions = tree_model.predict(X)
test_tree_predictions =  tree_model.predict(test[features])

train_error = mean_squared_error(train["cnt"], train_tree_predictions)
train_score = tree_model.score(X,y)
test_error = mean_squared_error(test["cnt"], test_tree_predictions)
test_score = tree_model.score(X_test,y_test)

print("Evaluating our decision tree model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))


tree_model = DecisionTreeRegressor(random_state=1, max_depth=10)
tree_model.fit(X,y)
train_tree_predictions = tree_model.predict(X)
test_tree_predictions =  tree_model.predict(test[features])

train_error = mean_squared_error(train["cnt"], train_tree_predictions)
train_score = tree_model.score(X,y)
test_error = mean_squared_error(test["cnt"], test_tree_predictions)
test_score = tree_model.score(X_test,y_test)

print("Evaluating our decision tree model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(min_samples_leaf=7)

rf_model.fit(X,y)
train_rf_predictions = rf_model.predict(X)
test_rf_predictions =  rf_model.predict(test[features])

train_error = mean_squared_error(train["cnt"], train_rf_predictions)
train_score = rf_model.score(X,y)
test_error = mean_squared_error(test["cnt"], test_rf_predictions)
test_score = rf_model.score(X_test,y_test)

print("Evaluating our random forest model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))
rf_model = RandomForestRegressor(n_estimators=20, min_samples_leaf=7)

rf_model.fit(X,y)
train_rf_predictions = rf_model.predict(X)
test_rf_predictions =  rf_model.predict(test[features])

train_error = mean_squared_error(train["cnt"], train_rf_predictions)
train_score = rf_model.score(X,y)
test_error = mean_squared_error(test["cnt"], test_rf_predictions)
test_score = rf_model.score(X_test,y_test)

print("Evaluating our random forest model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))
predictors = list(train.columns)
predictors.remove("cnt")
predictors.remove("casual")
predictors.remove("registered")
predictors.remove("dteday")

X_train = train[predictors]
y_train = train ["cnt"]

X_test = test[predictors]
y_test = test["cnt"]
tree_model = DecisionTreeRegressor(random_state=1, max_depth=10)
tree_model.fit(X_train,y_train)
train_tree_predictions = tree_model.predict(X_train)
test_tree_predictions =  tree_model.predict(test[predictors])

train_error = mean_squared_error(train["cnt"], train_tree_predictions)
train_score = tree_model.score(X_train,y_train)
test_error = mean_squared_error(test["cnt"], test_tree_predictions)
test_score = tree_model.score(X_test,y_test)

print("Evaluating our decision tree model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))

rf_model = RandomForestRegressor(n_estimators=20, min_samples_leaf=7)

rf_model.fit(X_train,y_train)
train_rf_predictions = rf_model.predict(X_train)
test_rf_predictions =  rf_model.predict(test[predictors])

train_error = mean_squared_error(train["cnt"], train_rf_predictions)
train_score = rf_model.score(X_train,y_train)
test_error = mean_squared_error(test["cnt"], test_rf_predictions)
test_score = rf_model.score(X_test,y_test)

print("Evaluating our random forest model")
print ("Training error is " +str(train_error))
print ("Training score is " +str(train_score))
print("Test error is " + str(test_error))
print ("Test score is " +str(test_score))
feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')