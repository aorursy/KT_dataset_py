import numpy as np
import pandas as pd
data = pd.read_csv("../input/avocado.csv")
data.head()
# make sure there aren't any missing values, if there are we need to use an Imputer
assert [col for col in data.columns if data[col].isnull().any()] == []
dates = [(int(mm), int(dd)) for mm, dd in [d.rsplit('-')[1:] for d in data['Date']]] 

data['month'] = pd.Series([mm[0] for mm in dates])
data['day'] = pd.Series([dd[1] for dd in dates])

features_num = ['Total Volume', '4046', '4225', '4770', 'Total Bags',
                'Small Bags', 'Large Bags', 'XLarge Bags', 'year',
                'month', 'day']
target = ['AveragePrice']

X = data[features_num].values
y = data[target].values.ravel()

from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

clf = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1, n_estimators=150))

scores = cross_val_score(clf, Xtrain, ytrain, cv=3, n_jobs=-1)
print(f"{round(np.mean(scores),3)*100}% accuracy")

clf.fit(Xtrain,ytrain)

print(mean_squared_error(y_pred=clf.predict(Xtest), y_true=ytest))
from xgboost import XGBRegressor

clf = make_pipeline(StandardScaler(), XGBRegressor(n_estimators=1000, learning_rate=0.2, early_stopping_rounds=5))

scores = cross_val_score(clf, Xtrain, ytrain, cv=3, n_jobs=-1)
print(f"{round(np.mean(scores),3)*100}% accuracy")

clf.fit(Xtrain, ytrain)

print(mean_squared_error(y_pred=clf.predict(Xtest),y_true=ytest))
data_with_categorical= pd.get_dummies(data.drop(columns=['Unnamed: 0', 'Date'], axis=1))
X = data_with_categorical.drop(columns='AveragePrice', axis=1).values
y = data_with_categorical['AveragePrice'].values.ravel()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)

clf = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1, n_estimators=100))

scores = cross_val_score(clf, Xtrain, ytrain, cv=3, n_jobs=-1)
print(f"{round(np.mean(scores),3)*100}% accuracy")

clf.fit(Xtrain,ytrain)

print(mean_squared_error(y_pred=clf.predict(Xtest), y_true=ytest))
clf = make_pipeline(StandardScaler(), XGBRegressor(n_estimators=1000, learning_rate=0.5, early_stopping_rounds=5))

scores = cross_val_score(clf, Xtrain, ytrain, cv=3, n_jobs=-1)
print(f"{round(np.mean(scores),3)*100}% accuracy")

clf.fit(Xtrain, ytrain)

print(mean_squared_error(y_pred=clf.predict(Xtest),y_true=ytest))
