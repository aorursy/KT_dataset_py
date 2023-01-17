import pandas as pd
raw_data = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
raw_data.head()
raw_data.shape
print(raw_data['Seller_Type'].unique())
print(raw_data['Transmission'].unique())
print(raw_data['Owner'].unique())
raw_data.isnull().sum()
raw_data.describe()
raw_data.drop(['Car_Name'], axis=1, inplace=True)
raw_data
raw_data['Years_Old'] = 2020 - raw_data['Year']
raw_data.head()
raw_data.drop(['Year'], axis=1, inplace=True)
raw_data
final_dataset = pd.get_dummies(raw_data, drop_first=True)
final_dataset.head()
final_dataset.corr()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

corrmat = final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))

g = sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
x = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]
x.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind="barh")
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15)
x_train.shape
y_train.shape
## Hyperparameters
import numpy as np

## Number of trees
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


from sklearn.model_selection import RandomizedSearchCV
random_grid = {'n_estimators' : n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf': min_samples_leaf
              }

print(random_grid)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(x_train, y_train)
predictions=rf_random.predict(x_test)
predictions
sns.distplot(y_test-predictions)
plt.scatter(y_test, predictions)
best_rf = rf_random.best_estimator_
best_rf.fit(x_train, y_train)
print(best_rf.score(x_test,y_test))
import pickle

file=open('random_forest_regression_model.pkl', 'wb')

pickle.dump(rf_random,file)