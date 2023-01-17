import pandas as pd
df = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
df.head()
df.shape
print(df['Seller_Type'].unique())
print(df['Owner'].unique())
print(df['Transmission'].unique())
print(df['Fuel_Type'].unique())
df.isnull().sum()
df.columns
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()
final_dataset['current_year'] =2020
final_dataset.head()
final_dataset['year_old'] = final_dataset['current_year'] - final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Year', 'current_year'], axis=1, inplace=True)
final_dataset.head()
final_dataset_dummy = pd.get_dummies(final_dataset, drop_first=True)
final_dataset_dummy.head()
final_dataset_dummy.corr()
import seaborn as sns
sns.pairplot(final_dataset_dummy)
import matplotlib.pyplot as plt
%matplotlib inline
corrmat = final_dataset_dummy.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset_dummy[top_corr_features].corr(), annot=True, cmap="RdYlGn")
X = final_dataset_dummy.iloc[:,1:]
y = final_dataset_dummy.iloc[:,0]
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
import numpy as np
n_estimators = [int(x) for x in np.linspace(100,1200,12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5,30,6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]
from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
rf_random.best_params_
predictions = rf_random.predict(X_test)
predictions
sns.distplot(y_test-predictions)
sns.scatterplot(y_test, predictions)
import pickle
pickle.dump(rf_random, open('random_forest_regression_model.pkl', 'wb'))
