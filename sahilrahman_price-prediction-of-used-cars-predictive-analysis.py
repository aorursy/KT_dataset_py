import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score
# This dataset contains information about used cars listed on www.cardekho.com



# df = pd.read_csv("/content/drive/My Drive/Colab Datasets/car_data.csv")

df = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
df.head()
df.shape
# print(df['fuel'].unique())

# print(df['seller_type'].unique())

# print(df['transmission'].unique())

# print(df['owner'].unique())





print(df['Fuel_Type'].unique())

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
df.isnull().sum()
df.describe()
# new_data = df.drop("name",axis=1)

# new_data.head()



new_data = df.drop("Car_Name",axis=1)

new_data.head()
new_data["current_year"] = 2020

new_data.head()
# new_data["no_of_years"] = new_data['current_year'] - new_data['year']

# new_data = new_data.drop(["year","current_year"], axis=1)

# new_data.head()





new_data["no_of_years"] = new_data['current_year'] - new_data['Year']

new_data = new_data.drop(["Year","current_year"], axis=1)

new_data.head()

new_data = pd.get_dummies(new_data, drop_first=True)

new_data.head()
new_data.shape
new_data.corr()
sns.pairplot(new_data)
cormat = new_data.corr()

top_corr_feat = cormat.index

plt.figure(figsize=(10,10))

g = sns.heatmap(new_data[top_corr_feat].corr(), annot=True, cmap='RdYlGn')
X = new_data.iloc[:,1:]

y = new_data.iloc[:,0]
X.head()
y.head()
model = ExtraTreesRegressor()

model.fit(X,y)
model.feature_importances_
important_feature = pd.Series(model.feature_importances_, index=X.columns)

important_feature.nlargest(5).plot(kind="barh")

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)
X_train
model = RandomForestRegressor()
# No. of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]



# No. of features to consider at every split

max_features = ['auto', 'sqrt']



# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]



# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
rf_random = RandomizedSearchCV(estimator = model, 

                               param_distributions = random_grid,

                               scoring = 'neg_mean_squared_error', 

                               n_iter = 10, 

                               cv = 5,

                               verbose = 2, 

                               random_state = 42, 

                               n_jobs = 1)
rf_random.fit(X_train, y_train)
rf_random.best_params_
predicted_y_values = rf_random.predict(X_test)
# plt.scatterplot(y_test, predicted_y_values)

plt.scatter(y_test, predicted_y_values)
sns.distplot(y_test- predicted_y_values)
accuracy = r2_score(y_test, predicted_y_values)

accuracy
radju = 1 - (((1-accuracy)*(len(new_data)-1))/(len(new_data) - len(new_data.columns) -1))
radju
df5 = pd.DataFrame({'Real Values':y_test, 'Predicted Values':predicted_y_values})

df5.head()