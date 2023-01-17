import os
import numpy as np
import pandas as pd
data_path = os.path.join('datasets', 'CarPrice_Assignment.csv')
cars = pd.read_csv(data_path)
cars.head()
cars.info()
cars.describe()
import matplotlib.pyplot as plt
import seaborn as sns
cars = cars.drop('car_ID', axis=1)
cars['CarName'] = cars['CarName'].str.split(' ', expand=True)
cars['CarName'].unique()
cars['CarName'] = cars['CarName'].replace({'maxda': 'mazda',
                                           'Nissan': 'nissan',
                                           'porcshce': 'porsche',
                                           'vokswagen': 'volkswagen',
                                           'vw': 'volkswagen',
                                           'toyouta': 'toyota'
                                          })
cars['doornumber'].unique()
cars['doornumber'] = cars['doornumber'].replace({'two': 2, 'four': 4})
cars['cylindernumber'].unique()
cars['cylindernumber'] = cars['cylindernumber'].replace({'four': 4,
                                                         'six': 6,
                                                         'five': 5,
                                                         'three': 3,
                                                         'twelve': 12,
                                                         'two': 2,
                                                         'eight': 8
                                                        })
cat_col = cars.select_dtypes(include=['object']).columns
num_col = cars.select_dtypes(exclude=['object']).columns
plt.rcParams['figure.figsize'] = [15, 8]
ax = cars['CarName'].value_counts().plot(kind='bar', stacked=True, colormap='Set1')
ax.title.set_text('Brands')
plt.xlabel('Brand', fontweight='bold')
plt.ylabel('Count of Cars', fontweight='bold')
plt.figure(figsize=(15,8))
plt.title('Price Distribution')
sns.distplot(cars['price'])
for i in range(0,4):
    batch=5*i
    sns.pairplot(
        data=cars,
        y_vars=['price'],
        x_vars=num_col[0+batch:5+batch],
        kind="reg"
    )
corr = cars[num_col].corr()
corr['price'].sort_values(ascending=False)
num_col_rel = ['enginesize', 'curbweight', 'horsepower', 'carwidth', 'cylindernumber', 'carlength', 'wheelbase', 'boreratio', 'price']
num_col_rel
plt.figure(figsize=(20,15))

for i in range(1, len(cat_col)):
    plt.subplot(3,3,i)
    sns.boxplot(data=cars, x=cat_col[i], y='price')
    
plt.show()
num_col_rel.extend(cat_col)
columns = num_col_rel
columns
cars = cars[columns]
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(cars, test_size=0.2, random_state=42)
train_set.head()
cars_data = train_set.drop('price', axis=1)
cars_label = train_set['price']
cars_data.head()
cars_label.head()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cat_col = cars_data.select_dtypes(include=['object']).columns
num_col = cars_data.select_dtypes(exclude=['object']).columns
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_col),
    ('cat', OneHotEncoder(drop='first'), cat_col)
])
cars_prepared = full_pipeline.fit_transform(cars_data)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
test_data = test_set.drop('price', axis=1)
test_prepared = full_pipeline.transform(test_data)
test_labels = test_set['price']
def display_scores(model):
    predictions = model.predict(test_prepared)
    mse = mean_squared_error(test_labels,predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_labels, predictions)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(cars_prepared, cars_label)
display_scores(lin_reg)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(cars_prepared, cars_label)
display_scores(forest_reg)
from xgboost import XGBRegressor
xg_reg = XGBRegressor()
xg_reg.fit(cars_prepared, cars_label)
display_scores(xg_reg)
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': np.arange(10, 200, 10), 'max_features': np.arange(2, 200, 2)},
    {'bootstrap': [False], 'n_estimators': np.arange(10, 200, 10), 'max_features': np.arange(2, 200, 2)}
]

forest_reg_grid = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg_grid, param_grid, cv=10, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
grid_search.fit(cars_prepared, cars_label)
grid_search.best_estimator_
forest_best_estimator = grid_search.best_estimator_
display_scores(forest_best_estimator)
param_grid = [
    {'n_estimators': np.arange(10, 200, 10), 'max_depth': np.arange(5, 10, 1)},
]

xgb_reg_grid = XGBRegressor()
grid_search = GridSearchCV(xgb_reg_grid, param_grid, cv=10, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)
grid_search.fit(cars_prepared, cars_label)
grid_search.best_estimator_
xgb_best_estimator = grid_search.best_estimator_
display_scores(xgb_best_estimator)
display_scores(forest_reg)
import joblib
joblib.dump(forest_reg, 'car_price_estimator.pkl')