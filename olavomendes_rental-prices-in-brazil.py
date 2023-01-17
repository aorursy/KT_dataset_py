import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
raw_data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
raw_data.head(20)
print('ROWS: ', raw_data.shape[0])
print('COLUMNS: ', raw_data.shape[1])
raw_data.info()
raw_data.describe().T
raw_data.isnull().sum()
plt.figure(figsize=(12, 6))
sns.distplot(raw_data['rent amount (R$)'])
plt.xticks(np.arange(raw_data['rent amount (R$)'].min(), raw_data['rent amount (R$)'].max(), step=3000));
plt.figure(figsize=(10, 7))

sns.boxplot(raw_data['rent amount (R$)'])
plt.xticks(np.arange(raw_data['rent amount (R$)'].min(), raw_data['rent amount (R$)'].max(), step=3000))

plt.show()
cities = raw_data['city'].unique()
cities
plt.figure(figsize=(18, 8))

i = 1
for city in cities:
    
    if city == 'São Paulo':
        continue
    
    plt.subplot(2, 3, i)
    plt.title(city)
    city_name = raw_data.loc[raw_data['city'] == city]
    sns.distplot(city_name['rent amount (R$)'])
    plt.xticks(np.arange(city_name['rent amount (R$)'].min(), city_name['rent amount (R$)'].max(), step=2000))
    i+=1
    

plt.tight_layout()
plt.show()
plt.figure(figsize=(18, 5))

sp = raw_data.loc[raw_data['city'] == 'São Paulo']
sns.distplot(sp['rent amount (R$)'])
plt.xticks(np.arange(sp['rent amount (R$)'].min(), sp['rent amount (R$)'].max(), step=2000))

plt.show()
plt.figure(figsize=(16, 8))

i = 1
step = 5000
for city in cities:
    if step < 2000:
        step = 2000
    plt.subplot(2, 3, i)
    plt.title(city)
    city_name = raw_data.loc[raw_data['city'] == city]
    sns.boxplot(city_name['rent amount (R$)'])
    plt.xticks(np.arange(city_name['rent amount (R$)'].min(), city_name['rent amount (R$)'].max(),
                        step=step))
    step-=3000
    i+=1

    

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 10))

numData = raw_data._get_numeric_data()
var_num_corr = numData.corr()

sns.heatmap(var_num_corr, vmin=-1, vmax=1, annot=True, linewidth=0.01, linecolor='black', cmap='RdBu_r')

plt.show()
var_num_corr['rent amount (R$)'].round(3)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=raw_data['rooms'], y=raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
sns.boxplot(x=raw_data['rooms'])
plt.xticks(np.arange(raw_data['rooms'].min(), raw_data['rooms'].max(), step=1))


plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=raw_data['bathroom'], y=raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
sns.boxplot(x=raw_data['bathroom'])
plt.xticks(np.arange(raw_data['bathroom'].min(), raw_data['bathroom'].max(), step=1))


plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=raw_data['parking spaces'], y=raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
sns.boxplot(x=raw_data['parking spaces'])
plt.xticks(np.arange(raw_data['parking spaces'].min(), raw_data['parking spaces'].max(), step=1))


plt.show()
plt.figure(figsize=(18, 6))

sns.regplot(x=raw_data['fire insurance (R$)'], y=raw_data['rent amount (R$)'], line_kws={'color': 'r'})
plt.xticks(np.arange(raw_data['fire insurance (R$)'].min(), raw_data['fire insurance (R$)'].max(), step=20))

plt.show()
furniture = raw_data['furniture'].value_counts()
pd.DataFrame(furniture)
plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.title('Furniture ratio')
plt.pie(furniture, labels = ['not furnished', 'furnished'], colors= ['r', 'g'], 
        explode = (0, 0.1), autopct='%1.1f%%')

plt.subplot(1, 2, 2)
plt.title('Furniture vs Rent amount')
sns.barplot(x=raw_data['furniture'], y=raw_data['rent amount (R$)'])

plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title('Acept or not acept')
sns.countplot(raw_data['city'], hue=raw_data['animal'])

plt.subplot(1, 2, 2)
plt.title('Boxplot')
sns.boxplot(x=raw_data['rent amount (R$)'], y=raw_data['animal'])
plt.xticks(np.arange(raw_data['rent amount (R$)'].min(), raw_data['rent amount (R$)'].max(), step=5000))

plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.regplot(x=raw_data['hoa (R$)'], y=raw_data['rent amount (R$)'], line_kws={'color': 'r'})
plt.xscale('log')
plt.yscale('log')

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.regplot(x=raw_data['property tax (R$)'], y=raw_data['rent amount (R$)'], line_kws={'color': 'r'})
plt.xscale('log')
plt.yscale('log')

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.regplot(x=raw_data['area'], y=raw_data['rent amount (R$)'], line_kws={'color': 'r'})
plt.xscale('log')
plt.yscale('log')

plt.show()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import gp_minimize

# ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
plt.figure(figsize=(8, 6))

sns.boxplot(raw_data['city'], raw_data['rent amount (R$)'])

plt.show()
# Grouping cities
city_group = raw_data.groupby('city')['rent amount (R$)']
# Quantile 1 = 25% of data
Q1 = city_group.quantile(.25)
Q3 = city_group.quantile(.75)

# IQR = Interquartile Range
IQR = Q3 - Q1

# Limits
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
# DataFrame to store the new data
new_data = pd.DataFrame()

for city in city_group.groups.keys():
    is_city = raw_data['city'] == city
    accepted_limit = ((raw_data['rent amount (R$)'] >= lower[city]) &
                     (raw_data['rent amount (R$)'] <= upper[city]))
    
    select = is_city & accepted_limit
    data_select = raw_data[select]
    new_data = pd.concat([new_data, data_select])

new_data.head()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title('With outliers')
sns.boxplot(raw_data['city'], raw_data['rent amount (R$)'])

plt.subplot(1, 2, 2)
plt.title('Without outliers')
sns.boxplot(new_data['city'], new_data['rent amount (R$)'])

plt.tight_layout(pad=5.0)
plt.show()
catTransformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numTransformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
cols = ['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance (R$)',
        'furniture']

X = new_data[cols]
X.head()
for col in X:
    X = X.astype('category')
X['fire insurance (R$)'] = X['fire insurance (R$)'].astype('int64')
X.info()
y = new_data['rent amount (R$)']
y
numFeatures = X.select_dtypes(include=['int64', 'float64']).columns
numFeatures
catFeatures = X.select_dtypes(include=['category']).columns
catFeatures
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numTransformer, numFeatures),
        ('categoric', catTransformer, catFeatures)
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
regressors = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    SVR(),
    LinearRegression(),
    XGBRegressor()
]
# Seed
np.random.seed(42)

for regressor in regressors:
    
    estimator = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    estimator.fit(X_train, y_train)
    preds = estimator.predict(X_test)
    
    print(regressor)

    print('MAE:', mean_absolute_error(y_test, preds))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, preds)))
    print('R2:', r2_score(y_test, preds))
    print('-' * 40)
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', XGBRegressor(random_state=42))
                      ])
params = {
        'model__learning_rate': np.arange(0.01, 0.1),
        'model__n_estimators': np.arange(100, 1000, step=50),
        'model__max_depth': np.arange(1, 20, step=2),
        'model__subsample': [0.8, 0.9, 1],
        'model__colsample_bytree': [0.8, 0.9, 1],
        'model__gamma': [0, 1, 3, 5]
         }
estimator = RandomizedSearchCV(pipe, cv=20, param_distributions=params, n_jobs=-1)
estimator.fit(X_train,y_train)
estimator.best_params_
estimator = XGBRegressor(colsample_bytree=0.8,
                           gamma=0, 
                           learning_rate=0.01, 
                           max_depth=5, 
                           n_estimators=950, 
                           subsample=1)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', estimator)
])
model.fit(X_train, y_train)
preds = model.predict(X_test)
print('MAE:', mean_absolute_error(y_test, preds))
print('RMSE:', np.sqrt(mean_squared_error(y_test, preds)))
print('R2:', r2_score(y_test, preds))
plt.figure(figsize=(12, 8))

sns.distplot(y_test, hist=False, color='b', label ='Actual')
sns.distplot(preds, hist=False, color='r', label = 'Predicted')

plt.show()
from joblib import dump, load
dump(model, 'model_2.joblib')
model = load('model_2.joblib')
