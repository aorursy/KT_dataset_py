import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
df = pd.read_csv('../input/used-cars-database/autos.csv',encoding='Latin-1')
df.head(3)
df_ford = df.query('brand == "ford" ')
df_ford.head(3)
df_ford.shape
df_ford.info()
df_ford.describe().T
nan_cols = df_ford.isna().sum()
nan_cols[nan_cols>0]

filt60k = df_ford['price']>60000
filt1_under_500 = (df_ford['price']>1) & (df_ford['price']<500)

print(f"Cars over 60.000€: {len(df_ford[filt60k])}")
print(f"Cars under 500€: {len(df_ford[filt1_under_500])}")
df_ford = df_ford.query('price < 60000 & price > 499')
df_ford.shape
plt.figure(figsize=(16,4))
sns.scatterplot(x='yearOfRegistration', y='price', data=df_ford, hue='yearOfRegistration')
plt.show()
min_date=1900
now = datetime.datetime.now()
max_date_current_year = now.year
df_ford.query("yearOfRegistration < @min_date | yearOfRegistration > @max_date_current_year")
df_ford = df_ford.query("yearOfRegistration > @min_date & yearOfRegistration <= @max_date_current_year")
plt.figure(figsize=(16,4))
sns.scatterplot(x='yearOfRegistration', y='price', data=df_ford, hue='price')
plt.show()
years_over_seventy = (df_ford['yearOfRegistration'] > 1969)
df_ford = df_ford.loc[years_over_seventy]
f = plt.figure(figsize=(16,4))
sns.distplot(df_ford['yearOfRegistration'], kde=True, bins=20, color='navy')
plt.show()
f = plt.figure(figsize=(16,4))
sns.countplot(x=df_ford['kilometer'], data=df_ford)
plt.show()
df_km_check = df_ford.copy()
df_km_check['km_category'] = df_km_check['kilometer'].apply(lambda x: 'Less or equal 100.000 km' if x<=100000 else 'Greater 100.000 km')
plt.figure(figsize=(16,4))
sns.countplot(df_km_check['km_category'])
plt.show()
f = plt.figure(figsize=(16,4))
sns.boxplot(x='kilometer', y='price', data=df_ford)
plt.show()
f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.countplot(df_ford['vehicleType'])
f.add_subplot(1,2,2)
sns.stripplot(y='price', x='vehicleType', data=df_ford)
plt.show()
vehicle_filt = ((df_ford['vehicleType'].notna()) & (~df_ford['vehicleType'].str.contains('andere', na=True)))
df_ford = df_ford.loc[vehicle_filt]
f = plt.figure(figsize=(18, 4))
f.add_subplot(1,2,1)
sns.boxplot(df_ford['powerPS'])
plt.title('All cars')
f.add_subplot(1,2,2)
sns.boxplot(df_ford.query("powerPS <= 460")['powerPS'])
plt.title('Cars <= 460ps')
plt.show()
ps_filt = (df_ford['powerPS']>460) & (df_ford['powerPS']<=800)
df_ford.loc[ps_filt, ['model', 'powerPS', 'kilometer','price']]
ps_filt = (df_ford['powerPS']>1) & (df_ford['powerPS']<=40)
df_ford.loc[ps_filt, ['model', 'powerPS', 'kilometer','price']]
df_ford = df_ford.query("powerPS > 34 & powerPS <= 460")
plt.figure(figsize=(16,4))
sns.countplot(y=df_ford['model'], order=df_ford['model'].value_counts().index)
plt.show()
filt = (df_ford['model'].isna())
print(f'Model information missing for {len(df_ford.loc[filt])} cars.')

filt_model = ( (df_ford['model'].notna()) & (~df_ford['model'].str.contains('andere', na=True)) )
df_ford = df_ford.loc[filt_model]
sorted_index = df_ford.groupby('model')['price'].mean().sort_values(ascending=False).index
plt.figure(figsize=(16,4))
sns.boxplot(x='model', y='price', data=df_ford, order=sorted_index)
plt.show()
plt.figure(figsize=(16,4))
sns.countplot(y=df_ford['fuelType'],data=df_ford)
plt.show()
df_ford = df_ford.query("fuelType=='benzin' | fuelType=='diesel' ")
plt.figure(figsize=(10,4))
my_colors = ['g', 'b', 'r']
df_ford['notRepairedDamage'].value_counts(dropna=False).plot(kind='bar', color=my_colors)
plt.show()
df_ford.loc[:,'notRepairedDamage'] = df_ford['notRepairedDamage'].fillna('nein')
df_ford = df_ford.query("notRepairedDamage != 'ja' ")
plt.figure(figsize=(10,4))
my_colors = ['seagreen', 'lightblue', 'crimson']
df_ford['gearbox'].value_counts(dropna=False).plot(kind='barh', color=my_colors)
plt.show()
relevant_cols = ['model', 'vehicleType', 'gearbox', 'powerPS', 'kilometer','fuelType', 'yearOfRegistration', 'price']
data = df_ford.loc[:,relevant_cols]
data.head(3)
f = plt.figure(figsize=(18,4))
f.subplots_adjust(hspace=0.2, wspace=0.4)

f.add_subplot(1,4,1)
sns.kdeplot(data['powerPS'])

f.add_subplot(1,4,2)
sns.kdeplot(data['kilometer'])

f.add_subplot(1,4,3)
sns.kdeplot(data['yearOfRegistration'])

f.add_subplot(1,4,4)
sns.kdeplot(data['price'])
plt.show()
plt.figure(figsize=(18,4))
sns.kdeplot(data['powerPS'])
sns.kdeplot(data['kilometer'])
sns.kdeplot(data['yearOfRegistration'])
sns.kdeplot(data['price'])
plt.show()
col_names = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]
scaler = StandardScaler()

#If you want to see the plot when MinMax scaled, uncomment next lines
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0,1))

df_s = scaler.fit_transform(data[col_names])
df_s = pd.DataFrame(df_s, columns=data[col_names].columns)

plt.figure(figsize=(18, 4))

plt.title('Result Standard Scaled')
sns.kdeplot(df_s['powerPS'])
sns.kdeplot(df_s['kilometer'])
sns.kdeplot(df_s['yearOfRegistration'])
sns.kdeplot(df_s['price'])
plt.show()
mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))

f = plt.figure(figsize=(20, 4))
f.subplots_adjust(hspace=0.2, wspace=0.4)

f.add_subplot(1,2,1)
heatmap1 = sns.heatmap(data.corr(), annot=True, mask=mask,cmap='RdYlGn', cbar=False)
heatmap1.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=10)

f.add_subplot(1,2,2)
heatmap2 = sns.heatmap(data.corr()[['price']].sort_values(by='price', ascending=False), annot=True, cmap='RdYlGn', cbar=False)
heatmap2.set_title('Features Correlating with Price', fontdict={'fontsize':14}, pad=10)
plt.show()
X = data.drop('price', axis=1)
y = data['price']
categorical_cols_OHE = [cname for cname in X.columns if X[cname].dtype =='object']
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
num_transformer = Pipeline(steps=[
    ('imput_num', SimpleImputer(strategy='mean')),
    #('stand_scaler', StandardScaler())
])
cat_ohe_transformer = Pipeline(steps=[
    ('impute_cat', SimpleImputer(strategy='most_frequent')),
    ('OHE', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat_ohe', cat_ohe_transformer, categorical_cols_OHE)
])
from xgboost import XGBRegressor
xgb_regr = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0, n_jobs=4)
pipe = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', xgb_regr)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_valid)
fig, ax = plt.subplots()
ax.scatter(y_pred, y_valid, edgecolors=('b'), color='navy')
ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=3)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
plt.show()
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred)
print('Performance:')
print('-'*30)
print(f'Mean Absolute Error: {round(mae, 2)}')
print(f'Mean Squared Error: {round(mse, 2)}')
print(f'Root Mean Squared Error: {round(rmse, 2)}')
print(f'R2 Score: {round(r2,2)}')