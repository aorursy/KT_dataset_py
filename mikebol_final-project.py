import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from scipy.stats import norm
from scipy import stats

from sklearn import ensemble
from matplotlib import style
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
%matplotlib inline
train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')
feature_names = train.columns.to_list()
df_feautre_names = train.select_dtypes(include=['float64', 'int64'])
num_features = pd.DataFrame(df_feautre_names)
# строим матрицу корреляций
corr = df_feautre_names.corr()
plt.figure(figsize=(10, 10))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale=1.2)
sns.heatmap(df_feautre_names.corr(), mask=mask, annot=True, fmt='.1f', linewidths=0.5, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()
plt.figure(figsize=(15, 8))
sns.swarmplot(train['Rooms'], train['Price'])
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.title('Price dependence on rooms')
plt.show()
scatter_square = pd.concat([train['Price'], train['Square']], axis=1)
scatter_square.plot.scatter(x='Square', y='Price', c='DarkBlue')
plt.title('Price dependence on Square')
plt.show()
scatter_lifesquare = pd.concat([train['Price'], train['LifeSquare']], axis=1)
scatter_lifesquare.plot.scatter(x='LifeSquare', y='Price', c='DarkBlue')
plt.title('Price dependence on LifeSquare')
plt.show()
train_features = train.select_dtypes(include=['float64', 'int64'])
df_train_features = pd.DataFrame(train_features)
df_train_features.hist(figsize=(12, 12), bins=15, grid=False, layout=(6, 3))
plt.show()
train = train.sort_values('Price')
train['Rooms'] = train['Rooms'].astype('int64')
test['Rooms'] = test['Rooms'].astype('int64')
train['HouseFloor'] = train['HouseFloor'].astype('int64')
test['HouseFloor'] = test['HouseFloor'].astype('int64')
# очистка выбросов Square
sort_by_square = train.sort_values(by='Square', ascending=False)
wasted_square = sort_by_square.loc[:, 'Square'][:3].index.to_list()
train = train.drop(wasted_square)
#train.sort_values(by='Square', ascending=False)
# Очистка выбросов HouseYear
sort_by_houseyear = train.sort_values(by='HouseYear', ascending=False)
wasted_houseyear = sort_by_houseyear.loc[:, 'HouseYear'][:2].index.to_list()
train = train.drop(wasted_houseyear)
#train.sort_values(by='HouseYear', ascending=False)
# Очистка выбросов Rooms
sort_by_rooms = train.sort_values(by='Rooms', ascending=False)
wasted_rooms = sort_by_rooms.loc[:, 'Rooms'][:3].index.to_list()
train = train.drop(wasted_rooms)
#train.sort_values(by='Rooms', ascending=False)
# Очистка выбросов LifeSquare
sort_by_lifesquare = train.sort_values(by='LifeSquare', ascending=False)
wasted_lifesquare = sort_by_lifesquare.loc[:, 'LifeSquare'][:6].index.to_list()
train = train.drop(wasted_lifesquare)
#train.sort_values(by='LifeSquare', ascending=False)
class FeatureCorrector:
    # класс для корректировки фичей
    
    def __init__(self):
        
        self.medians = None
        
    def fit(self, df):
        
        self.medians = df.median()
    
    def transform(self, df):
        
        df['Rooms_outlier'] = 0
        df.loc[(df['Rooms'] == 0) | (df['Rooms'] >= 6), 'Rooms_outlier'] = 1
        
        df.loc[df['Rooms'] == 0, 'Rooms'] = 1
        df.loc[df['Rooms'] >= 6, 'Rooms'] = self.medians['Rooms'] 
        df.loc[(df['Rooms'] > 5) & (df['Square'] < 50),'Rooms'] = 2
        
        df.loc[df['Square'] < 10,'Square']= df.loc[df['Square'] < 10,'Square'] * 10
        df.loc[df['Square'] < 1,'Square'] = df.loc[df['Square']< 1,'Square'] * 100 
        df.loc[df['Square'] > 250, 'Square'] = self.medians['Square']
        
        df['HouseFloor_outlier'] = 0
        df.loc[df['HouseFloor'] == 0, 'HouseFloor_outlier'] = 1
        
        df.loc[df['KitchenSquare'] < 3, 'KitchenSquare'] = 3
        df.loc[df['KitchenSquare'] > 1000, 'KitchenSquare'] = self.medians['KitchenSquare']
        
        df.loc[df['HouseFloor'] == 0, 'HouseFloor'] = self.medians['HouseFloor']
        df.loc[df['Floor'] > df['HouseFloor'], 'Floor'] = df.loc[df['Floor'] > df['HouseFloor'], 'HouseFloor']
        
        current_year = now = datetime.datetime.now().year
        
        df['HouseYear_outlier'] = 0
        df.loc[df['HouseYear'] > current_year, 'HouseYear_outlier'] = 1
        
        df.loc[df['HouseYear'] > current_year, 'HouseYear'] = self.medians['HouseYear']
        
        if 'Healthcare_1' in df.columns:
            df.drop('Healthcare_1', axis=1, inplace=True)
        
        df.loc[(df['LifeSquare'] > 200) &\
               (df['Square'] < 100),'LifeSquare'] = df.loc[(df['LifeSquare'] > 200) & (df['Square'] < 100),'LifeSquare'] / 10
        
        df['LifeSquare_nan'] = df['LifeSquare'].isna() * 1
        
        clause = (df['LifeSquare'].isna()) &\
                      (~df['Square'].isna()) &\
                      (~df['KitchenSquare'].isna())
        
        df.loc[clause, 'LifeSquare'] = df.loc[clause, 'Square'] * 0.65
        
        
        return df
class FeatureGenerator():
    # генерация фичей(точнее замена A и B в фиче на бинарный тип 0 и 1)
    
    def __init__(self):
        
        self.bin_as_num = None
    
    def fit(self, df):
        
        self.bin_as_num = {'A': 0, 'B': 1}
    
    def transform(self, df):
        
        df['Ecology_2'] = df['Ecology_2'].map(self.bin_as_num)
        df['Ecology_3'] = df['Ecology_3'].map(self.bin_as_num)
        df['Shops_2'] = df['Shops_2'].map(self.bin_as_num)
    
        return df
corrector = FeatureCorrector()

corrector.fit(train)

train = corrector.transform(train)
test = corrector.transform(test)
feature_gen = FeatureGenerator()

feature_gen.fit(train)

train = feature_gen.transform(train)
test = feature_gen.transform(test)
# заполнение полей NaN
train = train.fillna(method='pad')
test = test.fillna(method='pad')
y = train.Price
train.drop("Price", axis=1, inplace=True)
train.drop("Id", axis=1, inplace=True)
y
plt.figure(figsize=(15, 8))
plt.scatter(y, train.Square)
plt.ylabel('Price')
plt.xlabel('Square')
plt.show()
train = pd.get_dummies(train)
train.info()
rfr = RandomForestRegressor(n_estimators=300, max_depth=18, random_state=42, max_features=7)

cros_val = cross_val_score(rfr, train, y, 
                           scoring='r2', 
                           cv=KFold(n_splits=5, shuffle=True, random_state=42))
# среднее и ошибка
mean = cros_val.mean()
std = cros_val.std()

print(f'R2: {round(mean, 5)} +- {round(std, 5)}')
lgbmr = LGBMRegressor(max_depth=6,
                             num_leaves=12,
                             n_estimators=300,
                             random_state=42)

cros_val = cross_val_score(lgbmr, train, y, 
                           scoring='r2', 
                           cv=KFold(n_splits=5, shuffle=True, random_state=42))
# среднее и ошибка
mean = cros_val.mean()
std = cros_val.std()

print(f'R2: {round(mean, 5)} +- {round(std, 5)}')
gbr = ensemble.GradientBoostingRegressor(n_estimators=300, max_depth=6, min_samples_split=2,
                                           learning_rate=0.1, loss='ls', random_state=42)
cros_val = cross_val_score(gbr, train, y, 
                           scoring='r2', 
                           cv=KFold(n_splits=5, shuffle=True, random_state=42))
# среднее и ошибка
mean = cros_val.mean()
std = cros_val.std()

print(f'R2: {round(mean, 5)} +- {round(std, 5)}')
model = VotingRegressor([('gbr', gbr), ('rfr', rfr), ('lgbmr', lgbmr)])
cros_val = cross_val_score(model, train, y, 
                           scoring='r2', 
                           cv=KFold(n_splits=5, shuffle=True, random_state=42))
# среднее и ошибка
mean = cros_val.mean()
std = cros_val.std()

print(f'R2: {round(mean, 5)} +- {round(std, 5)}')
model.fit(train, y)
X_test = pd.get_dummies(test)
X_test.drop("Id", axis=1, inplace=True)



test["Price"] = model.predict(X_test)
test.loc[:, ['Id', 'Price']].to_csv('final_prediction.csv', index=False)
test.loc[:, ['Id', 'Price']].head(20)