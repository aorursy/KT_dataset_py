# Дан датасет с информацией о жилье

# В файле train.csv находятся данные для обучения

# В файле test.csv - для проверки

# Необходимо научиться прогнозировать цену на жилье на основании тренировочных данных

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor



from xgboost import XGBRegressor
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
data.head()
data.info()
# Приводим типы

data['Ecology_2'] = data['Ecology_2'].astype('category')

data['Ecology_3'] = data['Ecology_3'].astype('category')

data['Shops_2'] = data['Shops_2'].astype('category')

data['Rooms'] = data['Rooms'].astype('int')

data['HouseFloor'] = data['HouseFloor'].astype('int')
# Заполняем пропуски в данных

# Сначала это стоит делать самым простым способом 



data['Healthcare_1'] = data['Healthcare_1'].fillna(data['Healthcare_1'].mean());

data['LifeSquare'] = data['LifeSquare'].fillna(data['LifeSquare'].mean());



test['Healthcare_1'] = data['Healthcare_1'].fillna(test['Healthcare_1'].mean());

test['LifeSquare'] = data['LifeSquare'].fillna(test['LifeSquare'].mean());
data = pd.get_dummies(data)

test = pd.get_dummies(test)
fig, axs = plt.subplots(ncols=2)



sns.scatterplot('Square', 'Price', data=data, ax=axs[0])

sns.boxplot(x="Square", data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
data = data[data['Square'] < 140]

data = data[data['Square'] > 10]



data = data[data['Price'] > 30000]

data = data[data['Price'] < 600000]
fig, axs = plt.subplots(ncols=2)



sns.scatterplot('Square', 'Price', data=data, ax=axs[0])

sns.boxplot(x="Square", data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
# Что делать в таких случаях?

# Надо ли масштабировать test['Square']

# Стоит ли выбрасывать данные из data?



sns.boxplot(x="Square", data=test)

plt.show()
# мы видим, что в да

fig, axs = plt.subplots(ncols=2)



sns.countplot('Floor', data=data, ax=axs[0])

sns.countplot('HouseFloor', data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
# data['Floor'].value_counts()

fig, axs = plt.subplots(ncols=2)



sns.countplot('Floor', data=test, ax=axs[0])

sns.countplot('HouseFloor', data=test, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
data = data[data['HouseFloor'] <= 25]
fig, axs = plt.subplots(ncols=2)



sns.scatterplot('HouseYear', 'Price', data=data, ax=axs[0])

sns.boxplot(x="HouseYear", data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
data = data[data['HouseYear'] > 1900]

data = data[data['HouseYear'] < 2018]
fig, axs = plt.subplots(ncols=2)



sns.scatterplot('HouseYear', 'Price', data=data, ax=axs[0])

sns.boxplot(x="HouseYear", data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
fig, axs = plt.subplots(ncols=2)



sns.scatterplot('Rooms', 'Price', data=data, ax=axs[0])

sns.boxplot(x="Rooms", data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
data = data[data['Rooms'] < 10]
fig, axs = plt.subplots(ncols=2)



sns.scatterplot('Rooms', 'Price', data=data, ax=axs[0])

sns.boxplot(x="Rooms", data=data, ax=axs[1])

fig.set_size_inches(15, 5)

plt.show()
# Также в данных можно увидеть, что есть масса случаев, когда Floor > HouseFloor

# исправим это



np.where(data['Floor'] > data['HouseFloor'], data['Floor'], data['HouseFloor']);
fig, ax = plt.subplots(figsize=(10,8))

corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True, fmt=".1f", linewidths=.5, ax=ax);
# а теперь займемся feature-engineering-ом!

# мы видим, что наибольшее влияние на Price оказывают DistrictId, Rooms и Square

# попробуем смастерить из них одно поле

# ...

# после ряда экспериментов пришел к такому варианту:



def round_square(x, base=20):

    return int(base * round(float(x)/base))



data['Square_Class'] = data['Square'].apply(round_square)

test['Square_Class'] = test['Square'].apply(round_square)
# пришла пора разделить обучающую выборку на тренировочную и валидационную

# и начать обучение
x_train, x_valid, y_train, y_valid = train_test_split(

    data,

    data['Price'],

    test_size=0.2,

    random_state=42

)



mean_price = data['Price'].mean()

grouped = x_train.groupby(['Square_Class', 'Rooms'])[['Price']].mean().reset_index().rename(columns={'Price': 'mean_price'})



x_train_extended = pd.merge(x_train, grouped, on=['Square_Class', 'Rooms'], how='left')

x_valid_extended = pd.merge(x_valid, grouped, on=['Square_Class', 'Rooms'], how='left')



x_valid_extended['mean_price'] = x_valid_extended['mean_price'].fillna(mean_price);
x_train_extended = x_train_extended.drop('Price', axis=1)

x_valid_extended = x_valid_extended.drop('Price', axis=1)



scaler = StandardScaler()

scaler.fit(x_train_extended)



cols = x_train_extended.columns # ???

x_train_extended_scaled = scaler.transform(x_train_extended);
lr = LinearRegression()

reg = lr.fit(x_train_extended_scaled, y_train)



x_valid_extended_scaled = scaler.transform(x_valid_extended)

y_predict = lr.predict(x_valid_extended_scaled)



sqerr = mean_squared_error(y_valid, y_predict)

R2 = r2_score(y_valid, y_predict)



print(f'sqerr: {sqerr}, r2: {R2}')
rfreg = RandomForestRegressor(max_depth=14, random_state=42, n_estimators=1000)

reg2 = rfreg.fit(x_train_extended, y_train)



y_predict = reg2.predict(x_valid_extended)



sqerr = mean_squared_error(y_valid, y_predict)

R2 = r2_score(y_valid, y_predict)



print(f'sqerr: {sqerr}, r2: {R2}')
knn = KNeighborsRegressor(n_neighbors=6, weights='distance')

knn.fit(x_train_extended_scaled, y_train)



x_valid_extended_scaled = scaler.transform(x_valid_extended)

y_predict = knn.predict(x_valid_extended_scaled)



sqerr = mean_squared_error(y_valid, y_predict)

R2 = r2_score(y_valid, y_predict)

print(f'sqerr: {sqerr}, r2: {R2}')
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_model.fit(x_train_extended_scaled, y_train)



x_valid_extended_scaled = scaler.transform(x_valid_extended)

y_predict = xgb_model.predict(x_valid_extended_scaled)



sqerr = mean_squared_error(y_valid, y_predict)

R2 = r2_score(y_valid, y_predict)

print(f'sqerr: {sqerr}, r2: {R2}')
# Лучшие результаты показал XGBRegressor
# предсказание с использованием победителя



# заполняем для тестовых данных колонку mean_price на основании тренировочных данных

test_extended = pd.merge(test, grouped, on=['Square_Class', 'Rooms'], how='left')



# пропущенные строки заполняем средним значением цены

test_extended['mean_price'] = test_extended['mean_price'].fillna(mean_price);



test_predict = xgb_model.predict(test_extended)

test_predict