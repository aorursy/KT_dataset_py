# imports
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

# models
import sklearn.model_selection
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# data import
data0 = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
data0.head()
data0.info()
plt.figure(figsize = (10, 15))
sns.heatmap(data0.isnull(), cbar=False)
data0.describe(include='all')
data0.columns
cols_to_drop = ['url', 'vin', 'image_url', 'region_url', # скорее всего бесполезны, разве что отдельно обрабатывать изображения, но.. нет
                'county',      # всё пусто
                'lat', 'long', # геоданные
                'description', # в рамках этой задачи работать с текстом не будем
                'size'         # слишком много пустых значений. 
               ]
for i in cols_to_drop:
    if i in data0.columns:
        data0.drop(columns=[i], inplace=True)
data0.columns
data0.head()
data0.title_status.unique()
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data0.corr(), annot=True, linewidths=.1, cmap="coolwarm", square=True)
data0.info()
sns.kdeplot(data=data0['price'],label="Price" ,shade=True)
_ = sns.boxplot(data0.price)
_ = sns.boxplot(data0[data0.price<=50_000].price)
p99 = data0.price.quantile(0.99)
p99
fig, ax = plt.subplots(figsize=(18, 10))
_ = sns.boxplot(x=data0.price, y=data0.manufacturer, orient='h')
fig, ax = plt.subplots(figsize=(18, 10))
_ = sns.boxplot(x=data0[data0.price<=200_000].price, y=data0.manufacturer, orient='h')
data01 = data0[~data0['manufacturer'].isin(['tesla', 'ferrari', 'aston-martin'])]
print(f'99% обычных моделей дешевле {data01.price.quantile(0.99)}')
data01 = data01[data01.price <= data01.price.quantile(0.99)]
# обычные модели
fig, ax = plt.subplots(figsize=(18, 10))
_ = sns.boxplot(x=data01.price, y=data01.manufacturer, orient='h')
data02 = data0[data0['manufacturer'].isin(['tesla', 'ferrari', 'aston-martin'])]
print(f'99% дорогих моделей дешевле {data02.price.quantile(0.99)}')
data02 = data02[data02.price <= data02.price.quantile(0.99)]
# дорогие модели
fig, ax = plt.subplots(figsize=(18, 10))
_ = sns.boxplot(x=data02.price, y=data02.manufacturer, orient='h')
data1 = pd.concat([data01, data02])
print(data1.shape)
data1.head(1)
from sklearn.preprocessing import LabelEncoder
# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
data = data1.copy()
features = data.columns.values.tolist()
for col in features:
    if data[col].dtype in numerics: continue
    categorical_columns.append(col)
print('cat cols:', categorical_columns)
# Encoding categorical features
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        le.fit(list(data[col].astype(str).values))
        data[col] = le.transform(list(data[col].astype(str).values))
data.drop(columns=['id'], inplace=True)
data.head()
data.dropna().shape
data = data.dropna()
data.price.describe()
# почему-то такие есть, вряд ли нам стоит их предсказывать
data = data[data.price!=0]
fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.1, cmap="coolwarm", square=True)
_ = data.hist(data.columns, figsize=(14, 14), bins=25)
data.odometer.value_counts()
data.year.hist()
#import pandas_profiling as pp
#pp.ProfileReport(data)
X = data.drop(columns=['price'])
y = data['price']
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
X.head(3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
test_predictions = regressor.predict(X_test)

print('r2: ', regressor.score(X_test,y_test))
X = data.drop(columns=['price'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
# перебираем глубину
# перебираем мин кол-во для разделения
# максимально кол-во признаков для более случайной выборки
param_grid = {'max_depth': [i for i in range(13, 15)],
              'min_samples_split': [i for i in range(2, 5)],
              'max_features': [2, len(X.columns)-1]
             }

# инициализируем случайный лес с перебором по кросс-вал на выбранных выше праметрах
gs = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid, verbose=2)
gs.fit(X_train, y_train)

# best_params_ содержит в себе лучшие подобранные параметры, best_score_ лучшее качество
print()
gs.best_params_, gs.best_score_
def plot_feature_importances(model, columns):
    # выбираем кол-во признаков для отображения
    nr_f = 10
    # берем данные для графика
    # берем алгоритм с лучшими параметрами
    # по х мы берем важность признаков исходя из критерия Джини, сортируем их вместе с названиями колонок по возрастанию
    imp = pd.Series(data = model.best_estimator_.feature_importances_, 
                    index=columns).sort_values(ascending=False)
    # построили фигуру
    plt.figure(figsize=(7,5))
    # отобразили название
    plt.title("Важность признаков | Feature importance")
    # построили 10 самых важных признаков
    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')


# вызвали функцию отображения самых важных признаков
# тут dummies_columns - название всех колонок, типа encoded_df.columns
plot_feature_importances(gs, X_train.columns)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfc = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
results = cross_val_score(rfc, X_train, y_train, cv=skf)
print("CV accuracy score: {:.2f}%".format(results.mean()*100))

X = data.drop(columns=['price'])
y = np.asarray(data['price'])
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Создаем списки для сохранения точности на тренировочном и тестовом датасете
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

# Обучаем на тренировочном датасете
for ntrees in trees_grid:
    rfc = RandomForestRegressor(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best accuracy on CV is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100, 
                                                        trees_grid[np.argmax(test_acc.mean(axis=1))]))
plt.style.use('ggplot')
%matplotlib inline

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("N_estimators")
