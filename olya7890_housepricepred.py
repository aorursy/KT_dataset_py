import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.gridspec as gridspec

from datetime import datetime

from scipy.stats import skew 

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import matplotlib.style as style

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(10)
print("\ncolumn in training data set\n\n",train.columns.values)

print("\ncolumn in testing data set\n\n",test.columns.values)
#получение количества строк и столбцов данных для обучения и тестовых

print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")

print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']



#тренды 

for number in quantitative:

    g = sns.lmplot(x=number, y="SalePrice", data=train, line_kws={'color': 'yellow'})
#Информация о features. 

train.info()
## Gives use the count of different types of objects.

train.dtypes.value_counts()
# Проверка пропущенных значений (для обучающих данных)



def missing_percentage(df):

    

    #таблица возвращает два столбца: 

    # 1. общее количество пропущенных значений

    # 2. общий процент пропущенных значений

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])



missing_percentage(train)
# Проверка пропущенных значений (для тестовых данных)

missing_percentage(test)
# Выводы:

# 1. Есть несколько типов функций

# 2. Некоторые функции имеют пропущенные значения

# 3. Большинство функций типа object
# Построим гистограмму посмотреть нормально ли распределена целевая переменная SalePrice

# Для линейной регрессии важно, чтобы функции были нормально распределены



def plotting_3_chart(df, feature):

    

    import seaborn as sns

    import matplotlib.pyplot as plt

    import matplotlib.gridspec as gridspec

    from scipy import stats

    import matplotlib.style as style

    style.use('fivethirtyeight')

    ## Создание customized chart

    fig = plt.figure(constrained_layout=True, figsize=(8,15))

    # Создание сетки 

    grid = gridspec.GridSpec(ncols=3, nrows=3)

    # Настройка сетки гистограммы 

    ax1 = fig.add_subplot(grid[0, :3])

    ax1.set_title('Проверка данных на нормальность распределения')

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1) 



plotting_3_chart(train, 'SalePrice')
# Выводы:

# 1. Целевая переменная SalePrice распределена не нормально

# 2. Распределение смещено влево

# 3. Есть несколько выбросов
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']



#тренды 

for number in quantitative:

    g = sns.lmplot(x=number, y="SalePrice", data=train, line_kws={'color': 'yellow'})
# Получение отсортированной (от max до min) корреляции всех объектов с целевой переменной

(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]
combine_df = pd.concat([train.drop(['SalePrice'], axis=1), test], axis=0)
y = train['SalePrice'].values
# Набор данных, как обучающих, так и тестовых, имеет множество нулевых значений.  

# Нельзя отбросить все нулевые значения, поскольку некоторые тоже имеют значение.

# Например, нулевые значения для "GarageType" = Нет Garage и т.д.



combine_df['MSZoning'] = combine_df['MSZoning'].fillna(combine_df['MSZoning'].mode()[0])

combine_df["LotFrontage"] = combine_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

combine_df["Alley"] = combine_df["Alley"].fillna("None")

combine_df['MSZoning'] = combine_df['MSZoning'].fillna(combine_df['MSZoning'].mode()[0])

combine_df["LotFrontage"] = combine_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

combine_df["Alley"] = combine_df["Alley"].fillna("None")

combine_df['Utilities'] = combine_df['Utilities'].fillna(combine_df['Utilities'].mode()[0])

combine_df['Exterior1st'] = combine_df['Exterior1st'].fillna(combine_df['Exterior1st'].mode()[0])

combine_df['Exterior2nd'] = combine_df['Exterior2nd'].fillna(combine_df['Exterior2nd'].mode()[0])

combine_df["MasVnrType"] = combine_df["MasVnrType"].fillna(combine_df['MasVnrType'].mode()[0])

combine_df["MasVnrArea"] = combine_df["MasVnrArea"].fillna(combine_df['MasVnrArea'].mode()[0])

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    combine_df[col] = combine_df[col].fillna('None')

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    combine_df[col] = combine_df[col].fillna(0)

combine_df['Electrical'] = combine_df['Electrical'].fillna(combine_df['Electrical'].mode()[0])

combine_df['KitchenQual'] = combine_df['KitchenQual'].fillna(combine_df['KitchenQual'].mode()[0])

combine_df['Functional'] = combine_df['Functional'].fillna(combine_df['Functional'].mode()[0])

combine_df['FireplaceQu'] = combine_df['FireplaceQu'].fillna('None')

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    combine_df[col] = combine_df[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    combine_df[col] = combine_df[col].fillna(0)

combine_df['PoolQC'] = combine_df['PoolQC'].fillna('None')

combine_df['Fence'] = combine_df['Fence'].fillna('None')

combine_df['MiscFeature'] = combine_df['MiscFeature'].fillna('None')

combine_df['SaleType'] = combine_df['SaleType'].fillna(combine_df['SaleType'].mode()[0])

combine_df['MSSubClass'] = combine_df['MSSubClass'].astype(str)

combine_df['OverallCond'] = combine_df['OverallCond'].astype(str)

combine_df['OverallQual'] = combine_df['OverallQual'].astype(str)
# проверяем на нулевые значения

combine_df.isnull().sum().sum()
# Обучение модели для нашей задачи
combine_df = combine_df.drop(['Id'], axis=1)

combine_dummies = pd.get_dummies(combine_df)

result = combine_dummies.values
# Масштабирование

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

result = scaler.fit_transform(result)

X = result[:train.shape[0]]

test_values = result[train.shape[0]:]
# разделение данных 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



clf = LinearRegression()

clf = Lasso()



clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

y_train_pred = clf.predict(X_train)



# R ^ 2 (коэффициент детерминации) функция оценки регрессии.

from sklearn.metrics import r2_score



print("Train R^2: " , r2_score(y_train, y_train_pred))

print("Test R^2: ", r2_score(y_test, y_pred))



# Среднеквадратическая ошибка

from sklearn.metrics import mean_squared_error



print("Train RMSE: " , clf.score(X_train, y_train))

print("Test RMSE: ", clf.score(X_test, y_test))

plt.scatter(y_pred, y_test, alpha=.40, color='red')

plt.xlabel('predicted price')

plt.ylabel('actual sale price ')

plt.show()

final_labels = clf.predict(test_values)
final_labels