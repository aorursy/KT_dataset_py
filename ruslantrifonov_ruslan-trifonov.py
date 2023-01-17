path_data_train = '../input/realestatepriceprediction/train.csv'



path_data_test = '../input/realestatepriceprediction/test.csv'
import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

from sklearn.metrics import r2_score

import random

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import KFold, GridSearchCV
data_train = pd.read_csv(path_data_train)

data_train
data_train.info()
data_test = pd.read_csv(path_data_test)

data_test
data_test.info()
data_train.describe()
categorical_columns = [c for c in data_train.columns if data_train[c].dtype.name == 'object']

numerical_columns   = [c for c in data_train.columns if data_train[c].dtype.name != 'object']

print("Категориальные данные:\n{}".format(categorical_columns))

print("Числовые данные данные:\n{}".format(numerical_columns))
data_train[categorical_columns].describe()
for c in categorical_columns:

    print(data_train[c].unique())
data_train[numerical_columns].describe()
Price_Id_DistrictId = data_train[['Price','Id','DistrictId']]

px.scatter_matrix(Price_Id_DistrictId, dimensions = ['Price','Id','DistrictId'],color="Price")
%%time

Price_Rooms_Square_LifeSquare_KitchenSquare_Floor_HouseFloor = data_train[['Price','Rooms','Square','LifeSquare','KitchenSquare','Floor','HouseFloor']]

px.scatter_matrix(Price_Rooms_Square_LifeSquare_KitchenSquare_Floor_HouseFloor, dimensions = ['Price','Rooms','Square','LifeSquare','KitchenSquare','Floor','HouseFloor'],color="Price")
Price_HouseYear = data_train[['Price','HouseYear']]

px.scatter_matrix(Price_HouseYear, dimensions = ['Price','HouseYear'],color="Price")
Price_Ecology_1 = data_train[['Price','Ecology_1']]

px.scatter_matrix(Price_Ecology_1, dimensions = ['Price','Ecology_1'],color="Price")
Price_Social_1_Social_2_Social_3 = data_train[['Price','Social_1','Social_2','Social_3']]

px.scatter_matrix(Price_Social_1_Social_2_Social_3, dimensions = ['Price','Social_1','Social_2','Social_3'],color="Price")
Price_Healthcare_1_Helthcare_2= data_train[['Price','Healthcare_1','Helthcare_2']]

px.scatter_matrix(Price_Healthcare_1_Helthcare_2, dimensions = ['Price','Healthcare_1','Helthcare_2'],color="Price")

Price_Shops_1 = data_train[['Price','Shops_1']]

px.scatter_matrix(Price_Shops_1, dimensions = ['Price','Shops_1'],color="Price")
print("Статистические характеристики признака LifeSquare:\n{}".format(data_train['LifeSquare'].describe()))

print("Статистические характеристики Healthcare_1:\n{}".format(data_train['Healthcare_1'].describe()))

print("Уникальные значения признака LifeSquare:\n{}".format(data_train['LifeSquare'].unique()))

print("Уникальные значения признака Healthcare_1:\n{}".format(data_train['Healthcare_1'].unique()))
data_train_outliers = data_train.copy()
LifeSquare_mean = data_train_outliers["LifeSquare"].mean()

Healthcare_1_median = data_train_outliers["Healthcare_1"].median()



data_train_outliers["LifeSquare"].fillna(LifeSquare_mean, inplace=True)

data_train_outliers["Healthcare_1"].fillna(Healthcare_1_median, inplace=True)



print("Среднее признака LifeSquare:\n{}".format(LifeSquare_mean))

print("Медиана признака Healthcare_1:\n{}".format(Healthcare_1_median))

print(data_train_outliers.info())
data_train_outliers['Rooms'].value_counts()
data_train_outliers.loc[data_train_outliers['Rooms'].isin([0, 10, 19]), 'Rooms'] = data_train_outliers['Rooms'].median()
data_train_outliers.describe()
steps = []

scores = [] # <- записываем финальный score
steps.append('обработка пропусков, выбросов var1')



data_train_outliers = data_train_outliers[data_train_outliers['Square'].isnull() |

                     (data_train_outliers['Square'] < data_train_outliers['Square'].quantile(.99)) &

                     (data_train_outliers['Square'] > data_train_outliers['Square'].quantile(.01))]



data_train_outliers = data_train_outliers[data_train_outliers['LifeSquare'].isnull() |

                      (data_train_outliers['LifeSquare'] < data_train_outliers['LifeSquare'].quantile(.99)) &

                      (data_train_outliers['LifeSquare'] >data_train_outliers['LifeSquare'].quantile(.01))]



data_train_outliers = data_train_outliers[data_train_outliers['KitchenSquare'].isnull() |

                    (data_train_outliers['KitchenSquare'] < data_train_outliers['KitchenSquare'].quantile(.99)) &

                    (data_train_outliers['KitchenSquare'] > data_train_outliers['KitchenSquare'].quantile(.01))]
# steps.append('обработка пропусков, выбросов var2')



"""

...

...

...

"""
data_train_outliers.describe()
data_train_outliers.loc[data_train_outliers['LifeSquare'] < 10, 'LifeSquare'] = 10
data_train_outliers.loc[data_train_outliers['KitchenSquare'] < 3, 'KitchenSquare'] = 3
data_train_outliers['HouseFloor'].sort_values().unique()
data_train_outliers['Floor'].sort_values().unique()
data_train_outliers.loc[data_train_outliers['HouseFloor'] == 0, 'HouseFloor'] = data_train_outliers['HouseFloor'].median()
floor_outliers = data_train_outliers[data_train_outliers['Floor'] > data_train_outliers['HouseFloor']].index



data_train_outliers.loc[floor_outliers, 'Floor'] = data_train_outliers.loc[floor_outliers, 'HouseFloor'].apply(lambda x: random.randint(1, x))

data_train_outliers['HouseYear'].sort_values().unique()
data_train_outliers.loc[data_train_outliers['HouseYear'] == 20052011, 'HouseYear'] = 2011
data_train_outliers.loc[data_train_outliers['HouseYear'] > 2020, 'HouseYear'] = 2020
print("Значениея признака Ecology_2:\n{}".format(data_train['Ecology_2'].value_counts()))

print("Значениея признака Ecology_3:\n{}".format(data_train['Ecology_3'].value_counts()))

print("Значениея признака Shops_2:\n{}".format(data_train['Shops_2'].value_counts()))
print("Уникальные значения признака Ecology_2:\n{}".format(data_train['Ecology_2'].unique()))

print("Уникальные значения признака Ecology_3:\n{}".format(data_train['Ecology_3'].unique()))

print("Уникальные значения признака Shops_2:\n{}".format(data_train['Shops_2'].unique()))
category_A_B_map =  { 'A': 0,

                      'B': 1,}



data_train_outliers['Ecology_2'] = data_train_outliers['Ecology_2'].map(category_A_B_map)

data_train_outliers['Ecology_3'] = data_train_outliers['Ecology_3'].map(category_A_B_map)

data_train_outliers['Shops_2'] = data_train_outliers['Shops_2'].map(category_A_B_map)



print("Уникальные значения признака Ecology_2:\n{}".format(data_train_outliers['Ecology_2'].unique()))

print("Уникальные значения признака Ecology_3:\n{}".format(data_train_outliers['Ecology_3'].unique()))

print("Уникальные значения признака Shops_2:\n{}".format(data_train_outliers['Shops_2'].unique()))
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()



# from sklearn.preprocessing import RobustScaler

# scaler = RobustScaler()



# from sklearn.preprocessing import Normalizer

# scaler = Normalizer()



# from sklearn.preprocessing import PowerTransformer

# scaler = PowerTransformer()



from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer(n_quantiles=20, random_state=42) # Лучшие результаты



# from sklearn.preprocessing import MaxAbsScaler

# scaler =MaxAbsScaler()
data_train_outliers.columns.tolist()
feature_names = ['Id',

 'DistrictId',

 'Rooms',

 'Square',

 'LifeSquare',

 'KitchenSquare',

 'Floor',

 'HouseFloor',

 'HouseYear',

 'Ecology_1',

 'Ecology_2',

 'Ecology_3',

 'Social_1',

 'Social_2',

 'Social_3',

 'Healthcare_1',

 'Helthcare_2',

 'Shops_1',

 'Shops_2',

 'Price']



feature_names.remove('Price')



feature_names
target_data =data_train_outliers['Price']



data_train_for_scaling = scaler.fit_transform(data_train_outliers[feature_names].astype(float))



data_train_scaled = pd.DataFrame(data_train_for_scaling, columns=feature_names)



data_train_scaled.head()
feature_names = ['Rooms', 'Square', 'LifeSquare', 'KitchenSquare',

       'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Ecology_2',

       'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1',

       'Helthcare_2', 'Shops_1', 'Shops_2']
from sklearn.model_selection import train_test_split
X = data_train_scaled[feature_names]

y = target_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=45)
import seaborn as sns

from sklearn.metrics import r2_score as r2



def evaluate_preds(train_true_values, train_pred_values, val_true_values, val_pred_values):

    """

    Функция для оценки работы модели

    Parameters:

    train_true_values - целевая переменная из тренировочной части датасета

    train_pred_values - предсказания модели по тренировочной части

    val_true_values - целевая переменная из валидационной части датасета

    val_pred_values - предсказания модели по валидационной части

    Returns:

    R2 на тренировочной и валидационной части, 

    графики зависимости истинных значений от предсказаний

    """

    print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))

    print("Valid R2:\t" + str(round(r2(val_true_values, val_pred_values), 3)))

    

    plt.figure(figsize=(18,10))

    

    plt.subplot(121)

    sns.scatterplot(x=train_pred_values, y=train_true_values)

    plt.xlabel('Predicted values')

    plt.ylabel('True values')

    plt.title('Train sample prediction')

    

    plt.subplot(122)

    sns.scatterplot(x=val_pred_values, y=val_true_values)

    plt.xlabel('Predicted values')

    plt.ylabel('True values')

    plt.title('Test sample prediction')

    plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train, y_train)
y_pred_train = lr.predict(X_train)

y_pred = lr.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(lr, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.0001).fit(X_train, y_train)
y_pred_train = ridge.predict(X_train)

y_pred = ridge.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(lr, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1, max_iter=500).fit(X_train, y_train)
y_pred_train = lasso.predict(X_train)

y_pred = lasso.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(lasso, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
y_pred_train = reg.predict(X_train)

y_pred = reg.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(reg, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42).fit(X_train, y_train)
y_pred_train = regr.predict(X_train)

y_pred = regr.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(regr, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.tree import ExtraTreeRegressor

from sklearn.ensemble import BaggingRegressor

extra_tree = ExtraTreeRegressor(random_state=42)

reg = BaggingRegressor(extra_tree, random_state=22).fit(X_train, y_train)
y_pred_train = regr.predict(X_train)

y_pred = regr.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(regr, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.linear_model import PassiveAggressiveRegressor

regr = PassiveAggressiveRegressor(max_iter=100000, random_state=42, tol=1e-3).fit(X_train, y_train)
y_pred_train = regr.predict(X_train)

y_pred = regr.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(reg, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.ensemble import GradientBoostingRegressor

gbt = GradientBoostingRegressor(random_state=42, n_estimators=322).fit(X_train, y_train)
y_pred_train = gbt.predict(X_train)

y_pred = gbt.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(gbt, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.linear_model import RANSACRegressor

reg = RANSACRegressor(random_state=42).fit(X_train, y_train)
y_pred_train = reg.predict(X_train)

y_pred = reg.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(reg, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor(alpha=100).fit(X_train, y_train)
y_pred_train = huber.predict(X_train)

y_pred = huber.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(huber, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor(penalty='elasticnet', alpha=0.0001, l1_ratio=0.25, tol=1e-4).fit(X_train, y_train)
y_pred_train = sgdr.predict(X_train)

y_pred = sgdr.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(sgdr, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.svm import SVR

clf = SVR(C=55, epsilon=0.0001, gamma ='auto').fit(X_train, y_train)
y_pred_train = clf.predict(X_train)

y_pred = clf.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(clf, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
from sklearn.neural_network import MLPRegressor

mlpr = MLPRegressor(hidden_layer_sizes=220, 

                    alpha = 0.0001, 

                    activation = 'logistic', 

                    random_state = 42).fit(X_train, y_train)
y_pred_train = mlpr.predict(X_train)

y_pred = mlpr.predict(X_test)
print("R^2 на тренировочной выборке: {}".format(r2_score(y_train, y_pred_train)))

print("R^2 на тестовой выборке: {}".format(r2_score(y_test, y_pred)))
evaluate_preds(y_train, y_pred_train, y_test, y_pred)
cv_score = cross_val_score(mlpr, X, y, scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=21))

cv_score
data_test.head()
data_test_prep = data_test.copy()
LifeSquare_mean = data_test_prep["LifeSquare"].mean()

Healthcare_1_median = data_test_prep["Healthcare_1"].median()



data_test_prep["LifeSquare"].fillna(LifeSquare_mean, inplace=True)

data_test_prep["Healthcare_1"].fillna(Healthcare_1_median, inplace=True)



print("Среднее признака LifeSquare:\n{}".format(LifeSquare_mean))

print("Медиана признака Healthcare_1:\n{}".format(Healthcare_1_median))

print(data_test_prep.info())
category_A_B_map =  { 'A': 0,

                      'B': 1,}



data_test_prep['Ecology_2'] = data_test_prep['Ecology_2'].map(category_A_B_map)

data_test_prep['Ecology_3'] = data_test_prep['Ecology_3'].map(category_A_B_map)

data_test_prep['Shops_2'] = data_test_prep['Shops_2'].map(category_A_B_map)



print("Уникальные значения признака Ecology_2:\n{}".format(data_test_prep['Ecology_2'].unique()))

print("Уникальные значения признака Ecology_3:\n{}".format(data_test_prep['Ecology_3'].unique()))

print("Уникальные значения признака Shops_2:\n{}".format(data_test_prep['Shops_2'].unique()))
data_test_prep.keys()
feature_names = ['Id', 'DistrictId', 'Rooms', 'Square', 'LifeSquare', 'KitchenSquare',

       'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Ecology_2',

       'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1',

       'Helthcare_2', 'Shops_1', 'Shops_2']
data_test_for_scaling = scaler.fit_transform(data_test_prep.astype(float))



data_test_scaled = pd.DataFrame(data_test_for_scaling, columns=feature_names)



data_test_scaled.head()
feature_names = ['Rooms', 'Square', 'LifeSquare', 'KitchenSquare',

       'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Ecology_2',

       'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1',

       'Helthcare_2', 'Shops_1', 'Shops_2']
X = data_test_scaled[feature_names]
y_pred = gbt.predict(X)

y_pred
predictions = pd.DataFrame({

    'Id': data_test['Id'],

    'Price':y_pred

})
predictions.head()
# predictions.to_csv('Trifonov_Ruslan_predictions.csv', sep=',', index=False, encoding='utf-8')