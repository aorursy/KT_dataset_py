import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import ensemble

from matplotlib import style

from sklearn.ensemble import VotingRegressor

from sklearn.ensemble import RandomForestRegressor



style.use('fivethirtyeight')

%matplotlib inline



TEST_DATASET_PATH = '/kaggle/input/realestatepriceprediction/test.csv'

TRAIN_DATASET_PATH = '/kaggle/input/realestatepriceprediction/train.csv'



test_data = pd.read_csv(TEST_DATASET_PATH)

train_data = pd.read_csv(TRAIN_DATASET_PATH)
# Box plot

plt.figure(figsize=(12, 6))

sns.boxplot(train_data['Rooms'], train_data['Price'])

plt.xlabel('Rooms')

plt.ylabel('Price')

plt.title('Distribution of Price by Rooms')

plt.show();
# отбираем количественные признаки

df_num_features = train_data.select_dtypes(include='float64')



# Удаляем малозначащие столбцы

num_features = pd.DataFrame(df_num_features)

num_features.drop("Ecology_1", axis=1, inplace=True)

num_features.drop("Healthcare_1", axis=1, inplace=True)

num_features.hist(figsize=(10, 8), bins=20, grid=False);
corr = num_features.corr()

plt.figure(figsize = (8, 8))

mask = np.zeros_like(corr, dtype=np.bool)  # отрезаем лишнюю половину матрицы

mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1.4)

sns.heatmap(num_features.corr(), mask=mask, annot=True, fmt='.1f', linewidths=.5, cmap='GnBu')

plt.title('Correlation matrix')

plt.show();
test_data.info()
train_data = train_data.sort_values('Price')

test_data = test_data.sort_values('DistrictId')

train_data = train_data.fillna(method='pad')

test_data = test_data.fillna(method='pad')
from pylab import rcParams

rcParams['figure.figsize'] = 12, 6  # меняем размер графиков по умолчанию



plt.scatter(train_data.Price, train_data.Square)
X_train = pd.get_dummies(train_data)

X_train.drop("Price", axis=1, inplace=True)

X_train.drop("Id", axis=1, inplace=True)

y_train = train_data.Price



model1 = ensemble.GradientBoostingRegressor(n_estimators=442, max_depth=5, min_samples_split=2,

                                           learning_rate=0.1, loss='ls', random_state=42)



model2 = RandomForestRegressor(n_estimators=1442, max_depth=18, random_state=42, max_features=7)



model = VotingRegressor([('model1', model1), ('model2', model2)])



model.fit(X_train, y_train)
X_test = pd.get_dummies(test_data)

X_test.drop("Id", axis=1, inplace=True)

test_data["Price"] = model.predict(X_test)

test_data.loc[:, ['Id', 'Price']].to_csv('VR_predictions.csv', index=False)
# !pip install kaggle

# !kaggle competitions submit -c realestatepriceprediction -m "VotingRegressor" -f VR_predictions.csv