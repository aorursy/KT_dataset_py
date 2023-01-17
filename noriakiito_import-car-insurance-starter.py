# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
train_df
train_df = train_df.replace('?', np.NaN)
print(train_df.isnull().sum())
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
train_df['normalized-losses'] = imputer.fit_transform(train_df['normalized-losses'].values.reshape(-1, 1))
train_df['bore'] = imputer.fit_transform(train_df['bore'].values.reshape(-1, 1))
train_df['stroke'] = imputer.fit_transform(train_df['stroke'].values.reshape(-1, 1))
train_df['horsepower'] = imputer.fit_transform(train_df['horsepower'].values.reshape(-1, 1))
train_df['peak-rpm'] = imputer.fit_transform(train_df['peak-rpm'].values.reshape(-1, 1))
train_df['price'] = imputer.fit_transform(train_df['price'].values.reshape(-1, 1))
train_df = train_df.drop('num-of-cylinders', axis=1)
train_df = train_df.drop('fuel-system', axis=1)
train_df = train_df.drop('engine-type', axis=1)
train_df = train_df.drop('make', axis=1)
train_df = train_df.drop('stroke', axis=1)
train_df = train_df.drop('horsepower', axis=1)
train_df = train_df.drop('city-mpg', axis=1)
train_df = train_df.drop('compression-ratio', axis=1)
train_df = train_df.drop('aspiration', axis=1)
train_df = pd.get_dummies(train_df, drop_first=True)
train_df
test_df
test_df = test_df.replace('?', np.NaN)
print(test_df.isnull().sum())
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
test_df['normalized-losses'] = imputer.fit_transform(test_df['normalized-losses'].values.reshape(-1, 1))
test_df['bore'] = imputer.fit_transform(test_df['bore'].values.reshape(-1, 1))
test_df['stroke'] = imputer.fit_transform(test_df['stroke'].values.reshape(-1, 1))
test_df['horsepower'] = imputer.fit_transform(test_df['horsepower'].values.reshape(-1, 1))
test_df['peak-rpm'] = imputer.fit_transform(test_df['peak-rpm'].values.reshape(-1, 1))
test_df['price'] = imputer.fit_transform(test_df['price'].values.reshape(-1, 1))
test_df = test_df.drop('num-of-cylinders', axis=1)
test_df = test_df.drop('fuel-system', axis=1)
test_df = test_df.drop('engine-type', axis=1)
test_df = test_df.drop('make', axis=1)
test_df = test_df.drop('stroke', axis=1)
test_df = test_df.drop('horsepower', axis=1)
test_df = test_df.drop('city-mpg', axis=1)
test_df = test_df.drop('compression-ratio', axis=1)
test_df = test_df.drop('aspiration', axis=1)
test_df = pd.get_dummies(test_df, drop_first=True)
test_df
train_df = train_df.drop('drive-wheels_rwd', axis=1)
test_df = test_df.drop('drive-wheels_rwd', axis=1)
train_df = train_df.drop('width', axis=1)
test_df = test_df.drop('width', axis=1)
train_df = train_df.drop('length', axis=1)
test_df = test_df.drop('length', axis=1)
train_df = train_df.drop('curb-weight', axis=1)
test_df = test_df.drop('curb-weight', axis=1)
train_df = train_df.drop('price', axis=1)
test_df = test_df.drop('price', axis=1)
train_df = train_df.drop('bore', axis=1)
test_df = test_df.drop('bore', axis=1)
train_df = train_df.drop('engine-location_rear', axis=1)
test_df = test_df.drop('engine-location_rear', axis=1)
train_df = train_df.drop('fuel-type_gas', axis=1)
test_df = test_df.drop('fuel-type_gas', axis=1)
train_df = train_df.drop('body-style_hardtop', axis=1)
test_df = test_df.drop('body-style_hardtop', axis=1)
train_df = train_df.drop('body-style_wagon', axis=1)
test_df = test_df.drop('body-style_wagon', axis=1)
train_df = train_df.drop('highway-mpg', axis=1)
test_df = test_df.drop('highway-mpg', axis=1)
train_df = train_df.drop('peak-rpm', axis=1)
test_df = test_df.drop('peak-rpm', axis=1)
import matplotlib.pyplot as plt
import seaborn as sns

corr = train_df.corr()

plt.style.use('ggplot')
plt.figure(figsize=(15, 15)) 
sns.heatmap(corr, square=True, annot=True)
plt.show()
train_df['normalized-losses'] = train_df['normalized-losses'].apply(np.log)
test_df['normalized-losses'] = test_df['normalized-losses'].apply(np.log)
train_df['engine-size'] = train_df['engine-size'].apply(np.log)
test_df['engine-size'] = test_df['engine-size'].apply(np.log)
train_df['wheel-base'] = train_df['wheel-base'].apply(np.log)
test_df['wheel-base'] = test_df['wheel-base'].apply(np.log)
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')
plt.figure()
sns.pairplot(train_df)
plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_y = train_df['symboling'].to_numpy()
x = train_df.drop('symboling', axis=1)
train_df = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
train_X = x.to_numpy()
X_test = test_df.to_numpy()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.3)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor

estimators = [
        ('svr', SVR()),
        ('xgb', xgb.XGBRegressor(random_state=0))
        ]

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
)
reg.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = reg.predict(X_valid)  # 予測
np.sqrt(mean_squared_error(y_valid, y_pred))  # 評価
from sklearn.metrics import mean_squared_error

y_pred = reg.predict(train_X)  # 予測
np.sqrt(mean_squared_error(train_y, y_pred))  # 評価
p_test = reg.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)
submit_df['symboling'] = p_test
submit_df
submit_df.to_csv('submission13.csv')

