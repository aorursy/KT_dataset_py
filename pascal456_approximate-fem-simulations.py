# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

n_jobs = -2
cv=4

df_1000 = pd.read_csv("/kaggle/input/fem-simulations/1000randoms.csv")

test_size = 0.25
random_state = 42
features = ['ecc', 'N', 'gammaG', 'Esoil', 'Econc', 'Dbot', 'H1', 'H2', 'H3']
label = ['Mr_t', 'Mt_t', 'Mr_c', 'Mt_c']

X = df_1000[features]
y = df_1000[label]
# take only the first label ('Mr_t'), for demonstration
X_train, X_test, y_train, y_test = train_test_split(
        X, y.iloc[:,0],
        test_size = test_size,
        random_state = random_state
    )

pipe = make_pipeline(
    PowerTransformer(),
    xgb.XGBRegressor()
)

param_grid = {
    pipe.steps[-1][0] + '__' + 'max_depth': np.arange(2,6),
    pipe.steps[-1][0] + '__' + 'n_estimators': np.arange(100,1001,100)
}
model = GridSearchCV(
        pipe,
        param_grid = param_grid,
        n_jobs = n_jobs,
        cv = cv,
        verbose = 1
    )
model.fit(X_train, y_train)
print('--> best params:', model.best_params_)
y_hat = model.predict(X_test)

results = pd.DataFrame(
            {
                'R-Squared TRAIN': [r2_score(y_train, model.predict(X_train))],
                'R-squared TEST': [r2_score(y_test, y_hat)],
                'MAE TEST': [mean_absolute_error(y_test, y_hat)],
                'MSE TEST': [mean_squared_error(y_test, y_hat)],
                'RMSE TEST': [np.sqrt(mean_squared_error(y_test, y_hat))]
            },
            index=[pipe.steps[-1][0]]
        ).T
results
df_5184 = pd.read_csv("/kaggle/input/fem-simulations/5184doe.csv")

X_additional = df_5184[features]
y_additional = df_5184[label]
y_additional = y_additional.iloc[:,0]

y_hat_additional = model.predict(X_additional)

results = pd.DataFrame(
            {
                'R-squared': [r2_score(y_additional, y_hat_additional)],
                'MAE': [mean_absolute_error(y_additional, y_hat_additional)],
                'MSE': [mean_squared_error(y_additional, y_hat_additional)],
                'RMSE': [np.sqrt(mean_squared_error(y_additional, y_hat_additional))]
            },
            index=['additional ' + pipe.steps[-1][0]]
        ).T
results
span = max(y_additional) - min(y_additional)
rmse = np.sqrt(mean_squared_error(y_additional, y_hat_additional))
rmse_prop = 100 * (rmse / span)

print("span: ", span)
print("RMSE: ", np.sqrt(mean_squared_error(y_additional, y_hat_additional)))
print("therefore, the proportional RMSE is: {:.2f}%".format(rmse_prop))