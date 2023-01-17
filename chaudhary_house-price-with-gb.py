import pandas as pd
from sklearn import model_selection

df = pd.read_csv('../input/kc_house_data.csv', parse_dates='date yr_built yr_renovated'.split())

df_with_no_na = df.dropna()

columns_to_use = list(set(df.columns) - set('id price date yr_built yr_renovated zipcode lat long'.split()))

house_data, price = df_with_no_na.loc[:, columns_to_use], df['price']

train_X, test_X, train_y, test_y = model_selection.train_test_split(house_data, price, test_size=0.3)
df.describe()
df.info()
df.tail()
import seaborn as sns

sns.pairplot(house_data)
def train_model(model):

    return model.fit(train_X, train_y)

    

def test_model(model):

    from sklearn import metrics

    y_pred = model.predict(test_X)

    mae = metrics.regression.mean_absolute_error(test_y, y_pred)

    print('MAE: {}'.format(mae))

    mse = metrics.regression.mean_squared_error(test_y, y_pred)

    print('MSE: {}'.format(mse))

    evs = metrics.regression.explained_variance_score(test_y, y_pred)

    print('EVS: {}'.format(evs))

    r2 = metrics.regression.r2_score(test_y, y_pred)

    print('R2: {}'.format(r2))
from sklearn import linear_model

model_linear_regressor = train_model(linear_model.LinearRegression())

test_model(model_linear_regressor)
from sklearn import ensemble

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'criterion':'mse', 'loss': 'ls'}

model_gb_regressor = train_model(ensemble.GradientBoostingRegressor(**params))

test_model(model_gb_regressor)