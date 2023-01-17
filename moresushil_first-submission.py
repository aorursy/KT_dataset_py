import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Fill empty data with mean values

X_test_full.fillna(X_test_full.mean(), inplace=True)

X_full.fillna(X_full.mean(), inplace=True)





# Obtain target and predictors

y = X_full.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = X_full[features].copy()

X_test = X_test_full[features].copy()



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=10, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]
X_train.head()
from sklearn.ensemble import RandomForestRegressor





model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

model_6 = RandomForestRegressor(n_estimators=5000, random_state=2)

model_7 = RandomForestRegressor(n_estimators=1000, random_state=2)

model_8 = RandomForestRegressor(n_estimators=1000, criterion='mae', random_state=2)

model_9 = RandomForestRegressor(n_estimators=2000, min_samples_split=2, random_state=2)

model_10 = RandomForestRegressor(n_estimators=1000, max_depth=7, random_state=2)



models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]
from sklearn.metrics import mean_absolute_error





def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))
best_model = model_10
my_model = best_model
preds_test = my_model.predict(X_test)



output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
