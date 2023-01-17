import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/credit-risk-with-label/0c5a90_675dd154afbe46678c781af7cb5849b1.csv')
df.head()
df = df.drop('Unnamed: 0', axis='columns')
df.head()
df.isnull().sum()
df.shape
df['Saving accounts'].value_counts()
df['Saving accounts'].unique()
x = df.drop('Risk', axis='columns')
y = df['Risk']
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df['Risk'])
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
import category_encoders as ce

cat_columns = x.select_dtypes('object').columns
cat_columns
target_enc = ce.TargetEncoder(cols=cat_columns)
target_enc.fit(x_train[cat_columns], y_train)
x_train_final = x_train.join(target_enc.transform(x_train[cat_columns]).add_suffix('_target'))
x_test_final = x_test.join(target_enc.transform(x_test[cat_columns]).add_suffix('_target'))

x_train_final = x_train_final.drop(cat_columns, axis='columns')
x_test_final = x_test_final.drop(cat_columns, axis='columns')
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
my_model = XGBClassifier()

params = {
  'max_depth': range (2, 10, 1),
  'n_estimators': range(20, 320, 20),
  'max_depth' : range(20, 50, 10)}
def tuner(model, x_train, x_test, y_train, y_test, params):  
  grid_model = GridSearchCV(estimator=model,
                           param_grid=params,                           
                           n_jobs=10,
                           cv=10,
                           verbose=True)
  grid_model.fit(x_train_final, y_train)
  predictions = grid_model.predict(x_test_final)

  score = grid_model.score(x_test_final, y_test)
  mse = mean_squared_error(y_test, predictions)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, predictions)

  return score, mse, rmse, mae
result = tuner(my_model, x_train_final, x_test_final, y_train, y_test, params)
result
from catboost import CatBoostClassifier
my_model_cb = CatBoostClassifier()
params_for_catboost = {
  'max_depth': range (2, 10, 1),
  'n_estimators': range(20, 320, 20)}
result_3 = tuner(my_model_cb, x_train_final, x_test_final, y_train,y_test, params_for_catboost)
result_3
