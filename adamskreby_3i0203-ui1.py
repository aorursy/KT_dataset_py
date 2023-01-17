import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn import svm
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
kbins = KBinsDiscretizer(29, encode="ordinal")
y_stratify = kbins.fit_transform(df[['SalePrice']])
#df_train, df_test = train_test_split(df, test_size=0.35, stratify=y_stratify, random_state=4)
DF_TEST = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
categorical_inputs = []
numeric_inputs = []
output = ["SalePrice"]

for i in df.iloc[:, :-1]:
    try:
        corr = df.iloc[:, -1].corr(df[i])
        #if corr > 0.5 or corr < -0.5:
        if df[i].isnull().sum() < 200:
            numeric_inputs.append(i)
    except:
        if df[i].isnull().sum() < 200:
            categorical_inputs.append(i)
input_preproc = make_column_transformer(
    (make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown='ignore')),
    categorical_inputs),

    (make_pipeline(
        SimpleImputer(),
        StandardScaler()),
    numeric_inputs)
)

output_preproc = StandardScaler()
X_train = input_preproc.fit_transform(df[numeric_inputs+categorical_inputs])
Y_train = output_preproc.fit_transform(df[output]).ravel()
#X_traintest = input_preproc.transform(df_test[numeric_inputs+categorical_inputs])
#Y_traintest = output_preproc.transform(df_test[output])
X_test = input_preproc.transform(DF_TEST[numeric_inputs+categorical_inputs])
lgbm_values = {'learning_rate':[0.03,0.04,0.05,0.06,0.07],
               'max_depth':[18,19,20,21,22]}
xgb_values = {'eta':[0.08,0.09,0.1,0.11,0.12],
              'max_depth':[3,4,5,6,7,8,9,10]}
lasso_values = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
ridge_values = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

lgbm = GridSearchCV(LGBMRegressor(), param_grid = lgbm_values, scoring = 'neg_mean_squared_error', n_jobs=-1).fit(X_train, Y_train)
xgb = GridSearchCV(XGBRegressor(), param_grid = xgb_values, scoring = 'neg_mean_squared_error', n_jobs=-1).fit(X_train, Y_train)
lasso = GridSearchCV(Lasso(), param_grid = lasso_values, scoring = 'neg_mean_squared_error', cv=5).fit(X_train, Y_train)
ridge = GridSearchCV(Ridge(), param_grid = ridge_values, scoring = 'neg_mean_squared_error', cv=5).fit(X_train, Y_train)
model = LGBMRegressor()
model.fit(X_train, Y_train)

model = XGBRegressor()
model.fit(X_train, Y_train)

estimators = [
    ("xgb", xgb.best_estimator_),
    ("lgbm", lgbm.best_estimator_),
    ('svc', svm.SVR(gamma='auto')),
    ('lss', lasso.best_estimator_),
    ('rdg', ridge.best_estimator_),
]

model = VotingRegressor(estimators)
model.fit(X_train, Y_train)

#y_traintest = model.predict(X_traintest)
"""
y_train = model.predict(X_train)
mse_train = mean_squared_error(Y_train, y_train)
print("MSE train: {}".format(mse_train))
mae_train = mean_absolute_error(Y_train, y_train)
print("MAE train: {}".format(mae_train))
"""
"""
mse = mean_squared_error(Y_traintest, y_traintest)
print("MSE: {}".format(mse))
mae = mean_absolute_error(Y_traintest, y_traintest)
print("MAE: {}".format(mae))
"""
"""
plt.scatter(y_traintest, Y_traintest, alpha=.7,
            color='r')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Model')
plt.show()
"""
y_test = model.predict(X_test)
y_test = output_preproc.inverse_transform(y_test)
y_test = np.column_stack((DF_TEST.iloc[:,0],y_test))
y_test = pd.DataFrame({'Id':y_test[:,0].astype(int),'SalePrice':y_test[:,1]})
y_test.to_csv('Prices.csv', index=False)