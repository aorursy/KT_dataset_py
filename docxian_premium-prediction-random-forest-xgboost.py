import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import time



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



import shap



import xgboost as xgb
df = pd.read_csv('../input/insurance-charges/insurance.csv')
df.head()
df.shape
# target

df.charges.plot(kind='hist')

plt.title('Target = charges')

plt.grid()

plt.show()
# target is skewed => add log transformed target

df['log_charges'] = np.log10(df.charges)
# plot log target

df.log_charges.plot(kind='hist')

plt.title('log10(Target)')

plt.grid()

plt.show()
df.age.plot(kind='hist')

plt.title('Age')

plt.grid()

plt.show()
df.bmi.plot(kind='hist')

plt.title('BMI')

plt.grid()

plt.show()
df.sex.value_counts()
df.sex.value_counts().plot(kind='bar')

plt.title('Sex')

plt.grid()

plt.show()
df.children.value_counts()
df.children.value_counts().plot(kind='bar')

plt.title('Children')

plt.grid()

plt.show()
df.smoker.value_counts()
df.smoker.value_counts().plot(kind='bar')

plt.title('Smoker')

plt.grid()

plt.show()
df.region.value_counts()
df.region.value_counts().plot(kind='bar')

plt.title('Region')

plt.grid()

plt.show()
plt.scatter(df.age, df.bmi, alpha=0.25)

plt.title('BMI vs Age')

plt.grid()

plt.show()
df.columns
# define categorical and numeric features

features_cat = ['sex', 'children', 'smoker', 'region']

features_num = ['age', 'bmi']
for f in features_cat:

    plt.figure(figsize=(6,4))

    sns.violinplot(x=f, y='log_charges', data=df)

    my_title = 'Target vs ' + f

    plt.title(my_title)

    plt.grid()
# scatter plot

for f in features_num:    

    plt.scatter(df[f], df.log_charges, alpha=0.25)

    my_title = 'Target vs ' + f

    plt.title(my_title)

    plt.xlabel(f)

    plt.ylabel('log_charges')

    plt.grid()

    plt.show()
# smoothed visualization

for f in features_num:    

    my_title = 'Target vs ' + f

    sns.jointplot(data=df, x=f, y='log_charges', kind='kde')

    plt.title(my_title)

    plt.show()
# The target vs age plot looks nice, there seem to be separate regions.

# Let's have a closer look:

fig = px.scatter(df, x='age', y='log_charges',

                    color='smoker',

                    opacity=0.5)

fig.update_layout(title='Target vs Age / Smoker')

fig.show()
# same for BMI instead of Age

fig = px.scatter(df, x='bmi', y='log_charges',

                    color='smoker',

                    opacity=0.5)

fig.update_layout(title='Target vs BMI / Smoker')

fig.show()
# interactive 3D plot

fig = px.scatter_3d(df, x='age', y='bmi', z='log_charges',

                    color='smoker',

                    opacity=0.5)

fig.update_layout(title='Target vs Age and BMI (smoker via color)')

fig.show()

# select features

features = features_num + features_cat

features
X = df[features]

y = df.log_charges # let's use the log transformed target!
# dummy encoding

X = pd.get_dummies(X)
X.head()
# train / test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=180)
X_train.shape
X_test.shape
# define random forest model (regression)

regRF = RandomForestRegressor(n_estimators=500, max_features=6, random_state=1234)
# fit model

t1 = time.time()

regRF.fit(X_train, y_train)

t2 = time.time()

print('Elapsed time [sec]:', round(t2-t1,3))
# feature importance

feature_imp = pd.Series(regRF.feature_importances_, index=X.columns).sort_values(ascending=False)

feature_imp
# and plot

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.title('Variable Importance - RF')

plt.grid()

plt.show()
# predict values on test set

yhat_test = regRF.predict(X_test)
# metrics (on log transformed data)

print('RMSE Test (log trafo) :', round(np.sqrt(metrics.mean_squared_error(yhat_test, y_test)),6))

print('MAE Test  (log trafo) :', round(metrics.mean_absolute_error(yhat_test, y_test),6))

print('R^2       (log trafo) :', round(metrics.r2_score(yhat_test, y_test),6))
plt.scatter(y_test, yhat_test, alpha=0.5)

plt.title('RF Prediction vs Actual [Test] - log transformed data')

plt.grid()

plt.show()
# transform back to original "coordinates"

yhat_test_orig = 10**yhat_test

y_test_orig = 10**y_test
plt.scatter(y_test_orig, yhat_test_orig, alpha=0.5)

plt.title('RF Prediction vs Actual [Test] - Original Coordinates')

plt.grid()

plt.show()
# metrics in original coordinates

rmse_test_RF = round(np.sqrt(metrics.mean_squared_error(yhat_test_orig, y_test_orig)),6)

mae_test_RF = round(metrics.mean_absolute_error(yhat_test_orig, y_test_orig),6)

R2_test_RF = round(metrics.r2_score(yhat_test_orig, y_test_orig),6)

print('RMSE Test :', rmse_test_RF)

print('MAE Test  :', mae_test_RF)

print('R^2       :', R2_test_RF)
# check calibration

print('Mean actual [test] =        ', y_test_orig.mean())

print('Mean prediction RF [test] = ', yhat_test_orig.mean())

print('Ratio pred/act [test] =     ', yhat_test_orig.mean()/y_test_orig.mean())
# check calibration on training

yhat_train = regRF.predict(X_train)

yhat_train_orig = 10**yhat_train

y_train_orig = 10**y_train



print('Mean actual [train] =        ', y_train_orig.mean())

print('Mean prediction RF [train] = ', yhat_train_orig.mean())

print('Ratio pred/act [train] =     ', yhat_train_orig.mean()/y_train_orig.mean())
# scatter plot on training data

plt.scatter(y_train_orig, yhat_train_orig, alpha=0.5)

plt.title('RF Prediction vs Actual [Train] - Original Coordinates')

plt.grid()

plt.show()
regXGB = xgb.XGBRegressor(             

                    scale_pos_weight=1,

                    learning_rate=0.05,

                    colsample_bytree=0.7,

                    subsample=1,

                    n_estimators=100,

                    reg_alpha=0,

                    max_depth=4,

                    gamma=0,

                    random_state=1234)
n_estimators = [50,100,200,300,400,500]

max_depth = [4,6]

min_child_weight = [1,5,10]

colsample_bytree = [0.5,0.7]



parameter_grid = {'n_estimators': n_estimators,

                  'max_depth': max_depth,

                  'min_child_weight': min_child_weight,

                  'colsample_bytree': colsample_bytree}



print(parameter_grid)
# grid search

grid_cv = GridSearchCV(estimator = regXGB,

                       param_grid = parameter_grid,

                       cv=5,

                       n_jobs=-1,

                       verbose=1)



t1 = time.time()

grid_cv.fit(X_train,y_train)

t2 = time.time()
print('Elapsed time [sec]:', round(t2-t1,3))
# best parameters from grid search

grid_cv.best_params_
regXGB = grid_cv.best_estimator_
feature_impX = pd.Series(regXGB.feature_importances_, index=X.columns).sort_values(ascending=False)

feature_impX
# and plot

sns.barplot(x=feature_impX, y=feature_impX.index)

plt.title('Variable Importance - XGB')

plt.grid()

plt.show()
# predict values on test set

yhat_testX = regXGB.predict(X_test)
# metrics (on log transformed data)

print('RMSE Test (log trafo) :', round(np.sqrt(metrics.mean_squared_error(yhat_testX, y_test)),6))

print('MAE Test  (log trafo) :', round(metrics.mean_absolute_error(yhat_testX, y_test),6))

print('R^2       (log trafo) :', round(metrics.r2_score(yhat_testX, y_test),6))
plt.scatter(y_test, yhat_testX, alpha=0.5)

plt.title('XGB Prediction vs Actual [Test] - log transformed data')

plt.grid()

plt.show()
# transform back to original "coordinates"

yhat_testX_orig = 10**yhat_testX
plt.scatter(y_test_orig, yhat_testX_orig, alpha=0.5)

plt.title('XGB Prediction vs Actual [Test] - Original Coordinates')

plt.grid()

plt.show()
rmse_test_XGB = round(np.sqrt(metrics.mean_squared_error(yhat_testX_orig, y_test_orig)),6)

mae_test_XGB = round(metrics.mean_absolute_error(yhat_testX_orig, y_test_orig),6)

R2_test_XGB = round(metrics.r2_score(yhat_testX_orig, y_test_orig),6)

print('RMSE Test :', rmse_test_XGB)

print('MAE Test  :', mae_test_XGB)

print('R^2       :', R2_test_XGB)
# check calibration

print('Mean actual [test] =         ', y_test_orig.mean())

print('Mean prediction XGB [test] = ', yhat_testX_orig.mean())

print('Ratio pred/act [test] =      ', yhat_testX_orig.mean()/y_test_orig.mean())
# check calibration on training

yhat_trainX = regXGB.predict(X_train)

yhat_trainX_orig = 10**yhat_trainX

y_train_orig = 10**y_train



print('Mean actual [train] =         ', y_train_orig.mean())

print('Mean prediction XGB [train] = ', yhat_trainX_orig.mean())

print('Ratio pred/act [train] =      ', yhat_trainX_orig.mean()/y_train_orig.mean())
# scatter plot on training data

plt.scatter(y_train_orig, yhat_trainX_orig, alpha=0.5)

plt.title('XGB Prediction vs Actual [Train] - Original Coordinates')

plt.grid()

plt.show()
# init java script visualization

shap.initjs()

# create explainer object

explainer = shap.TreeExplainer(regXGB) 

# calc shap values

shap_values = explainer.shap_values(X)
# explain an individual observation

my_observation_index = 1

print(X.iloc[my_observation_index])

shap.force_plot(explainer.expected_value, shap_values[my_observation_index,:], X.iloc[my_observation_index,:])
# show all individual effects in one plot

shap.summary_plot(shap_values, X)
# calc "feature importance" using SHAP values

shap.summary_plot(shap_values, X, plot_type="bar")
# SHAP dependece plot

shap.dependence_plot("age", shap_values, X)
weight_RF = 0.8

weight_XGB = 1-weight_RF

# blend predictions on retransformed values

yhat_blend_test_orig = weight_RF*yhat_test_orig + weight_XGB*yhat_testX_orig
plt.scatter(y_test_orig, yhat_blend_test_orig, alpha=0.5)

plt.title('Blended Prediction vs Actual - Original Coordinates')

plt.grid()

plt.show()
rmse_test_blend = round(np.sqrt(metrics.mean_squared_error(yhat_blend_test_orig, y_test_orig)),6)

mae_test_blend = round(metrics.mean_absolute_error(yhat_blend_test_orig, y_test_orig),6)

R2_test_blend = round(metrics.r2_score(yhat_blend_test_orig, y_test_orig),6)

print('RMSE Test :', rmse_test_blend)

print('MAE Test  :', mae_test_blend)

print('R^2       :', R2_test_blend)
# compare RMSE

print('RMSE Test RF    :', rmse_test_RF)

print('RMSE Test XGB   :', rmse_test_XGB)

print('RMSE Test Blend :', rmse_test_blend)
# compare MAE

print('MAE Test RF    :', mae_test_RF)

print('MAE Test XGB   :', mae_test_XGB)

print('MAE Test Blend :', mae_test_blend)
# compare R^2

print('R^2 Test RF    :', R2_test_RF)

print('R^2 Test XGB   :', R2_test_XGB)

print('R^2 Test Blend :', R2_test_blend)
# check calibration

print('Mean actual [test] =         ', y_test_orig.mean())

print('Mean prediction XGB [test] = ', yhat_blend_test_orig.mean())

print('Ratio pred/act [test] =      ', yhat_blend_test_orig.mean()/y_test_orig.mean())
df_train_export = X_train.copy()

df_train_export['actual'] = y_train_orig

df_train_export['pred_RF'] = yhat_train_orig

df_train_export['pred_XGB'] = yhat_trainX_orig

df_train_export.head()
df_test_export = X_test.copy()

df_test_export['actual'] = y_test_orig

df_test_export['pred_RF'] = yhat_test_orig

df_test_export['pred_XGB'] = yhat_testX_orig

df_test_export.head()
# export

df_train_export.to_csv('export_train.csv')

df_test_export.to_csv('export_test.csv')
# check RF performance for non-smokers vs smokers

train_A = df_train_export[df_train_export.smoker_no==1]

train_B = df_train_export[df_train_export.smoker_no==0]

plt.scatter(train_A.actual, train_A.pred_RF, alpha=0.5, c='blue')

plt.scatter(train_B.actual, train_B.pred_RF, alpha=0.5, c='red')

plt.grid()

plt.show()
# check XGB performance for non-smokers vs smokers

train_A = df_train_export[df_train_export.smoker_no==1]

train_B = df_train_export[df_train_export.smoker_no==0]

plt.scatter(train_A.actual, train_A.pred_XGB, alpha=0.5, c='blue')

plt.scatter(train_B.actual, train_B.pred_XGB, alpha=0.5, c='red')

plt.grid()

plt.show()