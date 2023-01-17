# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split


pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x)) #Limiting 4 decimal
plt.rcParams["figure.figsize"] = [9,5]
plt.style.use('ggplot')


data_df = pd.read_csv("../input/forest-fires-data-set/forestfires.csv")
data_df.head()
target = 'area' 
data_df.shape
data_df.dtypes
data_df.columns
data_df.describe()
# Calculating missing data in feature columns
data_mis = (data_df.isnull().sum() / len(data_df)) * 100
data_mis = data_mis.drop(data_mis[data_mis == 0].index).sort_values(ascending=False)
data_mis = pd.DataFrame({'Percentage' :data_mis})
data_mis['Id'] = data_mis.index
data_mis.reset_index(drop=True,level=0, inplace=True)
data_mis.head()
dft = data_df.drop(columns=target)
cate_columns = dft.select_dtypes(include='object').columns.tolist()
nume_columns = dft.select_dtypes(exclude='object').columns.tolist()
print('Categorical columns: ',cate_columns)
print('Numerical columns: ',nume_columns)
print("Skew: \n{}".format(data_df.skew()))
print("Kurtosis: \n{}".format(data_df.kurtosis()))
plt.figure(figsize=(15,5))
ax = sns.kdeplot(data_df[target],shade=True,color='b')
plt.xticks([i for i in range(0,1250,50)])
plt.show()
plt.figure(figsize=(15,5))
ax = sns.kdeplot(data_df['FFMC'],shade=True,color='b')
plt.xticks([i for i in range(0,100,5)])
plt.show()
plt.figure(figsize=(15,5))
ax = sns.kdeplot(data_df['ISI'],shade=True,color='b')
plt.xticks([i for i in range(0,100,5)])
plt.show()
outl_dect = sns.boxplot(data_df[target])
outl_dect = sns.boxplot(data_df['FFMC'])
outl_dect = sns.boxplot(data_df['ISI'])
outl_dect = sns.boxplot(data_df['rain'])
outlier_columns = ['area','FFMC','ISI','rain']
np.log1p(data_df[outlier_columns]).skew()
np.log1p(data_df[outlier_columns]).kurtosis()
mask = data_df.loc[:,['FFMC']].apply(zscore).abs() < 3
data_df = data_df[mask.values]
data_df.shape
# Since most of the values in rain are 0.0, we can convert it as a categorical column
data_df['rain'] = data_df['rain'].apply(lambda x: int(x > 0.0))

outlier_columns.remove('rain')
data_df[outlier_columns] = np.log1p(data_df[outlier_columns])
data_df[outlier_columns].skew()
data_df[outlier_columns].kurtosis() 
data_df.describe()
data_sel = data_df.copy()
le = LabelEncoder() 
  
data_sel['day']= le.fit_transform(data_sel['day']) 
data_sel['month']= le.fit_transform(data_sel['month']) 
X, y = data_sel.iloc[:,:-1],data_sel.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
#                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg = xgb.XGBRegressor(base_score=0.3, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.24, gamma=0, learning_rate=0.01, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=102,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


#eval_set = [(X_test, y_test)]
eval_set = [(X_train, y_train), (X_test, y_test)]
xg_reg.fit(X_train, y_train, eval_metric=["rmse"],eval_set=eval_set, verbose=False)
preds = xg_reg.predict(X_test)
def calc_ISE(X_train, y_train, model):
    '''returns the in-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_train, y_train), rmse
    
def calc_OSE(X_test, y_test, model):
    '''returns the out-of-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_test, y_test), rmse

is_r2, ise = calc_ISE(X_train, y_train,xg_reg )
os_r2, ose = calc_OSE(X_test, y_test, xg_reg)

# show dataset sizes
data_list = (('R^2_in', is_r2), ('R^2_out', os_r2), 
             ('ISE', ise), ('OSE', ose))
for item in data_list:
    print('{:10}: {}'.format(item[0], item[1]))
print('train/test: ',ose/ise)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

xgb.plot_tree(xg_reg,num_trees=0)

plt.rcParams['figure.figsize'] = [15, 15]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [7, 7]
plt.show()

# retrieve performance metrics
results = xg_reg.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot RMSE
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()
xg_reg.save_model('0001.model_forest_fire')

