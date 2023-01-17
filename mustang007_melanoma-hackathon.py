import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns
Train = pd.read_csv('/kaggle/input/melanoma-data/Train.csv')
Test = pd.read_csv('/kaggle/input/melanoma-data/Test.csv')
sub = pd.read_csv('/kaggle/input/melanoma-data/sample_submission.csv')
Train.info()
for i in Train.columns:
    plot_data = [
        go.Histogram(
            x=Train[i],
            name=i,
        )
    ]
    plot_layout = go.Layout(
            title='Line Graph',
            yaxis_title='Values',
            xaxis_title=i,
            plot_bgcolor="#f8f8f8"
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    fig.show()
# Lets Check Skewness

for i in Train.columns:
    a = Train[i].skew()
    print('This columns ',i, ' has skewness = ',a)
# Train['err_malign'] = np.log1p(Train['err_malign'])
# Train.drop(columns='exposed_area', inplace=True)
# lets Check Corrleation
corr = Train.corr(method='spearman')
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True)
# lets check Mulit collinearity
Train.columns
for i in Train.columns:
    for j in Train.columns:
        if i != j:
            a = Train[i].corr(Train[j], method='spearman')
            if a > 0.7:
                print(i , ' and ',j,' has ', a)
            
            elif a < -0.7:
                 print(i , ' and ',j,' has ', a)
Train.columns

Train
Train['true_std'] = (Train['std_dev_malign'] - Train['malign_penalty'])
dddd

plt.figure(figsize=(20,10))
sns.boxplot(x="variable", y="value", data=pd.melt(Train))
sns.boxplot(Train['exposed_area'])
a = Train['exposed_area'].quantile(.99)
Train = Train[Train['exposed_area']< a]
sns.boxplot(Train['err_malign'])
a = Train['err_malign'].quantile(.98)
Train = Train[Train['err_malign']< a]
Train.skew()
# Train['mass_ratio'] = Train['mass_npea'] / Train['size_npear']
# Train.drop(columns='mass_npea', inplace=True)
# Train.drop(columns='std_dev_malign', inplace=True)
# Train.drop(columns='err_malign', inplace=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(Train.values, i) for i in range(Train.shape[1])]
vif["features"] = Train.columns
vif
Train.reset_index(drop=True)
X = Train.iloc[:8100, :]
test = Train.iloc[8100:, :]
X_train = X.drop(columns='tumor_size')
Y_train = X['tumor_size']
X_test = test.drop(columns='tumor_size')
Y_test = test['tumor_size']


from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
sc = StandardScaler()
pw = PowerTransformer()
mn = MinMaxScaler()
X_train = mn.fit_transform(X_train)
X_test = mn.transform(X_test)
# Y_train = Y_train.values.reshape(-1,1)
# Y_test = Y_test.values.reshape(-1,1)
# Y_train = pw.fit_transform(Y_train)
# Y_test = pw.transform(Y_test)
from lightgbm import LGBMRegressor
lgb_fit_params={"early_stopping_rounds":300, 
            "eval_metric" : 'rmse', 
            "eval_set" : [(X_test,Y_test)],
            'eval_names': ['valid'],
            'verbose':1000
           }

lgb_params = {'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse',
 'verbose': 0,
 'bagging_fraction': 0.8,
 'bagging_freq': 1,
 'lambda_l1': 0.01,
 'lambda_l2': 0.01,
 'learning_rate': 0.01,
 'max_bin': 255,
 'max_depth': 15,
 'min_data_in_bin': 1,
 'min_data_in_leaf': 1,
 'num_leaves': 250}
clf_lgb = LGBMRegressor(n_estimators=10000, **lgb_params, random_state=123456789, n_jobs=-1)
clf_lgb.fit(X_train, Y_train, **lgb_fit_params)
a= int(clf_lgb.best_iteration_)
a
model_lgbm = LGBMRegressor(bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.01,
              lambda_l2=0.01, learning_rate=0.01, max_bin=255, max_depth=20,
              min_data_in_bin=1, min_data_in_leaf=1,num_leaves = 250,
              n_estimators=a)
model_lgbm.fit(X_train, Y_train)

pred = model_lgbm.predict(X_test)

# model_lgbm = LGBMRegressor(bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.01,
#               lambda_l2=0.01, learning_rate=0.01, max_bin=255, max_depth=15,
#               min_data_in_bin=1, min_data_in_leaf=1,num_leaves = 320,
#               n_estimators=a)
# model_lgbm.fit(X_train, Y_train)

# pred = model_lgbm.predict(X_test)
# pred
from sklearn.metrics import mean_squared_error,r2_score
print(np.sqrt(mean_squared_error(pred, Y_test)))
print(r2_score(pred, Y_test))
Test
Test['true_std'] = (Test['std_dev_malign'] - Test['malign_penalty'])
Test = mn.transform(Test)
final = model_lgbm.predict(Test)
final
sub['tumor_size'] = final
sub.to_csv('final_lg.csv')
a = X_train[['damage_size','malign_penalty','malign_ratio']]
b = a.corr(method='spearman')
sns.heatmap(b, annot=True)
t = X_test[['damage_size','malign_penalty','malign_ratio']]
len(X_train.columns)