import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('/kaggle/input/time-series-starter-dataset/Month_Value_1.csv')
data.head()
data.dropna(inplace=True)
data['Period'] = pd.to_datetime(data['Period'],format='%d.%m.%Y')
x = data.drop(columns=['Sales_quantity','Period'])
y = data['Sales_quantity']
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.columns
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1)
model.fit(x_train, y_train)
pred = model.predict(x_test)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(pred,y_test))
data.index = data.Period
# Log Transformation
data['log'] = np.log(data.Sales_quantity)

datax = data.loc[:'2019-12-01']
datay = data.loc['2019-12-01':]
datay
exogenous_features = ['Revenue', 'Average_cost', 'The_average_annual_payroll_of_the_region']
import itertools
import warnings
warnings.filterwarnings('ignore')
p = d = q = range(0, 5)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 6) for x in list(itertools.product(p, d, q))]
best_pdq = (0, 1, 1)
best_seasonal_pdq = (1, 1, 1, 6)

import statsmodels.api as sm
best_model = sm.tsa.statespace.SARIMAX(datax['Sales_quantity'],
                                       exog=datax[exogenous_features],
                                      order=best_pdq,
                                        trend='n',
                                      seasonal_order=best_seasonal_pdq,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False,
                                      suppress_warnings=True)
best_results = best_model.fit()
best_results.plot_diagnostics(figsize=(14,10))
pred_dynamic = best_results.get_prediction(start=pd.to_datetime('2015-02-01'), exog=datax[exogenous_features])
pred_dynamic_ci = pred_dynamic.conf_int()

# conf_int = confidence Interval
result_predicted = pred_dynamic.predicted_mean
result_truth = datax['Sales_quantity'].iloc[1:]
import matplotlib.pyplot as plt
plt.plot(result_truth, label='original')
plt.plot(result_predicted, label='fitted Values')
# plt.plot(rng['future'], label='Future Values')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((result_predicted-result_truth)**2)/len(result_truth)))

x = best_results.forecast(steps=5, exog=datay[exogenous_features])
plt.plot(result_truth, label='original')
plt.plot(result_predicted, label='fitted Values')
plt.plot(x, label='Future Values')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((result_predicted-result_truth)**2)/len(result_truth)))

np.sqrt(mean_squared_error(x,datay['Sales_quantity']))
# ACtual RMSE with Test Data