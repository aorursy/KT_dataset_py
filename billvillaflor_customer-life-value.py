#load raw transactional data (customer_id & transaction_date are minimum requirements) 

import pandas as pd
from lifetimes.utils import calibration_and_holdout_data, calculate_alive_path
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import plot_frequency_recency_matrix, \
plot_probability_alive_matrix, plot_period_transactions, \
plot_calibration_purchases_vs_holdout_purchases, plot_history_alive
from matplotlib import pyplot as plt
from pylab import rcParams
from dateutil import relativedelta
from datetime import date, datetime

raw_data_path = '../input/cdnow_transaction_log.csv'
customer_id_col = 'cust'
txn_date_col = 'date'

raw_data = pd.read_csv(raw_data_path, parse_dates=[txn_date_col])
print(raw_data.head(5))
print(raw_data.describe())
#preprocess raw_data, adjust txn date to current

max_date = raw_data[txn_date_col].max().replace(hour=0, minute=0, second=0, microsecond=0)
max_date = max_date.replace(day=1)
current_date = date.today().replace(day=1)
difference = relativedelta.relativedelta(current_date, max_date)
months_to_pad = difference.years * 12 + difference.months
raw_data[txn_date_col] = raw_data[txn_date_col].transform(lambda x: x + relativedelta.relativedelta(months=months_to_pad))
print(raw_data.head(5))
#transform raw transactional data into frequency-recency-age summary matrix

max_date = raw_data[txn_date_col].max()
min_date = raw_data[txn_date_col].min()
mid_date = min_date + ((max_date - min_date) / 2)
mid_date = mid_date.replace(hour=0, minute=0, second=0, microsecond=0)
calibration_cutoff = mid_date
holdout_cutoff = max_date

summary = calibration_and_holdout_data(raw_data, customer_id_col, txn_date_col, \
                                       calibration_period_end=calibration_cutoff, \
                                       observation_period_end=holdout_cutoff)
print(summary.head(5))
#train Pareto/NBD model on calibration data

model = ParetoNBDFitter(penalizer_coef=0.0)
model.fit(summary['frequency_cal'], summary['recency_cal'], summary['T_cal'])
print(model)
#visualize trained model
rcParams['figure.figsize'] = 15, 10

#visualize customers that will purchase again
plot_frequency_recency_matrix(model)
plt.show()

#visualize customers that will be alive
plot_probability_alive_matrix(model)
plt.show()

#visualize comparison between actual and predicted customers purchase
plot_period_transactions(model)
plt.show()
#visualize top customers that are going to buy in the next periods

next_n_periods = 10
predictions = pd.DataFrame()
predictions['predicted_purchases'] = model.conditional_expected_number_of_purchases_up_to_time(next_n_periods, \
                                                                                        summary['frequency_cal'], \
                                                                                        summary['recency_cal'], \
                                                                                        summary['T_cal'])
print(predictions.sort_values(by='predicted_purchases', ascending=False).head(5))
#visualize trained model vs predictions in holdout dataset

plot_calibration_purchases_vs_holdout_purchases(model, summary)
plt.show()
#general usage of the trained model

#predicting future purchase on the next period
next_n_periods = 10
customer_id = 15562
customer = summary.loc[customer_id]
print(customer)
print(model.predict(next_n_periods, customer['frequency_cal'], customer['recency_cal'], customer['T_cal']))
#...continued

#analyzing probability of customer being alive on the next periods
#input
target_date = datetime(2018, 5, 30)
customer_id = 15562

customer_txn = raw_data.loc[raw_data[customer_id_col] == customer_id]
oldest_txn = customer_txn[txn_date_col].min()
days_since_first_txn = (target_date - oldest_txn).days + 1
plot_history_alive(model, days_since_first_txn, customer_txn, txn_date_col)
plt.show()

alive_path = calculate_alive_path(model, customer_txn, txn_date_col, days_since_first_txn)
alive_path = alive_path.to_frame('proba_alive')
alive_path['age'] = alive_path.index
alive_path[txn_date_col] = alive_path['age'].transform(lambda x : oldest_txn + relativedelta.relativedelta(days=x))
print(alive_path)
