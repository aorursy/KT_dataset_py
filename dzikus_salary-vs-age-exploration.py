import numpy as np 
import pandas as pd 
income_raw_data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
from ggplot import *
ggplot( aes(x='Age', y='MonthlyIncome'), data = income_raw_data) + \
geom_point() +\
theme_bw() + \
ggtitle('Monthly Income by Age - Raw Data')
ggplot( aes(x='Age', y='MonthlyIncome', colour='JobRole'), data = income_raw_data) + \
    geom_point() + \
    stat_smooth(span=0.3, level=0.95) + \
    theme_bw() + \
    ggtitle('Monthly Income by age grouped by JobRole')
bins=[-float('inf'), 18, 25, 35, 55, float('inf')]
labels = range(0, len(bins) - 1)
income_raw_data['AgeRange'] = pd.cut(income_raw_data['Age'], bins = bins, labels = labels, retbins=False)
range_stats = income_raw_data[['MonthlyIncome', 'AgeRange']].groupby('AgeRange').agg(['min', 'max', 'mean'])
range_stats.columns = range_stats.columns.get_level_values(1)
range_stats_to_merge = range_stats.reset_index()
range_stats = range_stats.stack()
range_stats = range_stats.reset_index()
range_stats.columns = ['AgeRange', 'stat', 'meanMonthlyIncome']
ggplot(aes(x = 'AgeRange', y='MonthlyIncome'), data = income_raw_data) + \
    geom_boxplot() + \
    theme_bw() + \
    ggtitle('Distribution of MonthlyIncome in age bins')
data_to_merge = income_raw_data[['Age', 'AgeRange', 'EmployeeNumber', 'MonthlyIncome']].copy()
data_to_merge['OriginalAgeRange'] = data_to_merge['AgeRange']
data_to_merge = data_to_merge.merge(range_stats_to_merge, on='AgeRange')
data_to_merge['distance_to_mean'] = (data_to_merge['MonthlyIncome'] - data_to_merge['mean']) 
predictions_users = data_to_merge[['EmployeeNumber', 'Age', 'OriginalAgeRange', 'MonthlyIncome', 'distance_to_mean']].copy()
predictions_users['key'] = 1
predictions_ranges = range_stats_to_merge[['AgeRange', 'mean']].copy()
predictions_ranges['key'] = 1
predictions_users = predictions_users.merge(predictions_ranges, on='key')
predictions_users['PredictedIncome'] = predictions_users['mean'] + predictions_users['distance_to_mean']
predictions_users = predictions_users[['EmployeeNumber', 'Age', 'OriginalAgeRange', 'MonthlyIncome', 'AgeRange', 'PredictedIncome']]
predictions_users.head(10)
ggplot(aes(x = 'AgeRange', y='PredictedIncome'), data = predictions_users) + \
    geom_boxplot() + \
    theme_bw() + \
    ggtitle('Distribution of predicted MonthlyIncome in age bins')
