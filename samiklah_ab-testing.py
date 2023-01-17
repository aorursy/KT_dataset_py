from datetime import datetime, timedelta,date

import pandas as pd

%matplotlib inline

from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.cluster import KMeans

import chart_studio.plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go

import plotly.figure_factory as ff
import sklearn

import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import warnings

warnings.filterwarnings("ignore")
pyoff.init_notebook_mode()
#create hv segment

df_hv = pd.DataFrame()

df_hv['customer_id'] = np.array([count for count in range(20000)])

df_hv['segment'] = np.array(['high-value' for _ in range(20000)])

df_hv['group'] = 'control'

df_hv.loc[df_hv.index<10000,'group'] = 'test' 

df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.6, 10000)

df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.5, 10000)
df_hv.group.value_counts()
df_hv.head(10)
df_hv.tail(10)
test_results = df_hv[df_hv.group == 'test'].purchase_count

control_results = df_hv[df_hv.group == 'control'].purchase_count



hist_data = [test_results, control_results]



group_labels = ['test', 'control']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, bin_size=.5,

                         curve_type='normal',show_rug=False)



fig.layout = go.Layout(

        title='High Value Customers Test vs Control',

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )





# Plot!

pyoff.iplot(fig)
from scipy import stats 

test_result = stats.ttest_ind(test_results, control_results)

print(test_result)
def eval_test(test_results,control_results):

    test_result = stats.ttest_ind(test_results, control_results)

    if test_result[1] < 0.05:

        print('result is significant')

    else:

        print('result is not significant')

        
eval_test(test_results,control_results)
#create hv segment

df_hv = pd.DataFrame()

df_hv['customer_id'] = np.array([count for count in range(20000)])

df_hv['segment'] = np.array(['high-value' for _ in range(20000)])

df_hv['prev_purchase_count'] = np.random.poisson(0.9, 20000)





df_lv = pd.DataFrame()

df_lv['customer_id'] = np.array([count for count in range(20000,100000)])

df_lv['segment'] = np.array(['low-value' for _ in range(80000)])

df_lv['prev_purchase_count'] = np.random.poisson(0.3, 80000)



df_customers = pd.concat([df_hv,df_lv],axis=0)

df_customers.head()
df_customers.tail()
len(df_customers)
df_test = df_customers.sample(frac=0.9)

df_control = df_customers[~df_customers.customer_id.isin(df_test.customer_id)]
df_test.segment.value_counts()
df_control.segment.value_counts()
df_test_hv = df_customers[df_customers.segment == 'high-value'].sample(frac=0.9)

df_test_lv = df_customers[df_customers.segment == 'low-value'].sample(frac=0.9)



df_test = pd.concat([df_test_hv,df_test_lv],axis=0)

df_control = df_customers[~df_customers.customer_id.isin(df_test.customer_id)]
df_test.segment.value_counts()
df_control.segment.value_counts()
#create hv segment

df_hv = pd.DataFrame()

df_hv['customer_id'] = np.array([count for count in range(30000)])

df_hv['segment'] = np.array(['high-value' for _ in range(30000)])

df_hv['group'] = 'A'

df_hv.loc[df_hv.index>=10000,'group'] = 'B' 

df_hv.loc[df_hv.index>=20000,'group'] = 'C' 

df_hv.group.value_counts()
df_hv.loc[df_hv.group == 'A', 'purchase_count'] = np.random.poisson(0.4, 10000)

df_hv.loc[df_hv.group == 'B', 'purchase_count'] = np.random.poisson(0.6, 10000)

df_hv.loc[df_hv.group == 'C', 'purchase_count'] = np.random.poisson(0.2, 10000)
a_stats = df_hv[df_hv.group=='A'].purchase_count

b_stats = df_hv[df_hv.group=='B'].purchase_count

c_stats = df_hv[df_hv.group=='C'].purchase_count



hist_data = [a_stats, b_stats, c_stats]



group_labels = ['A', 'B','C']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, bin_size=.5,

                         curve_type='normal',show_rug=False)



fig.layout = go.Layout(

        title='Test vs Control Stats',

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )





# Plot!

pyoff.iplot(fig)
def one_anova_test(a_stats,b_stats,c_stats):

    test_result = stats.f_oneway(a_stats, b_stats, c_stats)

    if test_result[1] < 0.05:

        print('result is significant')

    else:

        print('result is not significant')

        
one_anova_test(a_stats,b_stats,c_stats)
df_hv.loc[df_hv.group == 'A', 'purchase_count'] = np.random.poisson(0.5, 10000)

df_hv.loc[df_hv.group == 'B', 'purchase_count'] = np.random.poisson(0.5, 10000)

df_hv.loc[df_hv.group == 'C', 'purchase_count'] = np.random.poisson(0.5, 10000)
a_stats = df_hv[df_hv.group=='A'].purchase_count

b_stats = df_hv[df_hv.group=='B'].purchase_count

c_stats = df_hv[df_hv.group=='C'].purchase_count



hist_data = [a_stats, b_stats, c_stats]



group_labels = ['A', 'B','C']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, bin_size=.5,

                         curve_type='normal',show_rug=False)



fig.layout = go.Layout(

        title='Test vs Control Stats',

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )





# Plot!

pyoff.iplot(fig)
one_anova_test(a_stats,b_stats,c_stats)
#create hv segment

df_hv = pd.DataFrame()

df_hv['customer_id'] = np.array([count for count in range(20000)])

df_hv['segment'] = np.array(['high-value' for _ in range(20000)])

df_hv['group'] = 'control'

df_hv.loc[df_hv.index<10000,'group'] = 'test' 

df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.6, 10000)

df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.8, 10000)





df_lv = pd.DataFrame()

df_lv['customer_id'] = np.array([count for count in range(20000,100000)])

df_lv['segment'] = np.array(['low-value' for _ in range(80000)])

df_lv['group'] = 'control'

df_lv.loc[df_lv.index<40000,'group'] = 'test' 

df_lv.loc[df_lv.group == 'control', 'purchase_count'] = np.random.poisson(0.2, 40000)

df_lv.loc[df_lv.group == 'test', 'purchase_count'] = np.random.poisson(0.3, 40000)



df_customers = pd.concat([df_hv,df_lv],axis=0)

df_customers.head()
import statsmodels.formula.api as smf 

from statsmodels.stats.anova import anova_lm

model = smf.ols(formula='purchase_count ~ segment + group ', data=df_customers).fit()

aov_table = anova_lm(model, typ=2)
print(np.round(aov_table,4))
#create hv segment

df_hv = pd.DataFrame()

df_hv['customer_id'] = np.array([count for count in range(20000)])

df_hv['segment'] = np.array(['high-value' for _ in range(20000)])

df_hv['group'] = 'control'

df_hv.loc[df_hv.index<10000,'group'] = 'test' 

df_hv.loc[df_hv.group == 'control', 'purchase_count'] = np.random.poisson(0.8, 10000)

df_hv.loc[df_hv.group == 'test', 'purchase_count'] = np.random.poisson(0.8, 10000)





df_lv = pd.DataFrame()

df_lv['customer_id'] = np.array([count for count in range(20000,100000)])

df_lv['segment'] = np.array(['low-value' for _ in range(80000)])

df_lv['group'] = 'control'

df_lv.loc[df_lv.index<40000,'group'] = 'test' 

df_lv.loc[df_lv.group == 'control', 'purchase_count'] = np.random.poisson(0.2, 40000)

df_lv.loc[df_lv.group == 'test', 'purchase_count'] = np.random.poisson(0.2, 40000)



df_customers = pd.concat([df_hv,df_lv],axis=0)

import statsmodels.formula.api as smf 

from statsmodels.stats.anova import anova_lm

model = smf.ols(formula='purchase_count ~ segment + group ', data=df_customers).fit()

aov_table = anova_lm(model, typ=2)
print(np.round(aov_table,4))
from statsmodels.stats import power

ss_analysis = power.TTestIndPower()
#create hv segment

df_hv = pd.DataFrame()

df_hv['customer_id'] = np.array([count for count in range(20000)])

df_hv['segment'] = np.array(['high-value' for _ in range(20000)])

df_hv['prev_purchase_count'] = np.random.poisson(0.7, 20000)

df_hv.head()
purchase_mean = df_hv.prev_purchase_count.mean()

purchase_std = df_hv.prev_purchase_count.std()
print(np.round(purchase_mean,4),np.round(purchase_std,4))
effect_size = (0.75 - purchase_mean)/purchase_std
alpha = 0.05

power = 0.8

ratio = 1
ss_result = ss_analysis.solve_power(effect_size=effect_size, power=power,alpha=alpha, ratio=ratio , nobs1=None) 

print(ss_result)
def calculate_sample_size(c_data, column_name, target,ratio):

    value_mean = c_data[column_name].mean()

    value_std = c_data[column_name].std()

    

    value_target = value_mean * target

    

    effect_size = (value_target - value_mean)/value_std

    

    power = 0.8

    alpha = 0.05

    ss_result = ss_analysis.solve_power(effect_size=effect_size, power=power,alpha=alpha, ratio=ratio , nobs1=None) 

    print(int(ss_result))
calculate_sample_size(df_hv, 'prev_purchase_count', 1.05,1)