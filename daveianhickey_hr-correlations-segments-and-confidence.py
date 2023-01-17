import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



raw = pd.read_csv(r'../input/HR_comma_sep.csv')

print(raw.shape)

print(raw.head())
sal_dict = {'low': 1, 'medium': 2, 'high': 3}

raw['salary_map'] = raw['salary'].map(sal_dict)
#print("What department are they in?")

#print(raw['sales'].value_counts())



vc = raw['sales'].value_counts().plot(kind='bar', title="No. of Staff by Department")

plt.show()



raw_depts = raw['sales'].value_counts()

left = raw[raw["left"]==1]

left_depts = left['sales'].value_counts()



print("\n What % of the department have left?")

percent_dept_left = left_depts/raw_depts

print(percent_dept_left)

print("Avg. leavers per dept is ",percent_dept_left.mean())

#print(left['sales'].value_counts())
#Correlation Matrix

corr = raw.corr()

#corr = (corr)

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of correlations in the raw data')
stay_corr = raw[raw['left']==0]

stay_corr = stay_corr.corr()

sns.heatmap(stay_corr,

           xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title("Heatmap of Correlations for those who stayed")

           
leave_corr = raw[raw['left']==1]

leave_corr = leave_corr.corr()

sns.heatmap(leave_corr, annot = True,

           xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title("Heatmap of correlations for those who left")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

group_names = ['0.2', '0.4', '0.6', '0.8', '1'] #Create names for the groups

categories = pd.cut(raw['last_evaluation'], bins, labels=group_names) #Cut based a column

raw['eval_categories'] = pd.cut(raw['last_evaluation'], bins, labels=group_names) #Create new column
stay = raw[raw['left']==0]

fig = plt.figure(figsize=(15,16))



dfs = [raw,left,stay]

df_str = ['raw','left','stay']



for n in range(0,3):

    last_eval = (dfs[n])

    last_eval2 = (dfs[n]['last_evaluation'])

    num_proj = (dfs[n]['number_project'])



    ax = fig.add_subplot(3,3,n+1)

    ax.scatter(last_eval2,num_proj)

    ax.set_title("Correlation for " + df_str[n])

    ax.tick_params(bottom="off", top="off", left="off", right="off")    

    

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    plt.xlabel('Last Evaluation')

    plt.ylabel('Number of Projects')

    

    ax = fig.add_subplot(3,3,n+4)

    ax.boxplot(last_eval2.values)

    ax.set_title("last_eval boxplot for " + df_str[n])

    

    ax = fig.add_subplot(3,3,n+7)

    ax.boxplot(num_proj.values)

    ax.set_title("num_proj boxplot for " + df_str[n])

plt.show()



high_p = raw[raw['number_project']>4]

high_p = high_p[high_p['last_evaluation']>0.75]

high_p_left = high_p[high_p['left']==1]



high_p = high_p.corr()

high_p_left = high_p_left.corr()

sns.heatmap(high_p,annot = True,

            xticklabels = high_p.columns.values, 

            yticklabels = high_p.columns.values)



sns.plt.title('Correlations for high performers who have stayed')
sns.heatmap(high_p_left, annot = True,

            xticklabels = high_p_left.columns.values, 

            yticklabels = high_p_left.columns.values)

sns.plt.title("Correlations for high performers who have left")
print("Describe time spent at company for those who stayed")

print(stay['time_spend_company'].describe())

print("\n \n Describe time spent at company for those who left")

print(left['time_spend_company'].describe())

print("\n \n Time_spend_company value counts")

print(raw['time_spend_company'].value_counts())
low_leave_dept = raw[raw['sales'].isin(['RandD','management'])]

low_leave_dept.groupby(by='sales').mean().reset_index()

left.groupby(by='sales').mean().reset_index()

stay.groupby(by='sales').mean().reset_index()
high_perf_mean = raw['last_evaluation'].mean()

high_perf_std = np.std(raw['last_evaluation'])



high_perf_group = raw[raw['last_evaluation'] >= high_perf_mean + high_perf_std]

low_perf_group = raw[raw['last_evaluation'] <= high_perf_mean - high_perf_std]



left_high_perf_group = high_perf_group[high_perf_group['left'] == 1]['last_evaluation']

left_low_perf_group = low_perf_group[low_perf_group['left'] == 1]['last_evaluation']
from scipy.stats import chisquare



perf_data_sets = [high_perf_group, low_perf_group, raw] # Change the data sets to be sure it's not being skewed.

left_perf_data_sets = [left_high_perf_group, left_low_perf_group, left]



observed = [len(i) for i in left_perf_data_sets]

all_team = [len(i) for i in perf_data_sets]

expected = [o * 0.23 for o in all_team] # on avg, 23% of each dept left so this is our expected value



chisquare_value, pvalue = chisquare(observed, expected)



print(observed)

print(expected)



print("Chi-Square value is ", chisquare_value)

print("pvalue is ", pvalue)
proj_mean = raw['number_project'].mean()

proj_stddev = np.std(raw['number_project'])



lots_proj = raw[raw['number_project'] >= proj_mean + proj_stddev]

barely_proj = raw[raw['number_project'] <= proj_mean - proj_stddev]



left_lots_proj = lots_proj[lots_proj['left'] == 1]

left_barely_proj = barely_proj[barely_proj['left'] == 1]
proj_data_sets = [lots_proj, barely_proj]

left_proj_data_sets = [left_lots_proj, left_barely_proj]



observed = [len(i) for i in left_proj_data_sets] # cross section of the groups with those who had actually left

all_team = [len(i) for i in proj_data_sets]

expected = [o * 0.23 for o in all_team] # on avg, 23% of each dept left so this is our expected value



proj_chisquare, proj_pvalue = chisquare(observed, expected)



print(observed)

print(expected)



print("Chi-Square value is ", proj_chisquare)

print("pvalue is ", proj_pvalue)
time_mean = raw['time_spend_company'].mean()

time_stdD = np.std(raw['time_spend_company'])



oldish = raw[raw['time_spend_company'] >= time_mean + time_stdD]

newish = raw[raw['time_spend_company'] <= time_mean - time_stdD]



left_oldish = oldish[oldish['left'] ==1]

left_newish = newish[newish['left'] ==1]



time_sets = [oldish, newish, raw]

left_time_sets = [left_oldish, left_newish, left]



observed = [len(i) for i in left_time_sets] # cross section of the groups with those who had actually left

all_team = [len(i) for i in time_sets]

expected = [o * 0.23 for o in all_team] # on avg, 23% of each dept left so this is our expected value



time_chi, time_pvalue = chisquare(observed, expected)



print(observed)

print(expected)



print("time_chi is ", time_chi)

print("pvalue is ", time_pvalue)

print(type(pvalue))