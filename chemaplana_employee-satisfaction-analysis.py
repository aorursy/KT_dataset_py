import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



hr_base = pd.read_csv('../input/HR_comma_sep.csv')

hr_file = hr_base



print (hr_file.info())
print (hr_file.describe())
print (hr_file.head())
print (hr_file.corr())
hr_file.hist()
hr_file = hr_file.loc[:,['satisfaction_level','left']]

hr_file_sat = hr_file[hr_file.satisfaction_level>= 0.64]

hr_file_nsat = hr_file[hr_file.satisfaction_level< 0.64]



hr_file_sat = hr_file_sat.groupby('left').count()/len(hr_file_sat)

hr_file_nsat = hr_file_nsat.groupby('left').count()/len(hr_file_nsat)



ax = plt.axes()

ax.set_title('0.64 sat or higher')

sizes = hr_file_sat['satisfaction_level']

labels = 'Left(F)', 'Left(T)'

colors = ['b', 'orange']



plt.pie(sizes, colors=colors, labels=labels)

plt.show()
ax = plt.axes()

ax.set_title('lower than 0.64 sat')

sizes = hr_file_nsat['satisfaction_level']

labels = 'Left(F)', 'Left(T)'

colors = ['g', 'm']



plt.pie(sizes, colors=colors, labels=labels)

plt.show()
hr_file = pd.read_csv('../input/HR_comma_sep.csv', sep=',')

hr_file_sat = hr_file.loc[:,['satisfaction_level','left','sales','salary']]

hr_file_sat_left = hr_file_sat[hr_file_sat.left == 1]

hr_file_sat_left_gr = hr_file_sat_left.groupby('sales').mean()

hr_file_sat_nleft = hr_file_sat[hr_file_sat.left == 0]

hr_file_sat_nleft_gr = hr_file_sat_nleft.groupby('sales').mean()



ax = plt.axes()

ax.set_title('Avg satisfaction levels for left(T) and left(F) employees')

ax.set_ylim(0.2, 0.7)



xline = np.linspace(0,len(hr_file_sat_left_gr) - 1,len(hr_file_sat_left_gr))

width = 0.5



ax.set_xticks(xline + width)

ax.set_xticklabels(hr_file_sat_left_gr.index)



bar_1 = ax.bar(xline, hr_file_sat_left_gr['satisfaction_level'], width=width, color='r')

bar_2 = ax.bar(xline + width, hr_file_sat_nleft_gr['satisfaction_level'], width=width, 

               color='b')

plt.legend((bar_1[0], bar_2[0]), ('Left(true)' , 'Left(false)'), loc='lower right')

plt.setp(ax.set_xticklabels(hr_file_sat_left_gr.index), rotation=45)

plt.show()
hr_file_sat = hr_file.loc[:,['satisfaction_level','left','sales','salary']]

hr_file_sat_left = hr_file_sat[hr_file_sat.left == 1]

hr_file_sat_left_gr = hr_file_sat_left.groupby('sales').std()

hr_file_sat_nleft = hr_file_sat[hr_file_sat.left == 0]

hr_file_sat_nleft_gr = hr_file_sat_nleft.groupby('sales').std()



ax = plt.axes()

ax.set_title('St Dev satisfaction levels for left(T) and left(F) employees')

ax.set_ylim(0.2, 0.3)



xline = np.linspace(0,len(hr_file_sat_left_gr) - 1,len(hr_file_sat_left_gr))

width = 0.5



ax.set_xticks(xline + width)

ax.set_xticklabels(hr_file_sat_left_gr.index)



bar_1 = ax.bar(xline, hr_file_sat_left_gr['satisfaction_level'], width=width, color='r')

bar_2 = ax.bar(xline + width, hr_file_sat_nleft_gr['satisfaction_level'], width=width, 

               color='b')

ax.legend((bar_1[0], bar_2[0]), ('Left(true)' , 'Left(false)'), loc='upper right')

plt.setp(ax.set_xticklabels(hr_file_sat_left_gr.index), rotation=45)

plt.show()
hr_file = hr_file[hr_file.left == 1]



ax = plt.axes()

ax.set_title('Title')

ax.set_xlim(120,320)

ax.set_ylim(0,1)

plt.xlabel('Avg Monthly Hours')

plt.ylabel('Satisfaction Level')



hr_file['color'] = 'none'



hr_file.loc[hr_file.salary == 'low', ['color']] = 'b'

hr_file.loc[hr_file.salary == 'medium', ['color']] = 'g'

hr_file.loc[hr_file.salary == 'high', ['color']] = 'r'



plt.scatter(hr_file['average_montly_hours'], hr_file['satisfaction_level'],

	s=hr_file['number_project']**3, color=hr_file['color'], alpha=0.5)

plt.show()
hr_file = pd.read_csv('../input/HR_comma_sep.csv', sep=',')

hr_file = hr_file[hr_file.left == 1]



ax = plt.axes()

ax.set_title('Title')

ax.set_xlim(0.4,1.05)

ax.set_ylim(0,1.01)

plt.xlabel('Rating in last eval')

plt.ylabel('Satisfaction Level')



hr_file['color'] = 'none'



hr_file.loc[hr_file.sales == 'sales', ['color']] = 'r'

hr_file.loc[hr_file.sales == 'accounting', ['color']] = 'orange'

hr_file.loc[hr_file.sales == 'hr', ['color']] = 'yellow'

hr_file.loc[hr_file.sales == 'technical', ['color']] = 'g'

hr_file.loc[hr_file.sales == 'support', ['color']] = 'b'

hr_file.loc[hr_file.sales == 'management', ['color']] = 'm'

hr_file.loc[hr_file.sales == 'IT', ['color']] = 'grey'

hr_file.loc[hr_file.sales == 'product_mng', ['color']] = 'k'

hr_file.loc[hr_file.sales == 'marketing', ['color']] = 'pink'

hr_file.loc[hr_file.sales == 'RandD', ['color']] = 'c'



plt.scatter(hr_file['last_evaluation'], hr_file['satisfaction_level'],

	s=hr_file['number_project']**3, c=hr_file['color'], alpha=0.5)

plt.show()
ax = plt.axes()

ax.set_title('Title')

ax.set_xlim(0.42, 0.6)

ax.set_ylim(0.3, 0.5)

plt.xlabel('Rating in last eval')

plt.ylabel('Satisfaction Level')



plt.scatter(hr_file['last_evaluation'], hr_file['satisfaction_level'],

	s=hr_file['number_project']**3, c=hr_file['color'], alpha=0.5)

plt.show()
ax = plt.axes()

ax.set_title('Title')

ax.set_xlim(0.8, 1.01)

ax.set_ylim(0.9, 0.7)

plt.xlabel('Rating in last eval')

plt.ylabel('Satisfaction Level')



plt.scatter(hr_file['last_evaluation'], hr_file['satisfaction_level'],

	s=hr_file['number_project']**3, c=hr_file['color'], alpha=0.5)

plt.show()
ax = plt.axes()

ax.set_title('Title')

ax.set_xlim(0.75, 1.01)

ax.set_ylim(0.05, 0.15)

plt.xlabel('Rating in last eval')

plt.ylabel('Satisfaction Level')



plt.scatter(hr_file['last_evaluation'], hr_file['satisfaction_level'],

	s=hr_file['number_project']**3, c=hr_file['color'], alpha=0.5)

plt.show()
import statsmodels.api as sm



hr_file = hr_base

hr_file_dumm = pd.get_dummies(hr_file['salary'],prefix='sal')

hr_file = hr_file.join(hr_file_dumm.ix[:])

hr_file_dumm = pd.get_dummies(hr_file['sales'],prefix='dept')

hr_file = hr_file.join(hr_file_dumm.ix[:])



selected_columns = ['last_evaluation', 'number_project',

       'average_montly_hours', 'time_spend_company', 'Work_accident',

       'promotion_last_5years','sal_low', 'sal_medium', 'dept_RandD',

       'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing',

       'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical']



hr_file_train = hr_file.loc[0:13999,selected_columns]

hr_file_dependent = hr_file.loc[0:13999,['satisfaction_level']]

hr_file_train['intercept'] = 1



model = sm.OLS(hr_file_dependent, hr_file_train)

results = model.fit()



print (results.summary())
hr_file = hr_base

hr_file_dumm = pd.get_dummies(hr_file['salary'],prefix='sal')

hr_file = hr_file.join(hr_file_dumm.ix[:])

hr_file_dumm = pd.get_dummies(hr_file['sales'],prefix='dept')

hr_file = hr_file.join(hr_file_dumm.ix[:])



selected_columns = ['last_evaluation', 'number_project',

       'time_spend_company', 'Work_accident',

       'promotion_last_5years','sal_low', 'sal_medium']



hr_file_train = hr_file.loc[0:13999,selected_columns]

hr_file_dependent = hr_file.loc[0:13999,['satisfaction_level']]

hr_file_train['intercept'] = 1



model = sm.OLS(hr_file_dependent, hr_file_train)

results = model.fit()



print (results.summary())
hr_file['color'] = 'none'

hr_file.loc[hr_file.salary == 'low', ['color']] = 'b'

hr_file.loc[hr_file.salary == 'medium', ['color']] = 'g'

hr_file.loc[hr_file.salary == 'high', ['color']] = 'r'



ax = plt.axes()

ax.set_xlim(0.3, 1.05)

ax.set_ylim(0, 1.05)

plt.scatter(hr_file['last_evaluation'],hr_file['satisfaction_level'],

	color=hr_file['color'])

plt.xlabel('last evaluation')

plt.ylabel('satisfaction')

plt.show()