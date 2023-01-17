import numpy as np

import pandas as pd 
data = pd.read_csv('../input/HR_comma_sep.csv')
data.info()
data.head(5)
data.tail(5)
data.sample(10)
# RENAME column sale to department

data.rename(columns={'sales': 'department'}, inplace = True)



# Convert salary variable type to numeric

data['salary'] = data['salary'].map({'low':1, 'medium':2, 'high':3})
data.describe()
print(data['department'].value_counts())
print(data['salary'].value_counts())
table = data.pivot_table(values="satisfaction_level", index="department", columns="salary",aggfunc=np.count_nonzero)

table
%matplotlib inline 



import matplotlib.pyplot as plt



import seaborn as sns



sns.set()
f, axes = plt.subplots(2,2, figsize=(10,10), sharex=True)



plt.subplots_adjust(wspace=0.5)# adjust the space between the plots



sns.despine(left=True)



# plot a boxplot of satisfaction_level to see if there is outliers

sns.boxplot( x= 'satisfaction_level',  data=data, orient='v',ax=axes[0,0])



# plot a boxplot of last_evaluation to see if there is outliers

sns.boxplot( x= 'last_evaluation',  data=data, orient='v',ax=axes[0,1])



# plot a boxplot of number_project to see if there is outliers

sns.boxplot( x= 'number_project',  data=data, orient='v',ax=axes[1,0])



# plot a boxplot of average_montly_hours to see if there is outliers

sns.boxplot( x= 'average_montly_hours',  data=data, orient='v',ax=axes[1,1]);



#Put a ; at the end of the last line to suppress the printing of output 
plt.figure(figsize=(4,5))

sns.boxplot( x= 'time_spend_company',  data=data, orient='v');
corr = data.corr()

corr
sns.set(style='white')



mask = np.zeros_like(corr, dtype=np.bool)



mask[np.triu_indices_from(mask)] = True



# Inserir a figura

f, ax = plt.subplots(figsize=(13,8))



cmap = sns.diverging_palette(10,220, as_cmap=True)



#Desenhar o heatmap com a máscara

ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax= .5, annot=True, annot_kws= {'size':11}, square=True, xticklabels=True, yticklabels=True, linewidths=.5, 

           cbar_kws={'shrink': .5}, ax=ax)

ax.set_title('Correlation between variables', fontsize=20);

print(data['left'].value_counts()[1],"employees left the company")
# The plot show the amount o employees that stayed and left the company.

plt.figure(figsize=(4,5))

ax = sns.countplot(data.left)

total = float(len(data))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

plt.title('Stayed or Left', fontsize=14);
j = sns.factorplot(x='salary', y='left', kind='bar', data=data)

plt.title('Employees that left by salary level', fontsize=14)

j.set_xticklabels(['High', 'Medium', 'Low']);
h = sns.factorplot(x = 'salary', hue='department', kind ='count', size = 5,aspect=1.5, data=data, palette='Set1' )

plt.title("Salaries by department", fontsize=14)

h.set_xticklabels(['High', 'Medium', 'Low']);
sns.set()

plt.figure(figsize=(10,5))

sns.barplot(x='department', y='salary', hue='left', data=data)

plt.title('Salary Comparison', fontsize=14);
sns.factorplot(x='Work_accident', y='left', kind='bar', data=data)

plt.title('Employees that had work accident', fontsize=14);
print(data.Work_accident.sum())

print(data.Work_accident.mean())

print((data[data['left']==1]['Work_accident']).sum())
sns.factorplot(x='promotion_last_5years', y='left', kind='bar', data=data)

plt.title('Employees who have been promoted in the last 5 years', fontsize=14);
print(data.promotion_last_5years.sum())

print(data.promotion_last_5years.mean())
plt.figure(figsize =(7,5))

bins = np.linspace(1.0, 11,10)

plt.hist(data[data['left']==1]['time_spend_company'], bins=bins, alpha=1, label='Employees Left')

plt.hist(data[data['left']==0]['time_spend_company'], bins=bins, alpha = 0.5, label = 'Employee Stayed')

plt.grid(axis='x')

plt.xticks(np.arange(2,11))

plt.xlabel('time_spend_company')

plt.title('Years in the company', fontsize=14)

plt.legend(loc='best');
plt.figure(figsize =(7,7))

bins = np.linspace(0.305, 1.0001, 14)

plt.hist(data[data['left']==1]['last_evaluation'], bins=bins, alpha=1, label='Employees Left')

plt.hist(data[data['left']==0]['last_evaluation'], bins=bins, alpha = 0.5, label = 'Employee Stayed')

plt.title('Employees Performance', fontsize=14)

plt.xlabel('last_evaluation')

plt.legend(loc='best');
poor_performance_left = data[(data.last_evaluation <= 0.62) & (data.number_project == 2) & (data.left == 1)]

print('poor_performance_left:',len(poor_performance_left))



poor_performance_stayed = data[(data.last_evaluation > 0.62) & (data.number_project == 2) & (data.left == 1)]

print('poor_performance_stayed:',len(poor_performance_stayed))



print('\n')



high_performance_left= data[(data.last_evaluation <= 0.62) & (data.number_project >=5) & (data.left == 1)]

high_performance_stayed= data[(data.last_evaluation > 0.8) & (data.number_project >=5) & (data.left == 0)]

print('high_performance_left:',len(high_performance_left))

print('high_performance_stayed', len(high_performance_stayed))



plt.figure(figsize =(7,5))

bins = np.linspace(1.5,7.5, 7)

plt.hist(data[data['left']==1]['number_project'], bins=bins, alpha=1, label='Employees Left')

plt.hist(data[data['left']==0]['number_project'], bins=bins, alpha = 0.5, label = 'Employee Stayed')

plt.title('Number of projects', fontsize=14)

plt.xlabel('number_ projects')

plt.legend(loc='best');
plt.figure(figsize =(7,5))

bins = np.linspace(80,315, 15)

plt.hist(data[data['left']==1]['average_montly_hours'], bins=bins, alpha=1, label='Employees Left')

plt.hist(data[data['left']==0]['average_montly_hours'], bins=bins, alpha = 0.5, label = 'Employee Stayed')

plt.title('Working Hours', fontsize=14)

plt.xlabel('average_montly_hours')

plt.xlim((70,365))

plt.legend(loc='best');
groupby_number_projects = data.groupby('number_project').mean()

groupby_number_projects = groupby_number_projects['average_montly_hours']

print(groupby_number_projects)

plt.figure(figsize=(7,5))

groupby_number_projects.plot();
work_less_hours_left = data[(data.average_montly_hours < 200) & (data.number_project == 2) & (data.left == 1)]

print('work_less_hours_left:',len(work_less_hours_left))



work_more_hours_left = data[(data.average_montly_hours > 240) & (data.number_project >=5 ) & (data.left == 1)]

print('work_more_hours_left:',len(work_more_hours_left))



#<p><font color="red">Aqui você fala sobre a relação entre horas de trabalho e quantidade de projetos, mas isso não é exibido no gráfico</font></p>
plt.figure(figsize =(7,5))

bins = np.linspace(0.006,1.000, 15)

plt.hist(data[data['left']==1]['satisfaction_level'], bins=bins, alpha=1, label='Employees Left')

plt.hist(data[data['left']==0]['satisfaction_level'], bins=bins, alpha = 0.5, label = 'Employee Stayed')

plt.title('Employees Satisfaction', fontsize=14)

plt.xlabel('satisfaction_level')

plt.xlim((0,1.05))

plt.legend(loc='best');
groupby_time_spend = data.groupby('time_spend_company').mean()

groupby_time_spend['satisfaction_level']
sns.set()

sns.set_context("talk")

ax = sns.factorplot(x="number_project", y="satisfaction_level", col="time_spend_company",col_wrap=4, size=3, color='blue',sharex=False, data=data)

ax.set_xlabels('Number of Projects');
func_living = data[(data.last_evaluation >= 0.70) | (data.time_spend_company >=4) | (data.number_project >= 5)]



corr2 = func_living.corr()



sns.set(style='white')



mask = np.zeros_like(corr2, dtype=np.bool)



mask[np.triu_indices_from(mask)] = True



# Insert the graphic

f, ax = plt.subplots(figsize=(13,8))



cmap = sns.diverging_palette(10,220, as_cmap=True)



#Draw heat map mask

ax = sns.heatmap(corr2, mask=mask, cmap=cmap, vmax= .5, annot=True, annot_kws= {'size':11}, square=True, xticklabels=True, yticklabels=True, linewidths=.5, 

           cbar_kws={'shrink': .5}, ax=ax)

ax.set_title('Correlation: Why Valuable Employees Tend to Leave', fontsize=20);