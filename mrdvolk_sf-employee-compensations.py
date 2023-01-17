import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/employee-compensation.csv')
df.head()
df['Salaries'].hist(bins=20)
total_salary_description = df['Salaries'].describe()
print(total_salary_description)
negative_salaries = df[df['Salaries'] < 0]
negative_salaries_description = negative_salaries['Salaries'].describe()
print(negative_salaries_description)
negative_salaries_description['mean'] / (total_salary_description['count'] / negative_salaries_description['count'])
corrected_df = df[df['Salaries'] > 0].copy() #copying to avoid warnings later
groupped_by_department = corrected_df.groupby(by='Department')
departments_stats = groupped_by_department['Salaries'].agg(['mean', 'sum', 'count', 'std'])
departments_stats['sum'] = departments_stats['sum'] / 1000000000.0

top_departments_by_department_budget = departments_stats.sort_values(by='sum', ascending=False).iloc[0:10]
top_departments_by_average_salary = departments_stats.sort_values(by='mean', ascending=False).iloc[0:10]
def plot_salaries(entries, title, field_name):

    def calc_95_ci(entry):
        mean = entry['mean']
        z = 1.960
        std = entry['std']
        n = entry['count']
        diff = z*std/np.sqrt(n)
        return (mean - diff, mean + diff, diff)

    cb_dark_blue = (0/255, 107/255, 164/255)
    cb_orange = (255/255, 128/255, 14/255)
    cb_light_gray = (171/255, 171/255, 171/255)

    fig = plt.figure()
    fig.set_size_inches(18, 12)
    ax1 = fig.add_subplot(111)

    ax1.set_title(title)

    plt.sca(ax1)
    x_labels = entries.index
    plt.xticks(range(10), x_labels, rotation=60)

    sums = entries['sum'].values
    bar1 = ax1.bar([x for x in range(0, 10)], sums, width=0.8, color=cb_dark_blue)
    #ax1.set_ylim([0, 4])
    ax1.set_ylabel("Department " + field_name + " budget, billions of USD")

    ax2 = ax1.twinx()
    means = entries['mean'].values
    bar2 = ax2.bar(range(0, 10), means, width=0.3, color=cb_orange)
    #ax2.set_ylim([0, 170000])
    ax2.set_ylabel("Individual " + field_name + ", USD")

    fig.legend([bar1, bar2], ['Department ' + field_name + ' budget', 'Mean individual ' + field_name], loc=(0.55, 0.9), ncol=2)

    confidence_intervals = [calc_95_ci(entries.loc[x_labels[i]]) for i in range(0, len(x_labels))]
    salaries = entries['mean'].as_matrix()
    for i in range(0, len(x_labels)):
        plt.errorbar(i, salaries[i], xerr=0, yerr=confidence_intervals[i][2], capsize=10, color='black')

    plt.show()
plot_salaries(top_departments_by_department_budget, 'Top SF departments by department salary budgets', 'salaries')
plot_salaries(top_departments_by_average_salary, 'Top SF departments by average individual salary', 'salaries')
corrected_df['Guaranteed Compensations'] = corrected_df['Salaries'] + corrected_df['Total Benefits']

groupped_by_department = corrected_df.groupby(by='Department')
departments_compensation_stats = groupped_by_department['Guaranteed Compensations'].agg(['mean', 'sum', 'count', 'std'])
departments_compensation_stats['sum'] = departments_compensation_stats['sum'] / 1000000000.0

top_departments_by_compensation_budget = departments_compensation_stats.sort_values(by='sum', ascending=False).iloc[0:10]
top_departments_by_compensation = departments_compensation_stats.sort_values(by='mean', ascending=False).iloc[0:10]
plot_salaries(top_departments_by_compensation_budget, 'Top SF departments by department compensation budgets', 'compensations')
plot_salaries(top_departments_by_compensation, 'Top SF departments by guaranteed compensation', 'compensations')
all(top_departments_by_compensation_budget.index == top_departments_by_department_budget.index)
all(top_departments_by_compensation.index == top_departments_by_average_salary.index)
df_by_salary = top_departments_by_average_salary.reset_index(drop=False)[['Department', 'mean']]
df_by_salary[['Department by salary', 'Salary']] = df_by_salary[['Department', 'mean']]
df_by_salary = df_by_salary[['Department by salary', 'Salary']]

df_by_compensation = top_departments_by_compensation.reset_index(drop=False)[['Department', 'mean']]
df_by_compensation[['Department by compensation', 'Compensation']] = df_by_compensation[['Department', 'mean']]
df_by_compensation = df_by_compensation[['Department by compensation', 'Compensation']]

compare_df = pd.concat([df_by_salary, df_by_compensation], axis=1)
compare_df.head(10)
direct_comparison_df = top_departments_by_average_salary[['mean']].join(
    top_departments_by_compensation[['mean']], lsuffix=' salary', rsuffix=' compensation')


direct_comparison_df['Benefits to total Compensation, %'] = (
    (direct_comparison_df['mean compensation'] - direct_comparison_df['mean salary']) / direct_comparison_df['mean compensation']) * 100
direct_comparison_df = direct_comparison_df.reset_index(drop=False)
direct_comparison_df.plot(kind='bar',x='Department',y='Benefits to total Compensation, %')
