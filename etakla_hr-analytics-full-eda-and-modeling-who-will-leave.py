import numpy as np

import pandas as pd



import scipy.stats

from scipy.stats.stats import pearsonr



import matplotlib.pyplot as plt

%matplotlib inline







from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier





from IPython.display import display_html
#A function that returns the order of a group_by object according to the average of certain parameter param.

def get_ordered_group_index(df, group_by, param, ascending=False):

    return df.groupby(group_by)[param].mean().sort_values(ascending=ascending).index



def group_by_2_level_perc(df, level1, level2, level1_index_order = None, level2_index_order = None):

    #http://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby

    df_by_lvl1_lvl2 = df.groupby([level1, level2]).agg({level1: 'count'})

    df_by_lvl1_lvl2_perc = df_by_lvl1_lvl2.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

    #Reorder them in logical ascending order, but first make sure it is not an empty input

    if level1_index_order:

        df_by_lvl1_lvl2_perc = df_by_lvl1_lvl2_perc.reindex_axis(level1_index_order, axis=0, level=0)

    #If a second level order is passed, apply it, else use the default

    if level2_index_order:

        df_by_lvl1_lvl2_perc = df_by_lvl1_lvl2_perc.reindex_axis(level2_index_order, axis=0, level=1)

    return df_by_lvl1_lvl2_perc



#A function that adds some styling to the graphs, like custom ticks for axes, axes labels and a grid

def customise_2lvl_perc_area_graph(p, legend_lst, xtick_label = "", x_label="", y_label=""):

    #If custom ticks are passed, spread them on the axe and write the tick values

    if xtick_label:

        p.set_xticks(range(0,len(xtick_label)))

        p.set_xticklabels(xtick_label)

    #Create y ticks for grid. It will always be a percentage, so it is not customisable

    p.set_yticks(range(0,110,10)) 

    p.set_yticklabels(['{:3.0f}%'.format(x) for x in range(0,110,10)])

    p.set_yticks(range(0,110,5), minor=True) 



    #Draw grid and set y limit to be only 100 (By default it had an empty area at the top of the graph)

    p.xaxis.grid('on', which='major', zorder=1, color='gray', linestyle='dashed')

    p.yaxis.grid('on', which='major', zorder=1, color='gray', alpha=0.2)

    p.yaxis.grid('on', which='minor', zorder=1, color='gray', linestyle='dashed', alpha=0.2)

    p.set(ylim=(0,100))



    #Customise legend

    p.legend(labels=legend_lst, frameon=True).get_frame().set_alpha(0.2)



    #Put the axes labels

    if x_label:

        p.set_xlabel(x_label)

    if y_label:

        p.set_ylabel(y_label);
import seaborn as sns



s = pd.Series(data=[5850000, 6000000, 5700000, 13100000, 16331452], name='price_doc')

print(statsmodels.__version__)

print(sns.__version__)

_ = sns.distplot(s, bins=2, kde=True)
hr_df = pd.read_csv('../input/HR_comma_sep.csv')
print("The dataset has", hr_df.shape[1], "features and ", hr_df.shape[0], "entries")
hr_df.describe()
hr_df.head()
hr_by_left = hr_df.groupby('left')

employees_left = hr_by_left.get_group(1)

employees_stayed = hr_by_left.get_group(0)
fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))



sns.kdeplot(employees_left.satisfaction_level, ax=axs[0], shade=True, color="r")

kde_plot = sns.kdeplot(employees_stayed.satisfaction_level, ax=axs[0], shade=True, color="g")

kde_plot.legend(labels=['Left', 'Stayed'])



hist_plot = sns.distplot(hr_df.satisfaction_level, ax=axs[1])

box_plot = sns.boxplot(hr_df.satisfaction_level, ax=axs[2])



kde_plot.set(xlim=(0,1.1))

hist_plot.set(xlim=(0,1.1))

box_plot.set(xlim=(0,1.1));
fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))



sns.kdeplot(employees_left.last_evaluation, ax=axs[0], shade=True, color="r")

kde_plot = sns.kdeplot(employees_stayed.last_evaluation, ax=axs[0], shade=True, color="g")

kde_plot.legend(labels=['Left', 'Stayed'])



hist_plot = sns.distplot(hr_df.last_evaluation, ax=axs[1])

box_plot = sns.boxplot(hr_df.last_evaluation, ax=axs[2])



kde_plot.set(xlim=(0,1.1))

hist_plot.set(xlim=(0,1.1))

box_plot.set(xlim=(0,1.1));
fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))



sns.kdeplot(employees_left.number_project, ax=axs[0], shade=True, color="r")

kde_plot = sns.kdeplot(employees_stayed.number_project, ax=axs[0], shade=True, color="g")

kde_plot.legend(labels=['Left', 'Stayed'])



hist_plot = sns.distplot(hr_df.number_project, ax=axs[1], kde=False)

box_plot = sns.boxplot(hr_df.number_project, ax=axs[2])



kde_plot.set(xlim=(0,8))

hist_plot.set(xlim=(0,8))

box_plot.set(xlim=(0,8));
fig, axs = plt.subplots(nrows=3, figsize=(13, 4))



sns.kdeplot(employees_left.average_montly_hours, ax=axs[0], shade=True, color="r")

kde_plot = sns.kdeplot(employees_stayed.average_montly_hours, ax=axs[0], shade=True, color="g")

kde_plot.legend(labels=['Left', 'Stayed'])



hist_plot = sns.distplot(hr_df.average_montly_hours, ax=axs[1])

box_plot = sns.boxplot(hr_df.average_montly_hours, ax=axs[2])



kde_plot.set(xlim=(0,350))

hist_plot.set(xlim=(0,350))

box_plot.set(xlim=(0,350));
fig, axs = plt.subplots(nrows= 3, figsize=(13, 5))



sns.kdeplot(employees_left.time_spend_company, ax=axs[0], shade=True, color="r")

kde_plot = sns.kdeplot(employees_stayed.time_spend_company, ax=axs[0], shade=True, color="g")

kde_plot.legend(labels=['Left', 'Stayed'])



hist_plot = sns.distplot(hr_df.time_spend_company, ax=axs[1], kde=False)

box_plot = sns.boxplot(hr_df.time_spend_company, ax=axs[2])



kde_plot.set(xlim=(0,12))

hist_plot.set(xlim=(0,12))

box_plot.set(xlim=(0,12));
#TODO: CLEAN ME UP! Remove all the commented code

def annotate_bars(bar_plt, bar_plt_var, by=None, x_offset=0, y_offset=0, txt_color="white", fnt_size=12, fnt_weight='bold'):

    if by is None:

        for p in bar_plt.patches:

            bar_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /bar_plt_var.count(), 1) )+ "%", 

                             (p.get_x() + x_offset, p.get_height()-y_offset),

                             color=txt_color, fontsize=fnt_size, fontweight=fnt_weight)

    else:

        grouped = bar_plt_var.groupby(by)

        for p in bar_plt.patches:            

            #This part is tricky. The problem is that not each x-tick gets drawn in order, i.e. yes/no of the first group 

            #then yes/no of the second group located on the next tick, but rather all the yes on all the x-ticks get drawn first

            # then all the nos next. So we need to know we are using a patch that belongs to which tick (the x-tick) ultimately

            #refers to one of the groups. So, we get the x absolute coordinate, round it to know this patch is closest to which tick

            #(Assuming that it will always belong to its closest tick), then get the group count of that tick and use it as a total

            #to compute the percentage.

            total = grouped.get_group(bar_plot.get_xticks()[int(round(p.get_x()))]).count()

            bar_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /total, 1) )+ "%", 

                             (p.get_x() + x_offset, p.get_height()-y_offset),

                             color=txt_color, fontsize=fnt_size, fontweight=fnt_weight)
fig, axs = plt.subplots(ncols= 2, figsize=(13, 5))



work_accidents_plt = sns.countplot(hr_df.Work_accident, ax=axs[0]);

annotate_bars(bar_plt=work_accidents_plt, bar_plt_var=hr_df.Work_accident, x_offset=0.3, y_offset=1100)

    

bar_plot = sns.countplot(x=hr_df.Work_accident, hue=hr_df.left, ax=axs[1])

annotate_bars(bar_plt=bar_plot, by=hr_df.Work_accident, bar_plt_var=hr_df.Work_accident, x_offset=0.1, txt_color="black")

bar_plot.set(ylim=(0,14000));
employees_left_plt = sns.countplot(hr_df.left);



for p in employees_left_plt.patches:

    employees_left_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /hr_df.left.count(), 1) )+ "%", 

                                (p.get_x() + 0.3, p.get_height()-1100),

                                color='white', fontsize=12, fontweight='bold')
fig, axs = plt.subplots(ncols= 2, figsize=(13, 5))



promoted_5years_plt = sns.countplot(hr_df.promotion_last_5years, ax=axs[0]);

annotate_bars(bar_plt=promoted_5years_plt, bar_plt_var=hr_df.promotion_last_5years, x_offset=0.3, txt_color="black")

    

bar_plot = sns.countplot(x=hr_df.promotion_last_5years, hue=hr_df.left, ax=axs[1])

annotate_bars(bar_plt=bar_plot, by=hr_df.promotion_last_5years, bar_plt_var=hr_df.promotion_last_5years, x_offset=0.1, txt_color="black")

bar_plot.set(ylim=(0,16000));
#Create groups

employees_by_promotion = hr_df.groupby("promotion_last_5years")

employees_promoted = employees_by_promotion.get_group(1)

employees_not_promoted = employees_by_promotion.get_group(0)



#Get counts

employees_promoted_stayed = employees_promoted.groupby("left").get_group(0).left.count()

employees_promoted_left = employees_promoted.groupby("left").get_group(1).left.count()



employees_not_promoted_stayed = employees_not_promoted.groupby("left").get_group(0).left.count()

employees_not_promoted_left = employees_not_promoted.groupby("left").get_group(1).left.count()



#Create rows that makeup the contingency table

promoted_row = [employees_promoted_stayed, employees_promoted_left, employees_promoted_stayed + employees_promoted_left]

not_promoted_row = [employees_not_promoted_stayed, employees_not_promoted_left, employees_not_promoted_stayed + employees_not_promoted_left]

total_row = [employees_promoted_stayed+employees_not_promoted_stayed,

             employees_promoted_left+employees_not_promoted_left,

             hr_df.left.count()]



#Create the contingency table

contingency_table = pd.DataFrame({'Promoted': promoted_row ,

                                  'Not Promoted': not_promoted_row ,

                                  'Total, By Left': total_row},

                                 index = ['Stayed', 'Left', 'Total, by Promotion'], 

                                 columns = [ 'Promoted', 'Not Promoted', 'Total, By Left'])



display_html(contingency_table)
chi_squared, p, degrees_of_freedom, expected_frequency = scipy.stats.chi2_contingency( contingency_table )



print("Chi Squared: ", chi_squared)

print("p value: ", p)

print("Degrees of Freedom", degrees_of_freedom)

print("Expected Frequency for The Not Promoted Employees:", expected_frequency[0])

print("Expected Frequency for The Promoted Employees:", expected_frequency[1])
fig, axs = plt.subplots(figsize=(13, 4))



department_plt = sns.countplot(hr_df.sales, order = hr_df.sales.value_counts().index);



annotate_bars(bar_plt=department_plt, bar_plt_var=hr_df.sales, x_offset=0.2, y_offset=450, txt_color="black")
department_plt = sns.countplot(hr_df.salary, order = hr_df.salary.value_counts().index);



for p in department_plt.patches:

    department_plt.annotate(str( int(p.get_height()) ) + "\n" + str(round( (100.0* p.get_height()) /hr_df.salary.count(), 1) )+ "%", 

                                (p.get_x() + 0.3, p.get_height()-800),

                                color='white', fontsize=12, fontweight='bold')
plt.figure(figsize=(12, 8))



hr_corr = hr_df.corr()

sns.heatmap(hr_corr, 

            xticklabels = hr_corr.columns.values,

            yticklabels = hr_corr.columns.values,

            annot = True);
plt.figure(figsize=(10, 10))



sns.pairplot(hr_df,  hue="left");
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.average_montly_hours, order=get_ordered_group_index(hr_df, 'sales', 'average_montly_hours') )
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.satisfaction_level, order=get_ordered_group_index(hr_df, 'sales', 'satisfaction_level'))
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.number_project, order=get_ordered_group_index(hr_df, 'sales', 'number_project'))
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.promotion_last_5years, order=get_ordered_group_index(hr_df, 'sales', 'promotion_last_5years'))
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.last_evaluation, order=get_ordered_group_index(hr_df, 'sales', 'last_evaluation'))
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.time_spend_company, order=get_ordered_group_index(hr_df, 'sales', 'time_spend_company'))
fig, axs = plt.subplots(figsize=(13, 4))



bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.Work_accident, order=get_ordered_group_index(hr_df, 'sales', 'Work_accident'))
fig, axs = plt.subplots(figsize=(13, 4))



#Order the bars descendingly according to the PERCENTAGE % of those who left in each department

total_employees_by_dept = hr_df.groupby(["sales"]).satisfaction_level.count()

left_count_by_dept = hr_df[hr_df["left"] == 1].groupby(["sales"]).satisfaction_level.count()

percentages_left_by_dept = (left_count_by_dept / total_employees_by_dept).sort_values(ascending=False)

axe_name_order = percentages_left_by_dept.index



department_plt = sns.countplot(hr_df.sales, order = axe_name_order, color='g');

sns.countplot(employees_left.sales, order = axe_name_order, color='r');



department_plt.legend(labels=['Stayed', 'Left'])

department_plt.set(xlabel='Department\n Sorted for "Left" Percentage')



#Annotate the percentages of those who stayed. It was more straightforward to loop for each category (left, stayed) than

#doing all the work in one loop

#The zip creates an output that is equal to the shortest parameter, so we do not need to adjust the patches length, since

#the loop will stop after finishing the columns of those who stayed

for p, current_column in zip(department_plt.patches, axe_name_order):

    current_column_total = hr_df[hr_df['sales'] == current_column].sales.count()

    stayed_count = p.get_height() - employees_left[employees_left['sales'] == current_column].sales.count()

    department_plt.annotate(str(round( (100.0* stayed_count) /current_column_total, 1) )+ "%", 

                                (p.get_x() + 0.2, p.get_height()-10),

                                color='black', fontsize=12)

    

#In this loop, we want to use the patches located on the second half of patches list, which are the bars for those who left.

for p, current_column in zip(department_plt.patches[int(len(department_plt.patches)/2):], axe_name_order):

    current_column_total = hr_df[hr_df['sales'] == current_column].sales.count()

    left_count = p.get_height()

    department_plt.annotate(str(round( (100.0* left_count) /current_column_total, 1) )+ "%", 

                                (p.get_x() + 0.2, p.get_height()-10),

                                color='black', fontsize=12)
fig, axs = plt.subplots(figsize=(13, 4))



colours = ['green', 'red']

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.last_evaluation, hue=hr_df.left, palette=colours, order=hr_df.sales.value_counts().index)

bar_plot.set(xlabel='Department', ylabel='Average Evaluation')



plt.plot([-1, 10], [hr_df.last_evaluation.mean(), hr_df.last_evaluation.mean()], linewidth=1);



bar_plot.set(ylim=(0,1));
fig, axs = plt.subplots(figsize=(13, 4))



colours = ['green', 'red']

bar_plot = sns.barplot(x=hr_df.sales, y=hr_df.average_montly_hours, hue=hr_df.left, palette=colours, order=hr_df.sales.value_counts().index)

bar_plot.set(xlabel='Department', ylabel='Average Monthly Hours')



plt.plot([-1, 10], [hr_df.average_montly_hours.mean(), hr_df.average_montly_hours.mean()], linewidth=1);
#Group them according to salary range

low_salaried = hr_df[hr_df.salary == 'low']

mid_salaried = hr_df[hr_df.salary == 'medium']

high_salaried = hr_df[hr_df.salary == 'high']
#Group them then get the percentages

left_percent_dept = (hr_by_left.get_group(1).groupby(['sales']).salary.value_counts(normalize=True))*100

salary_percent_dept = (hr_df.groupby(['sales']).salary.value_counts(normalize=True))*100



fig, axs = plt.subplots(nrows= 2, figsize=(13, 7))



#Draw an area ploy for each group

salary_percent_dept.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[0])

left_percent_dept.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[1])



axs[0].set_title('All Employees')

axs[0].set_xlabel('')

axs[0].set_ylabel('Salary Percentage Makeup\n Stayed')



axs[1].set_title('Employees Who Left')

axs[1].set_xlabel('Department')

axs[1].set_ylabel('Salary Percentage Makeup \n Left')



plt.subplots_adjust(hspace=0.2)
low_salaried_leave_perc = 100*low_salaried.groupby(['sales', 'left']).salary.agg({'sales': 'count'})/low_salaried.groupby(['sales']).salary.agg({'sales': 'count'})

mid_salaried_leave_perc = 100*mid_salaried.groupby(['sales', 'left']).salary.agg({'sales': 'count'})/mid_salaried.groupby(['sales']).salary.agg({'sales': 'count'})

high_salaried_leave_perc = 100*high_salaried.groupby(['sales', 'left']).salary.agg({'sales': 'count'})/high_salaried.groupby(['sales']).salary.agg({'sales': 'count'})



fig, axs = plt.subplots(nrows= 3, figsize=(13, 9))



low_salaried_leave_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[0])

mid_salaried_leave_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[1])

high_salaried_leave_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[2])



axs[0].set_title('Low Salaried Employees')

axs[1].set_title('Medium Salaried Employees')

axs[2].set_title('High Salaried Employees')



axs[0].set_xlabel('')

axs[1].set_xlabel('')

axs[2].set_xlabel('Department')



axs[0].set_ylabel('Left to Stayed Makeup')

axs[1].set_ylabel('Left to Stayed Makeup')

axs[2].set_ylabel('Left to Stayed Makeup')



plt.subplots_adjust(hspace=0.3);
departments = list(set(hr_df.sales.values))

number_of_departments = len(departments)



fig, axs = plt.subplots(nrows= int(number_of_departments/2), ncols=2, figsize=(13, 20))



for i in range(number_of_departments):

    current_dep = departments[i]

    

    ratio_df = 100*hr_df[hr_df.sales == current_dep].groupby(['salary', 'left']).agg({'salary': 'count'})/hr_df[hr_df.sales == current_dep].groupby(['salary']).agg({'salary': 'count'})

    

    #plot the department

    ratio_df.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[int(i/2),i%2])

    axs[int(i/2),i%2].set_title(current_dep)

    axs[int(i/2),i%2].set_xlabel("")

    

axs[int(i/2),i%2].set_xlabel("Salary")

plt.subplots_adjust(hspace=0.3);
fig, axs = plt.subplots(figsize=(13, 4))



axe_name_order = hr_df.salary.value_counts().index



salary_plt = sns.countplot(hr_df.salary, order = axe_name_order, color='g');

sns.countplot(employees_left.salary, order = axe_name_order, color='r');



salary_plt.legend(labels=['Stayed', 'Left'])



#Annotate the percentages of those who stayed. It was more straightforward to loop for each category (left, stayed) than

#doing all the work in one loop

#The zip creates an output that is equal to the shortest parameter, so we do not need to adjust the patches length, since

#the loop will stop after finishing the columns of those who stayed

for p, current_column in zip(salary_plt.patches, axe_name_order):

    current_column_total = hr_df[hr_df['salary'] == current_column].salary.count()

    stayed_count = p.get_height() - employees_left[employees_left['salary'] == current_column].salary.count()

    salary_plt.annotate(str(round( (100.0* stayed_count) /current_column_total, 1) )+ "%", 

                                (p.get_x() + 0.35, p.get_height()-10),

                                color='black', fontsize=12)

    

#In this loop, we want to use the patches located on the second half of patches list, which are the bars for those who left.

for p, current_column in zip(salary_plt.patches[int(len(salary_plt.patches)/2):], axe_name_order):

    current_column_total = hr_df[hr_df['salary'] == current_column].salary.count()

    left_count = p.get_height()

    salary_plt.annotate(str(round( (100.0* left_count) /current_column_total, 1) )+ "%", 

                                (p.get_x() + 0.35, p.get_height()-10),

                                color='black', fontsize=12)
fig, axs = plt.subplots(figsize=(16, 4))



sns.stripplot(y = 'salary', x='average_montly_hours', hue='left', data=hr_df);
#A function to bin the average monthly hours into the categories described above

def work_load_cat(avg_mnthly_hrs):

    work_load = "unknown"

    if avg_mnthly_hrs < 168:

        work_load = "low"

    elif (avg_mnthly_hrs >= 168) & (avg_mnthly_hrs < 210):

        work_load = "average"

    elif (avg_mnthly_hrs >= 210) & (avg_mnthly_hrs < 252):

        work_load = "above_average"

    elif avg_mnthly_hrs >= 252:

        work_load = "workoholic"

        

    return work_load
hr_df['work_load'] = hr_df.average_montly_hours.apply(work_load_cat)



sns.countplot(x='work_load', hue='left', data=hr_df, order = ['low', 'average', 'above_average', 'workoholic']);


#Normalised stacked

departments = list(set(hr_df.sales.values))

number_of_departments = len(departments)



fig, axs = plt.subplots(nrows= int(number_of_departments/2), ncols=2, figsize=(13, 20))



for i in range(number_of_departments):

    current_dep = departments[i]

    

    ratio_df = 100*hr_df[hr_df.sales == current_dep].groupby(['work_load', 'left']).agg({'work_load': 'count'})/hr_df[hr_df.sales == current_dep].groupby(['work_load']).agg({'work_load': 'count'})

    ratio_df = ratio_df.reindex_axis(["low", "average", "above_average", "workoholic"], axis=0, level=0)

    #plot the department

    ratio_df.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs[int(i/2),i%2])

    axs[int(i/2),i%2].set_title(current_dep)

    axs[int(i/2),i%2].set_xlabel("")

    

axs[int(i/2),i%2].set_xlabel("work_load")

plt.subplots_adjust(hspace=0.3);
#A function to bin last evaluation into one of 5 categories

def last_evaluation_cat(last_evaluation):

    evaluation = "unknown"

    if last_evaluation < 0.45:

        evaluation = "very_low"

    elif (last_evaluation >= 0.45) & (last_evaluation < 0.55):

        evaluation = "mediocre"

    elif (last_evaluation >= 0.55) & (last_evaluation < 0.8):

        evaluation = "average"

    elif (last_evaluation >= 0.8) & (last_evaluation < 0.9):

        evaluation = "very_good"

    elif last_evaluation >= 0.9:

        evaluation = "excellent"

        

    return evaluation
hr_df['evaluation'] = hr_df.last_evaluation.apply(last_evaluation_cat)
sns.countplot(x='evaluation',  data=hr_df, order = ["unknown", 'very_low', 'mediocre', 'average', 'very_good', 'excellent']);
sns.countplot(x='evaluation',  hue = 'left', data=hr_df, order = ["unknown", 'very_low', 'mediocre', 'average', 'very_good', 'excellent']);
evaluation_index_order = ["unknown", 'very_low', 'mediocre', 'average', 'very_good', 'excellent']

evaluation_xticks = ['Very Low\n (eval < .45)', 'Mediocre\n ( .45 < eval < .55 )', 'Average\n ( .55 < eval < .8 )', 'Very Good\n ( .8 < eval < .9 )', 'Excellent\n ( .9 < eval)']

evaluation_x_label = "Company Evaluation for the Employee"
employees_by_eval_and_workload = group_by_2_level_perc(hr_df, 

                                                       'evaluation', 'work_load',

                                                       evaluation_index_order, ['low','average','above_average', 'workoholic'])#Index Order



workload_legend = ['Low Workload (< 40hrs/week)', 'Average Workload (40 < wl < 50 hrs/week)', 'Above Average Workload (50 < wl < 60hrs/week)', 'Workoholic Workload (wl > 60hrs/week)']



#Plot the Graph

p=employees_by_eval_and_workload.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



customise_2lvl_perc_area_graph(p, workload_legend, 

                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph

                               y_label="Percentage of Monthly Workload")
employees_by_eval_and_time_in_company_perc = group_by_2_level_perc(hr_df, 

                                                                   'evaluation', 'time_spend_company',

                                                                   evaluation_index_order)#Index Order



#Plot the Graph

p=employees_by_eval_and_time_in_company_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



time_spent_legend = [str(x) + " years" for x in range(2,9)] + ['10 years']



customise_2lvl_perc_area_graph(p, time_spent_legend, 

                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph

                               y_label="Percentage of Years in Company")
employees_by_eval_and_time_in_company_perc = group_by_2_level_perc(hr_df, 

                                                                   'evaluation', 'number_project',#Variables to groupby

                                                                   evaluation_index_order)#Index Order



#Plot the Graph

p=employees_by_eval_and_time_in_company_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



num_projects_legend = [str(x) + " projects" for x in range(2,8)]



customise_2lvl_perc_area_graph(p, num_projects_legend, 

                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph

                               y_label="Percentage of Number of Projects Assigned")
employees_by_eval_and_salary_perc = group_by_2_level_perc(hr_df, 

                                                          'evaluation', 'salary', 

                                                          evaluation_index_order, ['low', 'medium', 'high'])



#Plot the Graph

p=employees_by_eval_and_salary_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



num_projects_legend = ['Low', 'Medium', 'High']



customise_2lvl_perc_area_graph(p, num_projects_legend, 

                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph

                               y_label="Percentage of Salary Range")
#Create a satisfaction categories

#Arbitrary boundaries:

# < 4.5 low

# 4.5 < < 7.5 medium

# 7.5 < high

def rank_satisfaction(employee):

    level = "unknown"

    if employee.satisfaction_level < 0.45:

        level='low'

    elif employee.satisfaction_level < 0.75:

        level = 'medium'

    else:

        level = 'high'

    return level
hr_df['satisfaction'] = hr_df.apply(rank_satisfaction, axis=1)
employees_by_eval_and_satisfaction_perc = group_by_2_level_perc(hr_df, 

                                                                'evaluation', 'satisfaction', 

                                                                evaluation_index_order, ['low', 'medium', 'high'])



#Plot the Graph

p=employees_by_eval_and_satisfaction_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)





satisfaction_lvl_legend = ['Low', 'Medium', 'High']



customise_2lvl_perc_area_graph(p, satisfaction_lvl_legend, 

                               xtick_label=evaluation_xticks, x_label=evaluation_x_label, #Company Evaluation Graph

                               y_label="Percentage of Employee's Satisfaction Level")
y = hr_df.evaluation.copy()

X = hr_df.copy()

#Remove the label and other columns we have created that categorized continuous variables

X = X.drop(["left", "satisfaction", "last_evaluation", "evaluation", "work_load"], axis=1)
X.head()
print(y.isnull().any())

print(X.isnull().any())
#Use the label encoder from Scikit-learn to do the conversion

le_sales = LabelEncoder()

le_salary = LabelEncoder()

le_evaluation = LabelEncoder()



#Fit\Transform the data

le_sales.fit(X.sales)

le_salary.fit(X.salary)

le_evaluation.fit(y)



X.sales = le_sales.transform(X.sales)

X.salary = le_salary.transform(X.salary)

y = le_evaluation.transform(y)



#Convert the labels from integers to np so that the estimator wouldn't complain

y = np.float32(y)



min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)
#train, test = train_test_split(hr_df, test_size= 0.2)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
evaluation_decision_tree = DecisionTreeClassifier()



#Stratify split and train on 5 folds

skf = StratifiedKFold(n_splits=5)

counter = 1

for train_fold, test_fold in skf.split(X_train, y_train):

    evaluation_decision_tree.fit( X_train[train_fold], y_train[train_fold])

    print( str(counter) + ": ", evaluation_decision_tree.score(X_train[test_fold], y_train[test_fold]))

    counter += 1
features_order = ['satisfaction_level', 'number_project', 'average_montly_hours', 

                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']

feature_importance_dict = {key: val for key, val in zip(features_order, evaluation_decision_tree.feature_importances_)}



#http://stackoverflow.com/questions/20944483/python-3-sort-a-dict-by-its-values

print([(k, feature_importance_dict[k]) for k in sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)])
dept_eval_perc = 100*hr_df.groupby(['sales', 'evaluation']).salary.agg({'sales': 'count'})/hr_df.groupby(['sales']).salary.agg({'sales': 'count'})

dept_eval_perc = dept_eval_perc.reindex_axis(['very_low','mediocre', 'average','very_good', 'excellent'], axis=0, level=1)

fig, axs = plt.subplots(figsize=(15, 6))



dept_eval_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', ax=axs, ylim=(0,100))



axs.set_title('Evaluation By Department')

axs.set_xlabel('Department')

axs.set_ylabel('Evaluation % Makeup')



plt.subplots_adjust(hspace=0.3);
satisfaction_index_order = ["unknown", 'low', 'medium', 'high']

satisfaction_xticks = ['Low\n (satisf. < .45)', 'Medium\n ( .45 < satisf. < .75 )', 'High\n ( .9 < satisf.)']

satisfaction_x_label = "Employee's Satisfaction"
dept_legend = set(hr_df.sales.values)

satisfaction_legend = ['low', 'medium', 'high']



dept_low_satisf_order = (hr_df[hr_df['satisfaction'] == 'low'].groupby('sales').satisfaction_level.count()/hr_df.groupby('sales').satisfaction_level.count()).sort_values(ascending=False).index

dept_low_satisf_order = list(dept_low_satisf_order)



employees_by_satisf_and_department_perc = group_by_2_level_perc(hr_df, 

                                                                'sales', 'satisfaction', 

                                                                dept_low_satisf_order, satisfaction_index_order)



#Plot the Graph

p=employees_by_satisf_and_department_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)







customise_2lvl_perc_area_graph(p, satisfaction_legend, 

                               xtick_label=dept_legend, x_label='Department\n Ordered Descendigly for "Low Happiness" Percentage',

                               y_label="Percentage of Happiness Level")
employees_by_satisf_and_salary_perc = group_by_2_level_perc(hr_df, 

                                                                'satisfaction', 'salary', 

                                                                satisfaction_index_order, ['low', 'medium', 'high'])



salary_legend = ['Low Salary', 'Medium Salary', 'High Salary']

#Plot the Graph

p=employees_by_satisf_and_salary_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



customise_2lvl_perc_area_graph(p, salary_legend, 

                               xtick_label=satisfaction_xticks, x_label=satisfaction_x_label, #Employee Satisfaction Graph

                               y_label="Percentage of Employee's Salary Range")
employees_by_satisf_and_salary_perc = group_by_2_level_perc(hr_df, 

                                                                'time_spend_company', 'satisfaction', list(range(2,9)) + [10],

                                                                satisfaction_index_order )





#Plot the Graph

p=employees_by_satisf_and_salary_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



satisfaction_index_order = ["unknown", 'low', 'medium', 'high']

satisfaction_xticks = ['Low\n (satisf. < .45)', 'Medium\n ( .45 < eval < .75 )', 'High\n ( .9 < satisf.)']



customise_2lvl_perc_area_graph(p, ["Low Satisfaction", "Medium Satisfaction", "High  Satisfaction"], 

                               xtick_label=[0, 1,"2\n years", '3\n years', '4\n years', '5\n years', '6\n years', '7\n years', '8\n years', '10\n years'], 

                               x_label="Time Working for The Company",

                               y_label="Percentage of Employee's Salary Range")



p.set(xlim=(2,9))
employees_by_satisf_and_workload_perc = group_by_2_level_perc(hr_df, 

                                                                'satisfaction', 'work_load', 

                                                                satisfaction_index_order, ['low','average','above_average', 'workoholic'])



salary_legend = ['low','average','above_average', 'workoholic']

#Plot the Graph

p=employees_by_satisf_and_workload_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



customise_2lvl_perc_area_graph(p, salary_legend, 

                               xtick_label=satisfaction_xticks, x_label=satisfaction_x_label, #Employee Satisfaction Graph

                               y_label="Percentage of Employee's Salary Range")
low_work_load_employees = hr_df[hr_df['work_load'] == 'low']

pearsonr(low_work_load_employees.satisfaction_level, low_work_load_employees.average_montly_hours)[0]
sns.lmplot("satisfaction_level", "average_montly_hours", data=low_work_load_employees);
sns.lmplot("satisfaction_level", "average_montly_hours", hue="left", data=low_work_load_employees);
employees_by_satisf_and_proj_num_perc = group_by_2_level_perc(hr_df, 

                                                                'satisfaction', 'number_project', 

                                                                satisfaction_index_order,)



salary_legend = ["2 projects", '3 projects', '4 projects', '5 projects', '6 projects', '7 projects']#['low','average','above_average', 'workoholic']

#Plot the Graph

p=employees_by_satisf_and_proj_num_perc.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(15, 6), zorder=0)



customise_2lvl_perc_area_graph(p, salary_legend, 

                               xtick_label=satisfaction_xticks, x_label=satisfaction_x_label, #Employee Satisfaction Graph

                               y_label="Percentage of Employee's Salary Range")
#low_work_load_employees = hr_df[hr_df['work_load'] == 'low']

pearsonr(hr_df.number_project, hr_df.average_montly_hours)[0]
y = hr_df.satisfaction.copy()

X = hr_df.copy()

X = X.drop(["left", "satisfaction_level", "satisfaction", "evaluation", "work_load"], axis=1)
le_sales = LabelEncoder()

le_salary = LabelEncoder()

le_satisfaction = LabelEncoder()



le_sales.fit(X.sales)

le_salary.fit(X.salary)

le_satisfaction.fit(y)



X.sales = le_sales.transform(X.sales)

X.salary = le_salary.transform(X.salary)

y = le_salary.transform(y)
y = np.float32(y)
min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
happiness_decision_tree = DecisionTreeClassifier()



#Stratify split and train on 5 folds

skf = StratifiedKFold(n_splits=5)

counter = 1

for train_fold, test_fold in skf.split(X_train, y_train):

    happiness_decision_tree.fit( X_train[train_fold], y_train[train_fold])

    print( str(counter) + ": ", happiness_decision_tree.score(X_train[test_fold], y_train[test_fold]))

    counter += 1
features_order = ['last_evaluation', 'number_project', 'average_montly_hours', 

                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']

feature_importance_dict = {key: val for key, val in zip(features_order, happiness_decision_tree.feature_importances_)}

print(feature_importance_dict)
#Get the label

y = hr_df.left

X = hr_df.copy()

#Drop unnecessary\duplicate columns

X = X.drop(["left", "satisfaction", "evaluation", "work_load"], axis=1)



#Prepocess the data:

#Encode categorical variables into numeric representations

le_sales = LabelEncoder()

le_salary = LabelEncoder()



le_sales.fit(X.sales)

le_salary.fit(X.salary)



X.sales = le_sales.transform(X.sales)

X.salary = le_salary.transform(X.salary)



#Zero mean, 1 std

min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)



y = np.float32(y)



#Train\Test split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.35, random_state=42, stratify=y)



leaving_random_forest = RandomForestClassifier(n_estimators=100)



#Stratify split and train on 5 folds

skf = StratifiedKFold(n_splits=5)

counter = 1

for train_fold, test_fold in skf.split(X_train, y_train):

    leaving_random_forest.fit( X_train[train_fold], y_train[train_fold])

    print( str(counter) + ": ", leaving_random_forest.score(X_train[test_fold], y_train[test_fold]))

    counter += 1
features_order = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 

                  'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']

feature_importance_dict = {key: val for key, val in zip(features_order, leaving_random_forest.feature_importances_)}
#http://stackoverflow.com/questions/20944483/python-3-sort-a-dict-by-its-values

print([(k, feature_importance_dict[k]) for k in sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)])