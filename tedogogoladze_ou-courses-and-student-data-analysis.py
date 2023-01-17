# loading the datasets and needed libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

vle = pd.read_csv('../input/vle.csv')

assessments = pd.read_csv('../input/assessments.csv')

courses = pd.read_csv('../input/courses.csv')

studentAssessment = pd.read_csv('../input/studentAssessment.csv')

studentInfo = pd.read_csv('../input/studentInfo.csv')

studentRegistration = pd.read_csv('../input/studentRegistration.csv')

studentVle = pd.read_csv('../input/studentVle.csv')

# This function is returning count of rows and percentage allocation within a specific column, 
#and for each unique value of another column

# Function has 3 parameters: column of tab, row of the tab, and dataset.

def tabel(colname, rowname, data):
    tab = pd.crosstab(data[colname], data[rowname]) 
    tab.reset_index(level=0, inplace=True)
    
    tab['total'] = 0
    cols = [colname]
    for col in data[rowname].unique():
        tab['total'] = tab['total'] + tab[col]    
        cols.append(col)
    cols.append('total')
    for col in data[rowname].unique():
        ratiocol = col + '_%'
        tab[ratiocol] = round(100 * tab[col] / tab['total'], 2)
    return tab
# Showing the column results and ratios out of total

colname = 'age_band'
tabel(colname, 'final_result', studentInfo)
colname = 'region'
tabel(colname, 'final_result', studentInfo)

colname = 'highest_education'
tabel(colname, 'final_result', studentInfo)
colname = 'studied_credits'
credits = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
tab = pd.crosstab(studentInfo[studentInfo[colname].isin(credits)][colname], studentInfo[studentInfo[colname].isin(credits)].final_result) 

tab.reset_index(level=0, inplace=True)
tab.columns = [colname, 'Distinction', 'Fail', 'Pass', 'Withdrawn']
tab['total'] = tab.Distinction + tab.Fail + tab.Pass + tab.Withdrawn
tab['Distinction_%'] = round(100 * tab['Distinction'] / tab['total'], 2)
tab['Fail_%'] = round(100 * tab['Fail'] / tab['total'], 2)
tab['Pass_%'] = round(100 * tab['Pass'] / tab['total'], 2)
tab['Withdrawn_%'] = round(100 * tab['Withdrawn'] / tab['total'], 2)
tab

colname = 'imd_band'
tabel(colname, 'final_result', studentInfo)

colname = 'gender'
tabel(colname, 'final_result', studentInfo)
# this cell combines student ages table with vle table [vle_aged]
# and shows average number of clicks per module per presentation. (? might need improvements)


student_ages = studentInfo[['id_student', 'age_band']].drop_duplicates(keep=False)

vle_aged = pd.merge(studentVle, student_ages, how = 'left', left_on = 'id_student', right_on = 'id_student')
for course in courses.code_presentation.unique():
    vle_aged1 = vle_aged[(vle_aged['code_presentation'] == course)]
    tab = pd.crosstab(vle_aged1.age_band
                    , vle_aged1.code_module
                    , values =  vle_aged1.sum_click
                    , aggfunc = 'mean')
    print(course, tab)
# students info table merged with number of clicks on average.

clicks = studentVle.groupby(['id_student', 'code_module', 'code_presentation']).agg({'sum_click':'mean'})
clicks.reset_index(level=[0, 1, 2], inplace=True)
clicks

results_aged = pd.merge(studentInfo, 
                        clicks, 
                        how = 'left', 
                        left_on = ['id_student', 'code_module', 'code_presentation'], 
                        right_on = ['id_student', 'code_module', 'code_presentation'])

# This cell shows average clicks on each final esult for each courses. code presentation is not distincted.

tab = pd.crosstab(results_aged.code_module
                    , results_aged.final_result
                    , values =  results_aged.sum_click
                    , aggfunc = 'mean')

sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(tab, annot=True, cmap='Reds', fmt='g')

# It is shown here that higher number of clicks on average, results in more positive results.
# Defining the function for listing average clicks on Passed, Failed, Distincted or withdrawed courses.
# Function has 4 parameters. First two params - height and width of the main area of subplot will be allocated 
# with these filtering columns (in the next examples they are 'code_module' and 'code_presentation')
# third param is column that will be allocated on y axis (x axis will be constantly the final_result)
# fourth param is dataset.


def variable_results(filter_col=None, filter_col2='', param=None, data=None):
    i = 1
    nrow = results_aged[filter_col].unique().size
    ncol = results_aged[filter_col2].unique().size
    figure = plt.figure(figsize=(18,36))
    for val1 in results_aged[filter_col].unique():
        for val2 in results_aged[filter_col2].unique():
            ax = figure.add_subplot(nrow, ncol, i)
            ax.set_title(val2 + ' / ' + val1)
            df = []
            df = data[data[filter_col] == val1]
            if filter_col2 != '':
                df = df[df[filter_col2] == val2]
            tab = pd.crosstab(df[param]
                                , results_aged.final_result
                                , values =  results_aged.sum_click
                                , aggfunc = 'mean')
            if tab.empty == False:
                tab['Distinction'] = round(tab['Distinction'], 2) 
                tab['Pass'] = round(tab['Pass'], 2)
                tab['Fail'] = round(tab['Fail'], 2)
                tab['Withdrawn'] = round(tab['Withdrawn'], 2)
                sns.heatmap(tab, annot=True, fmt='g', cmap = 'RdYlGn')
            i = i + 1


variable_results('code_module', 'code_presentation', 'age_band', results_aged)
variable_results('code_module', 'code_presentation', 'imd_band', results_aged)
variable_results('code_module', 'code_presentation', 'highest_education', results_aged)
variable_results('code_module', 'code_presentation', 'gender', results_aged)
# Showing activity types with the number of sites

vle.groupby(['code_presentation', 'code_module','activity_type']).size()

#Activity types with number of websites
print(vle.groupby(['activity_type']).size())


# merge tables - add vle detailed information onto studentVle.

vle_details = pd.merge(studentVle, 
                        vle, 
                        how = 'left', 
                        left_on = ['code_module', 'code_presentation', 'id_site'], 
                        right_on = ['code_module', 'code_presentation', 'id_site'])


# Group activities for each site. aggregate mean and sum of the clicks.
vle_activity_totals  = vle_details.groupby(['id_site', 'activity_type']).agg({'sum_click': ['mean', 'sum']})
vle_activity_totals.reset_index(level= [0,1], inplace=True)
vle_activity_totals.columns = ['id_site', 'activity_type', 'mean_clicks', 'sum_clicks'] 

# Sites with top average activities
print(vle_activity_totals.sort_values('mean_clicks', ascending = False).head(6))


#Mean number of clicks on sites. each point represents site_id and mean activities on that.
# Boxplot shows popularity (in terms of clicks) of these sites, groupped by categories.
sns.set(rc={'figure.figsize':(15,12)})
g = sns.boxplot(x = vle_activity_totals[vle_activity_totals['mean_clicks'] <= 15].activity_type,
                y = vle_activity_totals[vle_activity_totals['mean_clicks'] <= 15].mean_clicks)
g.set_xticklabels(g.get_xticklabels(), rotation=30, size = 15)

fig = g.get_figure()
fig.savefig('interactions.png', bbox_inches='tight')
# All the same logic, but with the interactions ONLY BEFORE the course started.

vle_details = pd.merge(studentVle[studentVle['date'] < 0], 
                        vle, 
                        how = 'left', 
                        left_on = ['code_module', 'code_presentation', 'id_site'], 
                        right_on = ['code_module', 'code_presentation', 'id_site'])


# Group activities for each site. aggregate mean and sum of the clicks.
vle_activity_totals  = vle_details.groupby(['id_site', 'activity_type']).agg({'sum_click': ['mean', 'sum']})
vle_activity_totals.reset_index(level= [0,1], inplace=True)
vle_activity_totals.columns = ['id_site', 'activity_type', 'mean_clicks', 'sum_clicks'] 



# There is an outlier that needs to be filtered outfor better plot in the next cell.
print(vle_activity_totals.sort_values('mean_clicks', ascending = False).head(6))


sns.set(rc={'figure.figsize':(16,12)})
g = sns.boxplot(x = vle_activity_totals.activity_type, y = vle_activity_totals.mean_clicks)
g.set_xticklabels(g.get_xticklabels(), rotation=30, size = 15)

# lets remove OUTLIER to see better graph for overal image
vle_activity_totals = vle_activity_totals[vle_activity_totals['mean_clicks'] < 15]


sns.set(rc={'figure.figsize':(16,12)})
g = sns.boxplot(x = vle_activity_totals.activity_type, y = vle_activity_totals.mean_clicks)
g.set_xticklabels(g.get_xticklabels(), rotation=30, size = 15)

# - check influence of num_of_prev_attempts.studentInfo on final_result.studentInfo

colname = 'num_of_prev_attempts'
attempts_dt = tabel(colname, 'final_result', studentInfo)

attempts_dt
# - visualise activity_type.vle for different modules, 

figure = plt.figure(figsize=(18,16))
i = 1
for semester in vle['code_presentation'].unique():
    ax = figure.add_subplot(2, 2, i)
    ax.set_title(semester)
    df = []
    df = vle[vle['code_presentation'] == semester]
    
    tab = pd.crosstab(df.code_module, df.activity_type)
    sns.heatmap(tab, annot=True, fmt='g', cmap = 'Greens')
    i = i + 1
#plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.5)
# check average score.studentAssessment for each module to suggest which activities were most efficient

        
weights = assessments.groupby(['code_module', 'code_presentation']).agg({'weight':'sum'})
weights.reset_index(level=[0, 1], inplace=True)
weights.columns = ['code_module', 'code_presentation', 'sum_weight']
weights


assessments_totalled = pd.merge(studentAssessment, 
                        assessments, 
                        how = 'left', 
                        left_on = ['id_assessment'], 
                        right_on = ['id_assessment'])

assessments_totalled2 = pd.merge(assessments_totalled, 
                        weights, 
                        how = 'left', 
                        left_on = ['code_module', 'code_presentation'], 
                        right_on = ['code_module', 'code_presentation'])

assessments_totalled2['weight_ratio'] = assessments_totalled2['weight'] / assessments_totalled2['sum_weight']
assessments_totalled2['assessment_score'] = assessments_totalled2['score'] * assessments_totalled2['weight_ratio']
assessments_totalled2.head(6)

student_scores = assessments_totalled2.groupby(['code_presentation', 'code_module', 'id_student']).agg({'assessment_score':'sum'})
student_scores.reset_index(level=[0, 1, 2], inplace=True)
student_result_scores = pd.merge(student_scores, 
                        studentInfo, 
                        how = 'left', 
                        left_on = ['code_module', 'code_presentation', 'id_student'], 
                        right_on = ['code_module', 'code_presentation', 'id_student'])

student_result_scores = student_result_scores[[ 'code_presentation', 'code_module', 'id_student', 'assessment_score', 'final_result']]
student_result_scores


figure = plt.figure(figsize=(17,20))
i = 1

for module in student_result_scores['code_module'].unique():
    ax = figure.add_subplot(3, 3, i)
    ax.set_title(module)
    
    dt = student_result_scores[student_result_scores['code_module'] == module]
    
    sns.boxplot(x = student_result_scores.final_result, y = student_result_scores.assessment_score)
    #g.set_xticklabels(g.get_xticklabels(), rotation=30, size = 15)
    i = i + 1

# This boxplot will show, out of all results (pass/fail/distinct/withdraw) the students courses, how the 
#students were submitting submissions in time or delayed. Negative number on plot Y axis means the student was submitting 
#course assignments earlier than deadline. Positive number means vice-versa - student was submitting late.
assessments_totalled = pd.merge(studentAssessment, 
                        assessments, 
                        how = 'left', 
                        left_on = ['id_assessment'], 
                        right_on = ['id_assessment'])
assessments_totalled['delay'] = assessments_totalled['date_submitted'] - assessments_totalled['date']
assessments_totalled



student_delays = assessments_totalled.groupby(['code_presentation', 'code_module', 'id_student']).agg({'delay':'mean'})
student_delays.reset_index(level=[0, 1, 2], inplace=True)
student_delays

results_aged2 = pd.merge(results_aged, 
                        student_delays, 
                        how = 'left', 
                        left_on = ['id_student', 'code_module', 'code_presentation'], 
                        right_on = ['id_student', 'code_module', 'code_presentation'])

sns.set(rc={'figure.figsize':(14,10)})
g = sns.boxplot(x = results_aged2[ (results_aged2['delay'] > -150)].final_result
                , y = results_aged2[ (results_aged2['delay'] > -150)].delay)


fig = g.get_figure()
fig.savefig('delays.png', bbox_inches='tight')
