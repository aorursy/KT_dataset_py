%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter
df = pd.read_csv("../input/StudentsPerformance.csv")
df.isnull().sum()
df.head()
df_dummy = pd.get_dummies(df)
df_dummy.head()
check_male = df_dummy[df_dummy["gender_male"]==1]
check_female = df_dummy[df_dummy["gender_female"]==1]

check_group_A = df_dummy[df_dummy["race/ethnicity_group A"]==1]
check_group_B = df_dummy[df_dummy["race/ethnicity_group B"]==1]
check_group_C = df_dummy[df_dummy["race/ethnicity_group C"]==1]
check_group_D = df_dummy[df_dummy["race/ethnicity_group D"]==1]
check_group_E = df_dummy[df_dummy["race/ethnicity_group E"]==1]

check_master_degree = df_dummy[df_dummy["parental level of education_master's degree"]==1]
check_bachelor_degree = df_dummy[df_dummy["parental level of education_bachelor's degree"]==1]
check_associate_degree = df_dummy[df_dummy["parental level of education_associate's degree"]==1]
check_some_college = df_dummy[df_dummy["parental level of education_some college"]==1]
check_highschool = df_dummy[df_dummy["parental level of education_high school"]==1]
check_some_highschool = df_dummy[df_dummy["parental level of education_some high school"]==1]

check_lunch_free = df_dummy[df_dummy["lunch_free/reduced"]==1]
check_lunch_standard = df_dummy[df_dummy["lunch_standard"]==1]

check_course_yes = df_dummy[df_dummy["test preparation course_completed"]==1]
check_course_no = df_dummy[df_dummy["test preparation course_none"]==1]
male_math = check_male["math score"]
female_math = check_female["math score"]
group_A_math = check_group_A["math score"]
group_B_math = check_group_B["math score"]
group_C_math = check_group_C["math score"]
group_D_math = check_group_D["math score"]
group_E_math = check_group_E["math score"]
master_degree_math = check_master_degree["math score"]
bachelor_degree_math = check_bachelor_degree["math score"]
associate_degree_math = check_associate_degree["math score"]
some_college_math = check_some_college["math score"]
highschool_math = check_highschool["math score"]
some_highschool_math = check_some_highschool["math score"]
lunch_free_math = check_lunch_free["math score"]
lunch_standard_math = check_lunch_standard["math score"]
course_yes_math = check_course_yes["math score"]
course_no_math = check_course_no["math score"]
male_reading = check_male["reading score"]
female_reading = check_female["reading score"]
group_A_reading = check_group_A["reading score"]
group_B_reading = check_group_B["reading score"]
group_C_reading = check_group_C["reading score"]
group_D_reading = check_group_D["reading score"]
group_E_reading = check_group_E["reading score"]
master_degree_reading = check_master_degree["reading score"]
bachelor_degree_reading = check_bachelor_degree["reading score"]
associate_degree_reading = check_associate_degree["reading score"]
some_college_reading = check_some_college["reading score"]
highschool_reading = check_highschool["reading score"]
some_highschool_reading = check_some_highschool["reading score"]
lunch_free_reading = check_lunch_free["reading score"]
lunch_standard_reading = check_lunch_standard["reading score"]
course_yes_reading = check_course_yes["reading score"]
course_no_reading = check_course_no["reading score"]

male_writing = check_male["writing score"]
female_writing = check_female["writing score"]
group_A_writing = check_group_A["writing score"]
group_B_writing = check_group_B["writing score"]
group_C_writing = check_group_C["writing score"]
group_D_writing = check_group_D["writing score"]
group_E_writing = check_group_E["writing score"]
master_degree_writing = check_master_degree["writing score"]
bachelor_degree_writing = check_bachelor_degree["writing score"]
associate_degree_writing = check_associate_degree["writing score"]
some_college_writing = check_some_college["writing score"]
highschool_writing = check_highschool["writing score"]
some_highschool_writing = check_some_highschool["writing score"]
lunch_free_writing = check_lunch_free["writing score"]
lunch_standard_writing = check_lunch_standard["writing score"]
course_yes_writing = check_course_yes["writing score"]
course_no_writing = check_course_no["writing score"]
male_all = []
female_all = []
group_A_all = []
group_B_all = []
group_C_all = []
group_D_all = []
group_E_all = []
master_degree_all = []
bachelor_degree_all = []
associate_degree_all = []
some_college_all = []
highschool_all = []
some_highschool_all = []
lunch_free_all = []
lunch_standard_all = []
course_yes_all = []
course_no_all = []

for i in male_math.index:
    temp = (male_math[i] + male_reading[i] + male_writing[i]) / 3
    male_all.append(temp)
    
for i in female_math.index:
    temp = (female_math[i] + female_reading[i] + female_writing[i]) / 3
    female_all.append(temp)

for i in group_A_math.index:
    temp = (group_A_math[i] + group_A_reading[i] + group_A_writing[i]) / 3
    group_A_all.append(temp)

for i in group_B_math.index:
    temp = (group_B_math[i] + group_B_reading[i] + group_B_writing[i]) / 3
    group_B_all.append(temp)

for i in group_C_math.index:
    temp = (group_C_math[i] + group_C_reading[i] + group_C_writing[i]) / 3
    group_C_all.append(temp)
    
for i in group_D_math.index:
    temp = (group_D_math[i] + group_D_reading[i] + group_D_writing[i]) / 3
    group_D_all.append(temp)

for i in group_E_math.index:
    temp = (group_E_math[i] + group_E_reading[i] + group_E_writing[i]) / 3
    group_E_all.append(temp)
    
for i in master_degree_math.index:
    temp = (master_degree_math[i] + master_degree_reading[i] + master_degree_writing[i]) / 3
    master_degree_all.append(temp)
    
for i in bachelor_degree_math.index:
    temp = (bachelor_degree_math[i] + bachelor_degree_reading[i] + bachelor_degree_writing[i]) / 3
    bachelor_degree_all.append(temp)
    
for i in associate_degree_math.index:
    temp = (associate_degree_math[i] + associate_degree_reading[i] + associate_degree_writing[i]) / 3
    associate_degree_all.append(temp)
    
for i in some_college_math.index:
    temp = (some_college_math[i] + some_college_reading[i] + some_college_writing[i]) / 3
    some_college_all.append(temp)
    
for i in highschool_math.index:
    temp = (highschool_math[i] + highschool_reading[i] + highschool_writing[i]) / 3
    highschool_all.append(temp)
    
for i in some_highschool_math.index:
    temp = (some_highschool_math[i] + some_highschool_reading[i] + some_highschool_writing[i]) / 3
    some_highschool_all.append(temp)
    
for i in lunch_free_math.index:
    temp = (lunch_free_math[i] + lunch_free_reading[i] + lunch_free_writing[i]) / 3
    lunch_free_all.append(temp)
    
for i in lunch_standard_writing.index:
    temp = (lunch_standard_math[i] + lunch_standard_reading[i] + lunch_standard_writing[i]) / 3
    lunch_standard_all.append(temp)

for i in course_yes_writing.index:
    temp = (course_yes_math[i] + course_yes_reading[i] + course_yes_writing[i]) / 3
    course_yes_all.append(temp)
    
for i in course_no_writing.index:
    temp = (course_no_math[i] + course_no_reading[i] + course_no_writing[i]) / 3
    course_no_all.append(temp)
fig1,ax1=plt.subplots(figsize=(10,5))
ax1.set_title('Correlation Matrix')
sns.heatmap(df.corr(),annot=True,linewidths=.5,fmt='.6f',ax=ax1)
fig2,ax2=plt.subplots(figsize=(10,5))
ax2.set_title('Gender')
labels = ('Male', 'Female')
sizes = [len(check_male), len(check_female)]
explode = (0, 0)
ax2.pie(sizes, explode=explode, labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140) 
ax2.axis('equal')
plt.show()
fig3, ax3 = plt.subplots(figsize = (10,5))
ax3.set_title('Student Performance from Gender')
ax3.boxplot([male_all, female_all])
ax3.set_xticklabels(['Male', 'Female'])
n_bins = 5
fig4, ax4 = plt.subplots(figsize = (10,5))
labels = ['Male', 'Female']
ax4.hist((male_all, female_all), n_bins, density=True, histtype='bar', label=labels)
ax4.legend(prop={'size': 15}, loc='upper left')
ax4.set_title('Student Performance from Gender')
fig5, ax5 = plt.subplots(figsize=(10,5))
ax5.set_title('Race/Ethnicity')
labels = ('Group A', 'Group B', 'Group C', 'Group D', 'Group E')
sizes = [len(check_group_A), len(check_group_B), len(check_group_C), len(check_group_D), len(check_group_E)]
explode = (0, 0, 0.1, 0, 0)
ax5.pie(sizes, explode=explode, labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140) 
ax5.axis('equal')
plt.show()
fig6, ax6 = plt.subplots(figsize = (10,5))
ax6.set_title('Student Performance from Race/Ethnicity')
ax6.boxplot([group_A_all, group_B_all, group_C_all, group_D_all, group_E_all])
ax6.set_xticklabels(['Grpou A', 'Group B', 'Grpou C', 'Grpou D', 'Grpou E'])
n_bins = 5
fig7, ax7 = plt.subplots(figsize = (10,5))
labels = ['Grpou A', 'Group B', 'Grpou C', 'Grpou D', 'Grpou E']
ax7.hist((group_A_all, group_B_all, group_C_all, group_D_all, group_E_all), n_bins, density=True, histtype='bar', label=labels)
ax7.legend(prop={'size': 15}, loc='upper left')
ax7.set_title('Student Performance from Race/Ethnicity')
fig8, ax8 = plt.subplots(figsize=(10,5))
ax8.set_title('Parental Level of Education')
labels = ('Master', 'Bachelor', 'Associate', 'Some College', 'High school', 'Some High school')
sizes = [len(check_master_degree), len(check_bachelor_degree), len(check_associate_degree), len(check_some_college), len(check_highschool), len(check_some_highschool)]
explode = (0, 0, 0, 0.1, 0, 0)
ax8.pie(sizes, explode=explode, labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140) 
ax8.axis('equal')
plt.show()
fig9, ax9 = plt.subplots(figsize = (10,5))
ax9.set_title('Student Performance from Parental Level of Education')
ax9.boxplot([master_degree_all, bachelor_degree_all, associate_degree_all, some_college_all, highschool_all, some_highschool_all])
ax9.set_xticklabels(['Master', 'Bachelor', 'Associate', 'Some College', 'High school', 'Some High school'])
n_bins = 5
fig10, ax10 = plt.subplots(figsize = (10,5))
labels = ['Master', 'Bachelor', 'Associate', 'Some College', 'High school', 'Some High school']
ax10.hist((master_degree_all, bachelor_degree_all, associate_degree_all, some_college_all, highschool_all, some_highschool_all), n_bins, density=True, histtype='bar', label=labels)
ax10.legend(prop={'size': 15}, loc='upper left')
ax10.set_title('Student Performance from Parental Level of Education')
fig11, ax11 = plt.subplots(figsize=(10,5))
ax11.set_title('Lunch')
labels = ('Free Lunch', 'Standard Lunch')
sizes = [len(check_lunch_free), len(check_lunch_standard)]
explode = (0, 0)
ax11.pie(sizes, explode=explode, labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140) 
ax11.axis('equal')
plt.show()
fig12, ax12 = plt.subplots(figsize = (10,5))
ax12.set_title('Student Performance form Lunch')
ax12.boxplot([lunch_free_all, lunch_standard_all])
ax12.set_xticklabels(['Free Lunch', 'Standard Lunch'])
n_bins = 5
fig13, ax13 = plt.subplots(figsize = (10,5))
labels = ['Free Lunch', 'Standard Lunch']
ax13.hist((lunch_free_all, lunch_standard_all), n_bins, density=True, histtype='bar', label=labels)
ax13.legend(prop={'size': 15}, loc='upper left')
ax13.set_title('Student Performance form Lunch')
fig14, ax14 = plt.subplots(figsize=(10,5))
ax14.set_title('Preparation Course')
labels = ('Course Attended', 'Course Absented')
sizes = [len(check_course_yes), len(check_course_no)]
explode = (0, 0)
ax14.pie(sizes, explode=explode, labels=labels, 
        autopct='%1.1f%%', shadow=True, startangle=140) 
ax14.axis('equal')
plt.show()
fig15, ax15 = plt.subplots(figsize = (10,5))
ax15.set_title('Student Performance form Preparation Course')
ax15.boxplot([course_yes_all, course_no_all])
ax15.set_xticklabels(['Course Attended', 'Course Absented'])
n_bins = 5
fig16, ax16 = plt.subplots(figsize = (10,5))
labels = ['Course Attended', 'Course Absented']
ax16.hist((course_yes_all, course_no_all), n_bins, density=True, histtype='bar', label=labels)
ax16.legend(prop={'size': 15}, loc='upper left')
ax16.set_title('Student Performance form Preparation Course')
Two_sample_t_test_gender = stats.ttest_ind(male_all, female_all)
Anova_group = stats.f_oneway(group_A_all, group_B_all, group_C_all, group_D_all, group_E_all)
Anova_education = stats.f_oneway(master_degree_all, bachelor_degree_all, associate_degree_all, some_college_all, highschool_all, some_highschool_all)
Two_sample_t_test_lunch = stats.ttest_ind(lunch_free_all, lunch_standard_all)
Two_sample_t_test_course = stats.ttest_ind(course_yes_all, course_no_all)
dictionary_pvalue = {'p-value':[Two_sample_t_test_gender[1], Anova_group[1], Anova_education[1], Two_sample_t_test_lunch[1], Two_sample_t_test_course[1]]}
dataframe_pvalue = pd.DataFrame(dictionary_pvalue, index=['Two sample t-test by gender', 'Anova by group', 'Anova by parental education', 'Two sample t-test by lunch', 'Two sample t-test by course'])
dataframe_pvalue