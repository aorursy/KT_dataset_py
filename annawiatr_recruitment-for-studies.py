#uploading the libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
data = pd.read_csv('../input/placement-data-full-class/datasets_596958_1073629_Placement_Data_Full_Class.csv')
#checking the parameters
data.head()
data.shape
data.describe()
data.dtypes
data.isnull().sum()
plt.hist(data.status, color = 'pink')
#148 students got the place, 67 not.
#The number 67 correlates with the number of nulls in the salary table, so I conclude that there is no data of the earnings of those, who did not enter studies.
#I check this hypothesis.
data.groupby(['status'])['salary'].count()
#hypothesis confirmed, I complete the null in the column 'Not Placed' with 0
students = data.fillna(0)
students
# I check the string values
plt.hist(students['gender'], color = 'pink')
# There are 139 men and 76 women
plt.hist(students['ssc_b'], color = 'pink')
plt.hist(students['hsc_b'], color = 'pink')
plt.hist(students['hsc_s'], color = 'pink')
plt.hist(students['degree_t'], color = 'pink')
plt.hist(students['specialisation'], color = 'pink')
#changing 'status' column to 1 for 'Placed' and 0 for 'Not Placed'
mapper = {'Placed':1,'Not Placed':0}
students['status'] = students['status'].replace(mapper)
#deleting 'sl_no' column
del students['sl_no']
# As the 'salary' column has nothing to do with admission to university, rather with the status after graduation, for the time being, I am examining dependencies without this parameter
students_degree = students.loc[:,students.columns != 'salary']
# checking the relation between the variables concerning the exam results
sns.pairplot(students_degree, hue = 'status', palette = 'husl', corner = True, kind = 'reg')
plt.legend(['Placed', 'Not Placed'], loc='upper right')
sns.boxplot(data = data, x = 'degree_t', y = 'degree_p', hue = 'status', palette = 'husl')
pd.crosstab(data['status'], data['degree_t'])

# The Sci & Tech and Comm & Mgmt students had scored higher on the test. And also among students of these faculties
# Students of fields of study other than those mentioned above had significantly worse results in the test, and more than half of them did not receive a place at university.
sns.boxplot(data = data, x = 'ssc_b', y = 'ssc_p', hue = 'status', palette = 'husl')
pd.crosstab(data['status'], data['ssc_b'])

# I can't see any significant difference between Central and Others, so I'll treat them together.
# There is a significant discrepancy in SSC scores between the groups that got and did not enter college.
# It can therefore be concluded that a poor SSC result entails a problem in getting a place to study.
sns.boxplot(data = data, x = 'hsc_s', y = 'hsc_p', hue = 'status', palette = 'husl')
pd.crosstab(data['status'], data['hsc_s'])
# Commerce students achieved much higher results in the HSC test and as many as 79 of them got a place for studies. Among Sci & Tech students, by far the highest percentage was admitted to university.
# Students of other faculties had slightly worse results in the test, but in the case of Science students, as many as 2/3 obtained a place in studies
# The greatest discrepancy in the exam results is visible in the case of Art students. But here too, more than half (6 people) were admitted to university. However, the results of these six were significantly different from the results of the rest of the group.
sns.scatterplot(data.status, data.mba_p, color = 'pink')
# As can be seen from the diagram below, the MBA exam results show very little correlation with respect to getting a place to study.
# People who received such a place on the test achieved both very low results (slightly above 50%) and close to 80%.
# The same spread of grades can be seen among people who did not enter studies.
sns.scatterplot(data.status, data.etest_p, color = 'pink')
# The same correlation can be seen with the ETEST exam.
sns.heatmap(students_degree.corr(), linewidth=0.2, annot=True)
# I am checking the heatmap to see if the dependencies between the above variables are also confirmed here.
# Looking at the line "status" confirms the above observations: the SSC exam shows the greatest correlation between the exam and the student pass, followed by the HSC and the degree, while the MBA and ETEST scores do not seem to have any influence on the admission decision.

sns.countplot(data = data, x = 'specialisation', hue = 'status', palette = 'husl')
pd.crosstab(data['status'], data['specialisation'])
# In the case of specialization, MKT & Fin students show much higher efficiency - almost 3/4 of people received a place at university.
# In the case of the MKT & HR specialization, slightly more than half of the students received a place in studies.
sns.countplot(data = data, x = 'gender', hue = 'status', palette = 'husl')
pd.crosstab(data['status'], data['gender'])

# Looking at the gender criterion, we can see a slight difference between the percentages of men and women who entered college.
# In the case of women, it is nearly 65%, and in the case of men, it is just over 70%
sns.countplot(data = data, x = 'workex', hue = 'status', palette = 'husl')
pd.crosstab(data['status'], data['workex'])
# When it comes to previous work experience, we can observe a significant correlation. People who could boast some, were much more likely to get a place at university.
# This result was over 86%. Among people with no experience, this result was less than 60%
# Here I will look through financial factor, i.e. the level of salary achieved by ex students
sns.scatterplot(data.specialisation, data.salary, color = 'pink')
# As we can see, with the exception of a few outliers, graduates of both faculties can count on a similar salary.
# to explore the topic, I will divide the salary into groups and see what the situation is in each of them
plt.hist(data['salary'], color = 'pink', bins = 14)
#As we can observe, the vast majority of salaries fall in the area between $ 200,000 and $ 3,000,000 per year.
# I will narrow down the table to students who were granted a place at university, in the case of the rest we do not have data on remuneration.
placed = data[data['status'] == 'Placed']
len(placed)
# I will now check how salaries are distributed for each specialization. For this, I create two separate tablesstudents_hr = placed[placed['specialisation'] == 'Mkt&HR']
students_fin = placed[placed['specialisation'] == 'Mkt&Fin']
print(len(students_hr))
print(len(students_fin))
plt.hist(students_fin['salary'], bins = 12, alpha = 0.2, color = 'green')
plt.hist(students_hr['salary'], bins = 12, color = 'pink')
# In this perspective, we can see more clearly that students with the Mkt & Fin specialization have a chance of higher earnings.
# Nevertheless, the above observation remains valid - the vast majority of students from both groups are in the $ 200,000-325,000 range
# Mkt & Fin students are also likely to earn more than $ 450,000
two_three_HR = []
three_four_HR = []
four_five_HR = []
for i in students_hr.salary:
    if i < 300000:
        two_three_HR.append(i)
    elif i < 400000:
        three_four_HR.append(i)
    elif i < 500000:
        four_five_HR.append(i)
two_three_fin = []
three_four_fin = []
four_five_fin = []
over_half_milion = []
for i in students_fin.salary:
    if i < 300000:
        two_three_fin.append(i)
    elif i < 400000:
        three_four_fin.append(i)
    elif i < 500000:
        four_five_fin.append(i)
    elif i > 600000:
        over_half_milion.append(i)
print(str(round((len(two_three_fin))*100/(len(students_fin)), 2)) + "% students studying MKT & Fin and "+ str(round((len(two_three_HR))*100/(len(students_hr)), 2))+ "% of MKT & HR students earn between 200 and 300 thousand dollars a year.")
print(str(round((len(three_four_fin))*100/(len(students_fin)), 2)) + "% students studying MKT & Fin and "+ str(round((len(three_four_HR))*100/(len(students_hr)), 2))+ "% of MKT & HR students earn between 300 and 400 thousand dollars a year.")
print(str(round((len(four_five_fin))*100/(len(students_fin)), 2)) + "% students studying MKT & Fin and "+ str(round((len(four_five_HR))*100/(len(students_hr)), 2))+ "% of MKT & HR students earn more then 400 thousand dollars a year")
print("Only " + str(round((len(over_half_milion))*100/(len(students_fin)), 2)) +"%, " + str(len(over_half_milion)) + " people with MKT & Fin specialization earn over half a million dollars a year.")