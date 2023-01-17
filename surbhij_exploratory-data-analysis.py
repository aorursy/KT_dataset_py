#Importing the data
import pandas as pd
school=pd.read_csv('../input/2016 School Explorer.csv')
school=pd.DataFrame(school)
#Importing libraries for Exploratory Data Analysis
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#To check for duplicates
ids = school['School Name']
school[ids.isin(ids[ids.duplicated()])]
#Thus we can see that even though there are two school names that are dulicated but the records are not and therefore
#we conclude that there are no duplicate records
#Estimating community school wise average income 

#Removing dollar sign and comma from the school income estimate
school['School Income Estimate']=school['School Income Estimate'].astype(str)
school['School Income Estimate New']=school['School Income Estimate'].str.split("$").str[1]
school['School Income Estimate New']=school['School Income Estimate New'].str.split(" ").str[0]
school['School Income Estimate New']=school['School Income Estimate New'].replace({',':""},regex=True)
school['School Income Estimate New']=school['School Income Estimate New'].astype(float)

#Calculating average school income estimate for community school and Non-community school.
df1=school.groupby(by='Community School?')['School Income Estimate New'].mean()
df1=pd.DataFrame(df1)
print(df1)
df1.plot.bar()

#Insight:-
#Barplot shows that income estimate for community schools is very less than Non-community schools.
#Community school wise Economic Need Index
df2=school.groupby(by='Community School?')['Economic Need Index'].mean()
df2=pd.DataFrame(df2)
print(df2)
df2.plot.bar()

#Insight:-
#Barplot shows that Economic Need Index for community schools is more than Non-community schools.
#City wise school income estimate
school.groupby(by='City')['School Income Estimate New'].mean()
#Interpretating how Economic need index affects performance of students
#Replacing missing values with mean value
school['Economic Need Index']=school['Economic Need Index'].fillna(school['Economic Need Index'].mean())
#Distribution of Economic Need Index
sns.boxplot(x="Economic Need Index",data=school)

#Insight
# Economic Need Index is negatively skewed .i.e Economic Need Index values are high in schools
#Scatter plot to determine the effect of Economic Need Index on students performance for ELA subject for different Grades
sns.lmplot(x="Economic Need Index", y="Grade 3 ELA 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 4 ELA 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 5 ELA 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 6 ELA 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 7 ELA 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 8 ELA 4s - All Students", data=school)


#Insights:-
#Plots show the negative relationship between Economic Need Index and perfomance rating for ELA subject i.e. more the
#Economic need index less the performance
#Hence Higher economic index group should be more focused upon.
#Plotting scatter plot to determine the effect of Economic Need Index on students performance for Math subject
sns.lmplot(x="Economic Need Index", y="Grade 3 Math 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 4 Math 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 5 Math 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 6 Math 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 7 Math 4s - All Students", data=school)
sns.lmplot(x="Economic Need Index", y="Grade 8 Math 4s - All Students", data=school)

#Insights:-
#Plots show the negative relationship between Economic Need Index and perfomance rating for Math subject i.e. more the
#Economic need index less the performance
#Hence Higher economic index group should be more focused upon.

#Scatter plot showing the relationship between Average Math proficiency and Economic Need Index
sns.lmplot(x="Economic Need Index", y="Average Math Proficiency",data=school)

#Insights:-
#Plots show the negative relationship between Economic Need Index and average perfomance rating for Math subject i.e. more the
#Economic need index less the performance
#Hence Higher economic index group should be more focused upon.
#Scatter plot showing the relationship between Average ELA proficiency and Economic Need Index
sns.lmplot(x="Economic Need Index", y="Average ELA Proficiency",data=school)

#Insights:-
#Plots show the negative relationship between Economic Need Index and average perfomance rating for ELA subject i.e. more the
#Economic need index less the performance
#Hence Higher economic index group should be more focused upon.
#Scatter plot showing the relationship between Average Math proficiency and School Income Estiamte
sns.lmplot(x="School Income Estimate New", y="Average Math Proficiency",data=school)

#Insight:-
#Plot shows the positive relationship between school income estimate and average performance rating in Math subject
#i.e. Schools having higher income likely to have better performance in Math subject
#Scatter plot showing the relationship between Average ELA proficiency and School Income Estiamte
sns.lmplot(x="School Income Estimate New", y="Average ELA Proficiency",data=school)

#Insight:-
#Plot shows the positive relationship between school income estimate and average performance rating in ELA subject
#i.e. Schools having higher income likely to have better performance in ELA subject
#Determining the distribution of blacks and white in schools 
subset=school.loc[:,['School Name','Percent Black','Percent White']]
subset=pd.melt(subset,id_vars=['School Name'])
subset.iloc[:,2]=subset.iloc[:,2].replace({'%':""},regex=True)
subset.iloc[:,2]=subset.iloc[:,2].astype(float)


subset=subset.pivot_table(values='value',index='variable',aggfunc='mean')
print(subset)
subset.plot.bar()

#Insight:-
#Plot shows that there is a significant difference between average percentage of blacks and whites.
#i.e. on an average 32% of students are black in schools while only 13% are whites in schools
#Determining the distribution of Asian,Hispanic,blacks and white in schools 
subset=school.loc[:,['School Name','Percent Black','Percent White','Percent Asian','Percent Hispanic']]
subset=pd.melt(subset,id_vars=['School Name'])
subset.iloc[:,2]=subset.iloc[:,2].replace({'%':""},regex=True)
subset.iloc[:,2]=subset.iloc[:,2].astype(float)


subset=subset.pivot_table(values='value',index='variable',aggfunc='mean')
print(subset)
subset.plot.bar()

#Insight
#Plot shows the overall distribution of students from different groups in schools and students from hispanic group are highest
#while students from black group is second highest
# A plot that shows the distribution of 3 Math Grade Students who have scored 4. 
print(school['Grade 3 Math 4s - American Indian or Alaska Native'].sum())
print(school['Grade 3 Math 4s - Black or African American'].sum())
print(school['Grade 3 Math 4s - Hispanic or Latino'].sum())
print(school['Grade 3 Math 4s - Asian or Pacific Islander'].sum())
print(school['Grade 3 Math 4s - White'].sum())
print(school['Grade 3 Math 4s - Multiracial'].sum())
print(school['Grade 3 Math 4s - Limited English Proficient'].sum())
print(school['Grade 3 Math 4s - Economically Disadvantaged'].sum())

x={'Name':['Grade 3 Math 4s - American Indian or Alaska Native','Grade 3 Math 4s - Black or African American',
           'Grade 3 Math 4s - Hispanic or Latino','Grade 3 Math 4s - Asian or Pacific Islander',
           'Grade 3 Math 4s - White', 'Grade 3 Math 4s - Multiracial','Grade 3 Math 4s - Limited English Proficient',
       'Grade 3 Math 4s - Economically Disadvantaged'], 'Count':[35, 2869, 3581, 5121, 3829, 84, 675, 8494]}

df2=pd.DataFrame.from_dict(x)
print(df2)
plt.xticks(rotation=90)
sns.barplot(x='Name', y='Count', data=df2)
#Insight:-
#The Asians/Pacific Islanders have scored the best than other races for Math
#Hispancs are 2nd best for Math
# A plot that shows the distribution of 3 Grade Students who have scored 4. 
print(school['Grade 3 ELA 4s - American Indian or Alaska Native'].sum())
print(school['Grade 3 ELA 4s - Black or African American'].sum())
print(school['Grade 3 ELA 4s - Hispanic or Latino'].sum())
print(school['Grade 3 ELA 4s - Asian or Pacific Islander'].sum())
print(school['Grade 3 ELA 4s - White'].sum())
print(school['Grade 3 ELA 4s - Multiracial'].sum())
print(school['Grade 3 ELA 4s - Limited English Proficient'].sum())
print(school['Grade 3 ELA 4s - Economically Disadvantaged'].sum())

x={'Name':['Grade 3 ELA 4s - American Indian or Alaska Native','Grade 3 ELA 4s - Black or African American',
           'Grade 3 ELA 4s - Hispanic or Latino','Grade 3 ELA 4s - Asian or Pacific Islander',
           'Grade 3 ELA 4s - White', 'Grade 3 ELA 4s - Multiracial','Grade 3 ELA 4s - Limited English Proficient',
       'Grade 3 ELA 4s - Economically Disadvantaged'], 'Count':[12, 908, 1054, 1714, 1743, 50, 31, 2522]}

df2=pd.DataFrame.from_dict(x)
print(df2)
plt.xticks(rotation=90)
sns.barplot(x='Name', y='Count', data=df2)
#Insight:-
#The Asians and Whites have scored better than other races for ELA
# A plot that shows the distribution of 8 Grade Students who have scored 4. 
print(school['Grade 8 ELA 4s - American Indian or Alaska Native'].sum())
print(school['Grade 8 ELA 4s - Black or African American'].sum())
print(school['Grade 8 ELA 4s - Hispanic or Latino'].sum())
print(school['Grade 8 ELA 4s - Asian or Pacific Islander'].sum())
print(school['Grade 8 ELA 4s - White'].sum())
print(school['Grade 8 ELA 4s - Multiracial'].sum())
print(school['Grade 8 ELA 4s - Limited English Proficient'].sum())
print(school['Grade 8 ELA 4s - Economically Disadvantaged'].sum())

x={'Name':['Grade 8 ELA 4s - American Indian or Alaska Native','Grade 8 ELA 4s - Black or African American',
           'Grade 8 ELA 4s - Hispanic or Latino','Grade 8 ELA 4s - Asian or Pacific Islander',
          'Grade 8 ELA 4s - White', 'Grade 8 ELA 4s - Multiracial','Grade 8 ELA 4s - Limited English Proficient',
       'Grade 8 ELA 4s - Economically Disadvantaged'], 'Count':[17, 1166, 1934, 2869, 2457, 20, 2, 4979]}

df2=pd.DataFrame.from_dict(x)
print(df2)
plt.xticks(rotation=90)
sns.barplot(x='Name', y='Count', data=df2)
#Insight:-
#The Asians and Whites in Grade 8 have scored better than other races in ELA
# A plot that shows the distribution of 8 Math Grade Students who have scored 4. 
print(school['Grade 8 Math 4s - American Indian or Alaska Native'].sum())
print(school['Grade 8 Math 4s - Black or African American'].sum())
print(school['Grade 8 Math 4s - Hispanic or Latino'].sum())
print(school['Grade 8 Math 4s - Asian or Pacific Islander'].sum())
print(school['Grade 8 Math 4s - White'].sum())
print(school['Grade 8 Math 4s - Multiracial'].sum())
print(school['Grade 8 Math 4s - Limited English Proficient'].sum())
print(school['Grade 8 Math 4s - Economically Disadvantaged'].sum())

x={'Name':['Grade 3 Math 4s - American Indian or Alaska Native','Grade 3 Math 4s - Black or African American',
           'Grade 3 Math 4s - Hispanic or Latino','Grade 3 Math 4s - Asian or Pacific Islander',
           'Grade 3 Math 4s - White', 'Grade 3 Math 4s - Multiracial','Grade 3 Math 4s - Limited English Proficient',
      'Grade 3 Math 4s - Economically Disadvantaged'], 'Count':[4, 776, 1205, 2524, 1235, 3, 203, 3806]}

df2=pd.DataFrame.from_dict(x)
print(df2)
plt.xticks(rotation=90)
sns.barplot(x='Name', y='Count', data=df2)
#Insight:-
#The Asians/Pacific Islanders have scored the best than other races for Math
#Hispancs are 2nd best for Math
#Plot showing relationship between percentage of blacks in schools and average ELA proficiency
school['Percent Black']=school['Percent Black'].replace({'%':''},regex=True)
school['Percent Black']=school['Percent Black'].astype(float)
sns.jointplot('Percent Black','Average ELA Proficiency',data=school)

#Insight
#Plot shows that there is negative relationship between Average ELA proficiency and Percentage of black students
#i.e. schools having higher percentage of black students(80 to 100) have Average ELA proficiency below 3.
#Plot showing relationship between percentage of blacks in schools and average ELA proficiency
sns.jointplot('Percent Black', 'Average Math Proficiency', data=school)
#Insight
#Plot shows that there is negative relationship between Average Math proficiency and Percentage of black students
#i.e. schools having higher percentage of black students(80 to 100) have Average Math proficiency below 3.5.
#Plot showing relationship between percentage of hispanic in schools and average ELA proficiency
school['Percent Hispanic']=school['Percent Hispanic'].replace({'%':''},regex=True)
school['Percent Hispanic']=school['Percent Hispanic'].astype(float)
sns.jointplot('Percent Hispanic', 'Average ELA Proficiency', data=school)
#Insight
#Plot shows that there is negative relationship between Average ELA proficiency and Percentage of hispanic students
#i.e. schools having higher percentage of hispanic students(80 to 100) have Average ELA proficiency below 2.5.
#Plot showing relationship between percentage of hispanic in schools and average Math proficiency
sns.jointplot('Percent Hispanic', 'Average Math Proficiency', data=school)
#Insight
#Plot shows that there is negative relationship between Average Math proficiency and Percentage of hispanic students
#i.e. schools having higher percentage of hispanic students(80 to 100) have Average Math proficiency below 3.
#Plot showing relationship between percentage of Asian in schools and average ELA proficiency
school['Percent Asian']=school['Percent Asian'].replace({'%':''},regex=True)
school['Percent Asian']=school['Percent Asian'].astype(float)
sns.jointplot('Percent Asian', 'Average ELA Proficiency', data=school)

#Insight
#Plot shows that there is a positive relationship between Average ELA proficiency and Percentage of Asian students

#Plot showing relationship between percentage of Asian in schools and average Math proficiency
sns.jointplot('Percent Asian', 'Average Math Proficiency', data=school)

#Plot shows that there is a positive relationship between Average Math proficiency and Percentage of Asian students

#Scatter Plot showing the relationship between Average ELA Proficiency and students Attendance Rate

#Removing percentage symbol from student attendance rate
school['Student Attendance Rate']=school['Student Attendance Rate'].replace({'%':""},regex=True)
school['Student Attendance Rate']=school['Student Attendance Rate'].astype(float)
school['Student Attendance Rate']
school['Average ELA Proficiency'].astype(float)
sns.lmplot(x="Student Attendance Rate", y="Average ELA Proficiency", data=school)

#Insight:-
#Plot shows that while attendance rate for students is very high i.e. between 80 to 100 but performance rate in ELA
#is very less .i.e all the students have scored less than 4(not even passing marks)
#It shows that there is discrepancy in students attendance rate or teaching method is not effective.
#Scatter Plot showing the relationship between Average Math Proficiency and students Attendance Rate
sns.lmplot(x="Student Attendance Rate", y="Average Math Proficiency", data=school)


#Insight:-
#Plot shows that while attendance rate for students is very high i.e. between 80 to 100 but performance rate in Math
#is very less .i.e all the students have scored less than 4(not even passing marks)
#It shows that there is discrepancy in students attendance rate or teaching method is not effective.
#Scatter Plot showing the relationship between Average ELA Proficiency and Rigorous Instruction %

#Removing percentage symbol from rigorous instruction
school['Rigorous Instruction %']=school['Rigorous Instruction %'].replace({'%':""},regex=True)
school['Rigorous Instruction %']=school['Rigorous Instruction %'].astype(float)
sns.lmplot(x="Rigorous Instruction %", y="Average ELA Proficiency", data=school)

#Insight:-
#Plot shows that while rigorous instructions rate for students is very high i.e. between 60 to 100 but performance rate in ELA
#is very less .i.e all the students have scored less than 4(not even passing marks)
#There is discrepancy in the way rigorous instructions rate is measured.It could be that schools say that they give
#rigorous instructions however in comparison to the elite schools it is not rigorous enough.
#Scatter Plot showing the relationship between Average Math Proficiency and Rigorous Instruction %
sns.lmplot(x="Rigorous Instruction %", y="Average Math Proficiency", data=school)

#Insight:-
#Plot shows that while rigorous instructions rate for students is very high i.e. between 60 to 100 but performance rate in Maths
#is very less .i.e all the students have scored less than 4(not even passing marks)
#There is discrepancy in the way rigorous instructions rate is measured.It could be that schools say that they give
#rigorous instructions however in comparison to the elite schools it is not rigorous enough.
#Scatter Plot showing the relationship between Average Math Proficiency and Collaborative Teachers %
school['Collaborative Teachers %']=school['Collaborative Teachers %'].replace({'%':""},regex=True)
school['Collaborative Teachers %']=school['Collaborative Teachers %'].astype(float)
sns.jointplot(x='Collaborative Teachers %', y='Average Math Proficiency', data=school)

#Insight:-
#Plot shows that while Collaborative Teachers rate for students is very high i.e. between 60 to 100 but performance rate in Maths
#is very less .i.e all the students have scored less than 4(not even passing marks)
#i.e. teaching methods are not effective.
#Scatter Plot showing the relationship between Average ELA Proficiency and Collaborative Teachers %
sns.jointplot(x='Collaborative Teachers %', y='Average ELA Proficiency', data=school)

#Insight:-
#Plot shows that while Collaborative Teachers rate for students is very high i.e. between 60 to 100 but performance rate in ELA
#is very less .i.e all the students have scored less than 4(not even passing marks)
#i.e. teaching methods are not effective.
#Scatter Plot showing the relationship between Average ELA Proficiency and Supportive Environment %
school['Supportive Environment %']=school['Supportive Environment %'].replace({'%':""},regex=True)
school['Supportive Environment %']=school['Supportive Environment %'].astype(float)
sns.jointplot(x='Supportive Environment %', y='Average ELA Proficiency', data=school)

#Insight:-
#Plot shows that while Collaborative Teachers rate for students is very high i.e. between 80 to 100 but performance rate in ELA
#is very less .i.e all the students have scored less than 4(not even passing marks)
#There is discrepancy in the way supportive environment rate is measured.It could be that schools say that they have provided
#supportive environment however in comparison to the elite schools it is not supportive enough.
#Scatter Plot showing the relationship between Average Math Proficiency and Supportive Environment %

sns.jointplot(x='Supportive Environment %', y='Average Math Proficiency', data=school)

#Insight:-
#Plot shows that while Collaborative Teachers rate for students is very high i.e. between 70 to 100 but performance rate in Maths
#is very less .i.e all the students have scored less than 4(not even passing marks)
#There is discrepancy in the way supportive environment rate is measured.It could be that schools say that they have provided
#supportive environment however in comparison to the elite schools it is not supportive enough.
#Scatter Plot showing the relationship between Average ELA Proficiency and Effective School Leadership %
school['Effective School Leadership %']=school['Effective School Leadership %'].replace({'%':""},regex=True)
school['Effective School Leadership %']=school['Effective School Leadership %'].astype(float)
sns.jointplot(x='Effective School Leadership %', y='Average ELA Proficiency', data=school)

#Insight:-
#Plot shows that while Effective School Leadership rate for students is very high i.e. between 60 to 100 but performance rate in ELA
#is very less .i.e all the students have scored less than 4(not even passing marks)
#Schools are not able to bulid up effective leadership skills among students

#Scatter Plot showing the relationship between Average Math Proficiency and Effective School Leadership %
sns.jointplot(x='Effective School Leadership %', y='Average Math Proficiency', data=school)

#Insight:-
#Plot shows that while Effective School Leadership rate for students is very high i.e. between 60 to 100 but performance rate in Maths
#is very less .i.e all the students have scored less than 4(not even passing marks)
#Schools are not able to bulid up effective leadership skills among students
#Plot showing grade-wise percentage of number of students passed the ELA test
school['percent_grade3_ELA']=(school['Grade 3 ELA 4s - All Students']/school['Grade 3 ELA - All Students Tested'])*100
school['percent_grade4_ELA']=(school['Grade 4 ELA 4s - All Students']/school['Grade 4 ELA - All Students Tested'])*100
school['percent_grade5_ELA']=(school['Grade 5 ELA 4s - All Students']/school['Grade 5 ELA - All Students Tested'])*100
school['percent_grade6_ELA']=(school['Grade 6 ELA 4s - All Students']/school['Grade 6 ELA - All Students Tested'])*100
school['percent_grade7_ELA']=(school['Grade 7 ELA 4s - All Students']/school['Grade 7 ELA - All Students Tested'])*100
school['percent_grade8_ELA']=(school['Grade 8 ELA 4s - All Students']/school['Grade 8 ELA - All Students Tested'])*100
subset=school.loc[:,['School Name','percent_grade3_ELA','percent_grade4_ELA','percent_grade5_ELA','percent_grade6_ELA','percent_grade7_ELA','percent_grade8_ELA']]
subset=pd.melt(subset,id_vars=['School Name'])
subset=subset.pivot_table(values='value',index='variable',aggfunc='mean')
subset.plot.bar()

#Insight
#Plot shows that on average maximum percentage of students passed in Grade 4 in ELA subject
#Top five schools which have highest pass percentage in ELA subject starting from Grade 8 to 3
school.sort_values(['percent_grade8_ELA','percent_grade7_ELA','percent_grade6_ELA',
                'percent_grade5_ELA','percent_grade4_ELA','percent_grade3_ELA'],ascending=False)[['School Name','Community School?',
                                                                                            'percent_grade8_ELA','percent_grade7_ELA','percent_grade6_ELA',
                                                                                            'percent_grade5_ELA','percent_grade4_ELA','percent_grade3_ELA']].head()

#Plot showing grade-wise percentage of number of students passed the Math test
school['percent_grade4_Math']=(school['Grade 4 Math 4s - All Students']/school['Grade 4 Math - All Students Tested'])*100
school['percent_grade5_Math']=(school['Grade 5 Math 4s - All Students']/school['Grade 5 Math - All Students Tested'])*100
school['percent_grade6_Math']=(school['Grade 6 Math 4s - All Students']/school['Grade 6 Math - All Students Tested'])*100
school['percent_grade7_Math']=(school['Grade 7 Math 4s - All Students']/school['Grade 7 Math - All Students Tested'])*100
school['percent_grade8_Math']=(school['Grade 8 Math 4s - All Students']/school['Grade 8 Math - All Students Tested'])*100
subset=school.loc[:,['School Name','percent_grade4_Math','percent_grade5_Math','percent_grade6_Math','percent_grade7_Math','percent_grade8_Math']]
subset=pd.melt(subset,id_vars=['School Name'])
subset=subset.pivot_table(values='value',index='variable',aggfunc='mean')
subset.plot.bar()

#Insight
#Plot shows that on average maximum percentage of students passed in Grade 4 and Grade 8 have lowest pass paercentage in Math subject.
#Top five schools which have highest pass percentage in Math subject starting from Grade 8 to 4
school.sort_values(['percent_grade8_Math','percent_grade7_Math','percent_grade6_Math',
                'percent_grade5_Math','percent_grade4_Math'],ascending=False)[['School Name','Community School?',
                                                                                            'percent_grade8_Math','percent_grade7_Math','percent_grade6_Math','percent_grade4_Math']].head()
#Insight
#THE CHRISTA MCAULIFFE SCHOOL\I.S. 187  is the school which have good pass percentage in both subjects.
#Top five schools which have highest performance rate in Math and ELA
df6=school.loc[:,['School Name', 'Average ELA Proficiency', 'Average Math Proficiency']]
df6.sort_values(['Average Math Proficiency','Average ELA Proficiency'],ascending=False).head()

#THE CHRISTA MCAULIFFE SCHOOL\I.S. 187  also belongs top 5 schools having highest performance rate in both subjects.
#THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 is overall performing well 
#Distribution of grades in different schools
df4=school.loc[:,['School Name', 'Grade Low', 'Grade High']]
print(df4.groupby(['Grade Low', 'Grade High']).count())
print(df4.pivot_table(index='Grade Low', columns='Grade High', values='School Name', aggfunc='count'))
df5=df4.groupby(['Grade Low', 'Grade High']).count()
df5=df5.sort_values('School Name')
df5['Percent of Schools']=(df5['School Name']/(df5['School Name'].sum()))*100
print(df5)
#FEW STATISTICAL INFERENCES ABOUT DIFFERENT VARIABLES
from scipy.stats import chi2_contingency
#Testing the association between Collaborative Teachers Rating and community School
tab1=pd.crosstab(index = school["Collaborative Teachers Rating"],  columns=school["Community School?"], colnames = ['']) 
print(tab1)
chi,p,dof,ex=chi2_contingency(tab1)
print(p)

#INSIGHT
#Since p value is less than 0.05 hence we conclude that there is a significant relation between Collaborative Teachers Rating and community School
#Non-Community schools likely to have better Collaborative Teachers Rating.
#Testing the association between Student Achievement Rating and community School
tab2=pd.crosstab(index = school["Student Achievement Rating"],  columns=school["Community School?"], colnames = ['']) 
print(tab2)
chi,p,dof,ex=chi2_contingency(tab2)
print(p)

#INSIGHT
#Since p value is less than 0.05 hence we conclude that there is a significant relation between Student Achievement Rating and community School
#Non-Community schools likely to have better students achievement Rating.
#Testing the association between Student Achievement Rating and Rigorous Instruction Rating
tab2=pd.crosstab(index = school["Student Achievement Rating"],  columns=school["Rigorous Instruction Rating"], colnames = ['']) 
print(tab2)
chi,p,dof,ex=chi2_contingency(tab2)
print(p)

#INSIGHT
#Since p value is less than 0.05 hence we conclude that there is a significant relation between Student Achievement Rating and Rigorous Instruction Rating