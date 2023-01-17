#Descriptive Analysis of Placement Database of a Business Management University,Bangalore
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
%matplotlib inline

#Reading data frame
df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.describe()
#row = 215
#columns = 15
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis') 
#salary column has various nan values.
#nan values denote the student who are not placed.
#To remove nan values
df1 = df.dropna()
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis') 
#Maximum salary of placed student
sns.distplot(df1['salary'])
#Maximum salary amounts to about 940000
#Minimum salary of students had been placed amounts to 200000
#Count of placed and not plaved students
sns.countplot(x='status',data=df)
#Placed : Around 148 out of 215 (around 69% placed)
#Not placed :67 out of 215
#Count of Placed(Based on Gender)
sns.countplot(x='gender',data=df1)
#Male : 100 out of 148 (67% are placed)
#Female : 48 out of 148
#Salary 
#Men is placed with salary ranging from 200000 to 900000(above)max
#Women is placed with salary rangin from 200000 to 60000(above)max
#average salary for women
df1[df1['gender']=='F']['salary'].mean()
#average salary for men
df1[df1['gender']=='M']['salary'].mean()
#average salary for men is seen to be larger than women(in this dataset)
salary_low_female = df1[df1['gender']=='F']['salary'] < 267291.6666666667
#There are 32 women who are placed less than average salaray for given data(out of 48 women)
#66% of the women were placed above average salary 
salary_low_male = df1[df1['gender']=='M']['salary'] < 298910.0
#There are 62 men who are placed less than average salaray for given data(out of 48 women)
#58% of the men were placed above the average salary 
#Standard Deviation(Salary calculation)
np.std(df1['salary']) 
#Salary columm has very large variance from mean value, that is a lot of students were placed below avaerage salary and only few where near to the mean.
#Academic Details
#10th and 12th marks
#Average of 10th and 12th marks are computed
df['sum_10th_12th_marks'] = df['ssc_p'] + df['hsc_p']
df['total_marks'] = (df['sum_10th_12th_marks']/200)*100
value = df[df['status']=='Placed']['total_marks']
sns.barplot(x='status',y='total_marks',data=df)
#This computation has minimum percentage of 55
#This computation has maximum percentage of 87.5.
#Range:Students were selected for placements was from 54 to 87.5(overall percentage of 10th and 12th).
#Either minimum of 60 percentage in 10th or in 12th standard ,a student should have secured to be in list of placements.
#Education background played very important role in shortlisting students for placements.
#To convert catagorical plots to numeric values(0 or 1)
lb_make = LabelEncoder()
df["status3"] = lb_make.fit_transform(df["status"])
#Board of study
#0 - denoted placed
#1 - denotes not placed
sns.barplot(x='ssc_b',y='status3',data=df)
df[df['status']=='Placed']['ssc_b'].value_counts()
#Students who were placed had studied under Central board as compared to other boards.
#It would have been better for students who had studied under Central Board
#Stream chosen by student in school
#1- denotes Placed
#0- denotes not Placed
sns.barplot(x='hsc_s',y='status3',data=df)
df[df['status']=='Placed']['hsc_s'].value_counts()
#Company has preferred students to have background in commerce as compared to arts and Science.
#Strip plot
#Benifits of having commerce background: Students are awarded with higher salary as compared to other streams ,there r less outliers
sns.stripplot(x="hsc_s", y="salary", data=df1)
#Student school acamedic details had been very important factor in filtering out students and also their salary.
#UG details
#1 denotes placed
#0 denoted not placed
sns.barplot(x='degree_t',y='status3',data=df)
df[df['status']=='Placed']['degree_t'].value_counts()
#Students who had completed their degree in Communication and management has been placed the highest.
sns.boxplot(x='degree_t',y='salary',hue='hsc_s',data=df1)
#A student having commerce background and degree in communication management has been awarded with good amount of salary(from 200000 to 940000)
#A student having science background and degreen in sci and tech has been awarded salary(from 200000 to 450000 max(approx)
#Having commerce Background has been essential for higher salary.
#Student having science background and degree in sciandtech is also placed with good salary(200000 to 650000 approx(max))
#degree comes into role in deciding level salary for a student(as important factor).
df1["status2"] = lb_make.fit_transform(df1["status"])
#Degree pass Percentage
#0 - denotes student who are placed
sns.lmplot(x='degree_p',y='salary',hue='status2',data=df1,palette='coolwarm',scatter_kws={'s':100})
#degree pass percentage is not that important to be considered as a factor, as a student even with less pass percentage is placed with good amount if salary
#As we can see from plot,student who secured around 70-72 percentage has been placed with good salary.
#It has a negative correlation coefficient.
sns.barplot(x='degree_t',y='etest_p',hue='hsc_s',data=df1)
sns.lmplot(x='etest_p',y='salary',hue='status2',data=df1,scatter_kws={'s':100})
#Student having educational background in commerce, degree in communication management,with score above 80 in employability test has been placed with very high salary.
df1[df1['hsc_s']=='Commerce'][df1['degree_t']=='Comm&Mgmt'][df1['etest_p']>75]['salary'].min()
df1[df1['hsc_s']=='Science'][df1['degree_t']=='Sci&Tech'][df1['etest_p']>75]['salary'].min()
#Maximum salary for a student having background in commerce and with a degree in commnication management is 940000(Student with commerce field is preferred).
#Maximum salary for a student having science background and witha degree in science and technology is 450000.
#Minimum salary for both fields:200000.
#But if student is from commerce field he/she is awarded with higher salary as compared to other streams.
#Salary also depends on etest_p factor 
#Student having commerce backgorund,degree in communication and management ,with good etest score is awarded with good salary.
#If etest is poor,salary is less (student is awarded accordingngly).
#etest_p is directly proportional to salary(depends on streams also).
#Work experience
sns.boxplot(x='workex',y='salary',data=df1)
#Work factor has added to increase in salary
#As oberseved from dataset a person having commerce background,degree in communication and management and with a good etest score was awarded only 400000 
#as compared to other person who had all these qualifications with a work expeirnce was awarded 2000 extra(420000).
#work experience boosts salary.
#salary for student having work experience ranges from 300000 to 940000.
sns.lmplot(x='mba_p',y='salary',hue='status2',data=df1,palette='coolwarm',scatter_kws={'s':100})
#mba pass percentage is also not considered as an important factor in deciding salary for student as a person with least mba percentage is placed with good salary.
#Converting catagorical variables to 0 and 1
#1 denotes student is placed
#0 denotes student not placed
lb_make = LabelEncoder()
df["status3"] = lb_make.fit_transform(df["status"])
sns.barplot(x='specialisation',y='status3',data=df)
sns.boxplot(x="specialisation", y="salary", hue="gender",data=df1, palette="coolwarm")
#Company has preferred higher degree in marketing and finance.
#They have also offered good amount of salary(preferrable male) for this degree as compared to other degree,as it can be easily observed from outliers
#Analysis of person with highest salary
df1[df1['salary'] == 940000.0]['workex']
#student had work experience
df1[df1['salary'] == 940000.0]['specialisation']
#Had specialisation in MKtFin
df1[df1['salary'] == 940000.0]['ssc_p']
#Scored only 60 percent score in 10th.
df1[df1['salary'] == 940000.0]['hsc_p']
#Scored 68 % in 12 th
#Average of 12 th and 10th was around 63 percent
df1[df1['salary'] == 940000.0]['ssc_b']
#Central
df1[df1['salary'] == 940000.0]['degree_t']
#degree in communication and management
df1[df1['salary'] == 940000.0]['etest_p']
#Score : 82.66 (good score)
df1[df1['salary'] == 940000.0]['degree_p']
#72 (better as compared to school)
df1[df1['salary'] == 940000.0]['hsc_s']
#Commerce
value = df1[df1['salary'] == 940000.0]['mba_p']
#Score was above 64.34
#So we can observe that student was placed with average of 60 percentage in school and only with average scores in degree and mba pass percentage.
#He/She had good set of skills which seemed appropriate for job.(awarded wth good amount of salary)
#Skill seemed very important to company (Commerce or Science)
#Conclusion
#School Academic details has been essential in shortlisting the students for placements.
      #Requirements:
         #He/she should have scored minimum of 60 percentage either in 10th or 12th so that overall percentage comes above 55 percent.
             #Students who were not placed,many of them did not satisfy this basic criteria.
         #Preferrable for students to have background in Commerce(Company offered good amount of salary).
         #Science stream wasS also preffered ,but salary awarded was less as compared to Commerce stream students.
#UG details:
     #Requirements:
        #He/she should have good etest_score(increase in salary accordingly),even if degree pass percentage seemed very low.
        #Prefferable to have degree in communication and management.
        #Degree percentage is not considered as important factor,as a student with less degree has also got placed with good amount of salary.
#Mba details:
     #Requirements:
       #Prefferable to have degree in marketing and finance.
       #Mba pass percentage is not considered as an important factor.
#Work experience is optional ,it again results in increase in salary.
#Skillset is very essential for student as compared to marks scored by a student.
#Skillset reflects salary and should be appropriate for job.
