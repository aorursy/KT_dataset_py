import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', None)  
df = pd.read_csv('../input/2016 School Explorer.csv', sep = ',')
SHSAT = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv', sep = ',')

#df.shape, SHSAT.shape
# SHSAT's target students are those from grade 8 and 9
# they had 85 schools registered for grade 8 and 55 schools registered for grade 9
SHSAT[['Grade level','School name']].groupby(['Grade level']).count().plot(kind = 'bar', color = 'green', alpha = 0.5)
def f(x):
    return float(x.strip('%'))/100
df['Percent White'] = df['Percent White'].astype(str).apply(f)
df['Percent of Students Chronically Absent']=df['Percent of Students Chronically Absent'].astype(str).apply(f)
df['Rigorous Instruction %'] = df['Rigorous Instruction %'].astype(str).apply(f)
df['Collaborative Teachers %'] = df['Collaborative Teachers %'].astype(str).apply(f)
df['Supportive Environment %'] = df['Supportive Environment %'].astype(str).apply(f)
df['Effective School Leadership %'] = df['Effective School Leadership %'].astype(str).apply(f)
df['Strong Family-Community Ties %'] = df['Strong Family-Community Ties %'].astype(str).apply(f)
df['Trust %'] = df['Trust %'].astype(str).apply(f)
df['Percent ELL'] = df['Percent ELL'].astype(str).apply(f)
df['Student Attendance Rate'] = df['Student Attendance Rate'].astype(str).apply(f)

df['School Income Estimate'] = df['School Income Estimate'].str.replace(',', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace('$', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace(' ', '')
df['School Income Estimate'] = df['School Income Estimate'].astype(float)
s = df.fillna(0)
# Choose schools that have grade 8 and 9
s8 = s['Grades'].str.contains('8','9')
s_89 = s[s8]
print('School count: ', s_89['School Name'].count())
# Choosing shools which less than 50% students are white will help identify schools with more minority students
# There are 558 schools with minorities over 50%

minority = s_89.where(s_89['Percent White']<0.5).dropna()
print('School count: ', minority['School Name'].count())
# community schools vs. non community schools among the 171 schools
print(minority[['Community School?', 'School Name']].groupby(['Community School?']).count())
minority[['School Name', 
          'Community School?']
        ].groupby(['Community School?']
                 ).count().plot(kind = 'bar', legend = False)
target = minority.sort_values(by='Grade 8 ELA - All Students Tested'
                            ).where(minority['Grade 8 ELA - All Students Tested'] > 0
                                   ).dropna()
target['School Name'].count()
rigorous_instruction = target[['School Name','Location Code', 'City','Community School?', 
       'Rigorous Instruction %','Rigorous Instruction Rating'
      ]].groupby(['Rigorous Instruction Rating']).get_group('Not Meeting Target')

collaborative_teachers = target[['School Name','Location Code', 
                                 'City','Community School?', 
                                 'Collaborative Teachers %',
       'Collaborative Teachers Rating'
      ]].groupby(['Collaborative Teachers Rating']).get_group('Not Meeting Target')

effective_school_leadership = target[['School Name','Location Code', 'City','Community School?', 
                                      'Effective School Leadership %',
       'Effective School Leadership Rating'
      ]].groupby(['Effective School Leadership Rating']).get_group('Not Meeting Target')

strong_family_community_ties = target[['School Name','Location Code', 
                                       'City','Community School?', 
                                       'Strong Family-Community Ties %',
       'Strong Family-Community Ties Rating'
      ]].groupby(['Strong Family-Community Ties Rating']).get_group('Not Meeting Target')

trust = target[['School Name','Community School?', 'Location Code', 'City',
                'Trust %',
       'Trust Rating'
      ]].groupby(['Trust Rating']).get_group('Not Meeting Target')
student_achievement = target[['School Name','Community School?', 'Location Code', 'City',
       'Student Achievement Rating'
      ]].groupby(['Student Achievement Rating']).get_group('Not Meeting Target')
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,20))
rigorous_instruction[['Community School?', 
       'Rigorous Instruction Rating'
      ]].groupby(['Community School?'
                 ]).count().plot(kind = 'bar',
                                 ax=axes[0,0], 
                                 legend = False);axes[0,0].set_title('Rigorous Instruction Rating');
collaborative_teachers[['Community School?', 
       'Collaborative Teachers Rating'
      ]].groupby(['Community School?'
                 ]).count().plot(kind = 'bar',
                                 ax=axes[0,1], 
                                 legend = False);axes[0,1].set_title('Collaborative Teachers Rating');
effective_school_leadership[['Community School?', 
       'Effective School Leadership Rating'
      ]].groupby(['Community School?'
                 ]).count().plot(kind = 'bar', 
                                 color = 'green',
                                 ax=axes[1,0], legend = False);axes[1,0].set_title('Effective School Leadership Rating');
strong_family_community_ties[['Community School?', 
       'Strong Family-Community Ties Rating'
      ]].groupby(['Community School?'
                 ]).count().plot(kind = 'bar',
                                 color = 'orange',
                                 ax=axes[1,1],legend = False);axes[1,1].set_title('Strong Family-Community Ties Rating');
trust[['Community School?', 
       'Trust Rating'
      ]].groupby(['Community School?']).count().plot(kind = 'bar', 
                                                     alpha = 0.5, 
                                                     ax=axes[2,0],
                                                     legend = False);axes[2,0].set_title('Trust Rating');
student_achievement[['Community School?', 
                     'Student Achievement Rating'
                    ]].groupby(['Community School?']).count().plot(kind = 'bar', 
                                                                   color = 'yellow',
                                                                   legend = False,
                                                                   ax=axes[2,1]
                                                                   );axes[2,1].set_title('Student Achievement Rating');
Collaborative = 'Average rating: ',collaborative_teachers['Collaborative Teachers %'].mean()
Effective = effective_school_leadership['Effective School Leadership %'].mean()
Strong_family = strong_family_community_ties['Strong Family-Community Ties %'].mean()
Trust  = trust['Trust %'].mean()
rigorous_instruction_0 = target[['School Name','Location Code', 'City','Community School?', 
       'Rigorous Instruction %','Rigorous Instruction Rating'
      ]].groupby(['Rigorous Instruction Rating']).get_group(0)

collaborative_teachers_0 = target[['School Name','Location Code', 
                                 'City','Community School?', 
                                 'Collaborative Teachers %',
       'Collaborative Teachers Rating'
      ]].groupby(['Collaborative Teachers Rating']).get_group(0)

suppotive_environment_0 = target[['School Name','Location Code', 
                                 'City','Community School?', 
                                 'Supportive Environment %',
       'Supportive Environment Rating'
      ]].groupby(['Supportive Environment Rating']).get_group(0)

effective_school_leadership_0 = target[['School Name','Location Code', 'City','Community School?', 
                                      'Effective School Leadership %',
       'Effective School Leadership Rating'
      ]].groupby(['Effective School Leadership Rating']).get_group(0)

strong_family_community_ties_0 = target[['School Name','Location Code', 
                                       'City','Community School?', 
                                       'Strong Family-Community Ties %',
       'Strong Family-Community Ties Rating'
      ]].groupby(['Strong Family-Community Ties Rating']).get_group(0)

trust_0 = target[['School Name','Community School?', 'Location Code', 'City',
                'Trust %',
       'Trust Rating'
      ]].groupby(['Trust Rating']).get_group(0)
#print('Average rating: ',rigorous_instruction_0['Rigorous Instruction %'].mean())
#r = rigorous_instruction_0.where(rigorous_instruction_0['Rigorous Instruction %']==0).dropna()
#s =suppotive_environment_0.where(suppotive_environment_0['Supportive Environment %']==0).dropna()
#c = collaborative_teachers_0.where(collaborative_teachers_0['Collaborative Teachers %'] ==0).dropna()
#e = effective_school_leadership_0.where(effective_school_leadership_0['Effective School Leadership %']==0).dropna()
#st = strong_family_community_ties_0.where(strong_family_community_ties_0['Strong Family-Community Ties %'] == 0).dropna()
#t = trust_0.where(trust_0['Trust %'] == 0).dropna()
#print(r['School Name'])
#print(s['School Name'])
#print(c['School Name'])
#print(e['School Name'])
#print(st['School Name'])
#print(t['School Name'])
#print('School Count: ',suppotive_environment_0['School Name'].count())
#print('Average rating: ', suppotive_environment_0['Supportive Environment %'].mean())
#print('School Count: ',collaborative_teachers_0['School Name'].count())
#print('Average rating: ', collaborative_teachers_0['Collaborative Teachers %'].mean())
#collaborative_teachers_0.head().sort_values(by='Collaborative Teachers %').style.highlight_min()
#print('School count: ', effective_school_leadership_0['School Name'].count())
#print('Average rating: ',effective_school_leadership_0['Effective School Leadership %'].mean())
#effective_school_leadership_0.head(4).sort_values(by='Effective School Leadership %').style.highlight_min()
#print('School count: ',strong_family_community_ties_0['School Name'].count())
#print('Average rating: ',strong_family_community_ties_0['Strong Family-Community Ties %'].mean())
#strong_family_community_ties_0.head(4).sort_values(by='Strong Family-Community Ties %').style.highlight_min()
#print('School count: ', trust_0['School Name'].count())
#print('Average rating: ',trust_0['Trust %'].mean())
#trust_0.head(4).sort_values(by='Trust %').style.highlight_min()
tem = target.loc[[252,333]]
index = pd.concat([rigorous_instruction,collaborative_teachers,
               effective_school_leadership,
               strong_family_community_ties,
               trust,student_achievement, tem], join = 'inner').drop_duplicates()
target_new = target.loc[index.index]
#target_new.shape
SHSAT.rename(columns = {'School name':'School Name'}, inplace = True)
common_school = pd.merge(SHSAT, target_new, how = 'inner').drop_duplicates()

KAPPA = common_school.where(common_school['Year of SHST'] == 2016).dropna().set_index('School Name')
A = KAPPA['Rate of Enrollment on  10/31 %'] = KAPPA['Enrollment on 10/31']/KAPPA['Grade 8 ELA - All Students Tested']
B = KAPPA['Rate of Registration %'] = KAPPA['Number of students who registered for the SHSAT']/KAPPA['Enrollment on 10/31']
C = KAPPA['Rate of SHSAT Taken %'] = KAPPA['Number of students who took the SHSAT']/KAPPA['Number of students who registered for the SHSAT']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
KAPPA[['Grade 8 ELA - All Students Tested', 'Enrollment on 10/31',
       'Number of students who registered for the SHSAT',
       'Number of students who took the SHSAT']].plot(kind = 'bar',ax=axes[0,0])
KAPPA[['Percent ELL','Average ELA Proficiency', 'Percent White']].plot(kind = 'bar',ax=axes[0,1])
KAPPA[['Student Attendance Rate','Percent of Students Chronically Absent']].plot(kind = 'bar',ax=axes[1,0])
KAPPA[KAPPA.columns[32:44]].plot(kind = 'bar',ax=axes[1,1])
d = pd.Series({'Rate of Enrollment on  10/31 %':A.mean()*100,'Rate of Registration %':B.mean()*100,'Rate of SHSAT Taken %':C.mean()*100})
d.plot(kind = 'bar', figsize = (10,5))
print(round(d,2))
total = target_new['Grade 8 ELA - All Students Tested'].sum()
SHSAT_enrollment = int(total*A.mean())
SHSAT_Regi = int(SHSAT_enrollment*B.mean()) 
SHSAT_take = int(SHSAT_Regi*C.mean())
d =pd.Series({'Enrollment':SHSAT_enrollment, 'Registration':SHSAT_Regi,'SHSAT Will Be Taken': SHSAT_take})
d.plot(kind = 'bar', figsize = ((10,5)))
print(d)
target_new[['Percent ELL','Average ELA Proficiency','Percent White']].hist(figsize=(10,10), bins = 20, alpha = 0.5)
target_new[['Student Attendance Rate','Percent of Students Chronically Absent']].hist(figsize=(10,5), bins = 20)
target_new[target.columns[26:39]].hist(figsize=(10,10), bins = 20, color = 'orange')
target_new[target.columns[41:50]].hist(figsize=(20,15), bins = 20, color = 'green')
target_new[['Economic Need Index','School Income Estimate']].hist(figsize=(10,5), bins = 20, color = 'green')
target_new.plot.hexbin(x='Economic Need Index', y = 'School Income Estimate', gridsize=25)
not_meeting_target_rate = total/df['Grade 8 ELA - All Students Tested'].sum()
r = pd.Series({"Not Meeting Target Rate":total, "Total Grade 8 ELA Tested":df['Grade 8 ELA - All Students Tested'].sum()})
r.plot(kind='bar')
print('Among all the students that have taken the grade 8 ELA test in 2016, there were', round(not_meeting_target_rate*100, 2),'%', 'students belong to the not meeting target group.')