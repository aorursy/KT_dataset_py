import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
%matplotlib inline
import theano.tensor as T
import scipy.stats as stats
from scipy.interpolate import griddata

%config InlineBackend.figure_format = 'retina'
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
df = pd.read_csv("../input/2016 School NYC.csv")
df.head(2)
# df = df[(df['Grade High']=='08')& (df['City']=='BROOKLYN')]
df = df[(df['Grade High']=='8')]
df1 = df.copy()
df1 = df1[['Grade High','Economic Need Index','School Income Estimate','Percent Black / Hispanic','Percent White',
           'Percent of Students Chronically Absent','Average ELA Proficiency','Grade 6 Math - All Students Tested',
           'Grade 6 Math 4s - All Students', 'Supportive Environment Rating', 'Effective School Leadership Rating', 
           'Trust Rating','Student Achievement Rating','Percent Black','Percent Hispanic','Trust %', 
           'Collaborative Teachers %', 'Average Math Proficiency','Supportive Environment %','School Name']]
# Create column for percent of students that received a 4 in math from all students that took the exam.
df1['prcnt_sdnt_math_4s'] = (df1['Grade 6 Math 4s - All Students'] / df1['Grade 6 Math - All Students Tested']).round(3)
# convert data to floats
df1['Percent of Students Chronically Absent'] = df1['Percent of Students Chronically Absent'].str.rstrip('%').astype(float)
df1['Percent Black'] = df1['Percent Black'].str.rstrip('%').astype(float)
df1['Percent Hispanic'] = df1['Percent Hispanic'].str.rstrip('%').astype(float)

df1['Trust %'] = df1['Trust %'].str.rstrip('%').astype(float)
df1['Supportive Environment %'] = df1['Supportive Environment %'].str.rstrip('%').astype(float)
df1['Collaborative Teachers %'] = df1['Collaborative Teachers %'].str.rstrip('%').astype(float)
df1['Percent White'] = df1['Percent White'].str.rstrip('%').astype(float)
df1['School Income Estimate'] = df1['School Income Estimate'].str.strip('$')
df1['School Income Estimate'] = df1['School Income Estimate'].str.replace(",","").astype(float)

# drop the Null row
df1.dropna(subset=['Economic Need Index','prcnt_sdnt_math_4s'],inplace=True)
df1 = df1[(df1['Percent Black']>50) | (df1['Percent White']>50) | (df1['Percent Hispanic']>50)]
# Create race columns where the majority of students from each race represent the school

def f(row):
    if row['Percent Black'] > 50:
        val = 'African American'
    elif row['Percent White'] > 50:
        val = 'White'
    elif row['Percent Hispanic'] > 50:
        val = 'Hispanic'
    return val

df1['race'] = df1[['Percent Black','Percent White','Percent Hispanic']].apply(f, axis=1)
df1['race'].value_counts()
plt.figure(figsize=(14,8))
plt.scatter(df1['Economic Need Index'], df1['Average ELA Proficiency'], color='#b50d7c')

plt.title('Economic Need vs ELA average grade')
plt.xlabel("Economic Need Index", fontsize=14)
plt.ylabel("Average ELA Proficiency", fontsize=14)
plt.savefig("1Economic_need.png", bbox_inches='tight')
plt.show()
plt.figure(figsize=(14,8))
plt.scatter(df1['Economic Need Index'], df1['Average ELA Proficiency'], color='#b50d7c',
            s=df1['Percent of Students Chronically Absent']**1.8, alpha=0.3)

plt.title('Economic Need vs ELA average grade + % of Chronic Absence')
plt.xlabel("Economic Need Index", fontsize=14)
plt.ylabel("Average ELA Proficiency", fontsize=14)
#label marker
for idx, x in df1['Percent of Students Chronically Absent'][[266,1211]].iteritems():
    plt.annotate(x,(df1['Economic Need Index'][idx]+0.015,df1['Average ELA Proficiency'][idx]), 
                 fontsize=14)
plt.savefig("2Economic_need.png", bbox_inches='tight')
plt.show()
color= ['#111111' if x=='African American' else '#b50d7c' if x =='White' else '#999999' for x in df1['race']]

plt.figure(figsize=(14,8))
plt.scatter(df1['Economic Need Index'], df1['Average ELA Proficiency'], 
        color=color, s=df1['Percent of Students Chronically Absent']**1.8,alpha=0.9)

plt.title('Economic Need vs ELA average grade, by race')
plt.xlabel("Economic Need Index", fontsize=14)
plt.ylabel("Average ELA Proficiency", fontsize=14)   

# create labels and legend manually
import matplotlib.patches as mpatches
magenta_patch = mpatches.Patch(color='magenta', label='White')
black_patch = mpatches.Patch(color='black', label='African American')
gray_patch = mpatches.Patch(color='gray', label='Hispanic')
plt.legend(handles=[magenta_patch, black_patch, gray_patch])

plt.savefig("3Economic_need.png", bbox_inches='tight')
plt.show()
# Remove hispanic children from race
df1 = df1[(df1['Percent Black']>50) | (df1['Percent White']>50)]
sns.violinplot(x='Supportive Environment Rating', y='Percent of Students Chronically Absent', data=df1, 
               inner='box', color='lightgray')

sns.stripplot(x='Supportive Environment Rating', y='Percent of Students Chronically Absent', data=df1, 
              jitter=True, hue='race', 
              palette=sns.color_palette(["#111111","#b50d7c"]), size=6, alpha=0.7)

plt.xticks(rotation='horizontal')
fig=plt.gcf()
fig.set_size_inches(14,8)
plt.title("Supportive Environment Rating vs % of Students Chronically Absent")
plt.savefig("SupportiveEnvironment.png", bbox_inches='tight')
plt.show()
sns.barplot(x='race', y='Percent of Students Chronically Absent', data=df1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.title("% of Students Chronically Absent")
plt.savefig("Percnt_Chronically_Absent.png", bbox_inches='tight')
plt.show()
sns.boxplot(x='Trust Rating', y='Average Math Proficiency', data=df1, color='lightgray', order=['Not Meeting Target',
                                                      'Approaching Target',
                                                      'Meeting Target',
                                                      'Exceeding Target'])


sns.stripplot(x='Trust Rating', y='Average Math Proficiency', data=df1, jitter=True, size=5, hue='race', 
              palette=sns.color_palette(["#111111","#b50d7c"]), alpha=1, order=['Not Meeting Target',
                                                                                  'Approaching Target',
                                                                                  'Meeting Target',
                                                                                  'Exceeding Target'])
plt.xticks(rotation='horizontal')
fig=plt.gcf()
fig.set_size_inches(14,8)
plt.title("Trust Rating vs Average Math Proficiency")
plt.savefig("Trust_Rating.png", bbox_inches='tight')
plt.show()
sns.boxplot(x='Effective School Leadership Rating', y='Average ELA Proficiency', data=df1, 
            color='lightgray', order=['Not Meeting Target',
                                        'Approaching Target',
                                        'Meeting Target',
                                        'Exceeding Target'])

sns.stripplot(x='Effective School Leadership Rating', y='Average ELA Proficiency', data=df1, jitter=True, size=5, hue='race', 
              palette=sns.color_palette(["#111111","#b50d7c"]), order=['Not Meeting Target',
                                                                                  'Approaching Target',
                                                                                  'Meeting Target',
                                                                                  'Exceeding Target'])
plt.xticks(rotation='horizontal')
fig=plt.gcf()
fig.set_size_inches(14,8)
plt.title("Effective School Leadership Rating vs Average ELA Proficiency")
plt.savefig("Effective_School_Leadership.png", bbox_inches='tight')
plt.show()
sns.barplot(x='race', y='Average ELA Proficiency', data=df1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.title("Average ELA Proficiency")
plt.savefig("ELA Proficiency.png", bbox_inches='tight')
plt.show()