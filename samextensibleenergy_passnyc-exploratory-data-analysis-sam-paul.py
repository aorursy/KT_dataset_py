import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt
shsat = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
list(shsat)
shsat.head()
satdata = shsat  # creating a new dataframe , keeping the original as it is.
satdata['% taken exam'] = 100 *(satdata['Number of students who took the SHSAT']/satdata['Number of students who registered for the SHSAT'])
shsat_8 = satdata[satdata['Grade level']== 8]
shsat_9 = satdata[satdata['Grade level']== 9] # also creating a separate DF for grades 8 and 9 , if needed to be used later
plt.figure(figsize=(20,7))
shsat['Enrollment on 10/31'].plot.bar(alpha=0.10, color = 'Black', legend=True)
shsat['Number of students who registered for the SHSAT'].plot.bar(alpha=0.60 , color = 'yellow', legend=True)
shsat['Number of students who took the SHSAT'].plot.bar(edgecolor = 'black' , color = 'red', legend=True)
shsat_yr = pd.DataFrame(shsat.groupby(['Year of SHST']).sum()).reset_index() # generating a new DF after grouping in
shsat_yr.head()
ax = shsat_yr.plot(x='Year of SHST', y= ['Enrollment on 10/31','Number of students who registered for the SHSAT','Number of students who took the SHSAT'], 
                   kind='bar', figsize=(20, 8), legend=True, fontsize=12, edgecolor = 'black')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
plt.show()
shsat_nm = pd.DataFrame(shsat.groupby(['School name', 'Year of SHST' ]).sum()).reset_index() # generating a DF by School Name
shsat_nm['% taken exam'] = 100*(shsat_nm['Number of students who took the SHSAT']/shsat_nm['Number of students who registered for the SHSAT'])
shsat_nm['% cut-off'] = 50
shsat_nm.head()
ax =  shsat_nm.set_index('School name')[['% taken exam']].plot.bar(figsize=(20, 8), legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.49, 
                 color = 'hotpink') 

ax.set_xlabel("School", fontsize=12)
ax.set_ylabel("%-age Taken Exam after registration", fontsize=12)
plt.show()



plt.figure(figsize=(20,7))
shsat_nm['% taken exam'].plot.bar(x=['School name','Year of SHST'], alpha=0.70, color = 'hotpink', legend=True)
shsat_nm['% cut-off'].plot(alpha=0.99, color = 'darkmagenta', legend=True)
plt.show()
s_exp = pd.read_csv('../input/2016 School Explorer_new.csv')
# s_exp.dtypes
# list(s_exp)
## Check the dtypes of this file, I have converted the Grades_high column from object to Float type for easier manuevering.
data = s_exp # # creating a new dataframe , keeping the original as it is.
data['% Grade 3 ELA 4s - All Students'] = data['Grade 3 ELA 4s - All Students']/data['Grade 3 ELA - All Students Tested'] 
data['% Grade 4 ELA 4s - All Students'] = data['Grade 4 ELA 4s - All Students']/data['Grade 4 ELA - All Students Tested'] 
data['% Grade 5 ELA 4s - All Students'] = data['Grade 5 ELA 4s - All Students']/data['Grade 5 ELA - All Students Tested'] 
data['% Grade 6 ELA 4s - All Students'] = data['Grade 6 ELA 4s - All Students']/data['Grade 6 ELA - All Students Tested'] 
data['% Grade 7 ELA 4s - All Students'] = data['Grade 7 ELA 4s - All Students']/data['Grade 7 ELA - All Students Tested'] 
data['% Grade 8 ELA 4s - All Students'] = data['Grade 8 ELA 4s - All Students']/data['Grade 8 ELA - All Students Tested'] 


data['% Grade 3 Math 4s - All Students'] = data['Grade 3 Math 4s - All Students']/data['Grade 3 Math - All Students tested'] 
data['% Grade 4 Math 4s - All Students'] = data['Grade 4 Math 4s - All Students']/data['Grade 4 Math - All Students Tested'] 
data['% Grade 5 Math 4s - All Students'] = data['Grade 5 Math 4s - All Students']/data['Grade 5 Math - All Students Tested'] 
data['% Grade 6 Math 4s - All Students'] = data['Grade 6 Math 4s - All Students']/data['Grade 6 Math - All Students Tested'] 
data['% Grade 7 Math 4s - All Students'] = data['Grade 7 Math 4s - All Students']/data['Grade 7 Math - All Students Tested'] 
data['% Grade 8 Math 4s - All Students'] = data['Grade 8 Math 4s - All Students']/data['Grade 8 Math - All Students Tested'] 
d_8plus = data[data['Grade High']>7]
d_8minus = data[data['Grade High']<8] #also creating a separate DF for grades_high above and below 8, if needed to be used later
df1 = d_8plus[['Grade High', 'Economic Need Index', 'School Income Estimate', 'Percent Black / Hispanic','Percent of Students Chronically Absent',
            'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %','Effective School Leadership %', 'Strong Family-Community Ties %', 
            'Trust %']]  # creating a working dataframe
df1 = df1.dropna() # removing the Nan Values
x = df1['School Income Estimate']
plt.hist(x, bins=14)
plt.ylabel('No of times')
plt.show()
data['Total Students Enrolled'] =  (data['Grade 3 Math - All Students tested'] +   data['Grade 3 ELA - All Students Tested'] +
                                    data['Grade 4 Math - All Students Tested'] +   data['Grade 4 ELA - All Students Tested'] +
                                    data['Grade 5 Math - All Students Tested'] +   data['Grade 5 ELA - All Students Tested'] +
                                    data['Grade 6 Math - All Students Tested'] +   data['Grade 6 ELA - All Students Tested'] +
                                    data['Grade 7 Math - All Students Tested'] +   data['Grade 7 ELA - All Students Tested'] +
                                    data['Grade 8 Math - All Students Tested'] +   data['Grade 8 ELA - All Students Tested']  )/2
data['Per-Capita Income'] =  data['School Income Estimate'] / data['Total Students Enrolled']
d_8plus = data[data['Grade High']>7]
df =  d_8plus[['Per-Capita Income']]
df = df.dropna() # removing the Nan Values
x = df['Per-Capita Income']
plt.hist(x, bins=14)
plt.ylabel('No of times')
plt.show()
blk = d_8plus['Percent Black'].mean()
ELL = d_8plus['Percent ELL'].mean()
asn = d_8plus['Percent Asian'].mean()
hpn = d_8plus['Percent Hispanic'].mean()
wht = d_8plus['Percent White'].mean()

blk2 = d_8minus['Percent Black'].mean()
ELL2 = d_8minus['Percent ELL'].mean()
asn2 = d_8minus['Percent Asian'].mean()
hpn2 = d_8minus['Percent Hispanic'].mean()
wht2 = d_8minus['Percent White'].mean()
# Data to plot
labels = 'Black', 'ELL', 'Asian', 'Hispanic' , 'White'
sizes = [blk, ELL, asn, hpn, wht]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
explode = (0.05, 0, 0, 0.05,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode,  labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

labels2 = 'Black', 'ELL', 'Asian', 'Hispanic' , 'White'
sizes2 = [blk2, ELL2, asn2, hpn2, wht2]
colors2 = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
explode2 = (0.05, 0, 0, 0.05,0)  # explode 1st slice

# Plot
plt.pie(sizes2, explode=explode2,  labels=labels2, colors=colors2,
        autopct='%1.1f%%', shadow=True, startangle=140)

 
plt.axis('equal')
plt.show()
from string import ascii_letters

sns.set(style="white")

# Generate a large random dataset
d = df1[[ 'Economic Need Index', 'School Income Estimate', 'Percent Black / Hispanic']]

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df1['School Income Estimate'].corr(df1['Economic Need Index'])
df1['School Income Estimate'].corr(df1['Percent Black / Hispanic'])
df2 = d_8plus[['Economic Need Index','Percent Black / Hispanic','Percent of Students Chronically Absent','Trust %',
            'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %','Effective School Leadership %', 'Strong Family-Community Ties %', 
             '% Grade 3 ELA 4s - All Students', '% Grade 4 ELA 4s - All Students', '% Grade 5 ELA 4s - All Students', 
           '% Grade 6 ELA 4s - All Students', '% Grade 7 ELA 4s - All Students', '% Grade 8 ELA 4s - All Students']]
df2 = df2.dropna()
sns.set(style="white")

# Generate a large random dataset
d = df2

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df3 = d_8plus[['Economic Need Index','Percent Black / Hispanic','Percent of Students Chronically Absent','Trust %',
            'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %','Effective School Leadership %', 'Strong Family-Community Ties %', 
             '% Grade 3 Math 4s - All Students', '% Grade 4 Math 4s - All Students', '% Grade 5 Math 4s - All Students', 
           '% Grade 6 Math 4s - All Students', '% Grade 7 Math 4s - All Students', '% Grade 8 Math 4s - All Students']]
df3 = df3.dropna()
sns.set(style="white")

# Generate a large random dataset
d = df3

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
### Calling the All School Dataframe back again to compare the new findings above
data.head()
data.set_index("Location Code", inplace=True) # Setting Location code as the index of this data set
data_t = data.loc[['05M046','05M123', '05M129','05M148','05M161','05M286','05M302','05M362','05M499',
                   '05M514','05M670','84M065','84M284','84M336','84M341','84M350','84M384','84M388',
                   '84M481','84M709','84M726']] # sorting particular data from the whole data set 
data_t = pd.DataFrame(data_t).reset_index() # creating a new dataframe including only the school for our study purpose
# data_t.head()
# list(data_t)
ax =  data_t.set_index('School Name')[['Average Math Proficiency', 'Average ELA Proficiency']].plot.bar(figsize=(20, 8), legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.99, 
                 color = ['deeppink', 'crimson']) 
ax.set_xlabel("School", fontsize=12)
ax.set_ylabel("GPA", fontsize=12)
plt.show()

ax =  data_t.set_index('School Name')[['Economic Need Index', 'Percent Black / Hispanic', 'Percent of Students Chronically Absent']].plot.bar(figsize=(20, 8), legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.99, 
                 color = ['gold', 'goldenrod', 'sienna']) 
ax.set_xlabel("School", fontsize=12)
ax.set_ylabel("Ratio -  Percentage", fontsize=12)
plt.show()

ax =  data_t.set_index('School Name')[['Grade 8 Math 4s - Economically Disadvantaged', 'Grade 8 Math 4s - All Students', 
                     'Grade 8 ELA 4s - Economically Disadvantaged', 'Grade 8 ELA 4s - All Students']].plot.bar(figsize=(20, 8), 
                 legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.99, 
                 color = [ 'orchid', 'darkmagenta', 'paleturquoise',  'darkcyan']) 
ax.set_xlabel("School", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
plt.show()
ax =  data_t.set_index('School Name')[['Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %', 'Effective School Leadership %',
                    'Strong Family-Community Ties %', 'Trust %']].plot.bar(figsize=(20, 8), 
                 legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.99, 
                 color = [ 'orchid', 'darkmagenta', 'palevioletred',  'crimson', 'maroon', 'black']) 
ax.set_xlabel("School", fontsize=12)
ax.set_ylabel("Ratio - Percentage", fontsize=12)
plt.show()
ax =  satdata.set_index('School name')[['Enrollment on 10/31']].plot.bar(figsize=(20, 8), 
                 legend=True, fontsize=12, 
                 edgecolor = 'black', 
                 alpha=0.99, 
                 color = 'crimson') 
ax.set_xlabel("School", fontsize=12)
ax.set_ylabel("Enrollment Numbers", fontsize=12)
plt.show()



