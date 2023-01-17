# Load all the nessary libraries

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix

plt.style.use('ggplot')


# Load data from csv file 
import numpy as np
import pandas as pd

df = pd.read_csv('/kaggle/input/full-time-employees-by-job-classification.csv', na_values=['#NAME?']) # '#NAME?' in the datafile will be converted to NaN
df.head()
# remove rows with missing value (or 0) in Biweekly Low Rate/Biweekly High Rate
df = df.loc[ (df['Biweekly Low Rate'] > 0) & (df['Biweekly High Rate'] > 0), ]
print (df.head(5))
#Visualization
df_empByJob = df.copy()
df_empByJob.isnull().sum()
#Department Code:Department

dept_code = { "AAM":"Asian Art Museum",
"ADM":"General Services Agency - City Admin",
"ADP":"Adult Probation",
"AIR":"Airport Commission",
"ART":"Arts Commission",
"ASR":"Assessor/Recorder",
"BOS":"Board of Supervisors",
"CAT":"City Attorney",
"CFC":"Children and Families Commission",
"CHF":"Children, Youth & Their Families",
"CLP":"PUC Clean Power",
"CON":"Controller",
"CPC":"City Planning",
"CSC":"Civil Service Commission",
"CSS":"Child Support Services",
"CWP":"PUC Wastewater Enterprise",
"DAT":"District Attorney",
"DBI":"Department of Building Inspection",
"DPA":"Police Accountability",
"DPH":"Public Health",
"DPT":"Physical Therapy",             
"DPW":"General Services Agency - Public Works",
"DSS":"Human Services",
"ECD":"Department of Emergency Management",
"ECN":"Economic and Workforce Development",
"ENV":"Environment",
"ETH":"Ethics Commission",
"FAM":"Fine Arts Museum",
"FIR":"Fire Department",
"HHP":"PUC Hetch Hetchy",
"HOM":"Homelessness and Supportive Housing",
"HRC":"Human Rights Commission",
"HRD":"Human Resources",
"HSS":"Health Service System",
"JUV":"Juvenile Probation",
"LIB":"Public Library",
"LLB":"Law Library",
"MTA":"Municipal Transportation Agency",
"MYR":"Mayor",
"PAB":"Board of Appeals",
"PDR":"Public Defender",
"POL":"Police",
"PRT":"Port",
"PTC":"Parking and Traffic Commission",            
"PUC":"PUC Public Utilities Commission",
"REC":"Recreation and Park Commission",
"REG":"Elections",
"RET":"Retirement System",
"RNT":"Rent Arbitration Board",
"SCI":"Academy of Sciences",
"SHF":"Sheriff",
"TIS":"General Services Agency - Technology",
"TTX":"Treasurer/Tax Collector",
"TXC":"Taxi Commission",             
"USD":"County Education Office",
"WAR":"War Memorial",
"WOM":"Department of the Status of Women",
"WTR":"PUC Water Department"}

df_empByJob["Dept Level"]=df_empByJob["Dept Level"].replace(dept_code)
df_empByJob.sort_values(by = 'Biweekly Low Rate', ascending = False, inplace=True)

# Add Biweekly Mean Rate Column
col = df_empByJob.loc[: , "Biweekly Low Rate":"Biweekly High Rate"]
df_empByJob['Biweekly Mean Rate'] = col.mean(axis=1)

# test dataframe with '0' rates
test_df = df_empByJob[df_empByJob["Biweekly Mean Rate"]==0.00]

# drop rows with '0' Rates
df_emp_MeanRate = df_empByJob[df_empByJob["Biweekly Mean Rate"]!=0.00]

#Calculate Avg Rate
avgSalary = df_emp_MeanRate['Biweekly Mean Rate'].mean()

# Add Avg Score Column for above and below avg Rate
df_emp_MeanRate["Avg_Score"] = (df_emp_MeanRate["Biweekly Mean Rate"] > avgSalary).astype(int)
df_emp_MeanRate
# Plot Above and Below Avg Mean Rate
g = df_emp_MeanRate.groupby(['Dept Level', 'Avg_Score']).size().reset_index(name='count')
g

data = g.set_index('Dept Level').unstack().reset_index()
data.columns = ['Dept Level','Avg_Score', 'count']

sns.set(style="white")
d = sns.factorplot(x='Dept Level'
                   ,y= 'count'
                   ,hue='Avg_Score'
                   ,data=g
                   ,kind='bar'
                   ,aspect=5
                   )
d.set_xticklabels(rotation=90);
plt.show()
#plotting Mean Rate distribution, with vertical lines to represent the mean and median salary
sal_plot = df_emp_MeanRate[df_emp_MeanRate["Biweekly Mean Rate"].notnull()]
ax = sns.distplot(sal_plot["Biweekly Mean Rate"])
ax.axvline(sal_plot["Biweekly Mean Rate"].median(), lw=2.5, ls='dashed', color='black')
ax.axvline(sal_plot["Biweekly Mean Rate"].mean(), lw=2.5, ls='dashed', color='red')

# Plot 20 Job Titles
df_emp_MeanRate = df_emp_MeanRate.sort_values('Biweekly Mean Rate',ascending=False)
top20_Job = df_emp_MeanRate.head(20)
top20_Job
f, ax = plt.subplots(figsize=(10, 15)) 
ax.set_yticklabels(top20_Job['Job Title'], rotation='horizontal', fontsize='large')
g = sns.barplot(y = top20_Job['Job Title'], x= top20_Job['Biweekly Mean Rate'])
plt.axvline(avgSalary, color="red", linestyle="--");
plt.show()
print('Average BiWeekly Low Rate : {:8.2f} dollars'.format(df_emp_MeanRate['Biweekly Low Rate'].mean()))
print('Average Biweekly High Rate    : {:8.2f} dollars'.format(df_emp_MeanRate['Biweekly High Rate'].mean()))
# Biweekly Low rate and Biweekly High Rate
%matplotlib inline
df_emp_MeanRate[['Biweekly Low Rate', 'Biweekly High Rate']].hist(figsize=(15, 6), edgecolor='black', linewidth=1.2,bins=30, grid=False)
# Convert for one column
department = df_emp_MeanRate['Dept Level'].astype('category')
group = df_emp_MeanRate.groupby('Dept Level')
group.mean().head(10)
group.agg(['count', 'min', 'max', 'std', 'mean']).head()
# Avg Biweekly Rate per Department
average_MeanRate = group['Biweekly Mean Rate'].mean().sort_values(ascending=False)
average_MeanRate.plot(kind='bar', figsize=(15, 6))
plt.axhline(avgSalary, color="red", linestyle="--")
average_MeanRate = group['Biweekly Mean Rate'].sum().sort_values(ascending=False)
average_MeanRate.plot(kind='bar', figsize=(15, 6))
