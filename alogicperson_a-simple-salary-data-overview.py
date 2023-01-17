import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import csv
import pandas as pd

file = '../input/monthly_salary_brazil.csv'
f = open(file,'rt')
reader = csv.reader(f)

#once contents are available, I then put them in a list
csv_list = []
for l in reader:
    csv_list.append(l)
f.close()
#now pandas has no problem getting into a df
df = pd.DataFrame(csv_list)
#Solving the problem on line 845 and 847
i=1
while (i<10):
    df.loc[845,i] = df.loc[845,i+1]
    df.loc[847,i] = df.loc[847,i+1]
    i += 1
df.drop(labels=10,axis=1,inplace=True)

df.columns = df.iloc[0]
df = df.reindex(df.index.drop(0),)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
sns.set_style("darkgrid")
%matplotlib inline
def dist_info(data,target):
    print("="*90)
    print('Distribution info:')
    print("Mean: %f" %data[target].mean())
    print("STD: %f" %data[target].std())
    print("Skewness:%f" %data[target].skew())
    print("Kurtosis: %f" %data[target].kurt())
    print("="*90)
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (100*(data.isnull().sum()/data.isnull().count())).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
def Delta_mapping(p):
    mapping = [(15,'Under 15%'),(30,'15-30%'),(60,'30-60%'),(90,'60-90%'),
               (120,'90-120%'),(180,'120-180%'),(240,'180-240%'),(400,'240-400%'),
               (800,'400-800%')]
    for check, value in mapping:
        if p <= check:
            return value
# There are some details with the dataset that i needed to handle carefully,i'll mention them.
# first of all, we need to handle with the variables format.
df.info()
df = df.convert_objects(convert_numeric=True)
#Now it looks correct,let's proceed.
df.info()
df.head()
df = df[df.total_salary>0]
df[(df['Month_salary'] ==0) & (df['13_salary']==0) & 
    (df['eventual_salary']==0) & (df['indemnity']==0)&(df['extra_salary']==0) &
    (df['discount_salary']==0)].tail(4)
ddf = df[(df['Month_salary'] ==0) & (df['13_salary']==0) & 
    (df['eventual_salary']==0) & (df['indemnity']==0)&(df['extra_salary']==0) &
    (df['discount_salary']==0)].copy()

all_list = df.index
ddf_list =ddf.index
actual_list = all_list.difference(ddf_list)

df = df[df.index.isin(actual_list)].copy()
#Checking if our dataset has missing data:
sns.heatmap(df.isnull(),cbar=False)
missing_data(df)
# Limit salary 
limit = 21631.05
higher_values = df.groupby(['sector'])['total_salary'].mean().nlargest(15).values
higher_index = df.groupby(['sector'])['total_salary'].mean().nlargest(15).index
#the 15 higher Sector mean salaries 
plt.figure(figsize=(18,14))
plt.ylabel("Sector")
plt.xlabel("Total Salary")
pall = sns.color_palette("hls", 8)
sns.barplot(y=higher_index,x=higher_values,palette=pall)
higher_values = df.groupby(['job'])['total_salary'].mean().nlargest(15).values
higher_index = df.groupby(['job'])['total_salary'].mean().nlargest(15).index
#the 15 higher Job mean salaries
plt.figure(figsize=(18,14))
plt.ylabel("Job")
plt.xlabel("Total Salary")
pall =sns.color_palette("hls", 8)
sns.barplot(y=higher_index,x=higher_values,palette=pall)
plt.axvline(limit,linewidth=4,color='orange')
index_larg = df.total_salary.nlargest(100).index
job_larg = df[df.index.isin(index_larg)].sort_values(by='total_salary',ascending=False)['job'].values
salar_larg =df[df.index.isin(index_larg)].sort_values(by='total_salary',ascending=False)['total_salary'].values

plt.figure(figsize=(18,14))
plt.ylabel("Job")
plt.xlabel("Total Salary")
sns.barplot(y=job_larg,x=salar_larg,palette='GnBu_d')
plt.axvline(limit,linewidth=4,color='orange')
#let's look at the distribution of total_salary

sns.distplot(df.total_salary,fit=norm,fit_kws={'color':"red"})
fig = plt.figure()
res = stats.probplot(df.total_salary, plot=plt)
dist_info(df,'total_salary')
df['log_salary'] = df.total_salary.apply(np.log)
# with this transformation,let's look at it again and see if it worked : 

sns.distplot(df.log_salary,fit=norm,fit_kws={'color':"red"})
fig = plt.figure()
res = stats.probplot(df.log_salary, plot=plt)
dist_info(df,'log_salary')
v = []
print("Start")
print("="*90)
for i in range(len(df)):
    summ = df.iloc[i:i+1,3:9].sum(axis=1).values[0]
    tot = df.iloc[i:i+1,9:10]['total_salary'].values[0]
    v.append((1 - tot/summ)*100)
print("="*90)
print("End")
df['Delta_S'] = v
df['Delta Interval'] = df.Delta_S.apply(Delta_mapping)
df['Delta Interval'].value_counts()
plt.figure(figsize=(15,9))
sns.barplot(x=df['Delta Interval'].unique(),y=df['Delta Interval'].value_counts(normalize=True)*100)
sns.distplot(df['Delta_S'])
df = df[df.index!=990847].copy()
sns.distplot(df[(df.Delta_S>=0) & (df.Delta_S <=100)]['Delta_S'],fit=norm,fit_kws={'color':"red"})
dist_info(df[(df.Delta_S>=0) & (df.Delta_S <=100)],'Delta_S')
df['Over_Limit'] = df.total_salary.apply(lambda x:1 if x > limit else 0)
sns.distplot(df[df.Over_Limit==1]['log_salary'],color='green',kde_kws={'label':"Over Limit"})
sns.distplot(df[df.Over_Limit==0]['log_salary'],color='blue',kde_kws={'label':"Under Limit"})
plt.legend()
plt.xlim(5,13)
plt.xlabel("Log_Salary",fontsize=12)
plt.ylabel("Limit Density",fontsize=12)
print("Proportion of ID's with Salary under the limit : {} %".format(100*len(df[df.Over_Limit==0])/len(df)))
print("Proportion of ID's with Salary over the limit : {} %".format(100*len(df[df.Over_Limit==1])/len(df)))
