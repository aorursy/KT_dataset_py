import pandas as pd

job=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
job.head()

job['Salary Estimate']
#src.insert(2,"Average Salary",0)

#calculating avg salary
def avg_sal(x):
    x = x.replace("(Glassdoor est.)","")
    x = x.replace("$","")
    x = x.replace("K","000")
    if x.find("0-")!=-1:
        avg = (int(x.split("-")[0])+int(x.split("-")[1]))/2
    else:
        avg=0
    return avg
print(avg_sal)
job.insert(2,"Average Salary",0)
job["Average Salary"] = job["Salary Estimate"].apply(avg_sal)

job
job.drop(['Unnamed: 0','Salary Estimate'],axis=1,inplace=True)
job_high_rated=job[job.Rating >=job['Rating'].median()]
job_low_rated=job[job.Rating <job['Rating'].median()]
job.info()
job_high_rated_sal=job_high_rated[job_high_rated['Average Salary']>=job_high_rated['Average Salary'].median()]
job_high_rated_sal
job_high_rated_sal.nunique()
max_sal = job_high_rated_sal.groupby(by=['Company Name', 'Job Title', 'Rating', 'Sector', 'Location'])['Average Salary']
#.sort_values(by=(['Rating', 'Max Salary']),ascending=False).head(10)

max_sal = job_high_rated.groupby(['Company Name', 'Job Title', 'Rating', 'Sector', 'Location'])['Average Salary'].median().reset_index()
max_sal.sort_values(by=(['Rating', 'Average Salary']),ascending=False).head(10)

max_sal.nunique()
max_sal.shape
max_sal['Location'].value_counts(ascending=False)[:15].plot(kind='bar')
max_sal['Sector']=max_sal['Sector'].replace('-1','None')

max_sal['Sector'].value_counts(ascending=False)[:15].plot(kind='bar')
max_sal.corr()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
max_sal['Job Title'].value_counts()[:12].plot(kind='pie',autopct='%1.1f%%', labeldistance = None, pctdistance = 0.4, textprops={'fontsize': 10})
plt.legend( loc='best')
