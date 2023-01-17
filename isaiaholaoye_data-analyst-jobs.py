import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data_jobs = pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")
data_jobs.head()
data_jobs.info()
data_jobs.columns
# Droping Unnamed column
data_jobs.drop("Unnamed: 0",1,inplace=True)
# This shows that column Company contains a NaN Value
data_jobs.notnull().all()
# The 1860 index contained the NaN value
data_jobs[data_jobs["Company Name"].isnull()]["Company Name"]
data_jobs["Company Name"].dropna(inplace=True)
data_jobs = data_jobs.apply(lambda x: x.replace("-1",np.nan))
data_jobs = data_jobs.apply(lambda x: x.replace(-1,np.nan))
data_jobs = data_jobs.apply(lambda x: x.replace("-1.0",np.nan))
data_jobs = data_jobs.apply(lambda x: x.replace("Unknown / Non-Applicable",np.nan))
plt.figure(figsize=(14,8))
sns.heatmap(data_jobs.isnull(), yticklabels=False, cbar=False,cmap="viridis")
plt.tight_layout()
len(data_jobs[data_jobs["Rating"].isnull()]["Rating"])
def missing_value_perc(check):
    null_value = data_jobs[check].isnull()
    len_value = len([x for x in null_value if x])
    percentage = (len_value/2253) * 100
    
    print(f"Total missing values in {check} is ({len_value}) and it's {round(percentage,2)}%")
    
missing_value_perc("Job Title")
missing_value_perc("Salary Estimate")
missing_value_perc("Job Description")
missing_value_perc("Rating")
missing_value_perc("Company Name")
missing_value_perc("Location")
missing_value_perc("Headquarters")
missing_value_perc("Size")
missing_value_perc("Founded")
missing_value_perc("Type of ownership")
missing_value_perc("Industry")
missing_value_perc("Sector")
missing_value_perc("Revenue")
missing_value_perc("Competitors")
missing_value_perc("Easy Apply")

# droping columns with missing values greater that 70%
data_jobs.drop(["Easy Apply", "Competitors"],1,inplace=True)
# Removing the rating values(\n float)
data_jobs["Company Name"].dropna(inplace=True)
data_jobs["Company Name"] = data_jobs["Company Name"].apply(lambda x: x.split("\n")[0])
# Removing (Glassdoor)
data_jobs["Salary Estimate"].dropna(how="all",inplace=True)
data_jobs["Salary Estimate"] = data_jobs["Salary Estimate"].apply(lambda salary: salary.split()[0])
data_jobs["Salary Estimate"].dropna(how="all",inplace=True)
data_jobs.insert(loc=2, column="Salary min_Estimate", 
                 value=data_jobs["Salary Estimate"].apply(lambda x: x.split("-")[0]))
data_jobs.insert(loc=3, column="Salary max_Estimate", 
                 value=data_jobs["Salary Estimate"].apply(lambda x: x.split("-")[1]))
# Seperating city from location
loc_city = data_jobs["Location"].apply(lambda x: x.split(",")[1])   
data_jobs.insert(loc=8, column="Location City", value=loc_city)
# Removing cities from location
data_jobs["Location"] = data_jobs["Location"].apply(lambda x: x.split(",")[0])
data_jobs.head(3)
data_jobs["Rating"].head(20)
plt.figure(figsize=(14,5))
sns.set_style("whitegrid")
sns.set_palette("RdPu_r", 5, 1)
sns.set_context("paper", rc={"lines.linewidth": 1.5})
sns.distplot(data_jobs["Rating"])
print(data_jobs["Rating"].skew())
print(data_jobs["Rating"].median())
data_jobs["Rating"].describe()
data_jobs["Rating"].fillna(value=round(data_jobs["Rating"].mean(),1), inplace=True)
data_jobs[data_jobs["Salary min_Estimate"].isnull()]
data_jobs["Salary min_Estimate"].dropna(how="all",inplace=True)
data_jobs["Salary min_Estimate"] = data_jobs["Salary min_Estimate"].apply(
    lambda x: float(x.split("$")[1].strip('K')))
data_jobs["Salary min_Estimate"].describe()
plt.figure(figsize=(14,5))
sns.set_style("whitegrid")
sns.set_palette("prism", 5, 1)
sns.set_context("paper", rc={"lines.linewidth": 1.5})
sns.distplot(data_jobs["Salary min_Estimate"])
plt.title("MINIMUM SALARY DISTRIBUTION FOR DATA ANALYST JOBS")
data_jobs["Salary max_Estimate"].dropna(how="all",inplace=True)
data_jobs["Salary max_Estimate"] = data_jobs["Salary max_Estimate"].apply(
    lambda x: float(x.split("$")[1].strip('K')))
data_jobs["Salary max_Estimate"].describe()
plt.figure(figsize=(14,5))
sns.set_style("whitegrid")
sns.set_palette("rainbow_r", 5, 1)
sns.set_context("paper", rc={"lines.linewidth": 1.5})
sns.distplot(data_jobs["Salary max_Estimate"])
plt.title("MAXIMUM SALARY DISTRIBUTION FOR DATA ANALYST JOBS")
data_jobs.dropna(inplace=True)
plt.figure(figsize=(14,8))
sns.heatmap(data_jobs.isnull(), yticklabels=False, cbar=False,cmap="viridis")
plt.tight_layout()
data_jobs["Salary max_Estimate"].fillna(value=90.0,inplace=True)
data_jobs["Salary min_Estimate"].fillna(value=54.3,inplace=True)
data_jobs["Salary max_Estimate"].sort_values().unique()
data_jobs["Salary min_Estimate"].sort_values().unique()
data = data_jobs.groupby(["Job Title"])[["Salary min_Estimate","Salary max_Estimate"]].mean().sort_values(["Salary min_Estimate","Salary max_Estimate"],ascending=False)
data = data[(data["Salary max_Estimate"] == 132.0) & (data["Salary min_Estimate"] == 113.0)]
data.reset_index(inplace=True)
data
plt.figure(figsize=(13,6))
sns.set_context("paper", rc={"lines.linewidth": 1.5})
sns.barplot(x="Salary min_Estimate", y="Job Title", data=data)
plt.title("BEST JOB WITH THE MEAN OF THE MINIMUM SALARY")
plt.figure(figsize=(13,6))
sns.set_context("paper", rc={"lines.linewidth": 1.5})
sns.barplot(x="Salary max_Estimate", y="Job Title", data=data)
plt.title("BEST JOB WITH THE MEAN OF THE MAXIMUM SALARY")
data_jobs["Rating"].sort_values().unique()
data_jobs["Rating"].max()
data_jobs[data_jobs["Rating"] == 5.0][["Job Title","Rating"]]
data_jobs[data_jobs["Rating"] == 5.0]["Job Title"].unique()
data_jobs["Location"].value_counts()
print(data_jobs[data_jobs["Location"] == "New York"]["Job Title"].unique())