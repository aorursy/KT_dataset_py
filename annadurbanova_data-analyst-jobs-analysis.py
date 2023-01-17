import pandas as pd

import numpy as np

import holoviews as hv

import seaborn as sns

import matplotlib.pyplot as plt

hv.extension('bokeh')

!pip install hvplot

#!pip install pyzmq

#!pip install --upgrade pip

#!pip install -- pyzmq==17.0.0

#!conda install -c conda-forge pyzmq--y
import hvplot.pandas

import os

import glob

import panel as pn

import xarray as xr

import hvplot.xarray  

from hvplot import hvPlot







%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv', na_values="-1")

#data = pd.read_csv("DataAnalyst.csv", na_values="-1")

data = data.drop(columns="Unnamed: 0", axis=1)



##---------Salary-----------##



data[["Salary", "Source"]]=data["Salary Estimate"].str.split(" ",n=1, expand=True)

data[["Min Salary", "Max Salary"]]=data["Salary"].str.split("-", n=2, expand=True)



data = data.drop(["Salary", "Salary Estimate"], axis=1)



data["Min Salary"]= data["Min Salary"].str.replace("K", "000").str.replace('$', '').astype(float)

data["Max Salary"]= data["Max Salary"].str.replace("K", "000").str.replace('$', '').astype(float)



data["Source"] = data["Source"].str.replace("(Glassdoor est.)", "Glassdoor")



##---------Company Name-----------##



data["Company Name"]=data["Company Name"].str.split("\n", n=2).str[0]



##---------Location & Headquarters-----------##



data[["Location City", "Location Abb"]]=data["Location"].str.split(",", n=1, expand=True)

data=data.drop(["Location"], axis=1)



data[["Headquarter City", "Headquarter Abb"]]=data["Headquarters"].str.split(",", n=1, expand=True)

data=data.drop(["Headquarters"], axis=1)



##---------Company Size-----------##



company_size=data["Size"].str.split(n=3, expand=True)

data["Min Company Size, employees"]=(company_size[0]

    .replace("10000+","10000")

    .replace("Unknown",np.nan)

    .astype(float)

)

data["Max Company Size, employees"]=company_size[2].replace({None:np.nan}).astype(float)

data=data.drop("Size", axis=1)



##---------Founded & Ownership-----------##



data["Founded"]=data["Founded"].astype('Int64')

data["Type of ownership"]=data["Type of ownership"].replace("Unknown",np.nan)



##---------Revenue---------------##

data["Revenue"] = data["Revenue"].replace('Unknown / Non-Applicable', np.nan)

revenues=pd.CategoricalDtype(categories=[

'Less than $1 million (USD)', '$1 to $5 million (USD)','$5 to $10 million (USD)','$10 to $25 million (USD)','$25 to $50 million (USD)','$50 to $100 million (USD)','$100 to $500 million (USD)',

'$500 million to $1 billion (USD)','$1 to $2 billion (USD)','$2 to $5 billion (USD)','$5 to $10 billion (USD)','$10+ billion (USD)'

],ordered=True

)

data["Revenue"]=data["Revenue"].astype(revenues)



##---------Easy Apply---------------##



data["Easy Apply"]=data["Easy Apply"].replace({"True":True, pd.NA:False})

data = data[['Job Title', 'Job Description', 'Min Salary', 'Max Salary', 'Rating', "Company Name", "Location City", "Location Abb", "Headquarter City","Headquarter Abb", "Founded", "Type of ownership","Industry","Sector", "Revenue", "Competitors","Easy Apply","Source"]]



data.head()
#data["Max Salary"].hist()



sns.distplot(data["Max Salary"])
#Which top 10 Job Titles has the most salary?

data["Max Salary"].describe()

#Maximum Salary is 190000
## What is the median max salary for each industry

median_salary_high=(data

.groupby("Industry")

 [["Min Salary", "Max Salary", "Rating"]]

.median()

.sort_values(by="Max Salary", ascending=False)

.head(10)

)

median_salary_high
## Q2: What is the median min salary for each industry

median_salary_low=(data

.groupby("Industry")

 [["Min Salary", "Max Salary", "Rating"]]

.median()

.sort_values(by="Max Salary", ascending=True)

.head(10)

)

median_salary_low
plot_max_salary=median_salary_high.hvplot.bar(x="Industry", y="Max Salary", stacked=True, rot=90, width=800, height=400)

plot_min_salary=median_salary_low.hvplot.bar(x="Industry", y="Max Salary", stacked=True, rot=90, width=800, height=400)

plot_max_salary*plot_min_salary
median_job_title_high=(data

.groupby("Job Title")

 [["Min Salary", "Max Salary"]]

.median()

.sort_values(by="Max Salary", ascending=False)

.head(10)

)

median_job_title_high



 ## Min Salary of the Maximum salary is 38 000

median_job_title_low=(data

.groupby("Job Title")

[["Min Salary", "Max Salary"]]

.median()

.sort_values(by="Max Salary", ascending= True)

.head(10)

)

median_job_title_low

plot1=median_job_title_high.hvplot.bar(x="Job Title", y="Max Salary", stacked=True, rot=45, width=800, height=500, title="Job Titles with the highest salary")

plot2=median_job_title_low.hvplot.bar(x="Job Title", y="Max Salary", rot=45, stacked=True, width=1000, height=500, title = "Job titles with the lowest salary")

plot1*plot2
# Q3: Which US States have more Data Analyst jobs

salary_high=data["Max Salary"]>=190000

(data[salary_high]

.groupby("Location City")

[["Min Salary", "Max Salary"]]

.median()

.sort_values(by="Max Salary", ascending=True)

.head(10)

)

# Q4: Which Company Names have the highest salary for Data Analyst jobs

(data[salary_high]

.groupby("Company Name")

[["Min Salary", "Max Salary"]]

.median()

.sort_values(by="Max Salary", ascending=True)

.head(10)

)



#Q5: Which Type of organization have the maximum salara

type_high=(data[salary_high]

.groupby("Type of ownership")

[["Min Salary", "Max Salary"]]

.median()

.sort_values(by="Max Salary", ascending=True)

.head(10)

)
#Q6: Which Type of organization have the minimum salara

type_low=(data

.groupby("Type of ownership")

[["Min Salary", "Max Salary"]]

.median()

.sort_values(by="Max Salary", ascending=True)

.head(10)

)
high_plot=type_high.hvplot.bar(x="Type of ownership", y="Max Salary", stacked=True, rot=45, width=800, height=500, title="Type of ownership: low or high salary")

low_plot=type_low.hvplot.bar(x="Type of ownership", y="Max Salary", stacked=True, rot=45, width=800, height=500, title="Type of ownership: low or high salary")

high_plot*low_plot
## Q7: Rating

(data[salary_high]

.groupby("Job Title")

[["Rating","Max Salary"]]

.median()

.sort_values(by="Rating", ascending=False)

.head(10)

)



 