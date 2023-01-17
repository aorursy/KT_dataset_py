# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## This will gather necessary libraries for plotting.
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns 
from matplotlib import style 
import datetime as dt
style.use("ggplot")

# It is time to grab the data one by one.
resources = pd.read_csv("../input/Resources.csv", error_bad_lines = False, warn_bad_lines = False, low_memory = False)
schools = pd.read_csv("../input/Schools.csv", error_bad_lines = False, warn_bad_lines = False)
donors = pd.read_csv("../input/Donors.csv",low_memory = False)
donations = pd.read_csv("../input/Donations.csv", error_bad_lines = False, warn_bad_lines = False)
teachers = pd.read_csv("../input/Teachers.csv", error_bad_lines = False, warn_bad_lines = False)
projects = pd.read_csv("../input/Projects.csv", error_bad_lines = False, warn_bad_lines = False)
## lets look at their shapes respectively.
print(" Shape of the resources dataframe is: " , resources.shape)
print(" Shape of the schools dataframe is: ", schools.shape)
print(" Shape of the donors dataframe is: ", donors.shape)
print(" Shape of the donations dataframe is: ", donations.shape)
print(" Shape of the teachers dataframe is: ", teachers.shape)
print(" Shape of the projects dataframe is: ", projects.shape)
# Lets look at he first 5 rows for each dataframe to explore column names, indexes better. 
resources.head()
schools.head()
donors.head()
donations.head()
teachers.head()
projects.head()
# I would like to combine all dataframes in a one big dataframe to start my analysis. Normally you can perform analysis in each dataframe individually
# but this time I will try to do it in one big dataframe. I will look through ERD that I shared at the top.
data = pd.merge(projects, donations, how = "inner", on = "Project ID")

data.shape
data2 = pd.merge(data, donors, how = "inner", on = "Donor ID")
data2.shape
data3 = pd.merge(data2, schools, how = "inner", on = "School ID")
data3.shape
data4 = pd.merge(data3, teachers, how = "inner", on = "Teacher ID")
data4.shape
# It appears that when I want to combine remaining recources dataframe with the 
# data4 (which includes projects, donors, donations, teachers and schools) kernel dies since it creates seven times greater dataframe. So I decide to 
# not add resources for now.I will merge projects and resources later on. It looks like we obtained a nice 4.4 million rows and 34 columns dataframe. 
# I am already excited. Lets dive and see what we can dig from this dataframe.
data4.head(5)
a = data4.columns.values.tolist()  ##to get all column names in our dataframe as a list.
a
# Lets start with simpler questions: Which 10 states have the most number of schools that opened projects to gather donations ? 
# to answer this question we dont need our combined dataframe. WE should answer this question by looking only schools dataframe 
# since in our combined  dataframe we duplicated project ID's for each donation which also caused states to duplicate too.
s = schools["School State"].value_counts().sort_values(ascending = False).head(10)
s
# lets visualize this with a bar plot Since it has different categories ( states )
s.plot.bar()
plt.xlabel("states")
plt.ylabel("number of schools")
plt.title("Number of Schools involved in Projects by State")
plt.tight_layout()
plt.margins(0.05)


# Lets ask a more advanced version of this question :What are the top 10 states in which 
# schools gathered most amount of AVERAGE donations for their projects ? 
# This time we need our combined dataframe !
s2 = data4.groupby("School State")["Donation Amount"].mean().sort_values(ascending = False).head(10)
s2
# Lets visualize states that have more average donations per project than others.
s2.plot.barh()
plt.xlabel("Average Donations Per Project ( in dollars )")
plt.ylabel("States")
plt.axvline(data4.groupby("School State")["Donation Amount"].mean().mean(), color = "blue", linewidth = 2 )
plt.title("Top 10 States that gather donations more than average")
plt.tight_layout()
plt.margins(0.05)

# I want to further investigate the maximum, minimum, mean, median, 25th and 75th percentiles of "Donation Amount" column. What is the average donation
# amount acroos all projects ? What are the minimum and maximums ? 
mean = np.mean(data4["Donation Amount"].dropna())
median = np.median(data4["Donation Amount"].dropna())
percentiles = np.percentile(data4["Donation Amount"].dropna(), [25,75])
minimum = data4["Donation Amount"].dropna().min()
maximum = data4["Donation Amount"].dropna().max()
print("Mean donation amount is: ", np.round(mean,2))
print("Median donation amount is ", median)
print("25th and 75th percentiles are: ", percentiles)
print("Minimum and maximum donation amounts are :", minimum, "    ",maximum)

# I want to plot Empirical Cumulative Distribution Function(ECDF) of "Donation Amount" column.It will better visualize the existence of outliers 
# in the data.This method is pretty nifty for EDA. I recommend you to adapt this as a convention.
# Basically it shows in which percent my data has points greater or smaller than the value shown in the x axis.
# Although it is not clear here we can immediately see that almost 99 percent of our data lies in the range of 0 to 100.
x = np.sort(data4["Donation Amount"].dropna())
y = np.arange(1, len(x)+1) / len(x)
plt.plot(x,y,marker = ".", linestyle = "none")

# Now, I want to know in which states there are more donations done by donors. 
s3 = data4.groupby("Donor State")["Donation ID"].count().sort_values(ascending = False).head(15)
s3
s4 = schools["School State"].value_counts()
s5 = data4.groupby("Donor State")["Donation ID"].count()
df = pd.concat([s4,s5], axis = 1, keys = ["Projects", "Donations"])
df.head(10)

df.loc[:,df.isnull().any()]   ## returns column with any Nan values 
df = df.dropna()
# Now it is time to visualize this data for further insights.
df.plot.scatter(x = "Projects", y = "Donations")
plt.title("Projects vs Donations")
plt.tight_layout()
plt.margins(0.05)

np.corrcoef(df.Projects,df.Donations)   ## It is indeed a pretty strong correlation which is equal to 0.944
## In this part, I will try to explain how we can add data labels in scatter plot with the help of .annotate() method.
df.plot.scatter(x = "Projects", y = "Donations")
plt.title("Projects vs Donations")
for i, j in enumerate(df.index):
    plt.annotate(j,(df.Projects[i], df.Donations[i]))
plt.tight_layout()
plt.margins(0.05)
## to increase figure size : 
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

# Now, I would like to fit a linear model which will basically indicate the relationship between projects and donations.
slope, intercept = np.polyfit(df.Projects, df.Donations, 1)
x = np.array([df.Projects.min(), df.Projects.max()])
y =  slope * x + intercept 
plt.plot(x,y)
## Now, lets combine two graphs in the same plot: 
_ = df.plot.scatter(x = "Projects", y = "Donations")
_ = plt.title("Projects vs Donations")
slope, intercept = np.polyfit(df.Projects, df.Donations, 1)
x = np.array([df.Projects.min(), df.Projects.max()])
y =  slope * x + intercept 
_ = plt.plot(x,y)
_ = plt.tight_layout()
_ = plt.margins(0.05)
# Lets recall our dataframe to seek for new questions :
data4.head(1)
## How mant different project types exists ?  What is the total donation amount for each of them ? 
s6 = data4["Project Type"].value_counts()
s6
s7 = data4.groupby("Project Type")["Donation Amount"].sum().astype(int)
s7
plt.subplot(1,2,1)
plt.pie(s6, autopct = '%1.1f%%',labels = s6.index, startangle = 90)
plt.tight_layout()
plt.margins(0.05)
plt.legend(loc = "upper left")
plt.subplot(1,2,2)
plt.pie(s7, autopct = "%1.1f%%",labels = s7.index, startangle = 90)
plt.legend(loc = "upper left")
plt.tight_layout()
plt.margins(0.05)
fig = plt.gcf()
fig.set_size_inches(15, 5)
## How mant project subject category trees exists ? Which ones attracted the most donations ? 
data4["Project Subject Category Tree"].nunique()


# Which ones attracted the most donations ? 
s8 = data4.groupby("Project Subject Category Tree")["Donation Amount"].sum().sort_values(ascending = False).astype(int).head(15)
s8
## To convert it to million dollars.
s9 = s8 / 1000000
# lets visualize it : 
s9.plot.bar()
plt.xlabel("Project Subject Category")
plt.ylabel("Donation Amount ( million dollars )")
plt.title("Donation Amount by Project Subject")

data4[["Project Posted Date", "Project Fully Funded Date"]].isnull().sum()
## to see how many projects are still open, we may choose to remove that ones.

# to understand their formats. These two columns are initially object types. We will convert them to datetime.
data4[["Project Fully Funded Date", "Project Posted Date"]].head(10)
data4["Project Posted Date"] = pd.to_datetime(data4["Project Posted Date"])
data4["Project Posted Date"].dtype
data4["Project Fully Funded Date"] = pd.to_datetime(data4["Project Fully Funded Date"].dropna())
data4["Project Fully Funded Date"].dtype
data4["Funding Time"] = data4["Project Fully Funded Date"] - data4["Project Posted Date"]
data4[["Funding Time","Project Fully Funded Date","Project Posted Date"]].head()

data4[["Funding Time","Project Fully Funded Date","Project Posted Date"]].isnull().sum()
data5 = data4[pd.notnull(data4["Funding Time"])]   ## to drop NaT values.
data5[["Funding Time","Project Fully Funded Date","Project Posted Date"]].isnull().sum()
import datetime as dt
data5["Funding Time"] = data5["Funding Time"].dt.days 
data5[["Funding Time","Project Fully Funded Date","Project Posted Date"]].head()
mean_time_project_funding = data5.groupby("Project ID")["Funding Time"].mean()
overall_mean_time = mean_time_project_funding.mean()
overall_mean_time
wrong_overall_mean_time = data5["Funding Time"].mean()
wrong_overall_mean_time
states_project_funding_time = data5.groupby(["School State", "Project ID"])["Funding Time"].mean()
states_project_funding_time
states_average_funding_time = states_project_funding_time.groupby("School State").mean()
states_average_funding_time.round(0)   ## to get the exact days I rounded to 0 decimal.
ss = states_average_funding_time.round(0)
ss[ss < 32].sort_values().head(10)
fast_funding_states = ss[ss < 32].sort_values().head(10)
fast_funding_states.plot.bar()
plt.axhline(32, color ="m", linewidth = 2)
plt.ylim(0,40)
plt.xlabel("States")
plt.ylabel("Fully Funding Time  ( in days )")
plt.title("States that fund projects faster than others")


ss[ss > 32].sort_values(ascending = False).head(10)

slow_funding_states = ss[ss > 32].sort_values(ascending = False).head(10)
slow_funding_states.plot.bar()
plt.axhline(32, color = "m", linewidth = 2)
plt.ylim(0,40)
plt.xlabel("States")
plt.ylabel("Fully Funding Time  ( in days )")
plt.title("States that fund projects slower than others")
