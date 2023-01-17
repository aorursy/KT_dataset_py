import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings("ignore")



from scipy import stats

from scipy.stats import chi2_contingency 



import statsmodels.api as sm

from statsmodels.formula.api import ols
df=pd.read_csv("../input/311-service-requests-nyc/311_Service_Requests_from_2010_to_Present.csv")

df.head()
df.describe()
df.shape
# Converting the data into datetime format

df["Created Date"]=pd.to_datetime(df["Created Date"])

df["Closed Date"]=pd.to_datetime(df["Closed Date"])
#Creating the new column that consist the amount of time taken to resolve the complaint

df["Request_Closing_Time"]=(df["Closed Date"]-df["Created Date"])



Request_Closing_Time=[]

for x in (df["Closed Date"]-df["Created Date"]):

    close=x.total_seconds()/60

    Request_Closing_Time.append(close)

    

df["Request_Closing_Time"]=Request_Closing_Time
df["Agency"].unique()
#Univariate Distribution Plot for Request Closing Time

sns.distplot(df["Request_Closing_Time"])

plt.show
print("Total Number of Concerns : ",len(df),"\n")

print("Percentage of Requests took less than 100 hour to get solved   : ",round((len(df)-(df["Request_Closing_Time"]>100).sum())/len(df)*100,2),"%")

print("Percentage of Requests took less than 1000 hour to get solved : ",round((len(df)-(df["Request_Closing_Time"]>1000).sum())/len(df)*100,2),"%")
#Univariate Distribution Plot for Request Closing Time

sns.distplot(df["Request_Closing_Time"])

plt.xlim((0,5000))

plt.ylim((0,0.0003))

plt.show()
# Count plot to understand the type of the complaint raised

df['Complaint Type'].value_counts()[:10].plot(kind='barh',alpha=0.6,figsize=(15,10))

plt.show()
#Categorical Scatter Plot to understand which type of complaints are taking more time to get resolved

g=sns.catplot(x='Complaint Type', y="Request_Closing_Time",data=df)

g.fig.set_figwidth(15)

g.fig.set_figheight(7)

plt.xticks(rotation=90)

plt.ylim((0,5000))

plt.show()
# Count plot to know the status of the requests

df['Status'].value_counts().plot(kind='bar',alpha=0.6,figsize=(15,7))

plt.show()
#Count Plot for Coloumn Borough

plt.figure(figsize=(12,7))

df['Borough'].value_counts().plot(kind='bar',alpha=0.7)

plt.show()
#Percentage of cases in each Borough

for x in df["Borough"].unique():

    print("Percentage of Request from ",x," Division : ",round((df["Borough"]==x).sum()/len(df)*100,2))
#Unique Location Types

df["Location Type"].unique()
#Request Closing Time for all location Type sorted in ascending Order

pd.DataFrame(df.groupby("Location Type")["Request_Closing_Time"].mean()).sort_values("Request_Closing_Time")
#Request Closing Time for all City sorted in ascending Order

pd.DataFrame(df.groupby("City")["Request_Closing_Time"].mean()).sort_values("Request_Closing_Time")
#Percentage Of Missing Value

pd.DataFrame((df.isnull().sum()/df.shape[0]*100)).sort_values(0,ascending=False)[:20]
#Remove the column with very high percentage of missing value

new_df=df.loc[:,(df.isnull().sum()/df.shape[0]*100)<=50]
print("Old DataFrame Shape :",df.shape)

print("New DataFrame Shape : ",new_df.shape)
rem=[]

for x in new_df.columns.tolist():

    if new_df[x].nunique()<=3:

        print(x+ " "*10+" : ",new_df[x].unique())

        rem.append(x)
new_df.drop(rem,axis=1,inplace=True)
new_df.shape
#Remove columns that are not needed for our analysis

rem1=["Unique Key","Incident Address","Descriptor","Street Name","Cross Street 1","Cross Street 2","Due Date","Resolution Description","Resolution Action Updated Date","Community Board","X Coordinate (State Plane)","Y Coordinate (State Plane)","Park Borough","Latitude","Longitude","Location"]



new_df.drop(rem1,axis=1,inplace=True)
new_df.head()
g=sns.catplot(x="Complaint Type",y="Request_Closing_Time",kind="box",data=new_df)

g.fig.set_figheight(8)

g.fig.set_figwidth(15)

plt.xticks(rotation=90)

plt.ylim((0,2000))
anova_df=pd.DataFrame()

anova_df["Request_Closing_Time"]=new_df["Request_Closing_Time"]

anova_df["Complaint"]=new_df["Complaint Type"]



anova_df.dropna(inplace=True)

anova_df.head()
lm=ols("Request_Closing_Time~Complaint",data=anova_df).fit()

table=sm.stats.anova_lm(lm)

table
chi_sq=pd.DataFrame()

chi_sq["Location Type"]=new_df["Location Type"]

chi_sq["Complaint Type"]=new_df["Complaint Type"]



chi_sq.dropna(inplace=True)
data_crosstab = pd.crosstab( chi_sq["Location Type"],chi_sq["Complaint Type"])
stat, p, dof, expected = chi2_contingency(data_crosstab) 



alpha = 0.05

if p <= alpha: 

    print('Dependent (reject H0)') 

else: 

    print('Independent (H0 holds true)') 