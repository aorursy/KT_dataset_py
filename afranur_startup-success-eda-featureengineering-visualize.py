import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("../input/startup-success-prediction/startup data.csv")
df.head()
df.info()

# We have 49 columns and 923 rows, some columns have missing values
# Number of missing values

df.isnull().sum().sort_values(ascending=False).head(10)
# Let's look at percentage of missing values

print("Percentage of missing values in 'closed_at' column: % {:.2f}". format((df.closed_at.isnull().sum())/len(df)*100))
print("Percentage of missing values in 'Unnamed: 6' column: % {:.2f}". format((df["Unnamed: 6"].isnull().sum())/len(df)*100))
print("Percentage of missing values in 'age_last_milestone_year' column  : % {:.2f}". format((df.age_last_milestone_year.isnull().sum())/len(df)*100))
print("percentage of missing values in 'age_first_milestone_year' column : % {:.2f}". format((df.age_first_milestone_year.isnull().sum())/len(df)*100))
#  "Unnamed: 6", "Unnamed: 0", "id", "closed_at" columns are not necessary so drop it
df.drop(["Unnamed: 6"],axis=1, inplace=True)
df.drop(["Unnamed: 0"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)
df.drop(["closed_at"], axis=1, inplace=True)
for index, row in df.iterrows():
    if row['state_code']!=row['state_code.1']:
        print(index, row['state_code'], row['state_code.1'])

# "state_code" column and "state_code.1" column must be the same, so we should drop the "state_code.1" and also, 
# "state_code.1" column has a one missing value in the 515. row. 
df.drop(["state_code.1"], axis=1, inplace=True)
# "status_closed" column is for prediction of startup success and this is binary classification so we should convert numerical variable by using get_dummies( function) in pandas
df=pd.get_dummies(df, columns=["status"], drop_first=True)
# Year columns must be converted to datetime type

df.founded_at=pd.to_datetime(df.founded_at)
df.first_funding_at=pd.to_datetime(df.first_funding_at)
df.last_funding_at=pd.to_datetime(df.last_funding_at)

df.head(3)
# What does it mean "age_first_funding_year", "age_last_funding_year", "age_first_milestone_year", "age_last_milestone_year" , let's together analyze it

plt.figure(figsize=(18,3),dpi=100)

plt.subplot(1,4,1)
sns.scatterplot((df["first_funding_at"].dt.year - df["founded_at"].dt.year), df["age_first_funding_year"])
plt.xlabel("Difference 'Founded' and 'First Funding'")

plt.subplot(1,4,2)
sns.scatterplot((df["last_funding_at"].dt.year- df["founded_at"].dt.year), df["age_last_funding_year"])
plt.xlabel("Difference 'Founded' and 'Last Funding'");

plt.subplot(1,4,3)
sns.scatterplot(df["age_first_funding_year"], df["age_first_milestone_year"])

plt.subplot(1,4,4)
sns.scatterplot(df["age_last_funding_year"], df["age_last_milestone_year"]);


# As we see the graph, we can say high correlation between funding date and age funding. Difference between "last_funding_at" and "founded_at" is related "age_last_funding_year".
# "age_first_funding_year" and "age_last_funding_year" have negative values,it shouldn't be and also it can not be that "founded" date higher than "first_funding_at" and "last_funding_at"
# So we must get the absolute value of columns including negative value
age=["age_first_funding_year","age_last_funding_year","age_first_milestone_year","age_last_milestone_year"]

for a in range(len(age)):
    print("Is there any negative value in '{}' column  : {} ".format(age[a],(df[age[a]]<0).any()))
          
# Which rows have negative values? look at one of them
for index, rows in df.iterrows():
    if rows["age_first_funding_year"]<0:
        print(index, rows["age_first_funding_year"])
# we must get the absolute value of columns including negative value

df["age_first_funding_year"]=np.abs(df["age_first_funding_year"])
df["age_last_funding_year"]=np.abs(df["age_last_funding_year"])
df["age_first_milestone_year"]=np.abs(df["age_first_milestone_year"])
df["age_last_milestone_year"]=np.abs(df["age_last_milestone_year"])
age=["age_first_funding_year","age_last_funding_year","age_first_milestone_year","age_last_milestone_year"]

for a in range(len(age)):
    print("Is there any negative value in '{}' column  : {} ".format(age[a],(df[age[a]]<0).any()))
    
# Now, we get rid of negative values
# After we get the absolute, visualize relationships

plt.figure(figsize=(16,3),dpi=100)

plt.subplot(1,4,1)
sns.scatterplot(np.abs(df["first_funding_at"].dt.year - df["founded_at"].dt.year), df["age_first_funding_year"])
plt.xlabel("Difference 'Founded' and 'First Funding'")

plt.subplot(1,4,2)
sns.scatterplot(np.abs(df["last_funding_at"].dt.year- df["founded_at"].dt.year), df["age_last_funding_year"])
plt.xlabel("Difference 'Founded' and 'Last Funding'");

plt.subplot(1,4,3)
sns.scatterplot(df["age_first_funding_year"], df["age_first_milestone_year"])

plt.subplot(1,4,4)
sns.scatterplot(df["age_last_funding_year"], df["age_last_milestone_year"]);
# I will fill the missing values by mean() function

df["age_first_milestone_year"].fillna((df["age_first_milestone_year"].mean()), inplace=True)
df["age_last_milestone_year"].fillna((df["age_last_milestone_year"].mean()), inplace=True)

# After we get the absolute, visualize relationships

plt.figure(figsize=(16,3),dpi=100)

plt.subplot(1,2,1)
sns.scatterplot(df["age_first_funding_year"], df["age_first_milestone_year"])

plt.subplot(1,2,2)
sns.scatterplot(df["age_last_funding_year"], df["age_last_milestone_year"]);
df.describe()
# To find how much there are outliers in dataset, we should use only continuous variables, because rest of numerical variables are binary variables including 0 and 1

variable=["age_first_funding_year","age_last_funding_year","age_first_milestone_year","age_last_milestone_year"]

plt.figure(figsize=(17,3),dpi=100)
for i in range(len(variable)):
    plt.subplot(1,4,i+1)
    plt.title("{}". format(variable[i]))
    plt.boxplot(df[variable[i]]);
variable=["age_first_funding_year","age_last_funding_year","age_first_milestone_year","age_last_milestone_year"]

plt.figure(figsize=(17,3),dpi=100)
for i in range(len(variable)):
    plt.subplot(1,4,i+1)
    plt.title("{}". format(variable[i]))
    sns.distplot(df[variable[i]], color="orange");
    
# For only one column, analyze that number of outliers

from scipy.stats import zscore

zscores=zscore(df["age_first_funding_year"])

for threshold in range(1,8,1):
    print("Threshold value: {}". format(threshold))
    print("Number of outliers: {}".format(len(np.where(zscores>threshold)[0])))
    print("------------------------")

plt.figure(figsize=(15,7),dpi=100)

df["log_first_fundig"]=np.log(df["age_first_funding_year"]+1)
plt.subplot(2,4,1)
plt.xlabel("log_first_fundig")
plt.boxplot(df["log_first_fundig"])   

plt.subplot(2,4,5)
sns.distplot(df["log_first_fundig"] , color="green");


df["log_last_fundig"]=np.log(df["age_last_funding_year"]+1)
plt.subplot(2,4,2)
plt.xlabel("log_last_fundig")
plt.boxplot(df["log_last_fundig"])   

plt.subplot(2,4,6)
sns.distplot(df["log_last_fundig"], color="green")


df["log_first_milestone"]=np.log(df["age_first_milestone_year"]+1)
plt.subplot(2,4,3)
plt.xlabel("log_first_milestone")
plt.boxplot(df["log_first_milestone"])   

plt.subplot(2,4,7)
sns.distplot(df["log_first_milestone"], color="green")


df["log_last_milestone"]=np.log(df["age_last_milestone_year"]+1)
plt.subplot(2,4,4)
plt.xlabel("log_last_milestone")
plt.boxplot(df["log_first_fundig"])   

plt.subplot(2,4,8)
sns.distplot(df["log_last_milestone"], color="green");
plt.figure(figsize=(16,4),dpi=100)


# "avg_participants"  column has negative value but it shouldn't be, so firstly we should get the absolute of the column
df["avg_participants"]=np.abs(df["avg_participants"])


plt.subplot(1,4,1)
plt.title("Avg Participant Outliers")
plt.boxplot(df["avg_participants"])

plt.subplot(1,4,2)
plt.title("Histogram of Avg Participants")
sns.distplot(df["avg_participants"], color="green")

plt.subplot(1,4,3)
df["log_avg_participants"]=np.log(df["avg_participants"]+1)
plt.title("Logaritmic Avg Participants")
plt.boxplot(np.log(df["log_avg_participants"]))

plt.subplot(1,4,4)
plt.title("Histogram of Logaritmic Avg Participants")
sns.distplot(np.log(df["log_avg_participants"]), color="green");

# After we get the logaritmic of "avg_participant" column, we get rid of the outliers but anyway this column still is not normal distribution
df_state=df.groupby(["state_code"])["funding_total_usd"].sum().sort_values(ascending=False).reset_index().head(12)

plt.figure(figsize=(18,6), dpi=100)
plt.subplot(2,2,1)
plt.ylabel("First 10 state")
plt.xlabel("Total USD of Funding")
sns.barplot(df_state["state_code"],df_state["funding_total_usd"], palette="Greens")


plt.subplot(2,2,2)
df_funding=df.groupby(["state_code"])["funding_rounds"].sum().sort_values(ascending=False).reset_index().head(12)
sns.barplot(df_funding["state_code"], df_funding["funding_rounds"], palette="Greens")

plt.subplot(2,2,3)
sns.countplot(df["state_code"])
plt.xticks(rotation=55);

plt.figure(figsize=(18,4),dpi=100)
plt.xticks(rotation=45)
plt.title("Category Type Counts")
sns.countplot(df["category_code"], edgecolor=sns.color_palette("dark"));
plt.figure(figsize=(18,4),dpi=100)

plt.xticks(rotation=42)
plt.title("According to category top500")
sns.countplot(x=df["category_code"], hue=df["is_top500"], palette="Greens");
plt.figure(figsize=(18,4),dpi=100)

plt.xticks(rotation=42)
plt.title("According to category has_angel")
sns.countplot(x=df["category_code"], hue=df["has_angel"], palette="Greens");
plt.figure(figsize=(18,4),dpi=100)

plt.xticks(rotation=42)
plt.title("According to category status_closed")
sns.countplot(x=df["category_code"], hue=df["status_closed"], palette="Greens")
plt.legend(loc=1);
plt.figure(figsize=(18,4),dpi=100)

plt.xticks(rotation=42)
plt.title("According to category total USD")
sns.barplot(x=df["category_code"], y=df["funding_total_usd"], palette="Greens");
plt.figure(figsize=(18,4),dpi=100)

plt.xticks(rotation=42)
plt.title("According to category avg_participants")
sns.barplot(x=df["category_code"], y=df["avg_participants"], palette="Greens");
plt.figure(figsize=(16,4),dpi=100)

plt.subplot(1,3,1)
sns.barplot(df["is_top500"], df["funding_total_usd"], palette="Greens")

plt.subplot(1,3,2)
sns.barplot(df["has_angel"], df["funding_total_usd"], palette="Greens")

plt.subplot(1,3,3)
sns.countplot(df.milestones, palette="Greens");
plt.figure(figsize=(22,4),dpi=100)


col=["log_first_fundig","log_last_fundig","log_first_milestone","log_last_milestone","log_avg_participants"]


for i in range(len(col)):
    plt.subplot(1,5,i+1)
    sns.barplot(df["funding_rounds"],df[col[i]], palette="Greens");

plt.figure(figsize=(20,3),dpi=100)
plt.subplot(1,3,1)
sns.scatterplot(df["age_first_funding_year"],df["age_last_funding_year"], label="first&last funding", palette="Greens")
sns.scatterplot(df["age_first_milestone_year"], df["age_last_milestone_year"], label="first&last milestone", palette="Blues")
plt.legend()

plt.subplot(1,3,2)
sns.distplot(df["age_first_funding_year"], label="first_funding")
sns.distplot(df["age_last_funding_year"], label="last_funding")
sns.distplot(df["age_first_milestone_year"], label="first_milestone")
sns.distplot(df["age_last_milestone_year"], label="last_milestone")
plt.xlabel("first_funding, last_funding, first_milestone, last_milestone")
plt.legend()


plt.show()
# The most relational columns with target variable(status_closed) are below:

plt.figure(figsize=(4,8),dpi=100)

focus_cols = ['status_closed']
df_corr=df.corr().filter(focus_cols).drop(focus_cols)
sns.heatmap(df_corr, annot=True, fmt='.2f');
from scipy.stats import ttest_ind

# we get the null hypothesis that both groups have equal means.

ttest=ttest_ind(df["has_angel"],df["funding_total_usd"])
print("Is there any differences between means of has_angel and funding_total_usd?")
print("--"*40)
print("t statistic: {:3f} p_value: {:3f}". format(ttest[0],ttest[1]),"\n","\n")


print("Is there any differences between means of is_top500 and funding_total_usd?")
print("--"*40)
ttest2=ttest_ind(df["is_top500"],df["funding_total_usd"])
print("t statistic: {:3f} p_value: {:3f}". format(ttest2[0],ttest2[1]))


# In order to p_value is less than 0.05, we reject the H0 hypothesis so, there is not differences between mean of variables
# Test whether group differences are significant.


ttest_3=ttest_ind(df["funding_rounds"], df["log_first_fundig"])    
print("'funding_rounds' and 'log_first_fundig' t statistic: {:.4f}, p_value: {:.4f}". format(ttest_3[0], ttest_3[1]))
   
    
ttest_4=ttest_ind(df["funding_rounds"], df["log_last_fundig"])    
print("'funding_rounds' and 'log_last_fundig' t statistic: {:.4f}, p_value: {:.4f}". format(ttest_4[0], ttest_4[1]))

ttest_5=ttest_ind(df["funding_rounds"], df["log_first_milestone"])    
print("'funding_rounds' and 'log_first_milestone' t statistic: {:.4f}, p_value: {:.4f}". format(ttest_5[0], ttest_5[1]))

ttest_6=ttest_ind(df["funding_rounds"], df["log_last_milestone"])    
print("'funding_rounds' and 'log_last_milestone' t statistic: {:.4f}, p_value: {:.4f}". format(ttest_6[0], ttest_6[1]))

ttest_7=ttest_ind(df["funding_rounds"], df["log_avg_participants"])    
print("'funding_rounds' and 'log_avg_participants' t statistic: {:.4f}, p_value: {:.4f}". format(ttest_7[0], ttest_7[1]))

# In order to p_value is less than 0.05, rejected H0 hypothesis so, there is not difference between means
column=["log_first_fundig","log_last_fundig","log_first_milestone","log_last_milestone","log_avg_participants"]
colu=["age_first_funding_year","age_last_funding_year","age_first_milestone_year","age_first_milestone_year","avg_participants"]

plt.figure(figsize=(18,3), dpi=100)
for j in range(len(colu)):
    plt.subplot(1,5,j+1)
    sns.distplot(df[colu[j]], color="orange")
    
plt.figure(figsize=(18,3), dpi=100)
for i in range(len(column)):
    plt.subplot(1,5,i+1)
    sns.distplot(df[column[i]], color="green");
# Test it, whether these variables are normal distribution, for this i will use jarque-bera test function 

from scipy.stats import jarque_bera

dist=["log_first_fundig", "log_last_fundig", "log_first_milestone", "log_last_milestone", "log_avg_participants"]
jarq_df=pd.DataFrame(columns=["variable","test statistic","p_value"])


for d in range(len(dist)):
    jarq=jarque_bera(df[dist[d]])
    jarq_df=jarq_df.append({"variable":dist[d],
                   "test statistic":jarq[0],
                   "p_value":jarq[1]}, ignore_index=True)

display(jarq_df)    


# All of the variables are not the normal distribution because of rejected the H0 hypothesis.
# H0 --> have normal dstribution
# HA --> not normal distribution
from sklearn.preprocessing import normalize

df["norm_log_first_funding"]=normalize(np.array(df["log_first_fundig"]).reshape(1,-1)).reshape(-1,1)
df["norm_log_last_funding"]=normalize(np.array(df["log_last_fundig"]).reshape(1,-1)).reshape(-1,1)
df["norm_log_first_milestone"]=normalize(np.array(df["log_first_milestone"]).reshape(1,-1)).reshape(-1,1)
df["norm_log_last_milestone"]=normalize(np.array(df["log_last_milestone"]).reshape(1,-1)).reshape(-1,1)
df["norm_log_avg_participants"]=normalize(np.array(df["log_avg_participants"]).reshape(1,-1)).reshape(-1,1)
column2=["norm_log_first_funding","norm_log_last_funding","norm_log_first_milestone","norm_log_last_milestone","norm_log_avg_participants"]

plt.figure(figsize=(18,3), dpi=100)
for i in range(len(column2)):
    plt.subplot(1,5,i+1)
    sns.distplot(df[column2[i]], color="orange");
    
print("Minimum values is norm_log_first_funding", df["norm_log_first_funding"].min())
print("Maximum values is norm_log_first_funding", df["norm_log_first_funding"].max())   
# Still these columns are not normal distribution
# Let's now, try StandardScaler()

from sklearn.preprocessing import scale

df["scaled_log_first_funding"]=scale(df["log_first_fundig"])
df["scaled_log_last_funding"]=scale(df["log_last_fundig"])
df["scaled_log_first_milestone"]=scale(df["log_first_milestone"])
df["scaled_log_last_milestone"]=scale(df["log_last_milestone"])
df["scaled_log_avg_participants"]=scale(df["log_avg_participants"])
column3=["scaled_log_first_funding","scaled_log_last_funding","scaled_log_first_milestone","scaled_log_last_milestone","scaled_log_avg_participants"]

plt.figure(figsize=(18,3), dpi=100)
for i in range(len(column3)):
    plt.subplot(1,5,i+1)
    sns.distplot(df[column3[i]], color="orange");
from scipy.stats.mstats import winsorize

df["winsorize_first_funding"]=winsorize(df["age_first_funding_year"], (0,0.10))

# For "age_first_funding" column we analyze whether there are normal distribution

column4=["age_first_funding_year","log_first_fundig","winsorize_first_funding","norm_log_first_funding","scaled_log_first_funding"]

plt.figure(figsize=(18,3),dpi=100)

for i in range(len(column4)):
    plt.subplot(1,5,i+1)
    sns.distplot(df[column4[i]], color="orange");    

# None of this columns are not normal distribution but we need to select columns closer to normal distribution and these are logaritmic columns.

# I have decided to continue with only logaritmic columns in this dataset, so i will drop that is created new

df.drop(["winsorize_first_funding","scaled_log_avg_participants","scaled_log_last_milestone","scaled_log_first_milestone","scaled_log_last_funding","scaled_log_first_funding"],
             axis=1, inplace=True)
# Target variable must be in the end

cols = [col for col in df if col != 'status_closed'] + ['status_closed'] 
df=df[cols]

df.head()