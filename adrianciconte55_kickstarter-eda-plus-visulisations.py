import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
sns.set_style("white")
from collections import Counter
file = "../input/ks-projects-201801.csv"
df = pd.read_csv(file)
fontsize = 20
#Create Missing Values function to help determine what columns to keep and drop
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum().div(df.isnull().count()).sort_values(ascending=False))
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
missing_data(df)
df.drop(['usd pledged', 'usd_pledged_real', 'usd_goal_real'], axis=1, inplace=True)
df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
df['Days_Remaining'] = df['deadline'].sub(df['launched'])

df.rename(columns={'name':'Name', 'category':'Category', 'main_category':'Parent_Category', 'currency':'Currency',
                   'deadline':'Deadline', 'goal':'Goal', 'launched':'Launched', 'pledged':'Pledged', 'state':'Outcome',
                   'backers':'Backers', 'country':'Country'}, inplace=True)

df['Outcome'] = df["Outcome"].map({'failed':'Failed', 
                 'canceled':'Cancelled', 
                 'successful':'Sucessful', 
                 'live':'Live', 
                 'undefined':'Undefined', 
                 'suspended':'Suspended'})

#Time and Date Transformation
df['Month_No'] = df['Launched'].dt.month
df["Month_No"] = df["Month_No"].astype(int)
df["Month"] = [calendar.month_name[i] for i in df["Month_No"]]
df['Year'] = df['Launched'].dt.year
df["Year"] = df["Year"].astype(str)
df["Day_Name"] = df["Launched"].dt.weekday_name
df["Day_No"] = df['Launched'].dt.weekday

df = df[df['Year'] != '1970']
df = df.loc[(df["Country"] != 'N,0"')]
df.head()
ax1 = plt.subplot(121)
df["Parent_Category"].value_counts()[:50].plot(kind='bar', width=0.8, grid=True, colormap='tab20c', figsize=(22,7), ax=ax1)
plt.title('Kickstarter Count by Parent_Category', fontsize=fontsize)
plt.xlabel('Category', fontsize=fontsize)
plt.ylabel('Count by Category', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax2 = plt.subplot(122)
df["Category"].value_counts()[:50].plot(kind='bar', width=0.8, grid=True, colormap='tab20c', figsize=(22,7), ax=ax2)
plt.title('Kickstarter Count by Category', fontsize=fontsize)
plt.xlabel('Category', fontsize=fontsize)
plt.ylabel('Count by Category', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

df.groupby(["Day_Name", "Day_No"]
          )['Category'].count().reset_index(1).sort_values(by='Day_No').drop('Day_No',axis=1).plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Kickstarter Count by Day', fontsize=fontsize)
ax1.set_xlabel('Day', fontsize=fontsize)
ax1.set_ylabel('Count by Day', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Month", "Month_No"]
          )['Category'].count().reset_index(1).sort_values(by='Month_No').drop('Month_No',axis=1).plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Kickstarter Count by Month', fontsize=fontsize)
ax2.set_xlabel('Day', fontsize=fontsize)
ax2.set_ylabel('Count by Month', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)

df.groupby(['Year'])['Category'].count().plot(kind='bar', width=0.8, grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Kickstarter Count by Year', fontsize=fontsize)
ax3.set_xlabel('Day', fontsize=fontsize)
ax3.set_ylabel('Count by Year', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
df.pivot_table(index=['Country'], columns=["Outcome"], 
               values=["Backers"], aggfunc=(np.mean)).plot(kind='bar', width=0.8, grid=True, 
                                                           colormap='tab20c', figsize=(22,7))
plt.title('Outcome by Country', fontsize=fontsize)
plt.xlabel('Country', fontsize=fontsize)
plt.ylabel('Average Outcome by Country', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

df.groupby(["Parent_Category"])["Pledged"].agg('sum').sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Pledge Amount by Parent_Category', fontsize=fontsize)
ax1.set_xlabel('Parent_Category', fontsize=fontsize)
ax1.set_ylabel('Pledge Amount', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Category"])["Pledged"].agg('sum').sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Pledge Amount by Category', fontsize=fontsize)
ax2.set_xlabel('Category', fontsize=fontsize)
ax2.set_ylabel('Pledge Amount', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Country"])["Pledged"].agg('sum').sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Pledge Amount by Country', fontsize=fontsize)
ax3.set_xlabel('Country', fontsize=fontsize)
ax3.set_ylabel('Pledge Amount', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

df.groupby(["Parent_Category"])["Backers"].agg('sum').sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Backers by Parent_Category', fontsize=fontsize)
ax1.set_xlabel('Parent_Category', fontsize=fontsize)
ax1.set_ylabel('Backers', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Category"])["Backers"].agg('sum').sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Backers by Category', fontsize=fontsize)
ax2.set_xlabel('Category', fontsize=fontsize)
ax2.set_ylabel('Backers', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Country"])["Backers"].agg('sum').sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Backers by Country', fontsize=fontsize)
ax3.set_xlabel('Country', fontsize=fontsize)
ax3.set_ylabel('Backers', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
def success_rate(column_1):
    outcome_count = df.groupby([column_1, "Outcome"])["Outcome"].agg('count').to_frame().unstack()
    success_count = outcome_count.xs('Outcome', axis=1, drop_level=True)["Sucessful"]
    success_df = df.groupby([column_1])["Outcome"].agg('count').to_frame()
    success_df["Success_Count"] = success_count
    success_df["Success_Rate"] = success_df["Success_Count"].div(success_df["Outcome"])
    return success_df

fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

success_rate("Parent_Category")["Success_Rate"][:15].sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Success Rate by Parent_Category', fontsize=fontsize)
ax1.set_xlabel('Parent_Category', fontsize=fontsize)
ax1.set_ylabel('Success Rate', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)
ax1.axhline(success_rate("Parent_Category")["Success_Rate"].mean(), color='k', linestyle='--', linewidth=3, alpha=0.5)

success_rate("Category")["Success_Rate"][:15].sort_values(ascending=False)[:30].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Success Rate by Category', fontsize=fontsize)
ax2.set_xlabel('Category', fontsize=fontsize)
ax2.set_ylabel('Success Rate', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)
ax2.axhline(success_rate("Category")["Success_Rate"].mean(), color='k', linestyle='--', linewidth=3, alpha=0.5)

success_rate("Country")["Success_Rate"][:15].sort_values(ascending=False)[:30].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Success Rate by Country', fontsize=fontsize)
ax3.set_xlabel('Country', fontsize=fontsize)
ax3.set_ylabel('Success Rate', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)
ax3.axhline(success_rate("Country")["Success_Rate"].mean(), color='k', linestyle='--', linewidth=3, alpha=0.5)

plt.tight_layout()
plt.show()
def average_backer_value (cat_1):
    abv_df = df.groupby([cat_1])["Pledged", "Backers"].agg('sum')
    abv_df["Ave_Backer_Value"] = abv_df["Pledged"].div(abv_df["Backers"])
    return abv_df

fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

average_backer_value("Parent_Category")['Ave_Backer_Value'].sort_values(ascending=False)[:25].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Ave_Backer_Value by Parent_Category', fontsize=fontsize)
ax1.set_xlabel('Parent_Category', fontsize=fontsize)
ax1.set_ylabel('Ave_Backer_Investment', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)
ax1.axhline(average_backer_value("Parent_Category")['Ave_Backer_Value'].mean(), color='k', linestyle='--', linewidth=3, alpha=0.5)

average_backer_value("Category")['Ave_Backer_Value'].sort_values(ascending=False)[:25].plot(kind='bar', 
                                width=0.8, grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Ave_Backer_Value by Category', fontsize=fontsize)
ax2.set_xlabel('Category', fontsize=fontsize)
ax2.set_ylabel('Ave_Backer_Investment', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)
ax2.axhline(average_backer_value("Category")['Ave_Backer_Value'].mean(), color='k', linestyle='--', linewidth=3, alpha=0.5)

average_backer_value("Country")['Ave_Backer_Value'].sort_values(ascending=False)[:25].plot(kind='bar', 
                                width=0.8, grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Ave_Backer_Value by Country', fontsize=fontsize)
ax3.set_xlabel('Country', fontsize=fontsize)
ax3.set_ylabel('Ave_Backer_Investment', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)
ax3.axhline(average_backer_value("Country")['Ave_Backer_Value'].mean(), color='k', linestyle='--', linewidth=3, alpha=0.5)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(121)
df.groupby("Year", as_index=True)["Pledged"].sum().plot(kind='bar', 
                                                width=0.8, grid=True, colormap='tab20c', figsize=(22,7), ax=ax1)

plt.title('Pledges by Year', fontsize=fontsize)
plt.xlabel('Month', fontsize=fontsize)
plt.ylabel('Pledge', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax2 = plt.subplot(122)
df.groupby("Year", as_index=True)["Backers"].sum().plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', figsize=(22,7), ax=ax2)
plt.title('Backers by Year', fontsize=fontsize)
plt.xlabel('Month', fontsize=fontsize)
plt.ylabel('Backers', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
df[["Name", "Pledged"]].set_index("Name").sort_values(by="Pledged", ascending=False)[:50].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', figsize=(20,14))
plt.title('Pledge by Name', fontsize=fontsize)
plt.xlabel('Name', fontsize=fontsize)
plt.ylabel('Pledge', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
df[["Name", "Backers"]].set_index("Name").sort_values(by="Backers", ascending=False)[:50].plot(kind='bar', 
                                                     width=0.8, grid=True, colormap='tab20c', figsize=(20,14))


plt.title('Backers by Name', fontsize=fontsize)
plt.xlabel('Name', fontsize=fontsize)
plt.ylabel('Backers', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
ax1 = plt.subplot(121)
df.groupby(["Year", "Parent_Category"])["Backers"].sum().unstack().plot(kind='bar', 
                                                        figsize=(22,7), stacked=True, width=0.8, colormap='tab20c', ax=ax1)

plt.title('Parent_Category by Year ["Backers"]', fontsize=fontsize)
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel('Backers', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax2 = plt.subplot(122)
df.groupby(["Year", "Parent_Category"])["Pledged"].sum().unstack().plot(kind='bar', 
                                                        figsize=(22,7), stacked=True, width=0.8, colormap='tab20c', ax=ax2)

plt.title('Parent_Category by Year ["Pledged"]', fontsize=fontsize)
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel('Pledged', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
fig, (ax1, ax2) = plt.subplots(sharey=False, nrows=1, ncols=2, figsize=(22,7))

df.groupby(["Day_Name", "Day_No"]
          )['Pledged'].sum().reset_index(1).sort_values(by='Day_No').drop('Day_No',axis=1).plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Pledged by Day', fontsize=fontsize)
ax1.set_xlabel('Day', fontsize=fontsize)
ax1.set_ylabel('Pledged', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Month", "Month_No"]
          )['Pledged'].sum().reset_index(1).sort_values(by='Month_No').drop('Month_No',axis=1).plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Pledged by Month', fontsize=fontsize)
ax2.set_xlabel('Day', fontsize=fontsize)
ax2.set_ylabel('Pledged', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
fig, (ax1, ax2) = plt.subplots(sharey=False, nrows=1, ncols=2, figsize=(22,7))

df.groupby(["Day_Name", "Day_No"]
          )['Backers'].sum().reset_index(1).sort_values(by='Day_No').drop('Day_No',axis=1).plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Backers by Day', fontsize=fontsize)
ax1.set_xlabel('Day', fontsize=fontsize)
ax1.set_ylabel('Backers', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

df.groupby(["Month", "Month_No"]
          )['Backers'].sum().reset_index(1).sort_values(by='Month_No').drop('Month_No',axis=1).plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Backers by Month', fontsize=fontsize)
ax2.set_xlabel('Day', fontsize=fontsize)
ax2.set_ylabel('Backers', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
def success_rate_1(column_name):
    test = df.groupby([column_name])["Goal", "Pledged"].sum()
    test["Success_Rate"] = test["Pledged"].div(test["Goal"])*100
    test = test.sort_values(by="Success_Rate", ascending=False)
    return test

fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

success_rate_1("Parent_Category")["Success_Rate"].plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Pledge Vs Goal Percent', fontsize=fontsize)
ax1.set_xlabel('Parent_Category', fontsize=fontsize)
ax1.set_ylabel('Percent', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

success_rate_1("Category")["Success_Rate"][:25].plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Pledge Vs Goal Percent', fontsize=fontsize)
ax2.set_xlabel('Category', fontsize=fontsize)
ax2.set_ylabel('Percent', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)
ax2.axhline(100, color='k', linestyle='--', linewidth=3, alpha=0.5)

success_rate_1("Country")["Success_Rate"].plot(kind='bar', width=0.8, 
                                                                    grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Pledge Vs Goal Percent', fontsize=fontsize)
ax3.set_xlabel('Country', fontsize=fontsize)
ax3.set_ylabel('Percent', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
pledged_df = df.groupby(["Outcome", "Parent_Category"])["Pledged"].sum().unstack(1).apply(np.log)
backers_df = df.groupby(["Outcome", "Parent_Category"])["Backers"].sum().unstack(1).apply(np.log)

boxprops = dict(linestyle='-', linewidth=4, color='k')
medianprops = dict(linestyle='-', linewidth=4, color='k')
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

ax1 = plt.subplot(121)
pledged_df.plot(kind='box', figsize=(22,7), grid=True, boxprops=boxprops, 
                medianprops=medianprops, color=color, sym='r+', ax=ax1)
plt.title('Boxplot By Parent_Category ["Pledged": Log]', fontsize=fontsize)
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel('Pledged', fontsize=fontsize)
plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks(fontsize=fontsize)

ax2 = plt.subplot(122)
backers_df.plot(kind='box', figsize=(22,7), grid=True, boxprops=boxprops, 
                medianprops=medianprops, color=color, sym='r+', ax=ax2)
plt.title('Boxplot By Parent_Category ["Backers": Log]', fontsize=fontsize)
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel('Backers', fontsize=fontsize)
plt.xticks(fontsize=fontsize, rotation=90)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
df.groupby(["Parent_Category", "Outcome"])["Backers"].count().unstack()["Sucessful"].sort_values(ascending=False)
fig, (ax1, ax2, ax3) = plt.subplots(sharey=False, nrows=1, ncols=3, figsize=(22,7))

df[(df["Parent_Category"] == "Music") & 
   (df["Outcome"] == "Sucessful")].groupby(["Category"])["Pledged"].sum().sort_values(
    ascending=False).plot(kind='bar', width=0.8, grid=True, colormap='tab20c', ax=ax1)
ax1.set_title('Pledged by Sub Category [Music]', fontsize=fontsize)
ax1.set_xlabel('Category', fontsize=fontsize)
ax1.set_ylabel('Pledged', fontsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax1.tick_params(axis='x', labelsize=fontsize)

df[(df["Parent_Category"] == "Film & Video") & 
   (df["Outcome"] == "Sucessful")].groupby(["Category"])["Pledged"].sum().sort_values(
    ascending=False).plot(kind='bar', width=0.8, grid=True, colormap='tab20c', ax=ax2)
ax2.set_title('Pledged by Sub Category [Film & Video]', fontsize=fontsize)
ax2.set_xlabel('Category', fontsize=fontsize)
ax2.set_ylabel('Pledged', fontsize=fontsize)
ax2.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax2.tick_params(axis='x', labelsize=fontsize)

df[(df["Parent_Category"] == "Games") & 
   (df["Outcome"] == "Sucessful")].groupby(["Category"])["Pledged"].sum().sort_values(
    ascending=False).plot(kind='bar', width=0.8, grid=True, colormap='tab20c', ax=ax3)
ax3.set_title('Pledged by Sub Category [Games]', fontsize=fontsize)
ax3.set_xlabel('Category', fontsize=fontsize)
ax3.set_ylabel('Pledged', fontsize=fontsize)
ax3.tick_params(axis='y', labelsize=fontsize, rotation='auto')
ax3.tick_params(axis='x', labelsize=fontsize)

plt.tight_layout()
plt.show()
