import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from wordcloud import WordCloud, STOPWORDS



%matplotlib inline



df = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv",engine="python",encoding="UTF-8")



text = " ".join(review for review in df["Job Description"])

stopwords = set(STOPWORDS)

stopwords.update(["including","understand","must","use"])

wordcloud = WordCloud(stopwords=stopwords, max_font_size=100, max_words=50, background_color="black", min_font_size=8, width=1300, height=500).generate(text)



sns.set_context(context="paper")

plt.figure(figsize=[14,4],  dpi=150)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
# Remove observations having -1 values 

df = df[df["Revenue"]!="-1"]

df = df[df["Sector"]!="-1"]

df = df[df["Rating"]!=-1]

df = df[df["Founded"]!=-1]



# Replace Unknown  values by mode

df["Size"]=df["Size"].replace("Unknown",df["Size"].mode()[0])

df["Salary Estimate"]=df["Salary Estimate"].replace("-1",df["Salary Estimate"].mode()[0])

df['Easy Apply'] = df['Easy Apply'].str.replace('-1', 'Unknown') # Replace the -1 value to Unknown



# Split the  Salary Range

df['Salary Estimate'] = df['Salary Estimate'].map(lambda x: x.rstrip('(Glassdoor est.)'))

df['Salary Estimate'] = df['Salary Estimate'].str.replace('K', '')

df['Salary Estimate'] = df['Salary Estimate'].str.replace('$', '')

df['Min_salary_range'] = df['Salary Estimate'].map(lambda x:x.split("-")[0])

df['Max_salary_range'] = df['Salary Estimate'].map(lambda x:x.split("-")[1])



# Convert the datatype to numeric

df['Max_salary_range']=pd.to_numeric(df['Max_salary_range'])

df['Min_salary_range']=pd.to_numeric(df['Min_salary_range'])



#Get Location by country

df['JobLocation_Country'] = df['Location'].map(lambda x:x.split(", ")[1])

df['Headquarters_Country'] = df['Headquarters'].map(lambda x:x.split(", ")[1])



#Drop columns such as Salary Estimate,Unnamed: 0,Job Description

df.drop(["Salary Estimate", "Unnamed: 0","Job Description"], axis='columns', inplace=True)



#Remove unwanted string values

df['Revenue'] = df['Revenue'].str.replace('\(USD\)', ' ')

df['Revenue'] = df['Revenue'].str.strip()

df['Size'] = df['Size'].str.replace('employees', ' ')

df['Size'] = df['Size'].str.strip()



df.head()

# sns.pairplot(df, hue="JobLocation_Country")
# Create bins for Rating values

def change(x):

    if x < 0.5:

        return 0

    elif x > 0.5 and x <= 1.5:

        return 1

    elif x > 1.5 and x <= 2.5:

        return 2

    elif x > 2.5 and x <= 3.5:

        return 3

    elif x > 3.5 and x <= 4.5:

        return 4

    else:

        return 5

    

df['Rating'] = list(map(change,df['Rating']))
df.Size.nunique()

df["Size"]=df.Size.astype('category')

df["Size_level"] = df["Size"].cat.codes



df["Type of ownership"]=df["Type of ownership"].astype('category')

df["Ownership_level"] = df["Type of ownership"].cat.codes



df_CorrMatrix= df.corr()

fig, axes = plt.subplots(figsize=(12,4))

sns.set_context(context="paper", font_scale=2)

sns.heatmap(df_CorrMatrix, linecolor="white", linewidth=1)

axes.set_title("Correlation Between Features")
fig, axes = plt.subplots(figsize=(12,4))

axes.set_xticklabels([i for i in df.Size], rotation=20, ha="right")  

sns.set_context(context="paper", font_scale=2)

sns.set_style(style="whitegrid")

sns.barplot(x=df["Size"], y=df["Founded"])

                                     

axes.set_xlabel("Number of employees")

axes.set_ylabel("Foundation Year")

axes.set_title("Foundation year and number of employees")

plt.ylim(1900,2020)
by_JobLocation_Country = df.groupby("JobLocation_Country")

JobLocation_Country_count = by_JobLocation_Country.count()

JobLocation_Country_mean = by_JobLocation_Country.mean()



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,9), squeeze=False)

sns.set_style(style="darkgrid")

sns.set_context(context="paper", font_scale=2)

axes[0][0].set_xticklabels([])

axes[1][0].set_xticklabels([i for i in JobLocation_Country_mean.index], rotation=90, ha="right")

sns.lineplot(JobLocation_Country_mean.index,JobLocation_Country_mean.Min_salary_range, color ="#fa1302", linewidth=3, alpha=2, marker = "o", markersize=10, markerfacecolor="green", 

         markeredgewidth=2, markeredgecolor="yellow", label="Min_salary_range", ax=axes[0,0])

sns.lineplot(JobLocation_Country_mean.index,JobLocation_Country_mean.Max_salary_range, color ="#15217d", linewidth=3, alpha=2, marker = "o", markersize=10, markerfacecolor="green", 

         markeredgewidth=2, markeredgecolor="yellow", label="Max_salary_range", ax=axes[0,0])



sns.barplot(JobLocation_Country_count.index, JobLocation_Country_count["Company Name"], ax=axes[1,0])

axes[0][0].set_xlabel("")

axes[0][0].set_ylabel("Salary Range($1000 USD)")

axes[1][0].set_ylabel("Number of Companies")

axes[0][0].set_title("Changes in Salary range over Countries")

axes[1][0].set_title("No.of Companies in the same Countries")

axes[0][0].legend(loc = (0.7,0.75))

plt.tight_layout()
by_Sector = df.groupby("Sector")

Sector_Mean = by_Sector.mean()



fig, axes = plt.subplots(figsize=(20,4))

sns.set_style(style="darkgrid")

sns.set_context(context="paper", font_scale=2)



axes.set_xticklabels([i for i in Sector_Mean.index], rotation=70, ha="right")



sns.lineplot(Sector_Mean.index,Sector_Mean.Min_salary_range, color ="#fa1302", linewidth=3, alpha=2, marker = "o", markersize=10, markerfacecolor="green", 

         markeredgewidth=2, markeredgecolor="yellow", label="Min_salary_range")

sns.lineplot(Sector_Mean.index,Sector_Mean.Max_salary_range, color ="#15217d", linewidth=3, alpha=2, marker = "o", markersize=10, markerfacecolor="green", 

         markeredgewidth=2, markeredgecolor="yellow", label="Max_salary_range")



axes.set_xlabel("Sectors")

axes.set_ylabel("Salary Range($1000 USD))")

axes.set_title("Sectorwise Distribution of Salary")

axes.legend(loc = (0.5,0.72))
by_Sector = df.groupby("Sector")

Sector_Count = by_Sector.count()



fig, axes = plt.subplots(figsize=(20,4))

sns.set_style(style="darkgrid")

sns.set_context(context="paper", font_scale=2)



axes.set_xticklabels([i for i in Sector_Count.index], rotation=70, ha="right")



sns.barplot(Sector_Count.index, Sector_Count["Company Name"])

axes.set_xlabel("Sectors")

axes.set_ylabel("Count of Companies")

axes.set_title("Number of companies in each sector")

by_Revenue = df.groupby("Revenue")

Revenue_Mean = by_Revenue.mean().sort_values(by="Rating") 

Revenue_Count = by_Revenue.count().sort_values(by="Company Name") 

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,8), squeeze=False)



sns.set_style(style="whitegrid")

sns.set_context(font_scale=2)



color= {'Less than $1 million':'#c497a0','$1 to $5 million':'#db8495','$5 to $10 million':'#db5e77',

        '$10 to $25 million':"#de3c5c",'$25 to $50 million':'#d61e42','$50 to $100 million':'#e30b36',

        '$100 to $500 million':'#fc0334','$500 million to $1 billion':'#8ebfb5', '$1 to $2 billion':'#71bdad',

        '$2 to $5 billion':'#32ad93','$5 to $10 billion':'#15d1a9','$10+ billion':'#05fac6',

        'Unknown / Non-Applicable':'#181c1b'

       }



axes[0][0].set_xticklabels([i for i in Revenue_Mean.index], rotation=70, ha="right")

axes[0][1].set_xticklabels([i for i in Revenue_Mean.index], rotation=70, ha="right")

sns.barplot(x=Revenue_Mean.index, y=Revenue_Mean["Rating"], palette=color, ax=axes[0,0])

sns.barplot(x=Revenue_Count.index, y=Revenue_Count["Company Name"], palette=color, ax=axes[0,1])



axes[0][0].set_xlabel("Revenue")

axes[0][0].set_ylabel("Rating")

axes[0][0].set_title("Impact of Revenue on Ratings")

axes[0][0].set_ylim(3,4.25)



axes[0][1].set_xlabel("Revenue")

axes[0][1].set_ylabel("Count of Companies")

axes[0][1].set_title("Distribution of companies over Revenue")

axes[0][1].set_ylim(5,240)



plt.tight_layout()
by_JobTitle = df.groupby("Job Title")



by_JobTitle = df.groupby("Job Title")

JobTitle_mean = by_JobTitle.mean()



JobTitle_maxSort=JobTitle_mean.sort_values(by=['Max_salary_range'], ascending=False)

JobTitle_minSort=JobTitle_mean.sort_values(by=['Min_salary_range'])



JobTitle_maxSort_top20 = JobTitle_maxSort.iloc[0:20]

JobTitle_minSort_bottom20 = JobTitle_minSort.iloc[0:20]
JobTitle_maxSort_top20=JobTitle_maxSort_top20.reset_index()

JobTitle_maxSort_top20["Job Title"]=JobTitle_maxSort_top20["Job Title"].astype('category')

JobTitle_maxSort_top20["Level_JobTitle"] = JobTitle_maxSort_top20["Job Title"].cat.codes



Mapping = dict(enumerate(JobTitle_maxSort_top20["Job Title"].cat.categories))

fig, axes = plt.subplots(figsize=(12,5))

sns.set_style(style="darkgrid")

sns.set_context( font_scale=0.5)

sns.barplot(x=JobTitle_maxSort_top20.Level_JobTitle, y=JobTitle_maxSort_top20["Max_salary_range"])

axes.set_title("Top 20 Job titles gaining Maximum Salary ")

axes.set_xlabel("Jot Title Level")

axes.set_ylabel("Salary($1000 USD)")

axes.set_ylim(140,200)

plt.tight_layout()

print(Mapping)

JobTitle_minSort_bottom20=JobTitle_minSort_bottom20.reset_index()



JobTitle_minSort_bottom20["Job Title"]=JobTitle_minSort_bottom20["Job Title"].astype('category')

JobTitle_minSort_bottom20["Level_JobTitle"] = JobTitle_minSort_bottom20["Job Title"].cat.codes



Mapping = dict(enumerate(JobTitle_minSort_bottom20["Job Title"].cat.categories))



fig, axes = plt.subplots(figsize=(12,5))

sns.set_style(style="darkgrid")

sns.set_context( font_scale=0.5)

# axes.set_xticklabels([i for i in JobTitle_minSort_bottom20.Level_JobTitle], rotation=70, ha="right")

sns.barplot(x=JobTitle_minSort_bottom20.Level_JobTitle, y=JobTitle_minSort_bottom20["Min_salary_range"])



axes.set_xlabel("Jot Title Level")

axes.set_ylabel("Salary($1000 USD)")

axes.set_title("Bottom 20 Job title with Minimum Salary")

axes.set_ylim(22,28)

plt.tight_layout()

print(Mapping)