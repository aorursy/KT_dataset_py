import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style("whitegrid")
sns.set(font_scale=1.3)

multi = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv", header=1)
countries = pd.read_csv("../input/countries-classification-by-income/country_classification.tsv",
                       sep="\t")
# Add country group: developped or developping
countries["Group"] = countries["Income group"].apply(lambda x: "Developped countries" if x=="High income" else "Developing countries")

replacements = {'Egypt, Arab Rep.':'Egypt','Hong Kong SAR, China':'Hong Kong (S.A.R.)',
'Iran, Islamic Rep.':'Iran, Islamic Republic of...',"Korea, Dem. People's Rep.":'Republic of Korea',
'Russian Federation':'Russia','Korea, Rep.':'South Korea', 'United Kingdom':'United Kingdom of Great Britain and Northern Ireland',
'United States':'United States of America','Vietnam':'Viet Nam'}
countries['Economy'].replace(replacements, inplace=True)

# Add Group, Income Group and Region to multi choice question
#First rename column
multi.rename(columns={"In which country do you currently reside?": "Country"}, inplace=True)
for k in ["Group", "Income group", "Region"]:
    group_dict = dict(zip(countries.Economy, countries[k]))
    group_dict['I do not wish to disclose my location'] = "Undisclosed location"
    group_dict['Other'] = "Other"
    multi[k] = multi["Country"].apply(lambda x: group_dict[x])
#regions = multi["Region"].unique().tolist()
regions = ['North America',
 'East Asia & Pacific',
 'South Asia',
 'Latin America & Caribbean',
 'Europe & Central Asia',
 'Sub-Saharan Africa',
 'Middle East & North Africa']
income_group = multi["Income group"].unique().tolist()
ax = sns.countplot(y="Region", data=multi, order=multi["Region"].value_counts().index,
                  color="darkslateblue")
ax.set_xlabel("Respondents", labelpad=22)
ax.set_ylabel("Region", labelpad=20)
ax.set_title("What region are you from?", pad=22)
plt.show()
multi.rename(columns={"What is your gender? - Selected Choice": "Gender"}, inplace=True)

ax = sns.countplot(x="Region", data=multi, hue="Gender", order=multi["Region"].value_counts().index)
ax.set_xlabel("Respondents", labelpad=22)
ax.set_ylabel("Count", labelpad=20)
ax.set_title("What is your gender?", pad=22)
plt.xticks(rotation=90)
plt.show()
multi.rename(columns={'What is your age (# years)?': "Age"}, inplace=True)
age_df = pd.crosstab(multi.Age, multi.Region)*100/pd.crosstab(multi.Age, multi.Region).sum(axis=0)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,10))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,8):
    r = regions[i-1]
    y = age_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=age_df.index, y=y, palette=clrs)
    ax.set_xlabel("Age group", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
multi.rename(columns={'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': "Education"}, inplace=True)
multi["Education"]=multi["Education"].str.replace("Some college/university study without earning a bachelor’s degree","No bachelor's degree")
multi["Education"]=multi["Education"].str.replace("No formal education past high school","High school")
education_df = pd.crosstab(multi.Education, multi["Region"])*100/pd.crosstab(multi.Education, multi["Region"]).sum(axis=0)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,20))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,8):
    r = regions[i-1]
    y = education_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=education_df.index, y=y, palette=clrs)
    ax.set_ylim([0, 55])
    ax.set_xlabel("Education", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
multi.rename(columns={'Which best describes your undergraduate major? - Selected Choice': "Undergraduate major"}, inplace=True)
undergrad_major={"A business discipline \(accounting, economics, finance, etc.\)": "Business discipline",
"Computer science \(software engineering, etc.\)":"Computer Science",
"Engineering \(non-computer focused\)":"Engineering",
"Environmental science or geology":"Environmental science",
"Fine arts or performing arts":"Fine arts",
"Humanities \(history, literature, philosophy, etc.\)":"Humanitites",
"Information technology, networking, or system administration":"IT",
"Mathematics or statistics":"Maths/Stats",
"Medical or life sciences \(biology, chemistry, medicine, etc.\)":"Medical/life sciences",
"Social sciences \(anthropology, psychology, sociology, etc.\)":"Social sciences",
"Physics or astronomy":"Physics/astro",
"No formal education past high school":"High school",
"Some college/university study without earning a bachelor’s degree":"No bachelor's degree"}
for k,v in undergrad_major.items():
    multi["Undergraduate major"]=multi["Undergraduate major"].str.replace(k,v)
major_df = pd.crosstab(multi["Undergraduate major"], multi["Region"])*100/pd.crosstab(multi["Undergraduate major"], multi["Region"]).sum(axis=0)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,20))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,len(regions)+1):
    r = regions[i-1]
    y = major_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=major_df.index, y=y, palette=clrs)
    ax.set_ylim([0, 60])
    ax.set_xlabel("Major", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
multi.rename(columns={'Select the title most similar to your current role (or most recent title if retired): - Selected Choice': "Current role"}, inplace=True)
role_df = pd.crosstab(multi["Current role"], multi["Region"])*100/pd.crosstab(multi["Current role"], multi["Region"]).sum(axis=0)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,20))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,len(regions)+1):
    r = regions[i-1]
    y = role_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=role_df.index, y=y, palette=clrs)
    ax.set_ylim([0, 35])
    ax.set_xlabel("Current role", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
multi.rename(columns={'What is your current yearly compensation (approximate $USD)?': "Salary"}, inplace=True)
multi["Salary"]=multi["Salary"].str.replace("I do not wish to disclose my approximate yearly compensation","Undisclosed")
ordered_salary = ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000',
'60-70,000','70-80,000','80-90,000','90-100,000','100-125,000','125-150,000',
 '150-200,000','200-250,000','250-300,000','300-400,000','400-500,000','500,000+','Undisclosed']
salary_df = pd.crosstab(multi["Salary"], multi["Region"])*100/pd.crosstab(multi["Salary"], multi["Region"]).sum(axis=0)
salary_df=salary_df.reindex(ordered_salary)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,20))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,len(regions)+1):
    r = regions[i-1]
    y = salary_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=salary_df.index, y=y, palette=clrs)
    ax.set_ylim([0, 45])
    ax.set_xlabel("Salary", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
multi_salary = multi[multi["Current role"]=="Data Scientist"]
ds_salary_df = pd.crosstab(multi_salary["Salary"], multi_salary["Region"])*100/pd.crosstab(multi_salary["Salary"], multi_salary["Region"]).sum(axis=0)
ds_salary_df = ds_salary_df.reindex(ordered_salary)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,20))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,len(regions)+1):
    r = regions[i-1]
    y = ds_salary_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=ds_salary_df.index, y=y, palette=clrs)
    ax.set_ylim([0, 35])
    ax.set_xlabel("Salary", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
multi.rename(columns={'What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice': "Tool"}, 
             inplace=True)
tools={'Cloud-based data software & APIs \(AWS, GCP, Azure, etc.\)':'Cloud-based',
       'Basic statistical software \(Microsoft Excel, Google Sheets, etc.\)':'Basic SS',
       'Local or hosted development environments \(RStudio, JupyterLab, etc.\)':'Development',
       'Advanced statistical software \(SPSS, SAS, etc.\)':'Advanced SS',
       'Business intelligence software \(Salesforce, Tableau, Spotfire, etc.\)':'BI'}
for k,v in tools.items():
    multi["Tool"]=multi["Tool"].str.replace(k,v)
tool_df = pd.crosstab(multi["Tool"], multi["Region"])*100/pd.crosstab(multi["Tool"], multi["Region"]).sum(axis=0)
nrows, ncols = 3, 3
fig = plt.figure(figsize=(13,20))
#fig.subplots_adjust(hspace=2, wspace=0.3)
for i in range(1,len(regions)+1):
    r = regions[i-1]
    y = tool_df[r]
    p = np.sort(y.values)[-3]
    clrs = ['grey' if (x < p) else 'red' for x in y ]
    ax = fig.add_subplot(nrows, ncols, i)
    ax = sns.barplot(x=tool_df.index, y=y, palette=clrs)
    ax.set_ylim([0, 60])
    ax.set_xlabel("Tool", labelpad=15)
    ax.set_ylabel("Percentage", labelpad=10)
    ax.set_title(r, pad=15)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
