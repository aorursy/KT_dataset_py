
import numpy as np # linear algebra
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline
plt.style.use('seaborn-whitegrid')
import seaborn as sns
# color = sns.color_palette()

import random

from textwrap import wrap

import re

# from IPython.display import display, HTML

# import plotly.plotly as py1
# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.offline as offline
# from plotly import tools
# from plotly.offline import iplot

# py.init_notebook_mode(connected=True)
# offline.init_notebook_mode()


# Print all rows and columns
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 50)
# survey = pd.read_csv("../input/survey_results_public.csv",  warn_bad_lines=False, error_bad_lines=False, low_memory=False)
schema = pd.read_csv("../input/survey_results_schema.csv", low_memory=False).set_index("Column")
survey = pd.read_csv("../input/survey_results_public.csv", low_memory=False)

survey.sample(3)
schema.loc["OpenSource"].QuestionText
# countries = survey.Country.unique()
# for country in countries:
#     print(country)
latin_america = ["Belize","Costa Rica", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama", 
                 "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "French Guiana (département of France)", 
                 "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela", "Cuba", "Dominican Republic", "Haiti",
                 "Guadeloupe", "Martinique", "Puerto Rico", "Saint-Barthélemy", "Saint-Martin" ,
                 "Venezuela, Bolivarian Republic of..."]

lats = survey[survey["Country"].isin(latin_america)].copy()
lats.Country = lats.Country.replace(to_replace={"Venezuela, Bolivarian Republic of...": "Venezuela"})
print("Responses from latin america countries: %i" %lats.shape[0])
print("Percentage of latin america countries: %0.2f%%" %(100*lats.shape[0]/survey.shape[0]))
demografic_colums = ["Gender", "SexualOrientation", "EducationParents", "RaceEthnicity", "Age", "Country", "FormalEducation"]
countries = lats.Country.value_counts(ascending=False)#.to_frame()
f, ax = plt.subplots(1, 1, figsize=(15, 8))

ax = sns.barplot(x=countries.values, y=countries.index, alpha=0.7, ax=ax, palette="Set2")
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax.text(width, y+height/2,'%0.2f%%' %(100*width/lats.shape[0]) , ha="left",)
plt.xlabel("Participants", fontsize=14)
plt.ylabel("Country ", fontsize=14)
plt.title("Top 15 countries contestants", fontsize=16)
plt.show()

#Age 
age = lats.Age.to_frame() #loc[(mcr.Age > 16) & (mcr.Age <= 70),'Age'].astype(int).to_frame()
f, ax = plt.subplots(1, 1,  figsize=(12, 8))
pal = sns.cubehelix_palette(7, start=3, rot=0, dark=0.25, light=.75, reverse=True)
order = ['Under 18 years old', '18 - 24 years old', '25 - 34 years old',
         '35 - 44 years old', '45 - 54 years old',  '55 - 64 years old', '65 years or older']
ax = sns.countplot(x="Age", data=lats,palette= pal, order=order,  ax=ax)
ax.set_xlabel("Age", fontsize=14)
ax.set_ylabel("Respondents counts", fontsize=14)
ax.set_title("Age of contestants", fontsize=16)
plt.show()
# After seeing gender answers they are a little messy, for example quite a few respond male and female, why  didn't  just
# answer non-binary, in any case i'm gonna merge some of the answer
to_replace = {
    "Male;Non-binary, genderqueer, or gender non-conforming": "Non-binary, genderqueer, or gender non-conforming",
    "Female;Transgender" : "Transgender",
    "Female;Male": "Non-binary, genderqueer, or gender non-conforming",
    "Female;Non-binary, genderqueer, or gender non-conforming" : "Non-binary, genderqueer, or gender non-conforming",
    "Female;Male;Non-binary, genderqueer, or gender non-conforming": "Non-binary, genderqueer, or gender non-conforming",
    "Transgender;Non-binary, genderqueer, or gender non-conforming": "Non-binary, genderqueer, or gender non-conforming",
    "Female;Male;Transgender;Non-binary, genderqueer, or gender non-conforming": "Non-binary, genderqueer, or gender non-conforming"
}
lats.Gender = lats.Gender.replace(to_replace=to_replace)
# Removing outliers from 3 times lower and higher std()
lats_salary = lats[np.abs(lats.ConvertedSalary - lats.ConvertedSalary.mean()) <= (3*lats.ConvertedSalary.std())]

f, ax = plt.subplots(1, 2, figsize=(18, 8))

ax[0] = sns.boxplot(x="Gender", y="ConvertedSalary", data=lats_salary,  ax=ax[0], palette="Set2")
ax[0].set_xticklabels(['Male', 'Female', 'No-binary\nGenderqueer\nGender Non-confirming', "Transgender"])
ax[0].set_xlabel("Gender", fontsize=14)
ax[0].set_ylabel("Converted Salary", fontsize=14)
ax[0].set_title("Converted Salary by Gender", fontsize=16)

ax[1] = sns.regplot( x="Respondent", y="ConvertedSalary", data=lats_salary, fit_reg=False, ax=ax[1])
ax[1].set_xlabel("Respondent", fontsize=14)
ax[1].set_ylabel("Converted Salary", fontsize=14)
ax[1].set_title("Scatter plot Converted Salary", fontsize=16)

plt.show()
lats_salary_below200k =  lats_salary[lats_salary.ConvertedSalary < 200000]

f, ax = plt.subplots(1, 2, figsize=(18, 8))

ax[0] = sns.boxplot(x="Gender", y="ConvertedSalary", data=lats_salary_below200k,  ax=ax[0], palette="Set2")
ax[0].set_xticklabels(['Male', 'Female', 'No-binary\nGenderqueer\nGender Non-confirming', "Transgender"])
ax[0].set_xlabel("Gender", fontsize=14)
ax[0].set_ylabel("Converted Salary", fontsize=14)
ax[0].set_title("Converted Salary by Gender", fontsize=16)
# ax[0].axhline(lats_salary_below200k.ConvertedSalary.mean(), linestyle='dashed', color="black")

ax[1] = sns.regplot( x="Respondent", y="ConvertedSalary", data=lats_salary_below200k, fit_reg=False, ax=ax[1])
ax[1].set_xlabel("Respondent", fontsize=14)
ax[1].set_ylabel("Converted Salary", fontsize=14)
ax[1].set_title("Scatter plot Converted Salary", fontsize=16)
ax[1].axhline(lats_salary_below200k.ConvertedSalary.mean(), linestyle='dashed', color="black")

plt.show()
# return a list with ncolors length of colors in hex format
def palette_random(ncolors=10):
    col3 = []
    for x in range(0, ncolors):
        col3.append("#{:06x}".format(random.randint(0, 0xFFFFFF)))
    return col3
pal_country = sns.color_palette(palette_random(23))
sns.palplot(pal_country)
g = sns.lmplot(x="Respondent", y="ConvertedSalary", data=lats_salary_below200k, fit_reg=False, hue="Country",
               size=8, aspect=2, palette=pal_country)
# print(type(g))
f, ax = plt.subplots(1,1, figsize=(15, 8))
ax = sns.boxplot(y="Country", x="ConvertedSalary", data=lats_salary_below200k,  ax=ax, palette=pal_country)
ax.set_ylabel("Country", fontsize=14)
ax.set_xlabel("Converted Salary", fontsize=14)
ax.set_title("Converted Salary by country", fontsize=16)
ax.axvline(lats_salary_below200k.ConvertedSalary.mean(), linestyle='dashed', linewidth=3,color="red")

plt.show()
f, ax = plt.subplots(1,2, figsize=(15, 8))

ax[0] = sns.countplot(y="FormalEducation", data=lats_salary_below200k, ax=ax[0], palette="Set2")
ylabels = ["\n".join(wrap(l.get_text(), 30)) for l in ax[0].get_yticklabels()]
ax[0].set_yticklabels(ylabels, fontsize=10, rotation=0)
ax[0].set_ylabel("Formal Education", fontsize=14)
ax[0].set_xlabel("Count", fontsize=14)
ax[0].set_title("Count of formal education", fontsize=16)

for p in ax[0].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax[0].text(width, y+height/2,'%2i' %width , ha="left",)

ax[1] = sns.boxplot(y="FormalEducation", x="ConvertedSalary", data=lats_salary_below200k,  ax=ax[1], palette="Set2")
ax[1].set_yticklabels([""], fontsize=10, rotation=0)
ax[1].set_ylabel("", fontsize=14)
ax[1].set_xlabel("Converted Salary", fontsize=14)
ax[1].set_title("Salary by formal education", fontsize=16)
ax[1].axvline(lats_salary_below200k.ConvertedSalary.mean(), linestyle='dashed', color="red")

plt.tight_layout()
plt.show()
# get a DataFrame and a c column string
# Returns new dataframe with a one hot enconding from the column
def one_hot_encoding(df, c):
    df_new = pd.DataFrame()
    temp = df[c].str.split(';', expand=True)
    new_columns = pd.unique(temp.values.ravel())
    for new_c in new_columns:
        if new_c and new_c is not np.nan:
            # Create new column for each unique column
            df_new[c] = df[c].copy()
            idx = df[c].str.contains(new_c, regex=False).fillna(False)
            df_new.loc[idx, f"{new_c}"] = 1
    df_new.drop(c, axis=1, inplace=True)
    df_new = pd.get_dummies(df_new)    
    return df_new
race = one_hot_encoding(lats_salary_below200k, "RaceEthnicity").dropna(axis=0, how="all")
race["Respondent"] = lats.Respondent
race = race.merge(lats[["Respondent", "ConvertedSalary"]], on="Respondent", )
race.iloc[list(race.iloc[:, 0]==1), [0, -1]].ConvertedSalary.mean()
def sal_row(row):   
    for e in range(len(row)):
        if row[e]==1:
            row[e] = row.ConvertedSalary
    return row

race = race.apply(sal_row, axis=1)
race.describe()
# This funtion takes text as str and n_char as int
# return a new string with a newline "\n" as the n_char specified
def wrap_join(text, n_char):
    string = "\n".join(wrap(text, n_char))
    return string
f, ax = plt.subplots(1,1, figsize=(12, 8))
ax = race.plot(x="ConvertedSalary", y=race.columns[:-2], ax=ax, kind="box", colormap="Set1",)
#xlabels = ["\n".join(wrap(l.get_text(), 15)) for l in ax.get_xticklabels()]
xlabels = [wrap_join(l.get_text(), 15) for l in ax.get_xticklabels()]

ax.set_xticklabels(xlabels, fontsize=12, rotation=0)
# ax.axhline(race.iloc[:, 1].mean(), linestyle='dashed', color="black") 
ax.set_title("Salary by Race Ethnicity", fontsize=16)
ax.set_ylabel("Salary", fontsize=14)
ax.set_xlabel("Race Ethnicity", fontsize=14)
ax.grid(True)#color='lightpink', linestyle='-', linewidth=0.5, alpha=1)

plt.tight_layout()
plt.show()
health = ["WakeTime", "HoursComputer", "HoursOutside", "Exercise"]
titles = ["Wake up Time", "Hours in Computer", "Hours Outside", "Exercise"]
f, ax = plt.subplots(2, 2, figsize=(18, 18))
k = 0
pal = sns.color_palette("Set2")
for axi in ax.flat :
    axxi = lats[health[k]].value_counts().to_frame().plot(kind="pie", ax=axi, subplots=True, legend=False, startangle=180,
                                                          radius=0.85, fontsize=11, labeldistance=1.05, rotatelabels=False,
                                                          autopct="%0.0f%%", shadow=True, colors=pal)
    axi.set_title(titles[k], fontsize=16)
    axi.set_ylabel("")
    k+=1     
plt.show()
## # Search a string that appears in the columns of the dataframe
# Returns a list with the matches if not returns an empty list
def searchinColumns(df, match):
    match_col = []
    match = ("(?i)(.+|)%s(.+|)" %match)
    for column in df.columns:
        if re.match(match, column):
            match_col.append(column)
    return match_col
education_col = searchinColumns(lats, "education")
# print(education_col)
formal_education = ['Some college/university study without earning a degree',
                    'Bachelor’s degree (BA, BS, B.Eng., etc.)',
                    'Master’s degree (MA, MS, M.Eng., MBA, etc.)', 
                    'Other doctoral degree (Ph.D, Ed.D., etc.)']
group = lats_salary_below200k.groupby(by="FormalEducation")
group = group.Age.value_counts().to_frame()

f, ax = plt.subplots(2, 2, figsize=(14, 14))
k = 0
pal = sns.color_palette("Set3")
for axi in ax.flat :
#     group.Age.value_counts().to_frame().loc[formal_education[1]].plot(kind="pie", subplots=True)
    axxi = group.loc[formal_education[k]].plot(kind="pie", ax=axi, subplots=True, legend=False, startangle=180,
                                                          radius=0.85, fontsize=11, labeldistance=1.05, rotatelabels=False,
                                                          autopct="%0.0f%%", shadow=True, colors=pal)
    axi.set_title(formal_education[k], fontsize=16)
    axi.set_ylabel("")
    k+=1     
plt.show()
#Just a list to remember what columns I'll analyse later
work = ["OpenSource", "CompanySize", "DevType", "YearsCodingProf", "CareerSatisfaction", "HopeFiveYears", "CommunicationTools", "SelfTaughtTypes", "LanguageWorkedWith", "LanguageDesireNextYear",
        "DatabaseWorkedWith", "DatabaseDesireNextYear", "FrameworkWorkedWith", "IDE","OperatingSystem", "NumberMonitors", "VersionControl", ]
works = lats_salary_below200k.loc[:,work].copy()
works["ConvertedSalary"] = lats_salary_below200k.ConvertedSalary.copy()
# works.describe(include="all")
f, ax = plt.subplots(1, 3, figsize=(15, 5))
pal = sns.color_palette("Set3")
ax[0] =  sns.countplot(x="OpenSource", data=lats_salary_below200k, palette= pal,  ax=ax[0])   
ax[0].set_ylabel("", fontsize=14)
ax[0].set_xlabel("", fontsize=14)
ax[0].set_title(schema.loc["OpenSource"].QuestionText, fontsize=16)

ax[1] = sns.boxplot(x="OpenSource", y="ConvertedSalary", data=lats_salary_below200k,  ax=ax[1], palette=pal)
ax[1].set_ylabel("", fontsize=14)
ax[1].set_xlabel("", fontsize=14)
ax[1].set_xlabel("Salary by formal open source contributor", fontsize=16)
ax[1].axhline(lats_salary_below200k.ConvertedSalary.mean(), linestyle='dashed', color="red")


lats_formal = lats_salary_below200k[lats_salary_below200k.FormalEducation.isin(formal_education)]
ax[2] = sns.countplot(x="OpenSource", hue="FormalEducation", data=lats_formal, palette= pal,  ax=ax[2], hue_order=formal_education)
ax[2].set_ylabel("")
ax[2].set_xlabel("")
ax[2].set_title("Education and contributors", fontsize=16)

ax[2].legend(bbox_to_anchor=(0.55, 0.98), loc=2, borderaxespad=0.)
plt.show()


company = ['Fewer than 10 employees', '10 to 19 employees', '20 to 99 employees', '100 to 499 employees',
           '500 to 999 employees','1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees']
yticks = [comp.replace(" employees", "") for comp in company]
f, ax = plt.subplots(1, 2, figsize=(18, 5))

pal = sns.color_palette("tab10_r")
# pal = palette_random(8)

ax[0] =  sns.countplot(y="CompanySize", data=lats_salary_below200k, palette= pal,  ax=ax[0], order=company)   
ax[0].set_ylabel("", fontsize=14)
ax[0].set_xlabel("", fontsize=14)
ax[0].set_yticklabels(yticks)
ax[0].set_title(wrap_join(schema.loc["CompanySize"].QuestionText, 50), fontsize=16)
total = len(lats_salary_below200k.CompanySize.dropna())
accumulated = 0
for p in ax[0].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    accumulated += 100*width/total
    ax[0].text(width, y+height/2, "%0.2f%% %s" %(accumulated ," accumulated") , ha="left",)

ax[1] = sns.boxplot(y="CompanySize", x="ConvertedSalary", data=lats_salary_below200k,  ax=ax[1], palette=pal, order=company)
ax[1].set_ylabel("", fontsize=14)
ax[1].set_xlabel("", fontsize=14)
ax[1].set_yticklabels(yticks)
ax[1].set_title("Salary by company size", fontsize=16)
ax[1].axvline(lats_salary_below200k.ConvertedSalary.mean(), linestyle='dashed', color="red")

group = lats_salary_below200k.groupby("CompanySize").FormalEducation.value_counts().to_frame()
edcomp = pd.DataFrame(index=formal_education, columns=company)
for c in edcomp.columns:
    edcomp = edcomp.combine_first(group.loc[c].rename({'FormalEducation': c}, axis=1))
edcomp = edcomp.reindex(formal_education).reindex(company[::-1], axis=1).astype(int)#.transpose()
f, ax = plt.subplots(2,2, figsize=(11, 10))
k = 0
for axi in ax.flat:
    axi = edcomp.iloc[k,:].to_frame().plot(kind="pie", ax=axi, subplots=True, legend=False, startangle=0,
                  radius=0.85, fontsize=11, labeldistance=1.05, rotatelabels=False, labels=[""]*8,
                  autopct="%0.0f%%", shadow=True, colors=pal)[0]
    title = wrap_join(edcomp.iloc[k].to_frame().columns[0], 40)
    axi.set_title(title, fontsize=14)
    axi.set_ylabel("")
    axi.set_xlabel("")
    axi.set_yticklabels("")
    axi.set_xticklabels("")
    k+=1
plt.legend(yticks ,bbox_to_anchor=(0.30, 1.40), loc=0, borderaxespad=0.1, ncol=2)

# group = lats_salary_below200k.groupby("CompanySize").FormalEducation.value_counts().to_frame()
# edcomp = pd.DataFrame(index=formal_education, columns=company)
# for c in edcomp.columns:
#     edcomp = edcomp.combine_first(group.loc[c].rename({'FormalEducation': c}, axis=1))
# # edcomp = edcomp.reindex(formal_education).reindex(company[::-1], axis=1).astype(int).transpose()
edcomp = edcomp.transpose()

f, ax = plt.subplots(2,4, figsize=(13, 7))
k = 0
for axi in ax.flat:
    axi = edcomp.iloc[k,:].to_frame().plot(kind="pie", ax=axi, subplots=True, legend=False, startangle=0,
                  radius=0.85, fontsize=11, labeldistance=1.05, rotatelabels=False, labels=[""]*8,
                  autopct="%0.0f%%", shadow=True, colors=pal)[0]
    axi.set_title(yticks[k], fontsize=14)
    axi.set_ylabel("")
    axi.set_xlabel("")
    axi.set_yticklabels("")
    axi.set_xticklabels("")
    k+=1
plt.legend(formal_education, bbox_to_anchor=(0.70, 1.35), loc=0, borderaxespad=0.1, ncol=2) 
# plt.show()
plt.show()
devtype = list(lats_salary_below200k.DevType.dropna().str.split(";"))
devtype = pd.Series([d for dev in devtype for d in dev ], name='Dev Type').value_counts().to_frame()
f, ax = plt.subplots(1,1, figsize=(10, 7))
ax = devtype.plot(kind='barh', color='y', ax=ax)
plt.show()
onehot_devtype = one_hot_encoding(lats_salary_below200k, "DevType")
onehot_devtype["Country"] = lats_salary_below200k.Country
onehot_devtype["FormalEducation"] = lats_salary_below200k.FormalEducation
dsml = onehot_devtype[["Data scientist or machine learning specialist", "Country", "FormalEducation"]].dropna()
dsml

f, ax = plt.subplots(1, 2, figsize=(20, 7))
pal = sns.color_palette("Set3")
ax[0] =  sns.countplot(y="Country", data=dsml, palette= pal,  ax=ax[0])
# ax = dsml.plot(kind="barh", stacked=True, ax=ax)
ax[0].set_ylabel("", fontsize=14)
ax[0].set_xlabel("", fontsize=14)
ax[0].set_title("Data scientist or machine learning specialist by country", fontsize=15)

for p in ax[0].patches:
    width = p.get_width()
    height = p.get_height()
    x = p.get_x()
    y = p.get_y()
    ax[0].text(width, y+height/2, width , ha="left",)

ax[1] = sns.countplot(x="FormalEducation", data=dsml[dsml.isin(formal_education)], palette=pal, ax=ax[1], order=formal_education)
ticklabels = [wrap_join(l.get_text(), 15) for l in ax[1].get_xticklabels()]
ax[1].set_xticklabels(ticklabels)
ax[1].set_ylabel("", fontsize=14)
ax[1].set_xlabel("", fontsize=14)
ax[1].set_title("Count Data scientist or machine learning specialist by formal education", fontsize=15)

plt.show()
lang = lats_salary_below200k[["LanguageWorkedWith", "LanguageDesireNextYear", "ConvertedSalary", "FormalEducation", "Country"]]
lang_work = one_hot_encoding(lang, "LanguageWorkedWith").dropna(how="all", axis=1)
lang_desire = one_hot_encoding(lang, "LanguageDesireNextYear").dropna(how="all", axis=1, )
work_count = lang_work.sum(axis=0).sort_index().to_frame().rename({0:"Count"}, axis=1).sort_values(by="Count", ascending=False)
desire_count = lang_desire.sum(axis=0).sort_index().to_frame().rename({0:"Count"}, axis=1).sort_values(by="Count", ascending=False)


f, ax1 = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

labels = np.array(list(work_count.index))
w = np.array(work_count.Count)*-1
d = np.array(desire_count.Count)
ax1 = sns.barplot(x=labels, y=d, palette="Set2", ax=ax1)
ax1 = sns.barplot(x=labels, y=w, palette="Set2", ax=ax1)
ax1.axhline(0, linestyle='dashed', linewidth=1,color="black")
ax1.set_xticklabels(labels, fontsize=11, rotation=90)
ax1.set_yticklabels([0, 2000, 1500, 1000, 500, 0, 500, 1000, 1500, 2000])
ax1.grid( linestyle='-', linewidth=1)

ax1.text(30, 1200, "Desired", fontsize=16)
ax1.text(30, -1200, "Worked", fontsize=16)
ax1.set_title("Languages worked with and desire to work with next year", fontsize=16)
plt.show()

lang_sum = lang_work.fillna(0) + (lang_desire.fillna(0)+1)
lang_sum = lang_sum.astype(int).replace({1:"Worked with", 3:"Both", 2:"Desire to work"})
lang_values = pd.DataFrame(index=["Both", "Worked with", "Desire to work",])
for c in lang_sum.columns:
    lang_values = lang_values.combine_first(lang_sum[c].value_counts().to_frame())
lang_values = lang_values.transpose().fillna(0).astype(int).sort_values(by="Desire to work", ascending=False)#.drop("Both", axis=1)
lang_values = lang_values.reindex([ "Worked with", "Desire to work","Both"], axis=1)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
colormap = ListedColormap(sns.color_palette("Set2", n_colors=3 ))
ax1 = lang_values[:20].plot(kind='barh', stacked=True, colormap=colormap, ax=ax1, legend=False)
ax2 = lang_values[19:].plot(kind='barh', stacked=True, colormap=colormap, ax=ax2)
plt.yticks(fontsize=11)
plt.show()


 

