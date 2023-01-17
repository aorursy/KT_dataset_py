import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly
import plotly.offline as py
from collections import Counter
%matplotlib inline
import seaborn as sns
plt.style.use('fivethirtyeight')
py.init_notebook_mode(connected=True)
import os
def Questions_finder(Question_number):
    qlist=[]
    for x in df.columns:
        if x.find(Question_number)!=-1:
            qlist.append(x)
    return qlist

def Create_dictionary(data_f):
    dict1={}
    for i,columns in data_f.iterrows():
        for x in columns:
            if str(x)=="nan" or str(x)=="#NULL!" :
                continue
            if str(x) not in dict1:
                dict1[str(x)]=0
            dict1[str(x)]+=1
    return dict1
df_coded=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric.csv")
df_codebook=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Codebook.csv")
df_mapping=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv")
df=pd.read_csv("../input/HackerRank-Developer-Survey-2018-Values.csv")
df.shape
df.columns
x=df["q3Gender"].groupby(df["q3Gender"]).count().drop("Non-Binary").drop("#NULL!")
plt.figure(figsize=(12,8))
g = sns.barplot( x=list((x.index)), y=x.values, palette="winter")
plt.title('Male Female Count')
plt.ylabel("Count")
plt.xlabel("Gender")
plt.savefig('Gender Count.png')
countries_list=list(df['CountryNumeric'].unique())
female_male_ratio_country={}
for x in countries_list:
    data=df[df.CountryNumeric==x]
    xx=data["q3Gender"].groupby(data["q3Gender"]).count()
    if "Female" not in xx:
        avg=0
    else:
        avg=float(xx["Female"])/float((xx["Male"]))
        avg=avg*100
    avg=int(avg)
    
    female_male_ratio_country[x]=avg
d=Counter(female_male_ratio_country)
plt.figure(figsize=(12,8))
in_tuple=d.most_common()
g = sns.barplot( x=[xx[1] for xx in in_tuple][:5], y=[xx[0] for xx in in_tuple][:5], palette="winter")
plt.title('Female to male ratio')
plt.ylabel("Country")
plt.xlabel("Ratio in Percentage")
plt.savefig('female to male ratio.png')
x=df[df.CountryNumeric=="Papua New Guinea"]["q3Gender"].groupby(df["q3Gender"]).count()
print (x)
x=df["q2Age"].groupby(df["q2Age"]).count().drop("#NULL!")
plt.figure(figsize=(12,8))
g = sns.barplot( y=list((x.index)), x=x.values, palette="winter")
plt.title('Bar Chart showing the AGE group')
plt.ylabel("Count")
plt.xlabel("Age")
plt.savefig('Age_chart.png')
Q12_df=df[Questions_finder("q12")]
Top_in_company=Create_dictionary(Q12_df)
for_plot=Counter(Top_in_company)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Bar Chart Showing most required things in a company from a job seeker perspective')
plt.ylabel("Things Required")
plt.xlabel("number of people requiring it")
plt.savefig('things_required_in_a_company.png')
#what is required in a job by country and age:
Q12_df=df[df.CountryNumeric=="India"][Questions_finder("q12")]
Top_in_company=Create_dictionary(Q12_df)
for_plot=Counter(Top_in_company)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Bar Chart Showing most required things in a company from a job seeker perspective in India')
plt.ylabel("Things Required")
plt.xlabel("number of people requiring it")
plt.savefig('things_required_in_a_company.png')
#what is required in a job by country and age:
Q12_df=df[df.CountryNumeric=="United States"][Questions_finder("q12")]
Top_in_company=Create_dictionary(Q12_df)
for_plot=Counter(Top_in_company)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Bar Chart Showing most required things in a company from a job seeker perspective in US')
plt.ylabel("Things Required")
plt.xlabel("number of people requiring it")
plt.savefig('things_required_in_a_company.png')
q13_df=Questions_finder("q13")
q13_df=df[q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized ')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure.png')
q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="India"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in India')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_india.png')
q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="United States"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in United States of America')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_america.png')
q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="Ghana"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in Ghana')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_ghana.png')
q13_df=Questions_finder("q13")
q13_df=df[df.CountryNumeric=="Pakistan"][q13_df]
q13_df
evaluation_technique=Create_dictionary(q13_df)
for_plot=Counter(evaluation_technique)

to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('How a skill is quantized in Pakistan')
plt.ylabel("Method")
plt.xlabel("Count")
plt.savefig('skill_measure_pakistan.png')
# lets find out how many hiring managers participated in a survey
# Do you interview people as part of your company's hiring process? q16
q16_df=Questions_finder("q16")
q16_df=df[q16_df]
x=q16_df.q16HiringManager.groupby(q16_df.q16HiringManager).count()
plt.figure(figsize=(12,8))
g = sns.barplot( x=list((x.index)), y=x.values, palette="winter")
plt.title('Hiring Manager?')
plt.ylabel("Count")
plt.xlabel("Response")
plt.savefig('How_many_hiring Managers.png')
## Lets find out where in the world do most of the managers come from.
py.init_notebook_mode(connected=True)
newdf=df[df.q16HiringManager=="Yes"]
countries=newdf['CountryNumeric'].value_counts().to_frame()
data = [ dict(
        type = 'choropleth',
        locations = countries.index,
        locationmode = 'country names',
        z = countries['CountryNumeric'],
        text = countries['CountryNumeric'],
        colorscale ='Viridis',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Respondents who are Managers'),
      ) ]

layout = dict(
    title = 'Hiring Managers By Nationality',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='survey-world-map')
# Female Managers per country
py.init_notebook_mode(connected=True)

newdf=df[df.q3Gender=="Female"]
newdf=newdf[df.q16HiringManager=="Yes"]

countries=newdf['CountryNumeric'].value_counts().to_frame()
data = [ dict(
        type = 'choropleth',
        locations = countries.index,
        locationmode = 'country names',
        z = countries['CountryNumeric'],
        text = countries['CountryNumeric'],
        colorscale ='Viridis',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Respondents who are Female Managers'),
      ) ]

layout = dict(
    title = 'Female Hiring Managers By Nationality',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='Female Managers by country')
## Now we are going to find out what these 
challenges_faced_by_managers=Questions_finder("q17")
challenges_faced_by_managers=df[challenges_faced_by_managers]
challenges_count=Create_dictionary(challenges_faced_by_managers)
print (challenges_count)
for_plot=Counter(challenges_count)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Challenges Faced By Managers')
plt.ylabel("Challenges")
plt.xlabel("")
plt.savefig('Challenges_faced.png')
#WHAT TO THE MANAGERS LOOK FOR IN A CANDIDATE
ideal_candidate=df[Questions_finder("q20")]
x=Create_dictionary(ideal_candidate)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Ideal Candidate')
plt.ylabel("")
plt.xlabel("")
plt.savefig('idea_Candidate.png')
#language wars
#	Which of these core competencies do you look for in software developer candidates? Check all that apply.
ideal_language=df[Questions_finder("q22")]
x=Create_dictionary(ideal_language)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Core Language for a software developer all round the world')
plt.ylabel("")
plt.xlabel("")
plt.savefig('languages_required.png')
#language wars
#	Which of these core competencies do you look for in software developer candidates? Check all that apply.
ideal_language=df[df.CountryNumeric=="India"][Questions_finder("q22")]
x=Create_dictionary(ideal_language)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Core Language for a software developer in India')
plt.ylabel("")
plt.xlabel("")
plt.savefig('languages_required.png')
#language wars
#	Which of these core competencies do you look for in software developer candidates? Check all that apply.
ideal_language=df[df.CountryNumeric=="Pakistan"][Questions_finder("q22")]
x=Create_dictionary(ideal_language)
for_plot=Counter(x)
to_plot=for_plot.most_common()
plt.figure(figsize=(12,8))
g = sns.barplot( y=[x[0] for x in to_plot], x=[x[1] for x in to_plot], palette="winter")
plt.title('Core Language for a software developer in Pakistan')
plt.ylabel("")
plt.xlabel("")
plt.savefig('languages_required.png')