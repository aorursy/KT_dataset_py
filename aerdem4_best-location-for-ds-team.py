import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import squarify

import os
print(os.listdir("../input"))
response=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
qol=pd.read_csv('../input/quality-of-life/Quality of Life Index per Country - Sheet1.csv')
qol.head()
l  = [{"num_results": "576", "location": "China", "keyword": "Machine learning"},
{"num_results": "4135", "location": "India", "keyword": "Machine learning"},
{"num_results": "25723", "location": "United States", "keyword": "Machine learning"},
{"num_results": "53", "location": "Indonesia", "keyword": "Machine learning"},
{"num_results": "4", "location": "Pakistan", "keyword": "Machine learning"},
{"num_results": "253", "location": "Brazil", "keyword": "Machine learning"},
{"num_results": "36", "location": "Nigeria", "keyword": "Machine learning"},
{"num_results": "0", "location": "Bangladesh", "keyword": "Machine learning"},
{"num_results": "153", "location": "Russia", "keyword": "Machine learning"},
{"num_results": "248", "location": "Japan", "keyword": "Machine learning"},
{"num_results": "109", "location": "Mexico", "keyword": "Machine learning"},
{"num_results": "41", "location": "Philippines", "keyword": "Machine learning"},
{"num_results": "25", "location": "Egypt", "keyword": "Machine learning"},
{"num_results": "2", "location": "Ethiopia", "keyword": "Machine learning"},
{"num_results": "36", "location": "Vietnam", "keyword": "Machine learning"},
{"num_results": "2269", "location": "Germany", "keyword": "Machine learning"},
{"num_results": "3", "location": "Iran", "keyword": "Machine learning"},
{"num_results": "0", "location": "Democratic Republic of the Congo", "keyword": "Machine learning"},
{"num_results": "107", "location": "Turkey", "keyword": "Machine learning"},
{"num_results": "45", "location": "Thailand", "keyword": "Machine learning"},
{"num_results": "1669", "location": "France", "keyword": "Machine learning"},
{"num_results": "3823", "location": "United Kingdom", "keyword": "Machine learning"},
{"num_results": "241", "location": "Italy", "keyword": "Machine learning"},
{"num_results": "151", "location": "South Africa", "keyword": "Machine learning"},
{"num_results": "0", "location": "Myanmar", "keyword": "Machine learning"},
{"num_results": "0", "location": "Tanzania", "keyword": "Machine learning"},
{"num_results": "0", "location": "South Korea", "keyword": "Machine learning"},
{"num_results": "28", "location": "Colombia", "keyword": "Machine learning"},
{"num_results": "11", "location": "Kenya", "keyword": "Machine learning"},
{"num_results": "308", "location": "Spain", "keyword": "Machine learning"},
{"num_results": "139", "location": "Argentina", "keyword": "Machine learning"},
{"num_results": "0", "location": "Algeria", "keyword": "Machine learning"},
{"num_results": "0", "location": "Sudan", "keyword": "Machine learning"},
{"num_results": "369", "location": "Poland", "keyword": "Machine learning"},
{"num_results": "38", "location": "Ukraine", "keyword": "Machine learning"},
{"num_results": "0", "location": "Iraq", "keyword": "Machine learning"},
{"num_results": "3", "location": "Uganda", "keyword": "Machine learning"},
{"num_results": "1119", "location": "Canada", "keyword": "Machine learning"},
{"num_results": "3", "location": "Morocco", "keyword": "Machine learning"},
{"num_results": "10", "location": "Saudi Arabia", "keyword": "Machine learning"},
{"num_results": "108", "location": "Malaysia", "keyword": "Machine learning"},
{"num_results": "0", "location": "Uzbekistan", "keyword": "Machine learning"},
{"num_results": "9", "location": "Peru", "keyword": "Machine learning"},
{"num_results": "5", "location": "Venezuela", "keyword": "Machine learning"},
{"num_results": "3", "location": "Afghanistan", "keyword": "Machine learning"},
{"num_results": "3", "location": "Ghana", "keyword": "Machine learning"},
{"num_results": "0", "location": "Angola", "keyword": "Machine learning"},
{"num_results": "0", "location": "Mozambique", "keyword": "Machine learning"},
{"num_results": "0", "location": "Nepal", "keyword": "Machine learning"},
{"num_results": "0", "location": "Yemen", "keyword": "Machine learning"},
{"num_results": "0", "location": "Madagascar", "keyword": "Machine learning"},
{"num_results": "0", "location": "North Korea", "keyword": "Machine learning"},
{"num_results": "0", "location": "Ivory Coast", "keyword": "Machine learning"},
{"num_results": "0", "location": "Cameroon", "keyword": "Machine learning"},
{"num_results": "", "location": "Sri Lanka", "keyword": "Machine learning"},
{"num_results": "0", "location": "Niger", "keyword": "Machine learning"},
{"num_results": "141", "location": "Romania", "keyword": "Machine learning"},
{"num_results": "0", "location": "Burkina Faso", "keyword": "Machine learning"},
{"num_results": "0", "location": "Mali", "keyword": "Machine learning"},
{"num_results": "0", "location": "Syria", "keyword": "Machine learning"},
{"num_results": "0", "location": "Kazakhstan", "keyword": "Machine learning"},
{"num_results": "35", "location": "Chile", "keyword": "Machine learning"},
{"num_results": "0", "location": "Malawi", "keyword": "Machine learning"},
{"num_results": "585", "location": "Netherlands", "keyword": "Machine learning"},
{"num_results": "5", "location": "Ecuador", "keyword": "Machine learning"},
{"num_results": "0", "location": "Zambia", "keyword": "Machine learning"},
{"num_results": "0", "location": "Guatemala", "keyword": "Machine learning"},
{"num_results": "", "location": "Cambodia", "keyword": "Machine learning"},
{"num_results": "0", "location": "Senegal", "keyword": "Machine learning"},
{"num_results": "0", "location": "Chad", "keyword": "Machine learning"},
{"num_results": "0", "location": "Somalia", "keyword": "Machine learning"},
{"num_results": "0", "location": "Zimbabwe", "keyword": "Machine learning"},
{"num_results": "0", "location": "Guinea", "keyword": "Machine learning"},
{"num_results": "0", "location": "South Sudan", "keyword": "Machine learning"},
{"num_results": "0", "location": "Rwanda", "keyword": "Machine learning"},
{"num_results": "0", "location": "Tunisia", "keyword": "Machine learning"},
{"num_results": "234", "location": "Belgium", "keyword": "Machine learning"},
{"num_results": "0", "location": "Cuba", "keyword": "Machine learning"},
{"num_results": "0", "location": "Bolivia", "keyword": "Machine learning"},
{"num_results": "0", "location": "Benin", "keyword": "Machine learning"},
{"num_results": "0", "location": "Haiti", "keyword": "Machine learning"},
{"num_results": "62", "location": "Greece", "keyword": "Machine learning"},
{"num_results": "85", "location": "Czech Republic", "keyword": "Machine learning"},
{"num_results": "0", "location": "Burundi", "keyword": "Machine learning"},
{"num_results": "173", "location": "Portugal", "keyword": "Machine learning"},
{"num_results": "0", "location": "Dominican Republic", "keyword": "Machine learning"},
{"num_results": "332", "location": "Sweden", "keyword": "Machine learning"},
{"num_results": "2", "location": "Jordan", "keyword": "Machine learning"},
{"num_results": "", "location": "Azerbaijan", "keyword": "Machine learning"},
{"num_results": "51", "location": "Hungary", "keyword": "Machine learning"},
{"num_results": "6", "location": "Belarus", "keyword": "Machine learning"},
{"num_results": "80", "location": "United Arab Emirates", "keyword": "Machine learning"},
{"num_results": "0", "location": "Tajikistan", "keyword": "Machine learning"},
{"num_results": "0", "location": "Honduras", "keyword": "Machine learning"},
{"num_results": "455", "location": "Israel", "keyword": "Machine learning"},
{"num_results": "76", "location": "Austria", "keyword": "Machine learning"},
{"num_results": "297", "location": "Switzerland", "keyword": "Machine learning"},
{"num_results": "0", "location": "Papua New Guinea", "keyword": "Machine learning"},
{"num_results": "0", "location": "Togo", "keyword": "Machine learning"},
{"num_results": "362", "location": "Australia", "keyword": "Machine learning"},
{"num_results": "72", "location": "Denmark", "keyword": "Machine learning"},
{"num_results": "74", "location": "Finland", "keyword": "Machine learning"},
{"num_results": "395", "location": "Hong Kong", "keyword": "Machine learning"},
{"num_results": "277", "location": "Ireland", "keyword": "Machine learning"},
{"num_results": "33", "location": "New Zealand", "keyword": "Machine learning"},
{"num_results": "135", "location": "Norway", "keyword": "Machine learning"},
{"num_results": "572", "location": "Singapore", "keyword": "Machine learning"},
{"num_results": "102", "location": "Taiwan", "keyword": "Machine learning"}]

linkedin = pd.DataFrame.from_dict(l)
linkedin.columns = ["keyword", "Country", "linkedin_count"]
linkedin.head()
resp_coun=response['Country'].value_counts().to_frame()
sns.barplot(resp_coun['Country'],resp_coun.index,palette='inferno')
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
tree=response['Country'].value_counts().to_frame()
squarify.plot(sizes=tree['Country'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))
plt.rcParams.update({'font.size':20})
fig=plt.gcf()
fig.set_size_inches(40,15)
plt.show()
response['CompensationAmount']=response['CompensationAmount'].str.replace(',','')
response['CompensationAmount']=response['CompensationAmount'].str.replace('-','')
rates=pd.read_csv('../input/kaggle-survey-2017/conversionRates.csv')
rates.drop('Unnamed: 0',axis=1,inplace=True)

#[response["EmploymentStatus"] == "Employed full-time"]
salary=response[['CompensationAmount','CompensationCurrency', 'Age', 'GenderSelect','Country','CurrentJobTitleSelect',
                                                                       'JobSatisfaction']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
print('Median Salary is USD $',salary['Salary'].dropna().astype(int).median())
country_dict = {"People 's Republic of China": "China", 'Republic of China': "Taiwan"}
salary["Country"] = salary["Country"].apply(lambda x: country_dict[x] if x in country_dict else x)

set(salary["Country"]) - set(qol["Country"]), set(qol["Country"]) - set(salary["Country"])
plt.subplots(figsize=(15,8))
salary=salary[(salary['Salary']<200000) & (salary["Salary"] > 10000) & (salary["Age"] > 20) & (salary["Age"] < 55)]
print(salary.median())
sns.distplot(salary['Salary'])
plt.title('Salary Distribution',size=15)
plt.show()
f,ax=plt.subplots(1,1,figsize=(18,20))
sal_coun=salary.groupby('Country')['Salary'].median().sort_values(ascending=False).to_frame()
sns.barplot('Salary',sal_coun.index,data=sal_coun,palette='RdYlGn',ax=ax)
ax.axvline(salary['Salary'].median(),linestyle='dashed')
ax.set_title('Highest Salary Paying Countries')
ax.set_xlabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()
f,ax=plt.subplots(1,2,figsize=(25,15))
sns.countplot(y=response['MajorSelect'],ax=ax[0],order=response['MajorSelect'].value_counts().index)
ax[0].set_title('Major')
ax[0].set_ylabel('')
sns.countplot(y=response['CurrentJobTitleSelect'],ax=ax[1],order=response['CurrentJobTitleSelect'].value_counts().index)
ax[1].set_title('Current Job')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()
salary["JobSatisfaction"].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)

salary["JobSatisfaction"] = salary["JobSatisfaction"].astype(float)
salary["Salary"] = salary["Salary"].astype(float)

ranking = pd.merge(salary.groupby("Country").agg({"Salary": "median", "JobSatisfaction": "median", "Age": "count"}).reset_index().rename(columns={"Age": "DS_count"}), 
                   qol, on="Country")

from scipy.stats import rankdata

for c in ["Salary", "JobSatisfaction", "Quality of Life Index", "DS_count"]:
    ranking["rank_" + c] = rankdata(ranking[c]).astype(int)

ranking = ranking.drop("Rank", axis=1)

ranking = pd.merge(ranking, linkedin, on="Country")
ranking["linkedin_count"] = rankdata(ranking["linkedin_count"].astype(int))

ranking["linkedin_count"] = ranking["linkedin_count"].astype(int)
len(ranking)
import plotly.offline as py
py.init_notebook_mode(connected=True)

from ipywidgets import *

def update(salary_w, satisfaction_w, quality_life_w, ds_count_w, linkedin_count_w):
    ranking["Score"] = salary_w*(len(ranking)-ranking["rank_Salary"]) + satisfaction_w*(ranking["rank_JobSatisfaction"]) \
                    + quality_life_w*(ranking["rank_Quality of Life Index"]) + ds_count_w*(ranking["rank_DS_count"]) + linkedin_count_w*(ranking["linkedin_count"])
    ranking["Score"] /= salary_w + satisfaction_w + quality_life_w + ds_count_w + linkedin_count_w
    ranking["Score"] = ranking["Score"].astype(int)
    col = "Score"
    satisfy_job=ranking.groupby(['Country'])[col].median().sort_values(ascending=True).to_frame()
    data = [ dict(
            type = 'choropleth',
            autocolorscale = False,
            colorscale = 'Jet',
            reversescale = True,
            showscale = True,
            locations = satisfy_job.index,
            z = satisfy_job[col],
            locationmode = 'country names',
            text = satisfy_job[col],
            marker = dict(
                line = dict(color = 'rgb(200,200,200)', width = 0.5)),
                colorbar = dict(autotick = True, tickprefix = '', 
                title = col)
                )
           ]

    layout = dict(
        title = '{} By Country'.format(col),
        geo = dict(
            showframe = True,
            showocean = True,
            oceancolor = 'rgb(200,200,255)',
            projection = dict(
            type = 'chloropleth',

            ),
            lonaxis =  dict(
                    showgrid = False,
                    gridcolor = 'rgb(102, 102, 102)'
                ),
            lataxis = dict(
                    showgrid = False,
                    gridcolor = 'rgb(102, 102, 102)'
                    )
                ),
            )
    fig = dict(data=data, layout=layout)
    py.iplot(fig, validate=False, filename='worldmap2010')

style = {'description_width': '100px'}
layout = {'width': '100px'}

salary_scaler = IntSlider(description="salary", min=0.0, max=100, step=1, value=50, style=style, layout=layout)
satisfaction_scaler = IntSlider(description="job satisfaction", min=0.0, max=100, step=1, value=50, style=style, layout=layout)
quality_life_scaler = IntSlider(description="quality of life", min=0.0, max=100, step=1, value=50, style=style, layout=layout)
ds_count_scaler = IntSlider(description="DS count", min=0.0, max=100, step=1, value=50, style=style, layout=layout)
linkedin_scaler = IntSlider(description="linkedin job ads", min=0.0, max=100, step=1, value=50, style=style, layout=layout)


w = interactive_output(update, 
         {"salary_w":salary_scaler, 
         "satisfaction_w":satisfaction_scaler,
         "quality_life_w":quality_life_scaler,
         "ds_count_w":ds_count_scaler,
         "linkedin_count_w": linkedin_scaler})

ui = VBox([salary_scaler, satisfaction_scaler, quality_life_scaler, ds_count_scaler, linkedin_scaler], style={"width": "1000px"}, layout=Layout(
    display='strict',
    flex_flow='row',
    border='solid 2px', width= "1000px"
))

display(ui, w)

quality_life_scaler.set_trait(name="value", value=51)
ranking
ranking.to_csv("data.csv")
