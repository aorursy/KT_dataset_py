import plotly as pycred
import plotly.offline  as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
import pycountry
from sklearn import preprocessing
survey_data = pd.read_csv('../input/survey_results_public.csv')
survey_schema = pd.read_csv('../input/survey_results_schema.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")
survey_data.head()
labels = survey_data['Hobby'].value_counts().index
values = survey_data['Hobby'].value_counts(1).values 
colors = ['yellowgreen', 'lightcoral']
explode = (0.2, 0) 
plt.pie(values, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=40)
plt.axis('equal')
plt.title('Percentage of people who code a hobby?')
labels1 = survey_data['OpenSource'].value_counts().index
values1 = survey_data['OpenSource'].value_counts(1).values 
plt.title('Do you contribute to open source projects?')
colors = [ 'lightcoral','yellowgreen']
explode = (0.1, 0) 
plt.pie(values1, explode=explode, labels=labels1, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=260)
plt.axis('equal')

plt.show()
input_countries = survey_data['Country']
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

survey_data['CountryCode'] = [countries.get(country, 'Unknown code') for country in survey_data['Country']]


data = [ dict(
        type = 'choropleth',
        locations = survey_data['CountryCode'].value_counts().index ,
        z = survey_data['CountryCode'].value_counts().values ,
        autocolorscale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'In which country do you currently reside?'),
      ) ]

layout = dict(
    title = 'In which country do you currently reside?',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='In which country do you currently reside' )


Top10 = survey_data["Country"].dropna().value_counts().head(10)

df = pd.DataFrame({'Country': Top10.index,
                   'Number of Participant': Top10.values},index=Top10.index,columns=['Number of Participant'])
ax = sns.barplot(x=Top10.index, y="Number of Participant", data=df)
plt.xticks(rotation=90)
plt.show()
Filtered_Salary = survey_data[survey_data['ConvertedSalary'].notnull()]
Filtered_Salary = Filtered_Salary[Filtered_Salary.ConvertedSalary != 0]
malefemale=Filtered_Salary[Filtered_Salary['Gender'].isin(["Male","Female"])]
ax= sns.boxplot(x='ConvertedSalary',y='Gender',data=malefemale)
plt.xticks(rotation=90)
plt.show()

sns.set(style="ticks")
Sal_age=malefemale[['ConvertedSalary','Gender','Age','YearsCoding','FormalEducation','HoursComputer']]
g=sns.factorplot(x="ConvertedSalary", y="Age", hue="Gender", col="Gender", data=Sal_age,
                   capsize=.2, palette="muted", size=8, aspect=.75,order=[ 'Under 18 years old','18 - 24 years old', '25 - 34 years old',
                                                                             '35 - 44 years old',
       '45 - 54 years old', '55 - 64 years old', '65 years or older'])
g.despine(left=True)
plt.show()
g1=sns.factorplot(x="FormalEducation", y="ConvertedSalary", hue="Gender", data=Sal_age,x_estimator=np.mean
                  , palette="muted", size=8, label="Men Vs. Women Salary based on education")
plt.xticks(rotation=90)

plt.show()
g1=sns.factorplot(x="HoursComputer", y="ConvertedSalary", hue="Gender", data=Sal_age, palette="muted", size=8)

plt.show()
#Define a generic function using Pandas replace function
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

malefemale['CodedJobSatisfaction']= coding(malefemale["JobSatisfaction"], {'Extremely dissatisfied':0,'Moderately dissatisfied':1,'Slightly dissatisfied':2,'Neither satisfied nor dissatisfied':3,'Slightly satisfied':4,'Moderately satisfied':5,'Extremely satisfied':6})
sns.lmplot(x="CodedJobSatisfaction", y="ConvertedSalary", hue="Gender", data=malefemale,x_estimator=np.mean,
           markers=["o", "x"], palette="Set1", size=8);
plt.show()
sns.pointplot(x='Age', y="CodedJobSatisfaction",hue="Gender", data=malefemale, markers=["o", "x"], linestyles=["-", "--"], aspect=2)
plt.xticks(rotation=90)
plt.show()