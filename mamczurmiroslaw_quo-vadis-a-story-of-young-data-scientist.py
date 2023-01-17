# Import Python packages
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 5000)
sns.set(style="whitegrid")
Responce = pd.read_csv('../input/kaggle-survey-2018/freeFormResponses.csv').loc[1:, :]
MultipleResponce = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv').loc[1:, :]
SurveySchema = pd.read_csv('../input/kaggle-survey-2018/SurveySchema.csv').loc[1:, :]
MultipleResponce_tmp = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
MultipleResponce['Q9_avg'] = MultipleResponce['Q9'].apply(lambda x: 5 if x == '0-10,000'
                            else (15 if x == '10-20,000' else (25 if x == '20-30,000' else (35 if x == '30-40,000' 
                            else (45 if x == '40-50,000' else (55 if x == '50-60,000' else (65 if x == '60-70,000' 
                            else (75 if x == '70-80,000' else (85 if x == '80-90,000' else (95 if x == '90-100,000' 
                            else (112 if x == '100-125,000' else (137 if x == '125-150,000' else (175 if x == '150-200,000' 
                            else (225 if x == '200-250,000' else (275 if x == '250-300,000' else (400 if x == '300-400,000' 
                            else (450 if x == '400-500,000' else (500 if x == '500,000+' else 0)
                            ))))))))))))))))) #.value_counts()
MultipleResponce['Q2_avg'] = MultipleResponce['Q2'].apply(lambda x: 20 if x == '18-21'
                            else (23 if x == '22-24' else (27 if x == '25-29' else (32 if x == '30-34' 
                            else (37 if x == '35-39' else (42 if x == '40-44' else (47 if x == '45-49' 
                            else (55 if x == '50-59' else (65 if x == '60-79' else (75 if x == '70-79' 
                            else (80 if x == '80+' else 30)
                            ))))))))))
MultipleResponce['Q8_avg'] = MultipleResponce['Q8'].apply(lambda x: 1 if x == '0-1'
                            else (2 if x == '1-2' else (5 if x == '5-10' else (3 if x == '2-3' 
                            else (4 if x == '3-4' else (15 if x == '10-15' else (5 if x == '4-5' 
                            else (20 if x == '15-20' else (25 if x == '20-25' else (30 if x == '30 +' 
                            else (99 if x == '25-30' else 0)
                            ))))))))))
MultipleResponce['Q24_avg'] = MultipleResponce['Q24'].apply(lambda x: 2 if x == '1-2 years'
                            else (5 if x == '3-5 years' else (1 if x == '< 1 year' else (10 if x == '5-10 years' 
                            else (20 if x == '10-20 years' else (0 if x == 'I have never written code' else (30 if x == '20-30 years' 
                            else (99 if x == '30+ ' else 0)
                            )))))))
MultipleResponce['Q25_avg'] = MultipleResponce['Q25'].apply(lambda x: 1 if x == '< 1 year'
                            else (2 if x == '1-2 years' else (3 if x == '2-3 years' else (4 if x == '3-4 years' 
                            else (10 if x == '5-10 years' else (5 if x == '4-5 years' else (15 if x == '10-15 years' 
                            else (20 if x == '20+ years' else (0 if x == 'I have never studied machine learning but plan to learn in the future'
                            else (-1 if x == 'I have never studied machine learning and I do not plan to' else 0)
                            )))))))))
MultipleResponce['Q2'] = MultipleResponce['Q2'].apply(lambda x: '50-59' if x in ['50-54','55-59']
                            else ('60+' if x in ['60-69','70-79','80+'] else x ))
MultipleResponce['Q3'] = MultipleResponce['Q3'].apply(lambda x: 'Hong Kong' if x == 'Hong Kong (S.A.R.)'
                            else ('Iran' if x == 'Iran, Islamic Republic of...'
                            else ('United Kingdom' if x == 'United Kingdom of Great Britain and Northern Ireland'
                            else ('United States' if x == 'United States of America'
                            else ('Vietnam' if x == 'Viet Nam'
                            else ('Other' if x == 'I do not wish to disclose my location'       
                            else x
                            )))))) 
MultipleResponce['Q4'] = MultipleResponce['Q4'].apply(lambda x: 'High school' if x == 'No formal education past high school'
                            else ('High school' if x == 'Some college/university study without earning a bachelor’s degree'
                            else x
                            ))
MultipleResponce['Q9'] = MultipleResponce['Q9'].apply(lambda x: 'Do not want to say' if x == 'I do not wish to disclose my approximate yearly compensation'
                            else x
                            )
MultipleResponce['Q10_new'] = MultipleResponce['Q10'].apply(lambda x: 'We are exploring ML methods' if x == 'We are exploring ML methods (and may one day put a model into production)'
                            else ('We are exploring ML methods' if x == 'We use ML methods for generating insights (but do not put working models into production)'
                            else x
                            ))

def prep_data(df, group_col, target):
    data = df.copy()
    data = pd.DataFrame(data.groupby([group_col])[target].agg(['size','mean','sum']))
    data['mean'] = np.round(data['mean'],1)
    data = data.reset_index()
    data.columns = ['val', 'cnt', 'mean','sum']
    return data.sort_values(by = 'mean', ascending=False)
data = prep_data(pd.DataFrame(MultipleResponce), 'Q9', 'Q9_avg')
data['cnt'] = np.round(100*data['cnt']/data['cnt'].sum(),1)

sns.set(rc={'figure.figsize':(12,7)}, style="whitegrid")
ax = sns.barplot(x='cnt', y='val', data=data, 
                 order=['Do not want to say', 
                        '0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000', '50-60,000', '60-70,000',  '70-80,000', '80-90,000','90-100,000',
                        '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000', '300-400,000', '400-500,000', '500,000+'], 
                 palette=np.array(['#dc0000', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'
                                   , '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'])            
                )
ax.set_title("Yearly salary in $USD",fontsize=16,weight='bold')
ax.set_xlabel("Percentage %",fontsize=12)
ax.set_ylabel("", fontsize=12)
ax.text(15, 4, 'One on four do not want \nto say about salary', ha='left', fontsize=14, color = '#dc0000')
#, ha="left", va="center", size=10, bbox=dict(boxstyle="square", fc="w")
sns.despine(offset=10, trim=True)

plt.show()

MultipleResponce = MultipleResponce[MultipleResponce['Q9'] != 'Do not want to say']
data = prep_data(pd.DataFrame(MultipleResponce), 'Q9', 'Q9_avg')
data['sum'] = data['sum'] /1000
plt.subplots(1, 2, figsize = (18, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='sum', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'
                                    , '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000'])                )
ax.set_title("Yearly income for whole group",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ MILLION",fontsize=12)
ax.set_ylabel("", fontsize=12)
ax.text(35, 0.6, '63 people \nearned abount 31.5 million $', ha='left', fontsize=16, color = '#00cc00')
ax.text(35, 17.5, '4.4k people (28%) \nearned abount 22 million $', ha='left', fontsize=16, color = '#dc0000')
sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'
                                    , '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000'])
                )
ax.set_title("Number of people by yearly income",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.text(900,9, '63 richest people earned as much \nas 6,300 people from the smallest \ngroup would earn', ha='left', fontsize=16, color = 'black')
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
data = prep_data(pd.DataFrame(MultipleResponce.fillna('Not employed')), 'Q6', 'Q9_avg')
MultipleResponce['Q6'] = MultipleResponce['Q6'].apply(lambda x: 'Other' if x in data[data['cnt'] < 100]['val'].unique() else x)
data = prep_data(pd.DataFrame(MultipleResponce.fillna('Not employed')), 'Q6', 'Q9_avg')
plt.subplots(1, 2, figsize = (18, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#00cc00','#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'
                                   , '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000', '#dc0000'])
                )
ax.set_title("Yearly income ($ k) by current job",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("", fontsize=12)
ax.text(40, 15, 'More then 5.3k people (28%) \nare students or unemployed', ha='left', fontsize=16, color = '#dc0000')
sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                 palette=np.array(['#00cc00', '#00cc00','#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'
                                   , '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000', '#dc0000'])
                )
ax.set_title("Number of people by current job",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.text(600,1.5, 'People managing people or \nproducts are paid the best (6%)', ha='left', fontsize=16, color = '#00cc00')
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
MultipleResponce = MultipleResponce[MultipleResponce['Q6'] != 'Student']
MultipleResponce = MultipleResponce[MultipleResponce['Q6'] != 'Not employed']
MultipleResponce = MultipleResponce[MultipleResponce['Q7'] != 'I am a student']
MultipleResponce = MultipleResponce[MultipleResponce['Q9_avg'] != 0]

const_mean = round(MultipleResponce['Q9_avg'].mean(),1)
const_median = round(MultipleResponce['Q9_avg'].median(),1)

#print(const_mean)
#print(const_median)
data = prep_data(pd.DataFrame(MultipleResponce), 'Q2', 'Q9_avg')

plt.subplots(1, 2, figsize = (18, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff', '#c8c8c8', '#dc0000'])            
                )
ax.set_title("Yearly income ($ k) by age",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("Age group", fontsize=12)
ax.text(80, 6.3, 'The largest group are people \nafter graduation starting \nwith their careers (29%)', ha='left', fontsize=16, color = '#1245ff')
ax.text(60, 3.3, 'Perfect income dependence', ha='left', rotation = '52', fontsize=16, color = 'black')

ax.annotate("",
            xy=(130, 0.1), xycoords='data',
            xytext=(30, 8), textcoords='data',
            arrowprops=dict(arrowstyle="fancy", 
                            color="0.5",
                            patchB=0,
                            shrinkB=20,
                            connectionstyle="arc3,rad=0.0",
                            ),
            )


sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff', '#c8c8c8', '#dc0000'])  
                )


ax.set_title("Number of people by age",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
data = prep_data(pd.DataFrame(MultipleResponce), 'Q8', 'Q9_avg')

plt.subplots(1, 2, figsize = (18, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff', '#c8c8c8', '#c8c8c8', '#dc0000', '#dc0000', '#dc0000'])            
                )
ax.set_title("Yearly income by work experience (in years)",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("Work experience is years", fontsize=12)
ax.text(67, 3.5, 'Perfect income dependence', ha='left', rotation = '56', fontsize=16, color = 'black')

ax.annotate("",
            xy=(145, 0.1), xycoords='data',
            xytext=(45, 10), textcoords='data',
            arrowprops=dict(arrowstyle="fancy", 
                            color="0.5",
                            patchB=0,
                            shrinkB=2,
                            connectionstyle="arc3,rad=0.0",
                            ),
            )


sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff', '#c8c8c8', '#c8c8c8', '#dc0000', '#dc0000', '#dc0000'])
                )

ax.text(1200, 4.3, 'The first large group of ML precursors \nfrom before 2014', ha='left', fontsize=16, color = '#1245ff')
ax.text(1200, 7.3, 'More and more interest in the field \nrelated to data year by year', ha='left', fontsize=16, color = '#dc0000')
ax.annotate("",
            xy=(2700, 10), xycoords='data',
            xytext=(1600, 8), textcoords='data',
            arrowprops=dict(arrowstyle="fancy", 
                            color="0.5",
                            patchB=0,
                            shrinkB=2,
                            connectionstyle="arc3,rad=-0.3",
                            ),
            )
ax.text(2200, 8.2, '+ 50% !', ha='left', fontsize=16, rotation = -35, color = '#dc0000', weight='bold')

ax.set_title("Number of people by work experience  (in years)",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
data = prep_data(pd.DataFrame(MultipleResponce), 'Q26', 'Q9_avg')

plt.subplots(1, 2, figsize = (18, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'])
                )
#ax.text(15, 2, 'The first large group of ML precursors \nfrom before 2014', ha='left', fontsize=16, color = '#1245ff')
ax.set_title("Do you consider yourself to be a data scientist?",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'])
                )

ax.set_title("Do you consider yourself to be a data scientist?",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
data = prep_data(pd.DataFrame(MultipleResponce), 'Q4', 'Q9_avg')

plt.subplots(1, 2, figsize = (18, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000'])            
                )
ax.set_title("Yearly income ($ k) by education",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("", fontsize=12)

sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                palette=np.array(['#00cc00', '#c8c8c8','#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000'])   
                )

ax.set_title("Number of people by education",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
data = prep_data(pd.DataFrame(MultipleResponce.fillna('No information')), 'Q37', 'Q9_avg')
MultipleResponce['Q37'] = MultipleResponce['Q37'].apply(lambda x: 'Other' if x in data[data['cnt'] < 100]['val'].unique() else x)
data = prep_data(pd.DataFrame(MultipleResponce.fillna('No information')), 'Q37', 'Q9_avg')
MultipleResponce['Q37'] = MultipleResponce['Q37'].apply(lambda x: 'University Courses' if x == 'Online University Courses'
                            else x
                            )

data = prep_data(pd.DataFrame(MultipleResponce.fillna('No information')), 'Q37', 'Q9_avg')

plt.subplots(1, 2, figsize = (20, 9), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(1, 2, 1)

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff', '#c8c8c8','#c8c8c8', '#c8c8c8', '#dc0000', '#dc0000'])            
                )
ax.set_title("Yearly income ($ k) by online education",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("", fontsize=12)
ax.text(55, 8.6, 'Why???', ha='left', fontsize=16, color = '#dc0000')

sns.despine(offset=10, trim=True)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='cnt', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff', '#c8c8c8','#c8c8c8', '#c8c8c8', '#dc0000', '#dc0000'])            
                )

ax.text(3000, 3.2, 'Every second person did not provide \nany information...', ha='left', fontsize=16, color = '#1245ff')
ax.text(3000, 5.2, '...but that means that up to 50% \nlearn online!', ha='left', fontsize=16, color = '#1245ff')

ax.set_title("Number of people by online education",fontsize=18,weight='bold')
ax.set_xlabel("Number of people from the survey",fontsize=12)
ax.set_ylabel("Online platforms", fontsize=12)
sns.despine(offset=10, trim=True)
data = prep_data(pd.DataFrame(MultipleResponce.fillna('No information')), 'Q37', 'Q2_avg')
sns.set(rc={'figure.figsize':(12,7)}, style="whitegrid")
ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#c8c8c8', '#00cc00','#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#1245ff'
                                   , '#dc0000', '#dc0000'])            
                )

ax.set_title("Avarage age by online education",fontsize=16,weight='bold')
ax.set_xlabel("Avarage age",fontsize=12)
ax.set_ylabel("Online platforms", fontsize=12)
sns.despine(offset=10, trim=True)

plt.show()
data = prep_data(pd.DataFrame(MultipleResponce.fillna('I do not know')), 'Q10_new', 'Q9_avg')

ax = sns.barplot(x='mean', y='val', data=data, 
                 palette=np.array(['#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#dc0000'])            
                )
ax.set_title("Yearly income ($ k) by company usage of ML methods",fontsize=18,weight='bold')
ax.set_xlabel("Yearly income in $ thousands",fontsize=12)
ax.set_ylabel("", fontsize=12)
ax.text(45, 4.2, 'Maybe soon these companies will wake up \nfrom the winter sleep', ha='left', fontsize=16, color = '#dc0000')

sns.despine(offset=10, trim=True)

# download data with country name 4 hole world
all_country = pd.read_csv('../input/survey-data/2014_world_gdp_with_codes.csv')
all_country.columns = ['country', 'mean', 'CODE']
all_country = all_country[['country', 'mean']]
all_country['mean'] = 0

locations = pd.DataFrame(MultipleResponce.groupby(['Q3'])['Q9_avg'].agg(['mean']))
locations['mean'] = np.round(locations['mean'],0)
locations = locations.reset_index()
locations.columns = ['country', 'mean']
locations = locations.sort_values(by = 'country', ascending=True)
locations.head()

locations = pd.merge(all_country, locations, how='left', on='country', sort=True).fillna(0)
locations = locations.drop(columns=['mean_x'])
locations.columns = ['country', 'mean']

data = [ dict(
        type = 'choropleth',
        locations = locations['country'],
        locationmode = 'country names',
        z = locations['mean'],
        text = locations['country'],
        colorscale = [[0,"rgb(235, 0, 0)"],[0.5,"rgb(235, 235, 0)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Avarage yearly salary in $'),
      ) ]

layout = dict(
    title = '<b>Avarage yearly salary in $</b>',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
BigMacIndex = pd.read_csv('../input/bigmacindex/BigMacIndex.csv')
BigMacIndex = BigMacIndex[BigMacIndex['date'] == '2018-07-01']
BigMacIndex.rename(columns={'name':'Q3'}, inplace=True)
BigMacIndex['Q3'] = BigMacIndex['Q3'].apply(lambda x: 'United Kingdom' if x == 'Britain'
                            else x)
BigMacIndex['BigMacIndex'] = BigMacIndex['local_price']/BigMacIndex['dollar_ex']
BigMacIndex = BigMacIndex[['Q3','BigMacIndex']]

BigMacIndex.sort_values(by='BigMacIndex').head()

MultipleResponce = pd.merge(MultipleResponce, BigMacIndex, left_index=False, right_index=False, how='left')
MultipleResponce['BigMacIndex'] = MultipleResponce['BigMacIndex'].fillna(MultipleResponce['BigMacIndex'].mean())

MultipleResponce['Q9_BigMacDaily'] = (1000*MultipleResponce['Q9_avg']/365)/MultipleResponce['BigMacIndex']
locations = pd.DataFrame(MultipleResponce.groupby(['Q3'])['Q9_BigMacDaily'].agg(['mean']))
locations['mean'] = np.round(locations['mean'],0)
locations = locations.reset_index()
locations.columns = ['country', 'mean']
locations = locations.sort_values(by = 'country', ascending=True)

locations = pd.merge(all_country, locations, how='left', on='country', sort=True).fillna(0)
locations = locations.drop(columns=['mean_x'])
locations.columns = ['country', 'mean']

data = [ dict(
        type = 'choropleth',
        locations = locations['country'],
        locationmode = 'country names',
        z = locations['mean'],
        text = locations['country'],
        colorscale = [[0,"rgb(235, 0, 0)"],[0.5,"rgb(235, 235, 0)"],[1,"rgb(235, 235, 235)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Daily salary in BiG Macs'),
      ) ]

layout = dict(
    title = '<b>Avarage daily salary in Big Macs </b>',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
tmp = iplot( fig, validate=False, filename='d3-world-map' )
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import gc

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def kfold_lightgbm(df, num_folds, stratified = False, debug= False, n_estimators=1000, max_depth=4, early_stopping_rounds=100,
                  num_leaves = 16, learning_rate = 0.02, reg_alpha = 0.04, reg_lambda = 0.07, subsample = 0.85,
                  colsample_bytree = 1.0, min_split_gain = 0.02, min_child_weight = 40,
                  min_child_samples = 20, min_data_in_leaf = 20):
    # Divide in training/validation and test data
    #train_df = df[df['TARGET'].notnull()]
    #test_df = df[df['TARGET'].isnull()]
    train_df, test_df = train_test_split(df,test_size=0.2, random_state=2018)
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=2018)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2018)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    
    sub_preds_train = np.zeros(train_df.shape[0])
    
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            n_jobs = 4,
            n_estimators = n_estimators, 
            learning_rate = learning_rate, 
            num_leaves = num_leaves, 
            subsample = subsample, 
            colsample_bytree = colsample_bytree,
            max_depth = max_depth, 
            reg_alpha = reg_alpha, 
            reg_lambda = reg_lambda, 
            min_split_gain = min_split_gain, 
            min_child_weight = min_child_weight, 
            min_child_samples = min_child_samples,
            min_data_in_leaf = min_data_in_leaf,
            silent = -1,
            verbose = -1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= early_stopping_rounds)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds_train += clf.predict_proba(train_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
         
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    fpr, tpr, _ = roc_curve(train_df['TARGET'], oof_preds)

    #display_importances(feature_importance_df)
    return feature_importance_df, fpr, tpr, roc_auc_score(train_df['TARGET'], oof_preds)

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:10].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    #plt.savefig('lgbm_importances01.png')
probka = MultipleResponce.copy()
probka['TARGET'] = probka['Q9_BigMacDaily'].apply(lambda x: 1 if x >= 60 else 0)
probka['Time from Start to Finish (seconds)'] = probka['Time from Start to Finish (seconds)'].astype('int64')

char_list = []
out_list = []

for c in probka.columns:
    if ("OTHER" in c) or ("TEXT" in c) or ("Q34" in c) or ("Q35" in c) or (c in ['Q9','Q9_avg','Q9_BigMacDaily','BigMacIndex', 'Q47_Part_16', 'Q2', 'Time from Start to Finish (seconds)']):
        out_list.append(c)
    else:
        char_list.append(c)     

probka = probka[char_list]

# Categorical features with One-Hot encode
probka, cat_cols = one_hot_encoder(probka, False) 
feats1 = [f for f in probka.columns]
feat_importance1, fpr1, tpr1, roc_score1  = kfold_lightgbm(probka[feats1], num_folds= 4, stratified= False, debug= False, n_estimators = 1000, early_stopping_rounds = 50)

tmp1 = feat_importance1.groupby(['feature'])['importance'].agg(['mean','sum', 'max'])
tmp1.sort_values(by = 'mean', ascending = False).head(10)
feats2 = [f for f in probka.columns if ('Q3' not in f)]
feat_importance2, fpr2, tpr2, roc_score2 = kfold_lightgbm(probka[feats2], num_folds= 4, stratified= False, debug= False, n_estimators = 1000, early_stopping_rounds = 50)

tmp2 = feat_importance2.groupby(['feature'])['importance'].agg(['mean','sum', 'max'])
tmp2.sort_values(by = 'mean', ascending = False).head(10)
f1 = tmp1.sort_values(by = 'mean', ascending = False).head(10)
f2 = tmp2.sort_values(by = 'mean', ascending = False).head(10)

index1 = pd.Index(['Q8_Years of experience','Q2_Age', 'Q3_Live in USA', 'Q24_Expirience in writing code to analyze data', 'Q7_Academics education'
             ,'Q13_Use Notepad++','Q10_We have well established ML methods','Q42_Important is revenue and business goals'
             , 'Q25_How many years have you used machine learning methods','Q11_My role: Build prototypes to explore applying ML to new areas'])
f1 = f1.set_index(index1, inplace = False)

index2 = pd.Index(['Q8_Years of experience','Q2_Age', 'Q24_Expirience in writing code to analyze data', 'Q7_Academics education'
             ,'Q13_Use Notepad++', 'Q25_How many years have you used machine learning methods','Q10_We have well established ML methods'
             ,'Q42_Important is revenue and business goals','Q11_My role: Build prototypes to explore applying ML to new areas'
             ,'Q10_We do not use ML methods'])
f2 = f2.set_index(index2, inplace = False)

plt.subplots(2, 1, figsize = (9, 18), sharey = True, gridspec_kw = {'wspace': 0.05})

plt.subplot(2, 1, 1)

ax = sns.barplot(x='mean', y=f1.index, data=f1, 
                 palette=np.array(['#00cc00', '#00cc00','#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'])            
                )
ax.set_title("Top 10 char - all data",fontsize=18,weight='bold')
ax.set_xlabel("Importance",fontsize=12)
ax.set_ylabel("", fontsize=12)

sns.despine(offset=10, trim=True)

plt.subplot(2, 1, 2)
ax = sns.barplot(x='mean', y=f2.index, data=f2, 
                palette=np.array(['#00cc00', '#00cc00','#00cc00', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8', '#c8c8c8'])            
                )

ax.set_title("Top 10 char - without countries",fontsize=18,weight='bold')
ax.set_xlabel("Importance",fontsize=12)
ax.set_ylabel("", fontsize=12)
sns.despine(offset=10, trim=True)
trace0 = go.Scatter(x=[0, 1], y=[0, 1], 
                    mode='lines', 
                    line=dict(color='lightgray', width=1, dash='dash'),
                    showlegend=False)

trace1 = go.Scatter(x=fpr1, y=tpr1, 
                    mode='lines', 
                    line=dict(color='#00cc00', width=3),
                    name='Model predicting income (All Characteristis)'
                   )

trace2 = go.Scatter(x=fpr2, y=tpr2, 
                    mode='lines', 
                    line=dict(color='#dc0000', width=3),
                    name='Model predicting income <br>(without affecting the place of living in the world)'
                   )

layout = go.Layout(title='<b>ROC Curve</b><br><i>Gini: ' + str(round(100*(2*roc_score1 - 1),1)) +'% vs ' + str(round(100*(2*roc_score2 - 1),1)) +'% </i> ', 
                     height=400, 
                     width=700,
                     showlegend=True,
                     legend=dict(x=0.4, y=0.5),
                     xaxis=dict(title='% of negatives'),
                     yaxis=dict(title='% of positives'),
                    )

fig = go.Figure(data=[trace1, trace2], layout=layout)

iplot(fig)
data_m = prep_data(pd.DataFrame(MultipleResponce[MultipleResponce['Q1']=='Male']), 'Q4', 'Q9_avg')
data_f = prep_data(pd.DataFrame(MultipleResponce[MultipleResponce['Q1']=='Female']), 'Q4', 'Q9_avg')
data_f['mean'] = -1*data_f['mean']

layout = go.Layout(title = '<b>Does the academic title affect your income?</b>',
                   yaxis=go.layout.YAxis(automargin=True),
                   xaxis=go.layout.XAxis(
                       range=[-70, 70],
                       tickvals=[-60, -40, -20, 0, 20, 40, 60],
                       ticktext=[60, 40, 20, 0, 20, 40, 60],
                       title='Avarage yearly income in thousands of $'),
                   barmode='overlay',
                   bargap=0.1, 
                   annotations=[
                    dict(
                        x = -50, y = 5, text='<b>In every group <br>womens earn less</b>',showarrow=False,
                    ),]
                  )

data = [go.Bar(y=data_f['val'],
               x=data_m['mean'],
               orientation='h',
               name='Men',
               hoverinfo='x',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=data_f['val'],
               x=data_f['mean'],
               orientation='h',
               name='Women',
               text=-1*data_f['mean'],
               hoverinfo='text',
               marker=dict(color='pink')
               )]

iplot(dict(data=data, layout=layout)) 

MultipleResponce['Q8_new'] = MultipleResponce['Q8'].apply(lambda x: '0-5' if x in ['0-1','1-2','2-3','3-4','4-5']
                            else ('20+' if x in ['20-25','25-30','30 +'] else x ))

data_m = prep_data(pd.DataFrame(MultipleResponce[MultipleResponce['Q1']=='Male']), 'Q8_new', 'Q9_avg')
data_f = prep_data(pd.DataFrame(MultipleResponce[MultipleResponce['Q1']=='Female']), 'Q8_new', 'Q9_avg')

data_f['mean'] = -1*data_f['mean']

layout = go.Layout(title = '<b>... and work experience?</b>',
                   yaxis=go.layout.YAxis(automargin=True),
                   xaxis=go.layout.XAxis(
                       range=[-150, 150],
                       tickvals=[-120, -80, -40, 0, 40, 80, 120],
                       ticktext=[120, 80, 40, 0, 40, 80, 120],
                       title='Avarage yearly income in thousands of $'),
                   barmode='overlay',
                   bargap=0.1, 
                   annotations=[
                    dict(
                        x = -100, y = 3.8, text='<b>In every group with <br>5+ experience <br>womens earn less',showarrow=False,
                    ),]
                  )

data = [go.Bar(y=data_f['val'],
               x=data_m['mean'],
               orientation='h',
               name='Men',
               hoverinfo='x',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=data_f['val'],
               x=data_f['mean'],
               orientation='h',
               name='Women',
               text=-1*data_f['mean'],
               hoverinfo='text',
               marker=dict(color='pink')
               )]

iplot(dict(data=data, layout=layout)) 
