#importing libraries
import pandas as pd # For DataFrame Manipulation
import numpy as np #For Data Manipulatoion
import matplotlib.pyplot as plt # For Visualization
import seaborn as sns #For Visualization
import plotly as py  #For Visualization
import plotly.graph_objs as go
## read data from "multipleChoiceResponses.csv" file
data=pd.read_csv('../input/multipleChoiceResponses.csv', skiprows=[1])
country=data['Q3'].value_counts()
country.sort_index(inplace=True)

py.offline.init_notebook_mode(connected=True)

country=pd.DataFrame(country)
country['country']=country.index
count = [ dict(
        type = 'choropleth',
        locations = country['country'],
        locationmode='country names',
        z = country['Q3'],
        text = country['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick =False,
            title = 'Number of Respondents(In Thousands)'),
      ) ]
layout = dict(
    title = 'Number of Respondent from across the Globe',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=count, layout=layout )
py.offline.iplot( fig, validate=False, filename='d3-world-map' )
feq=data['Q2'].value_counts()
total=feq.sum(axis=0)
feq.sort_index(inplace=True)
ax=feq.plot.bar(figsize=(10, 8))
plt.title("Age wise DS", fontsize=20)
plt.xlabel('Age group', fontsize=12)
plt.ylabel('Number of Respondent', fontsize=12)
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()-.03, i.get_height()+.5, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=10, weight='bold',
               color='dimgrey')

feq=data[['Q2', 'Q3']].rename(columns={'Q2':'Age Group', 'Q3':'Country of Residence'})
feq=feq.groupby(['Country of Residence','Age Group'])['Age Group'].size()
feq=pd.DataFrame(feq)
feq['Country/Age Group']=feq.index
feq.rename(columns={'Age Group':'Count'}, inplace=True)
feq['Country'] = feq['Country/Age Group'].str.get(0)
feq['Age Group'] = feq['Country/Age Group'].str.get(1)
feq.drop(['Country/Age Group'], axis=1, inplace=True)
feq1=feq.groupby(['Country'])['Count'].transform(max) == feq['Count']
feq=feq[feq1]
feq.index = np.arange(0, len(feq))

py.offline.init_notebook_mode(connected=True)
count = [ dict(
        type = 'choropleth',
        locations = feq['Country'],
        locationmode='country names',
        z = feq['Count'],
        text = feq['Age Group'],
        colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick =True,
            title = 'Number of Respondents(In Thousands)'),
      ) ]
layout = dict(
    title = 'Age Group with Higher Respondent from Each Country',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=count, layout=layout )
py.offline.iplot( fig, validate=False, filename='d3-world-map' )
#India : 4417 respondent
data_india=data['Q3'].str.contains('India')
data_india=data[data_india]

#USA:  4716 respondent
data_usa=data['Q3'].str.contains('United States of America')
data_usa=data[data_usa]

#China:  1644 respondent
data_china=data['Q3'].str.contains('China')
data_china=data[data_china]

#Russia:  879 respondent
data_russia=data['Q3'].str.contains('Russia')
data_russia=data[data_russia]


#Brazil:  736 respondent
data_brazil=data['Q3'].str.contains('Brazil')
data_brazil=data[data_brazil]

#UK:  702 respondent
data_uk=data['Q3'].str.contains('United Kingdom')
data_uk=data[data_uk]
age_india=data_india.groupby('Q2').size()
age_usa=data_usa.groupby('Q2').size()
age_china=data_china.groupby('Q2').size()
age_russia=data_russia.groupby('Q2').size()
age_brazil=data_brazil.groupby('Q2').size()
age_uk=data_uk.groupby('Q2').size()

age_data=pd.concat([age_india, age_usa, age_china, age_russia, age_brazil, age_uk], axis=1, sort=True)
age_data.columns=['Ind','USA', 'China', 'Russia', 'Brazil', 'UK']
age_data['Total'] = age_data.sum(axis=1)
age_data=age_data.fillna(0)

age_data.drop(['Total'], axis=1).plot(kind='barh', stacked=True, figsize=(20, 15), width=.8)
df_total=age_data['Total']
df_rel = age_data[age_data.columns[0:6]].div(df_total, 0)*100
for n in df_rel:
    for i, (cs, ab, pc, tot) in enumerate(zip(age_data.iloc[:, 0:].cumsum(1)[n], age_data[n], df_rel[n], df_total)):
        plt.text(tot, i, str(tot), va='center', fontsize=12)
        if pc >= 15 and tot>100:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', fontsize=12, rotation=90)
        
        
plt.legend(fontsize=20)
plt.xlabel("Number of Repondent", fontsize=20)
plt.ylabel("Age Group", fontsize=20)
plt.yticks(rotation=45, fontsize=12)
plt.xticks(fontsize=12)
plt.title("Number of Respondent from different Countries (Top 6)", fontsize=25, color='r')
degree=pd.DataFrame(data['Q4'].value_counts())
degree.sort_index(inplace=True)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = degree.index
sizes = degree['Q4']
explode = (0.01,0.01,0.01,0.01, 0.01, 0.01, 0.01)
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=-110, textprops={'weight':'bold'})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Percenatge of Respondent in Different Degree Programs ", fontsize=14, weight='bold')
plt.show()
major_india=age_india=data_india.groupby('Q4').size()
major_usa=data_usa.groupby('Q4').size()
major_china=data_china.groupby('Q4').size()
major_russia=data_russia.groupby('Q4').size()
major_brazil=data_brazil.groupby('Q4').size()
major_uk=data_uk.groupby('Q4').size()
major_data=pd.concat([major_india, major_usa, major_china, major_russia, major_brazil, major_uk], axis=1, sort=True)
major_data.columns=['Ind','USA', 'China', 'Russia', 'Brazil', 'UK']
major_data['Total'] = major_data.sum(axis=1)
major_data=major_data.fillna(0)

major_data.drop(['Total'], axis=1).plot(kind='barh', stacked=True, figsize=(20, 12), width=.8)
df_total=major_data['Total']
df_rel = major_data[major_data.columns[0:6]].div(df_total, 0)*100
for n in df_rel:
    for i, (cs, ab, pc, tot) in enumerate(zip(major_data.iloc[:, 0:].cumsum(1)[n], major_data[n], df_rel[n], df_total)):
        plt.text(tot, i, str(tot), va='center', fontsize=15)
        if pc >= 10 and tot>200:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', fontsize=15, rotation=90)
        
        
plt.legend(fontsize=20)
plt.xlabel("Number of Repondent", fontsize=20)
plt.yticks(rotation=45, fontsize=14, weight='bold')
plt.ylabel("Major Level", fontsize=20)
plt.xticks(fontsize=14, weight='bold')
plt.title("Major Level  of Respondents in Top 6 Countries ", fontsize=25, color='r')
majors_stream=data['Q5'].value_counts()
majors_stream.sort_index(inplace=True)
#majors_stream.sb.barh(figsize=(10,8))
plt.figure(figsize=(10,10))
total=majors_stream.sum(axis=0)
ax=sns.barplot(y=majors_stream.index, x=majors_stream.values)
plt.title("Undergarduate Major of Respondents", fontsize=20, color='r')
plt.xticks( weight='bold')
plt.yticks( weight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='black')
profession= data['Q6'].value_counts()
plt.figure(figsize=(15,8))
tot=profession.sum(axis=0)
ax=sns.barplot(y=profession.index, x=profession.values)
plt.title(" Current Role of the Respondents", fontsize=15)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=12,
color='black')
df=data.groupby(['Q6','Q7'])['Q6'].count()
df=pd.DataFrame(df)
df1=pd.DataFrame(df.index)
df1.columns=['Role/Ind']
df1['Role'] = df1['Role/Ind'].str.get(0)
df1['Industry'] = df1['Role/Ind'].str.get(1)
arr=df['Q6'].values
df=pd.DataFrame(arr)
df.columns=['Count']
frame=[df, df1]
result = pd.concat(frame, axis=1, sort=True)
result.drop(['Role/Ind'], axis=1, inplace=True)

data_heatmap= result.pivot(values='Count', index='Industry', columns='Role')
plt.figure(figsize=(14,10))
plt.xlabel("Role", weight='bold', fontsize=12)
plt.ylabel("Industry", weight='bold', fontsize=12)
plt.title("Profession Of Respondents in Different Industry", fontsize=20, color='blue')
sns.heatmap(data_heatmap, annot=True, linewidths=.5, cmap="YlGnBu", vmin=0, vmax=3800, fmt='g', cbar_kws={'label':'Number of Respondents'})
plt.show()
frame=[data_usa, data_india, data_china, data_russia, data_brazil, data_uk]
cp= pd.concat(frame, axis=0, sort=False)
arr=['Data Scientist','Student', 'Data Analyst', 'Software Engineer','Research Scientist', 'Consultant']
cp=cp.loc[cp['Q6'].isin(arr)]
cp=pd.DataFrame(cp.groupby(['Q3'])['Q6'].value_counts()).rename(columns={'Q6': 'Count'})
cp['Country/Profession']=cp.index
cp.index = np.arange(0, len(cp))
cp['Country'] = cp['Country/Profession'].str.get(0)
cp['Profession'] = cp['Country/Profession'].str.get(1)
cp=cp.drop('Country/Profession', axis=1)
cp=cp.pivot(values='Count', index='Profession', columns='Country')
cp['Total'] = cp.sum(axis=1)


cp.drop(['Total'], axis=1).plot(kind='barh', stacked=True, figsize=(20, 15), width=.8)
df_tot=cp['Total']
df_r =cp[cp.columns[0:6]].div(df_tot, 0)*100
for n in df_r:
    for i, (cs, ab, pc, tot) in enumerate(zip(cp.iloc[:, 0:].cumsum(1)[n], cp[n], df_r[n], df_tot)):
        plt.text(tot, i, str(tot), va='center', fontsize=15)
        if pc >= 5 and tot>10:
            plt.text(cs - ab/2, i, str(np.round(pc, 1)) + '%', va='center', ha='center', fontsize=15, rotation=90)
        
        
plt.legend(fontsize=20)
plt.xlabel("Number of Repondents", fontsize=20)
plt.yticks(rotation=45, fontsize=13, weight='bold')
plt.xticks(fontsize=13, weight='bold')
plt.ylabel("Profession", fontsize=20)
plt.title("Current Role in Top 6 Countries", fontsize=25, color='r')

plt.grid(True)
experience=pd.DataFrame(data['Q8'].value_counts())
experience['Years']=experience.index
experience.index = np.arange(0, len(experience))

compensation=data['Q9'].value_counts()
compensation=pd.DataFrame(compensation)
compensation["Compensation"]=compensation.index
compensation.Compensation.replace('I do not wish to disclose my approximate yearly compensation', 'Undisclosed', inplace=True)
compensation.index = np.arange(0, len(compensation))


fig, ax =plt.subplots(1,2, figsize=(20,7))
a=sns.barplot(y=experience['Years'], x=experience['Q8'],ax=ax[0])
b=sns.barplot(y=compensation['Compensation'], x=compensation['Q9'], ax=ax[1])
a.set_xlabel("Number of respondents",fontsize=14)
b.set_xlabel("Number of respondents",fontsize=14)
a.set_ylabel("Experience",fontsize=14)
b.set_ylabel("Compensation",fontsize=14)
year_in_profession=pd.DataFrame(data.groupby(['Q6','Q8'])['Q8'].count()).rename(columns={'Q6': 'Job ROle', 'Q8':'Count'})
abc=pd.DataFrame(year_in_profession.index)
abc.columns=['Role/Year']
year_in_profession = year_in_profession.reset_index(drop=True)
abc['Job Roles']=abc['Role/Year'].str.get(0)
abc['Experience (Years)']=abc['Role/Year'].str.get(1)
abc.drop(['Role/Year'], axis=1, inplace=True)
frame=[abc, year_in_profession]
year_in_profession = pd.concat(frame, axis=1, sort=True)
ndf = year_in_profession.pivot_table('Count', ['Experience (Years)'], 'Job Roles')
ndf=ndf.fillna(0)

plt.figure(figsize=(15,8))


plt.title("Work Experience in Current Job", fontsize=20, color='blue')
ax=sns.heatmap(ndf, annot=True, linewidths=.5, cmap="BuPu", vmin=0, vmax=2050, fmt='g', cbar_kws={'label': 'Number of Respondents'})
count_comp=pd.DataFrame(data.groupby(['Q3'])['Q9'].value_counts()).rename(columns={'Q9':'Count'})
count_comp['country/compensation']=count_comp.index
count_comp['Country']=count_comp['country/compensation'].str.get(0)
count_comp['Compensation']=count_comp['country/compensation'].str.get(1)
count_comp.drop('country/compensation', axis=1, inplace=True)
count_comp=count_comp[count_comp['Compensation']!='I do not wish to disclose my approximate yearly compensation']
ind=count_comp.groupby(['Country'])['Count'].transform(max) == count_comp['Count']
count_comp=count_comp[ind]
count_comp.index= np.arange(0, len(count_comp))
count_comp

py.offline.init_notebook_mode(connected=True)
abc = [ dict(
        type = 'choropleth',
        locations = count_comp['Country'],
        locationmode='country names',
        z = count_comp['Count'],
        text = count_comp['Compensation'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick =False,
            title = 'Number of Respondents'),
      ) ]
layout = dict(
    title = 'Number of Respondent from across the Globe',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=abc, layout=layout )
py.offline.iplot( fig, validate=False, filename='d3-world-map' )


ml_method=data['Q10'].value_counts(sort=False)
plt.figure(figsize=(10,5))
ml_method.plot.barh()
tools=data['Q12_MULTIPLE_CHOICE'].value_counts(sort=False)

ide=data[['Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7', 'Q13_Part_8',
          'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11', 'Q13_Part_12', 'Q13_Part_13']]
nide=ide.apply(pd.Series.value_counts)
nide=nide.fillna(0)
nide['Number of Respondents']=nide.sum(axis=1)
nide['IDE']=nide.index
nide.index = np.arange(0, len(nide))
nide=nide[1:]



fig, ax =plt.subplots(2,1, figsize=(10,6))
plt.suptitle('Pimary Tools and IDEs Used by Respondents',fontsize=15)
a=sns.barplot(x=tools.values, y=tools.index, ax=ax[0])
b=sns.barplot(x=nide['IDE'], y=nide['Number of Respondents'], ax=ax[1])
b.set_ylabel("Number of respondents",fontsize=12)
a.set_ylabel("Primary Tools to Analyze the Data",fontsize=12)
b.set_xlabel("IDE",fontsize=12)
plt.sca(ax[1])
plt.xticks(rotation=70)

hosted_notebook=data[['Q14_Part_1', 'Q14_Part_2', 'Q14_Part_3', 'Q14_Part_4', 'Q14_Part_5', 'Q14_Part_6', 'Q14_Part_7', 'Q14_Part_8',
          'Q14_Part_9', 'Q14_Part_10', 'Q14_Part_11']]
hosted_notebook=hosted_notebook.fillna('0')
hosted_notebook=hosted_notebook.apply(pd.Series.value_counts)
hosted_notebook['Number of Respondents']=hosted_notebook.sum(axis=1)
hosted_notebook['Hosted Notebook']=hosted_notebook.index
hosted_notebook=hosted_notebook[hosted_notebook['Hosted Notebook']!= 'None']
hosted_notebook.index = np.arange(0, len(hosted_notebook))
hosted_notebook=hosted_notebook[1:]
plt.figure(figsize=(8,8))
ax=sns.barplot(y=hosted_notebook['Hosted Notebook'], x=hosted_notebook['Number of Respondents'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
total=hosted_notebook['Number of Respondents'].sum(axis=0)
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=12,
color='black')
cloud=data[['Q15_Part_1', 'Q15_Part_2', 'Q15_Part_3', 'Q15_Part_4', 'Q15_Part_5', 'Q15_Part_6', 'Q15_Part_7']]
cloud=cloud.apply(pd.Series.value_counts)
cloud['Number of Respondents']=cloud.sum(axis=1)
cloud['Cloud Computing Service Used']=cloud.index
cloud.index = np.arange(0, len(cloud))
cloud=cloud.fillna(0)

py.offline.init_notebook_mode(connected=True)
fig = {
  "data": [
    {
      "values": cloud['Number of Respondents'],
      "labels": cloud['Cloud Computing Service Used'],
      "name": "Cloud Service",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Cloud Computing Service Used",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text":"",
                "x": 0.20,
                "y": 0.5
            }
        ]
    }
}
py.offline.iplot( fig, validate=False, filename='donut' )

pro_lang=pd.DataFrame(data['Q17'].value_counts())
rec_lang=pd.DataFrame(data['Q18'].value_counts()).rename(columns={'Q18':'Count'})


fig, ax =plt.subplots(1,2, figsize=(15,6))
a=sns.barplot(y=pro_lang.index, x=pro_lang['Q17'],ax=ax[0])
b=sns.barplot(y=rec_lang.index, x=rec_lang['Count'], ax=ax[1])
plt.suptitle('Program Languages',fontsize=18)

b.set_xlabel("Number of respondents(k)",fontsize=14)
a.set_xlabel("'Number of Respondents (k)",fontsize=14)
b.set_ylabel("")
a.title.set_text('Used')
b.title.set_text('Recommended')
ml_library=pd.DataFrame(data['Q20'].value_counts()).rename(columns={'Q20':'Count'})
visualization_library=pd.DataFrame(data['Q22'].value_counts()).rename(columns={'Q22':'Count'})

fig, ax =plt.subplots(1,2, figsize=(15,6))
a=sns.barplot(y=ml_library.index, x=ml_library['Count'],ax=ax[0])
b=sns.barplot(y=visualization_library.index, x=visualization_library['Count'], ax=ax[1])
b.set_xlabel("Number of respondents(k)",fontsize=14)
a.set_xlabel("Number of Respondents (k)",fontsize=14)
plt.suptitle("Libraries:", fontsize=16)
a.title.set_text('Machine Learning')
b.title.set_text('Visualization')

active_coding=pd.DataFrame(data['Q23'].value_counts()).rename(columns={'Q23':'Count'})
active_coding['Percent of Time'] = active_coding.index
active_coding.index = np.arange(0, len(active_coding))
active_coding['Percent of Time'] = pd.DataFrame({'active_coding': ['50%-74%', '25%-49%', '1%-25%', '75%-99%', '0%', '100%']})

consider_ds=data['Q26'].value_counts()
py.offline.init_notebook_mode(connected=True)
fig = {
  "data": [
    {
      "values": active_coding['Count'],
      "labels": active_coding['Percent of Time'],
      "domain": {"x": [.52, 1]},
      "name": "Time Spend in coding",
      "hoverinfo":"label+name",
      "hole": .6,
      "type": "pie"
    },
     {
      "values": consider_ds.values,
      "labels": consider_ds.index,
      "domain": {"x": [0, .48]},
      "name": "Data Scientist?",
      "hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    }],
  "layout": {
        "title":"Are you a Data Scientist? Time Spend in Coding?",
        'showlegend': False,
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text":"",
                "x": 0.20,
                "y": 0.5
            }
        ]
    }
}
py.offline.iplot( fig, validate=False, filename='donut' )
%matplotlib inline
exp_vis=pd.DataFrame(data['Q24']).rename(columns={'Q24':'Experience in Analizing data'})
exp_vis=exp_vis.apply(pd.Series.value_counts)
exp_vis['Experience']=exp_vis.index
exp_vis['Experience'].replace(to_replace=['I have never written code but I want to learn', 'I have never written code and I do not want to learn'], value=['0;Want to learn',"0;Don't want to learn"],
                                         inplace=True)
exp_vis.index = np.arange(0, len(exp_vis))
exp_ml=pd.DataFrame(data['Q25']).rename(columns={'Q25':'Experience with ML Libraries'})
exp_ml=exp_ml.apply(pd.Series.value_counts)
exp_ml['Experience']=exp_ml.index
exp_ml['Experience'].replace(to_replace=['I have never studied machine learning but plan to learn in the future', 'I have never studied machine learning and I do not plan to'], value=['0;Plan to learn',"0;Don't want to learn"],
                                         inplace=True)
exp_ml.index = np.arange(0, len(exp_ml))

fig, ax =plt.subplots(1,2, figsize=(6,8))
plt.subplots_adjust(left=-1)
a=sns.barplot(y=exp_vis['Experience'], x=exp_vis['Experience in Analizing data'],ax=ax[0])
b=sns.barplot(y=exp_ml['Experience'], x=exp_ml['Experience with ML Libraries'], ax=ax[1])
a.set_xlabel("Number of respondents",fontsize=12)
b.set_xlabel("Number of respondents",fontsize=12)
a.set_title("Experience in Analizing data",fontsize=14)
b.set_title("Experience with ML Libraries",fontsize=14)
a.set_ylabel("Experience",fontsize=14)
b.set_ylabel("")
plt.sca(ax[1])
plt.yticks(rotation=45)
plt.sca(ax[0])
plt.yticks(rotation=45)

data_type=data['Q32'].value_counts()

plt.figure(figsize=(18,7))
ax=sns.barplot(x=data_type.index, y=data_type.values)
ax.set_ylabel("Number of Respondents",fontsize=16)
ax.set_xlabel("Data Type",fontsize=16)
data_source=data[['Q33_Part_1', 'Q33_Part_2', 'Q33_Part_3', 'Q33_Part_4', 'Q33_Part_5', 'Q33_Part_6', 'Q33_Part_7',
                  'Q33_Part_8', 'Q33_Part_9', 'Q33_Part_10', 'Q33_Part_11']]
data_source=data_source.apply(pd.Series.value_counts)
data_source["Number of Respondents"]=data_source.sum(axis=1)
data_source['Data Source']=data_source.index
data_source.index=np.arange(0,len(data_source))
sns.set_style("darkgrid")
plt.figure(figsize=(10,7))
ax=sns.barplot(y=data_source['Data Source'], x=data_source['Number of Respondents'])
ax.set_xlabel("Number of Respondents",fontsize=16)
ax.set_ylabel("Data Source",fontsize=16)
online_platform=pd.DataFrame(data['Q37'].value_counts())
plt.figure(figsize=(16,8))
sns.set_style("darkgrid")
ax=sns.barplot(y=online_platform.index, x=online_platform['Q37'])        
for p in ax.patches:
    width = p.get_width()
    plt.text(5+p.get_width(), p.get_y()+0.5*p.get_height(),
             width,
             ha='left', va='center')
ax.set_xlabel("Number of respondent", fontsize=12)
ax.set_ylabel("Online Learning Platform", fontsize=12)
plt.show()
media_source=data[['Q38_Part_1', 'Q38_Part_2', 'Q38_Part_3', 'Q38_Part_4', 'Q38_Part_5', 'Q38_Part_6', 'Q38_Part_7',
                  'Q38_Part_8', 'Q38_Part_9', 'Q38_Part_10', 'Q38_Part_11', 'Q38_Part_12', 'Q38_Part_13', 'Q38_Part_14', 
                  'Q38_Part_15', 'Q38_Part_16', 'Q38_Part_17', 'Q38_Part_18']]
media_source=media_source.apply(pd.Series.value_counts)
media_source['Number of Respondents']=media_source.sum(axis=1)
media_source['Media Sources']=media_source.index

media_source.index=np.arange(0,len(media_source))
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax=sns.barplot(y=media_source['Media Sources'], x=media_source['Number of Respondents'])
for p in ax.patches:
    width = p.get_width()
    plt.text(5+p.get_width(), p.get_y()+0.5*p.get_height(),
             width,
             ha='left', va='center')

ax.set_xlabel("Number of Respondents",fontsize=16)
ax.set_ylabel("Media Sources",fontsize=16)
plt.show()