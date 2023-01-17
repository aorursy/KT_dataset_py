import pandas as pd

import geopandas as gpd

from matplotlib import pyplot as plt

import altair as alt

import seaborn as sns

sns.set(style="darkgrid")
GSI = pd.read_csv('../input/GSI_Private_Projects_Retrofit.csv')

GSW_Reg = pd.read_csv('../input/GSI_Private_Projects_Regs.csv')

GSW_Reg
print(GSI.columns.values)
print(GSW_Reg.columns.values)
len(GSW_Reg.columns)
len(GSI.columns)
GSI.GRANTAMOUNT.isnull().sum()
len(GSI)
vertical_stack = pd.concat([GSI, GSW_Reg], axis=0,sort='none')



vertical_stack

df = vertical_stack

df['Approval_Time'] = df['APPROVALDATE'].str[11:16]

df['APPROVALDATE'] = df['APPROVALDATE'].str[:10]

df
#subset important variables 

df2 = df.drop(['GRANTAMOUNT', 'OBJECTID','TRACKING','TRACKINGNUMBER', 'SMIP','GARP'], axis=1)
projects = df2[['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

       'GREEN_ROOF', 'POROUS_PAVEMENT', 'SUBSURFACE_DETENTION_BASIN',

       'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

       'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE']]



#add a new column for the total projects



projects['Totals'] = ''

#projects['Projects'] =''

projects2 = projects

# pd.melt?

projects2 = pd.melt(projects2, id_vars=[('Totals')], value_vars=['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

        'GREEN_ROOF', 'POROUS_PAVEMENT', 'SUBSURFACE_DETENTION_BASIN',

        'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

        'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE'],var_name='projects', value_name='Project_Counts')



# pd.melt(projects).groupby(id_vars=[('A', 'D')],['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

#        'GREEN_ROOF', 'POROUS_PAVEMENT', 'SUBSURFACE_DETENTION_BASIN',

#        'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

#        'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE'])['Count'].sum().reset_index()

projects2
projects2 = projects2.groupby('projects').sum()[['Project_Counts']]

projects2
projects3 = projects2

projects3 = projects3.reset_index()

projects3['projects'] = projects3['projects'].str.replace('_', ' ')

projects3['projects'] = projects3['projects'].apply(lambda x: x.title())

projects3
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_title('Green Stormwater Infrastructure Projects', color='C0',size=16)



plt.style.use('fivethirtyeight')



ax.hlines(projects3.projects, xmin=0, xmax=projects3['Project_Counts'])

ax.plot(projects3['Project_Counts'], projects3.projects, "o",markersize=10, color='#2ca25f',linewidth=15)





ax.set_ylabel('GSI Projects',fontsize=11)

ax.set_xlabel('Sum of all Projects',fontsize=11)                  

                  



ax.set_xlim(0, 1030)
df3 = df2[['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

       'GREEN_ROOF',  'POROUS_PAVEMENT',

       'PROGRAM', 'SUBSURFACE_DETENTION_BASIN',

       'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

       'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE', 

       'Approval_Time']]



df4 = pd.melt(df3, id_vars=[('PROGRAM')], value_vars=['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

        'GREEN_ROOF', 'POROUS_PAVEMENT', 'SUBSURFACE_DETENTION_BASIN',

        'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

        'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE'],var_name='projects', value_name='Project_Counts')

df4['projects'] = df4['projects'].str.replace('_', ' ')

df4['projects'] = df4['projects'].apply(lambda x: x.title())

df4=df4.sort_values('Project_Counts', ascending=False)

df4



chrt= sns.catplot(x='projects',y='Project_Counts',hue='PROGRAM',linewidth=0.2,kind='bar',ci=None,data=df4)

chrt.set_xticklabels(rotation=90)

chrt.set(xlabel='Project Categories', ylabel='Number of Projects')

chrt._legend.set_title('Program Type')
#need dependencies to render charts 

alt.data_transformers.enable('json')

alt.renderers.enable('kaggle')

df5 = vertical_stack[['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

       'GREEN_ROOF',  'POROUS_PAVEMENT',

       'PROGRAM', 'SUBSURFACE_DETENTION_BASIN',

       'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

       'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE', 

       'APPROVALDATE']]



df6 = pd.melt(df5, id_vars=[('APPROVALDATE')], value_vars=['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

        'GREEN_ROOF', 'POROUS_PAVEMENT', 'SUBSURFACE_DETENTION_BASIN',

        'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

        'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE'],var_name='projects', value_name='Project_Counts')

df6['projects'] = df6['projects'].str.replace('_', ' ')

df6['projects'] = df6['projects'].apply(lambda x: x.title())

#df6=df6.sort_values('APPROVALDATE', ascending=False)

df6


alt.Chart(df6).mark_bar().encode(x=alt.X("APPROVALDATE:O",timeUnit="year", sort='ascending', axis=alt.Axis(title='Years')),y=alt.Y('sum(Project_Counts)',axis=alt.Axis(title='Total Number of Projects')),color='projects',tooltip=['projects:N', 'APPROVALDATE:T', 'sum(Project_Counts):Q']).transform_filter(alt.datum['Project_Counts'] > 0).interactive()

#make a data frame with project categories, names, xy and time

CNXYT_DF = df2[['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

       'GREEN_ROOF',  'POROUS_PAVEMENT',

       'PROGRAM', 'SUBSURFACE_DETENTION_BASIN',

       'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

       'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE', 

       'APPROVALDATE','NAME','X','Y','OVERALLSTATUSCATEGORY']] 



CNXYT_DF2 = pd.melt(CNXYT_DF, id_vars=['PROGRAM','NAME','X','Y','APPROVALDATE','OVERALLSTATUSCATEGORY'], value_vars=['BIOINFILTRATION', 'BIORETENTION', 'CISTERN',

         'GREEN_ROOF', 'POROUS_PAVEMENT', 'SUBSURFACE_DETENTION_BASIN',

         'SUBSURFACE_INFILTRATION_BASIN', 'SURFACE_DETENTION_BASIN',

         'SURFACE_INFILTRATION_BASIN', 'WQ_TREATMENT_DEVICE'],var_name='projects', value_name='Project_Counts')





CNXYT_DF2
alt.Chart(CNXYT_DF2).mark_rect().encode(

    alt.X('Project_Counts:Q', stack='center',

        axis=alt.Axis(title='Number of Projects', tickSize=2)

    ),

    alt.Y('projects:N',axis=None),

      alt.Color('NAME:N',

        scale=alt.Scale(scheme='tableau10')

    ),tooltip=['Project_Counts:Q','NAME:N']

).properties(title='Organizations with highest number of Projects').transform_filter(alt.datum.Project_Counts > 16).interactive()



interval = alt.selection_interval(encodings=['x'])

## totals over the years

big_chart =alt.Chart(df6).mark_line(point=True).encode(x=alt.X("APPROVALDATE:T", timeUnit="year", axis=alt.Axis()), y="sum(Project_Counts):Q", tooltip=['sum(Project_Counts):Q'],

).interactive()



small_chart = alt.Chart(CNXYT_DF2).mark_circle(size=100).encode(alt.X('APPROVALDATE:T',timeUnit='datemonth'),alt.Y('sum(Project_Counts):Q'),alt.X2('OVERALLSTATUSCATEGORY:N'),tooltip=['Project_Counts','OVERALLSTATUSCATEGORY'],shape='OVERALLSTATUSCATEGORY',color=alt.Color('Project_Counts:Q',bin=True,scale=alt.Scale(scheme='plasma'))) 



chart = small_chart.properties(

    width=800,

    height=200

).encode(

    x=alt.X('APPROVALDATE:T',scale=alt.Scale(domain=interval.ref()))

)



view = chart.properties(

    width=800,

    height=60,

    selection=interval

)

chart & view
