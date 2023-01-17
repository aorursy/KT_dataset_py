from IPython.display import YouTubeVideo

YouTubeVideo('wR9Tsdqgmuk',900,400)
import numpy as np 

import pandas as pd

from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
df=pd.read_csv('../input/washington-state-home-mortgage-hdma2016/Washington_State_HDMA-2016.csv')# load the dataset
df.head()
df['action_taken_name'].value_counts()
df=df[df['action_taken_name']!="Application withdrawn by applicant"]

df=df[df['action_taken_name']!='Loan purchased by the institution']
fips=pd.read_excel("../input/2016-state-county-fips-codes/all-geocodes-v2016.xlsx",converters={'County Code (FIPS)': lambda x: str(x)})

fips=fips[fips["State Code (FIPS)"]==53]# state code for washington

fips['county_code']=fips['State Code (FIPS)'].astype(str).str.cat(fips['County Code (FIPS)'].astype(str))

fips=fips.drop(labels=['Summary Level','State Code (FIPS)','County Code (FIPS)','County Subdivision Code (FIPS)','Place Code (FIPS)','Consolidtated City Code (FIPS)'],axis=1)

fips.columns=['county_name','county_code']
df=pd.merge(df,fips,how="left",on="county_name",sort=False)#merging it into original dataset
county1=pd.DataFrame(df['county_code'].value_counts())

county1=county1.reset_index()

county1.columns=['county_code','number of loans']#renaming columns
fips=county1['county_code'].tolist()

values=county1['number of loans'].tolist()

endpts = list(np.mgrid[min(values):max(values):13j])



colorscale = ["#141d43","#15425a","#0a6671","#26897d","#67a989","#acc5a6","#e0e1d2",

              "#f0dbce","#e4ae98","#d47c6f","#bb4f61","#952b5f","#651656","#330d35"] 



fig = ff.create_choropleth(

    fips=fips, values=values, scope=['Washington'], show_state_data=True,

    colorscale=colorscale, binning_endpoints=endpts, round_legend_values=True,

    plot_bgcolor='rgb(229,229,229)',

    paper_bgcolor='rgb(229,229,229)',

    legend_title='Number of Loans by county',

    county_outline={'color': 'rgb(255,255,255)', 'width': 0.2}, exponent_format=False

)

iplot(fig, filename='loans_washington')
df['loan_status']=["approved" if x=="Loan originated" else "not approved" for x in df['action_taken_name']]
df_approved=df[df['loan_status']=='approved']

df_notapproved=df[df['loan_status']=='not approved']
county2=pd.DataFrame(df_approved['county_code'].value_counts())

county2=county2.reset_index()

county2.columns=['county_code','number of loans approved']

county2=pd.merge(county2,county1,how="left",on="county_code",sort=False)

l=[]

for x in range(county2.shape[0]):

    l.append(county2['number of loans approved'][x]/county2['number of loans'][x])

county2['approval rate']=[x*100 for x in l]
fips=county2['county_code'].tolist()

values=county2['approval rate'].tolist()

endpts = list(np.mgrid[min(values):max(values):13j])

colorscale = ["#141d43","#15425a","#0a6671","#26897d","#67a989","#acc5a6","#e0e1d2",

              "#f0dbce","#e4ae98","#d47c6f","#bb4f61","#952b5f","#651656","#330d35"]

fig = ff.create_choropleth(

    fips=fips, values=values, scope=['Washington'], show_state_data=True,

    colorscale=colorscale, binning_endpoints=endpts, round_legend_values=True,

    plot_bgcolor='rgb(229,229,229)',

    paper_bgcolor='rgb(229,229,229)',

    legend_title='Number of approved Loans by county',

    county_outline={'color': 'rgb(255,255,255)', 'width': 0.2},

    exponent_format=True,

)

iplot(fig, filename='approved_loans_washington')
df_purpose=pd.crosstab(df['loan_purpose_name'],df['loan_status'])

df_purpose=df_purpose.reset_index()

df_purpose.columns=['purpose','approved count','not approved count']

l=[]

for x in range(3):

    l.append(df_purpose['approved count'][x]/(df_purpose['approved count'][x]+ df_purpose['not approved count'][x]))

df_purpose['percent approved']=[x*100 for x in np.array(l)]

df_purpose['percent not approved']=[100-x for x in df_purpose['percent approved']]
trace1=go.Bar(

x= df_purpose['purpose'],

y= df_purpose['approved count'],

name='approved',

marker=dict(

    color='#009393'))

trace2=go.Bar(

x= df_purpose['purpose'],

y=df_purpose['not approved count'],

name='not approved',

marker=dict(

        color='#930000'))

trace3=go.Bar(

x=df_purpose['purpose'],

y=df_purpose['percent approved'],

name='percent approved',

marker=dict(

    color='#8eb48b'))

trace4=go.Bar(

x=df_purpose['purpose'],

y=df_purpose['percent not approved'],

name="percent not approved",

marker=dict(

        color='#7fc780'))



fig = tools.make_subplots(rows=1, cols=2,subplot_titles=('Approved loans for different purposes','Percent of loans approved for differnet purposes'))

fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,1)

fig.append_trace(trace3,1,2)

fig.append_trace(trace4,1,2)



fig['layout'].update(height=600, width=900,barmode='stack')

iplot(fig)
df_type=pd.crosstab(df['loan_type_name'],df['loan_status'])

df_type=df_type.reset_index()

df_type.columns=['type','approved count','not approved count']

l=[]

for x in range(4):

    l.append(df_type['approved count'][x]/(df_type['approved count'][x]+ df_type['not approved count'][x]))

df_type['percent approved']=[x*100 for x in l]

df_type['percent not approved']=[100-x for x in df_type['percent approved']]
trace1 = {"x":df_type['percent approved'] ,

          "y": df_type['type'] ,

          "marker": {"color": "rgba(255, 182, 193, .9)", "size": 20},

          "mode": "markers",

          "name": "percent approved",

          "type": "scatter"

}



trace2 = {"x": df_type['percent not approved'],

          "y": df_type['type'],

          "marker": {"color": "rgba(152, 0, 0, .8)", "size": 20},

          "mode": "markers",

          "name": "percent not approved",

          "type": "scatter",

}

data = [trace1, trace2]

layout = go.Layout(title="Loan Status for different type of loans",

                  height=500,

                  width=700,

                  autosize=False,

                  margin=go.layout.Margin(

        l=150,

        r=50,

        b=100,

        t=100,

        pad=4

    ))



fig = go.Figure(data=data, layout=layout)

iplot(fig)#let's plot
sns.set(style="white", palette="deep", font_scale=1.2, 

        rc={"figure.figsize":(15,9)})

ax = sns.scatterplot(x="loan_amount_000s", y="applicant_income_000s", hue="loan_status",data=df)
df_temp=df[df['loan_amount_000s']<20000]

ax = sns.scatterplot(x="loan_amount_000s", y="applicant_income_000s", hue="loan_status",data=df_temp)
df_owner=pd.crosstab(df['owner_occupancy_name'],df['loan_status'])

df_owner=df_owner.reset_index()

df_owner.columns=['owner_occupancy','approved','not approved']

l=[]

for x in range(3):

    l.append(df_owner['approved'][x]/(df_owner['approved'][x]+ df_owner['not approved'][x]))

df_owner['percent approved']=[x*100 for x in l]

df_owner['percent not approved']=[100-x for x in df_owner['percent approved']]
df_hoepa=pd.crosstab(df['hoepa_status_name'],df['loan_status'])

df_hoepa=df_hoepa.reset_index()

df_hoepa.columns=['hoepa_status','approved','not approved']

l=[]

for x in range(2):

    l.append(df_hoepa['approved'][x]/(df_hoepa['approved'][x]+ df_hoepa['not approved'][x]))

df_hoepa['percent approved']=[x*100 for x in l]

df_hoepa['percent not approved']=[100-x for x in df_hoepa['percent approved']]
trace1=go.Bar(

x= df_owner['owner_occupancy'],

y= df_owner['percent approved'],

name='percent approved',

marker=dict(

    color='rgb(158,202,225)'))

trace2=go.Bar(

x= df_owner['owner_occupancy'],

y=df_owner['percent not approved'],

name='percent not approved',

marker=dict(

        color='rgba(219, 64, 82, 0.7)'))

trace3=go.Bar(

x=df_hoepa['hoepa_status'],

y=df_hoepa['percent approved'],

name='percent approved',

marker=dict(

    color='rgba(204,204,204,1)'))

trace4=go.Bar(

x=df_hoepa['hoepa_status'],

y=df_hoepa['percent not approved'],

name="percent not approved",

marker=dict(

        color='rgba(222,45,38,0.8)'))



fig = tools.make_subplots(rows=1, cols=2,subplot_titles=('Approval % for owner occupancy','Approval % for HOEPA Status'))

fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,1)

fig.append_trace(trace3,1,2)

fig.append_trace(trace4,1,2)



fig['layout'].update(height=600, width=900,barmode='group')

iplot(fig)
df['hud_median_family_income_000s']=[x/1000 for x in df['hud_median_family_income']]

df_approved['hud_median_family_income_000s']=[x/1000 for x in df_approved['hud_median_family_income']]

df_notapproved['hud_median_family_income_000s']=[x/1000 for x in df_notapproved['hud_median_family_income']]

approved_msamd_diff=df_approved.groupby('msamd_name').mean()

not_approved_msamd_diff=df_notapproved.groupby('msamd_name').mean()
trace0 = go.Scatter(

    x = approved_msamd_diff.index,

    y = approved_msamd_diff['hud_median_family_income_000s'],

    mode = 'lines+markers',

    name = 'Neighbourhood median family income',

    line = dict(

        color = "#009393")

)

trace1 = go.Scatter(

    x = approved_msamd_diff.index,

    y = approved_msamd_diff['applicant_income_000s'],

    mode = 'lines+markers',

    name = 'Applicant income',

    line= dict( color= "#230405")

)

data=[trace0,trace1]

layout = dict(title = 'Difference in neighborhood median family income and applicant income for approved loans  ',

              xaxis = dict(title = 'MSA/MD'),

              yaxis = dict(title = 'Income'),

              margin=go.layout.Margin(

        l=50,

        r=50,

        b=200,

        t=100,

        pad=4

    )

              )



fig = dict(data=data, layout=layout)



iplot(fig)
trace0 = go.Scatter(

    x = not_approved_msamd_diff.index,

    y = not_approved_msamd_diff['hud_median_family_income_000s'],

    mode = 'lines+markers',

    name = 'Neighbourhood median family income',

    line = dict(

        color = "#009393")

)

trace1 = go.Scatter(

    x = not_approved_msamd_diff.index,

    y = not_approved_msamd_diff['applicant_income_000s'],

    mode = 'lines+markers',

    name = 'Applicant income',

    line= dict( color= "#230405")

)

data=[trace0,trace1]

layout = dict(title = 'Difference for the loans not approved',

              xaxis = dict(title = 'MSA/MD'),

              yaxis = dict(title = 'Income'),

               margin=go.layout.Margin(

        l=50,

        r=50,

        b=200,

        t=100,

        pad=4

    )

              )



fig = dict(data=data, layout=layout)



iplot(fig)
df_property=pd.crosstab(df['property_type_name'],df['loan_status'])

df_property=df_property.reset_index()



l=[]

for x in range(df_property.shape[0]):

    l.append(df_property['approved'][x]/(df_property['approved'][x]+ df_property['not approved'][x]))

df_property['percent approved']=[x*100 for x in l]

df_property['percent not approved']=[100-x for x in df_property['percent approved']]

df_property['property_type_name']=df_property['property_type_name'].replace("One-to-four family dwelling (other than manufactured housing)",'1-4 Family dwelling')

trace1 = go.Bar(

    y=df_property['property_type_name'],

    x=df_property['percent approved'],

    name='percent approved',

    orientation = 'h',

    marker = dict(

        color = '#7bc043 '

        

    )

)

trace2 = go.Bar(

    y=df_property['property_type_name'],

    x=df_property['percent not approved'],

    name='percent not approved',

    orientation = 'h',

    marker = dict(

        color = '#fdf498 '

       

    )

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='stack',

    title="Effect of Property Type",

    

    margin=go.layout.Margin(

        l=200,

        r=50,

        b=100,

        t=100,

        pad=4

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace0 = go.Scatter(

    x = approved_msamd_diff.index,

    y = approved_msamd_diff['tract_to_msamd_income'],

    mode = 'lines+markers',

    name = 'approved',

    line = dict(

        color = "#a77d5f")

)

trace1 = go.Scatter(

    x = not_approved_msamd_diff.index,

    y = not_approved_msamd_diff['tract_to_msamd_income'],

    mode = 'lines+markers',

    name = 'not approved',

    line= dict( color= "#930000")

)

data=[trace0,trace1]

layout = dict(title = '',

              xaxis = dict(title = 'MSA/MD'),

              yaxis = dict(title = 'tract_to_msamd_income'),

              margin=go.layout.Margin(

        l=50,

        r=50,

        b=200,

        t=100,

        pad=4

    )

              )



fig = dict(data=data, layout=layout)



iplot(fig)
df_lien=pd.crosstab(df['lien_status_name'],df['loan_status'])

df_lien=df_lien.reset_index()

df_lien.columns=['lien_status','approved','not approved']

l=[]

for x in range(3):

    l.append(df_lien['approved'][x]/(df_lien['approved'][x]+ df_lien['not approved'][x]))

df_lien['percent approved']=[x*100 for x in l]

df_lien['percent not approved']=[100-x for x in df_lien['percent approved']]
trace1 = go.Bar(

    y=df_lien['lien_status'],

    x=df_lien['percent approved'],

    name='percent approved',

    orientation = 'h',

    marker = dict(

        color = 'rgba(71, 58, 131, 0.8)',

        line = dict(

            color = 'rgba(38, 24, 74, 0.8)',

            width = 3)

    )

)

trace2 = go.Bar(

    y=df_lien['lien_status'],

    x=df_lien['percent not approved'],

    name='percent not approved',

    orientation = 'h',

    marker = dict(

        color = 'rgba(190, 192, 213, 1)',

        line = dict(

            color = 'rgba(164, 163, 204, 0.85)',

            width = 3)

    )

)



data = [trace1, trace2]

layout = go.Layout(

    barmode='group',

    title="Effect of lien status",

    margin=go.layout.Margin(

        l=200,

        r=50,

        b=100,

        t=100,

        pad=4

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
df['applicant_income_range'] = np.nan

l = [df]

for col in l:

    col.loc[col['applicant_income_000s'] <= 100, 'applicant_income_range'] = 'Low'

    col.loc[(col['applicant_income_000s'] > 100) & (col['applicant_income_000s'] <= 200), 'applicant_income_range'] = 'Medium'

    col.loc[col['applicant_income_000s'] > 200, 'applicant_income_range'] = 'High'
df_approved=df[df['loan_status']=='approved']

df_notapproved=df[df['loan_status']=='not approved']
trace0 = go.Box(

    y=df_approved['loan_amount_000s'],

    x=df_approved['applicant_income_range'],

    name='approved',

    marker=dict(

        color='#3D9970'

    )

)

trace1 = go.Box(

    y=df_notapproved['loan_amount_000s'],

    x=df_notapproved['applicant_income_range'],

    name='not approved',

    marker=dict(

        color='#FF4136'

    )

)

data = [trace0, trace1]

layout = go.Layout(

    yaxis=dict(

        title='',

        zeroline=False

    ),

    boxmode='group'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_approved1=df_approved[df_approved['loan_amount_000s']<1500]

df_notapproved1=df_notapproved[df_notapproved['loan_amount_000s']<1500]
trace0 = go.Box(

    y=df_approved1['loan_amount_000s'],

    x=df_approved1['applicant_income_range'],

    name='approved',

    marker=dict(

        color='#3D9970'

    )

)

trace1 = go.Box(

    y=df_notapproved1['loan_amount_000s'],

    x=df_notapproved1['applicant_income_range'],

    name='not approved',

    marker=dict(

        color='#FF4136'

    )

)

data = [trace0, trace1]

layout = go.Layout(

    yaxis=dict(

        title='',

        zeroline=False

    ),

    boxmode='group'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_sex=pd.crosstab(df['applicant_sex_name'],df['loan_status'])

df_sex=df_sex.reset_index()

df_sex.columns=['sex','approved','not approved']

l=[]

for x in range(df_sex.shape[0]):

    l.append(df_sex['approved'][x]/(df_sex['approved'][x]+ df_sex['not approved'][x]))

df_sex['percent approved']=[x*100 for x in np.array(l)]

df_sex['percent not approved']=[100-x for x in df_sex['percent approved']]

df_sex['sex']=df_sex['sex'].replace('Information not provided by applicant in mail, Internet, or telephone application','Info not provided')
df_ethnicity=pd.crosstab(df['applicant_ethnicity_name'],df['loan_status'])

df_ethnicity=df_ethnicity.reset_index()

df_ethnicity.columns=['ethnicity','approved','not approved']

l=[]

for x in range(df_ethnicity.shape[0]):

    l.append(df_ethnicity['approved'][x]/(df_ethnicity['approved'][x]+ df_ethnicity['not approved'][x]))

df_ethnicity['percent approved']=[x*100 for x in np.array(l)]

df_ethnicity['percent not approved']=[100-x for x in df_ethnicity['percent approved']]

df_ethnicity['ethnicity']=df_ethnicity['ethnicity'].replace('Information not provided by applicant in mail, Internet, or telephone application','Info not provided')
df_race=pd.crosstab(df['applicant_race_name_1'],df['loan_status'])

df_race=df_race.reset_index()

df_race.columns=['race','approved','not approved']

l=[]

for x in range(df_race.shape[0]):

    l.append(df_race['approved'][x]/(df_race['approved'][x]+ df_race['not approved'][x]))

df_race['percent approved']=[x*100 for x in np.array(l)]

df_race['percent not approved']=[100-x for x in df_race['percent approved']]

df_race['race']=df_race['race'].replace('Information not provided by applicant in mail, Internet, or telephone application','Info not provided')
trace0=go.Bar(

x=df_ethnicity['ethnicity'],

y=df_ethnicity['percent approved'],

name='percent approved',

marker=dict(color='#051e3e '))

trace1=go.Bar(x=df_ethnicity['ethnicity'],

              y=df_ethnicity['percent not approved'],

              name='percent not approved',

             marker=dict(

             color='#851e3e '))

trace2=go.Bar(x=df_sex['sex'],

             y=df_sex['percent approved'],

             name='percent approved',

             marker=dict(color='#96ceb4  '))

trace3=go.Bar(x=df_sex['sex'],

             y=df_sex['percent not approved'],

             name='percent not approved',

             marker=dict(color='#ff6f69  '))

trace4=go.Scatter(x=df_race['race'],

                 y=df_race['percent approved'],

                 name='percent approved')

trace5=go.Scatter(x=df_race['race'],

                 y=df_race['percent not approved'],

                 name='percent not approved')

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Applicant Race','Applicant sex', 'Applicant ethnicity'))



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 2, 1)

fig.append_trace(trace5, 2, 1)



fig['layout'].update( height=900,width=1000,paper_bgcolor = "rgb(255, 248, 243)",margin=go.layout.Margin(

        l=50,

        r=50,

        b=200,

        t=100,

        pad=4

    ))

iplot(fig)
df_reason=pd.DataFrame(df_notapproved['denial_reason_name_1'].value_counts())

df_reason=df_reason.reset_index()

df_reason.columns=['reason','number of loans']
trace0 = go.Bar(

    x=df_reason['reason'],

    y=df_reason['number of loans'],

    

    marker=dict(

        color='rgb(158,202,225)',

        line=dict(

            color='rgb(8,48,107)',

            width=1.5,

        )

    ),

    opacity=0.6

)



data = [trace0]

layout = go.Layout(

    title='Major loan denial reasons',height=500,

                  width=700,

                  autosize=False,

                  margin=go.layout.Margin(

        l=50,

        r=50,

        b=200,

        t=100,

        pad=4

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)