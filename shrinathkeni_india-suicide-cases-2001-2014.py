!pip install chart_studio

import pandas as pd

import numpy as np

import seaborn as snd

import matplotlib.pyplot as plt

%matplotlib inline

import chart_studio.plotly as py

from plotly import tools

#from plotly.offline import init_notebook_mode,iplot

#init_notebook_mode(connected=False)

import plotly.figure_factory as ff

import plotly.graph_objects as go

import plotly.express as px 

#help(go.Figure.write_html)

from plotly.subplots import make_subplots

import math
suicide_india_data=pd.read_csv("../input/suicide-india-dataset/Suicides_by_causes_state.csv")

suicide_india_data_2018=pd.read_csv("../input/suicide-india-dataset-2018/NCRB-ADSI-2018.csv")
#Removing Null values

suicide_india_data.isna().sum()

suicide_india_data=suicide_india_data.dropna()
#Removing Total (Uts), Total (All India) , Total (States) from  States column

suicide_india_data=suicide_india_data[~(suicide_india_data["STATE/UT"].isin(["Total (States)","Total (Uts)","Total (All India)",]))]
#Replacing State names because of name conventions 

suicide_india_data=suicide_india_data.replace({"A And N Islands":"A & N Islands"},regex=True)

suicide_india_data=suicide_india_data.replace({"Dandn Haveli":"D & N Haveli"},regex=True)

suicide_india_data=suicide_india_data.replace({"Delhi Ut":"Delhi (Ut)"},regex=True)

suicide_india_data=suicide_india_data.replace({"Jammu And Kashmir":"Jammu & Kashmir"},regex=True)

suicide_india_data=suicide_india_data.replace({"Daman And Diu":"Daman & Diu"},regex=True)
#Removing "Total" and "Total Illness" from  CAUSE Column And Grouping Some "Suicide Causes" which belong in the same Category but are different due to name conventions



## Removing "Total" and "Total Illness" from  CAUSE Column

suicide_india_data=suicide_india_data[~(suicide_india_data["CAUSE"].isin(["Total","Total Illness"]))]



##Grouping Some "Suicide Causes" which belong in the same Category but are different due to name conventions

suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(['Bankruptcy or.*'], 'Bankrupt & Indebtness'),regex=True)



suicide_india_data=suicide_india_data.replace(

dict.fromkeys(["Cancellation.*","Non Settlement.*","Extra.*","Dowry.*","Divorce"],"Marriage Related Issues"),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(['Other Prolonged Illness',"Illness","Other prolonged illness"], 'Other Illness'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(['Suspected/.*',"Love.*"], 'Love Affairs'),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Illness (Aids/STD)"], 'AIDS/STD'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Not having Children(Barrenness/Impotency","Impotency/Infertility","Not having Children (Barrenness/Impotency"], 'Barrenness/Impotency/Infertility'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Physical Abuse.*"], 'Physical Abuse/Rape'),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Drug.*"], 'Drug Abuse'),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Death.*"], 'Death of Dear Person'),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Failuer.*"], 'Failure in Examination'),regex=True)



#suicide_india_data=suicide_india_data.replace(

 #   dict.fromkeys(["Other Causes (Please Specity)","Other causes","Others"], 'Other Causes'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Other Family problems"], 'Family Problems'),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Other Family problems"], 'Family Problems'),regex=True)



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Fall in social reputation","Fall in Social Reputation"], 'Social Reputation'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Insanity/Mental illness"], 'Insanity/Mental Illness'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Ideological causes/Hero worshipping"], 'Ideological Causes/Hero Worshipping'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Causes not known","Causes Not known","Other Causes (Please Specity)","Other causes","Others"], 'Unknown Causes'))



suicide_india_data=suicide_india_data.replace(

    dict.fromkeys(["Property dispute"], 'Property Dispute'))

#Data Pre Processing of data set of the year 2018



##Changing columns into rows

suicide_india_data_2018_new=suicide_india_data_2018.melt(id_vars=["Sl. No.","Category", "State/UT/City"], 

        var_name="CAUSE", 

        value_name="COUNT")
#Dropping columns

suicide_india_data_2018_new=suicide_india_data_2018_new.drop(["Sl. No.","Category"],axis=1)

#Removing rows which contains the string Total in CAUSE columns

suicide_india_data_2018_new=suicide_india_data_2018_new[~suicide_india_data_2018_new.CAUSE.str.contains("Total")]
#Changing name conventions and grouping similiar Causes into same Category



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Marriage.*"],"Marriage Related Issues"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Bankruptcy.*"],"Bankruptcy or Indebtedness"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Ideological.*"],"Ideological Causes/Hero Worshipping"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Impotency/Infertility.*"],"Impotency/Infertility"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Unemployment.*"],"Unemployment"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Illegitimate.*"],"Illegitimate Pregnancy"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Drug.*"],"Drug Abuse/Alcoholic Addiction"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Family.*"],"Family Problems"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Failure.*"],"Failure in Examination"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Fall.*"],"Failure in Examination"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Love.*","Suspected.*"],"Love Affairs"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Poverty.*"],"Poverty"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Physical.*"],"Physical Abuse (Rape, etc.)"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Professional.*"],"Professional/Career Problem"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Property Dispute.*"],"Property Dispute"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Illness.*(Cancer).*"],"Cancer"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Illness.*(AIDS/STD).*"],"AIDS/STD"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Illness.*(Other Prolonged Illness).*"],"Other Prolonged Illness"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Illness.*(Insanity/Mental illness).*"],"Insanity/Mental illness"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Illness.*(Paralysis).*"],"Paralysis"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Death.*"],"Death of Dear Person"),regex=True)



suicide_india_data_2018_new=suicide_india_data_2018_new.replace(

dict.fromkeys(["Causes Not Known.*","Other Causes.*"],"Unknown Causes"),regex=True)

suicide_india_data_2018_new_sum=suicide_india_data_2018_new.agg([sum]).reset_index()
suicide_india_data_2018_new_sum=suicide_india_data_2018_new_sum.drop(["State/UT/City","CAUSE"],axis=1)



suicide_india_data_2018_new_sum=suicide_india_data_2018_new_sum.rename(columns ={"COUNT":"Total Cases"})

suicide_india_data_2018_new_sum=suicide_india_data_2018_new_sum.rename(columns ={"index":"Year"})

suicide_india_data_2018_new_sum=suicide_india_data_2018_new_sum.replace({"sum":"2018"},regex=True)

suicide_year_data=suicide_india_data[["Year"]]

suicide_year_data["Total Cases"]=suicide_india_data["Grand Total"]

pd.set_option('precision', 0)

suicide_year_data=suicide_year_data.groupby(["Year"]).sum().reset_index()

suicide_year_data.head(20)







fig = px.bar(suicide_year_data, x="Year", y='Total Cases',color="Total Cases",text="Total Cases",color_continuous_scale=px.colors.sequential.Viridis)



fig.update_traces( textposition='outside')

fig.update_layout(title_text='Suicide Cases From 2001-2014')

fig.update_xaxes(dtick=1)

fig.show()
suicide_gender=suicide_india_data[["Total Male","Total Female"]].sum().rename_axis('Sex').reset_index()

suicide_gender=suicide_gender.rename(columns ={0:"Suicide Count"})

suicide_gender=suicide_gender.replace({"Total Male":"Male"},regex=True)

suicide_gender=suicide_gender.replace({"Total Female":"Female"},regex=True)
suicide_gender.head()
fig = px.bar(suicide_gender,x="Sex",y="Suicide Count",text="Suicide Count",color="Sex",title="Male vs Female Suicide Count")

fig.update_traces( textposition='inside',insidetextanchor = "middle")

fig.show()
suicide_gender_data=suicide_india_data[["Year","Total Male","Total Female"]]

suicide_gender_data=suicide_gender_data.groupby(["Year"]).sum()

#suicide_gender_data.set_index("Year")

suicide_gender_data.head(20)
suicide_gender_data=suicide_gender_data.reset_index()
# Setting Bars

bar1=suicide_gender_data["Total Male"]

bar2=suicide_gender_data["Total Female"]





fig = go.Figure()

fig.add_trace(go.Bar(

    x=suicide_gender_data["Year"],

    y=bar1,

    text =bar1,

    name='Male',

    marker_color='#003049'

    

))

fig.add_trace(go.Bar(

    x=suicide_gender_data["Year"],

    y=bar2,

    text=bar2,

    name='Female',

    marker_color='#fcbf49'

   

))



fig.update_traces(textposition='auto')



fig.update_yaxes(

        title_text = "Suicide Count")



fig.update_xaxes(

        title_text = "Years",dtick=1)



fig.update_layout(barmode='group',title_text='Suicide Cases(Male vs Female) From 2001-2014')



fig.show()
suicide_age_male_sum=suicide_india_data[["Male upto 14 years","Male 15-29 years","Male 30-44 years","Male 45-59 years","Male 60 years and above"]].sum().rename_axis("Age Group (Male)").reset_index()

suicide_age_female_sum=suicide_india_data[["Female upto 14 years","Female 15-29 years","Female 30-44 years","Female 45-59 years","Female 60 years and above"]].sum().rename_axis("Age Group (Female)").reset_index()
# Male Data

#suicide_age_male_sum=suicide_age_male_sum

suicide_age_male_sum=suicide_age_male_sum.rename(columns ={0:"Suicide Count"})

suicide_age_male_sum=suicide_age_male_sum.replace({"Male upto 14 years":"Upto 14 years"},regex=True)

suicide_age_male_sum=suicide_age_male_sum.replace({"Male 15-29 years":"15-29 years"},regex=True)

suicide_age_male_sum=suicide_age_male_sum.replace({"Male 30-44 years":"30-44 years"},regex=True)

suicide_age_male_sum=suicide_age_male_sum.replace({"Male 45-59 years":"45-59 years"},regex=True)

suicide_age_male_sum=suicide_age_male_sum.replace({"Male 60 years and above":"60 years and above"},regex=True)

suicide_age_male_sum.head()
# Female Data

suicide_age_female_sum=suicide_age_female_sum.rename(columns ={0:"Suicide Count"})

suicide_age_female_sum=suicide_age_female_sum.replace({"Female upto 14 years":"Upto 14 years"},regex=True)

suicide_age_female_sum=suicide_age_female_sum.replace({"Female 15-29 years":"15-29 years"},regex=True)

suicide_age_female_sum=suicide_age_female_sum.replace({"Female 30-44 years":"30-44 years"},regex=True)

suicide_age_female_sum=suicide_age_female_sum.replace({"Female 45-59 years":"45-59 years"},regex=True)

suicide_age_female_sum=suicide_age_female_sum.replace({"Female 60 years and above":"60 years and above"},regex=True)

suicide_age_female_sum.head()
label1 = suicide_age_male_sum["Age Group (Male)"]

label2= suicide_age_female_sum["Age Group (Female)"]



value1= suicide_age_male_sum["Suicide Count"]

value2= suicide_age_female_sum["Suicide Count"]



# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels= label1, values= value1, name="Male",sort=False),

              1, 1)

fig.add_trace(go.Pie(labels= label2, values= value2, name="Female",sort=False),

              1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.45, hoverinfo="label+percent+name+value",textinfo="value",rotation = 90,textposition="inside",

                 marker=dict(colors =  ['yellow' ,'darkblue','red','royalblue','lime'],

                     line=dict(color='black',width=1.0)))



fig.update_layout(

    title_text="Total Suicide cases for different Age Group",legend_traceorder="normal",legend_title="Age Group",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Male', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Female', x=0.82, y=0.5, font_size=20, showarrow=False)])





fig.show()
suicide_age_male=suicide_india_data[["Year","Male upto 14 years","Male 15-29 years","Male 30-44 years","Male 45-59 years","Male 60 years and above"]]
suicide_age_male=suicide_age_male.groupby(["Year"]).sum().reset_index()

suicide_age_male.head(20)


Count=["Male upto 14 years", "Male 15-29 years", "Male 30-44 years","Male 45-59 years","Male 60 years and above"]





fig = px.bar(suicide_age_male,x="Year", y=Count,title="Male Sucide Cases of All age Group",color_discrete_sequence=px.colors.qualitative.G10,labels={'variable':'Age Group',"value":"Count"})





fig.update_xaxes(

        title_text = "Years",dtick=1,)



fig.update_yaxes(

        title_text = "Count",tickformat="d")



fig.update_layout(legend_title= "Male Age Group")



fig.show()
suicide_age_female=suicide_india_data[["Year","Female upto 14 years","Female 15-29 years","Female 30-44 years","Female 45-59 years","Female 60 years and above"]]
suicide_age_female=suicide_age_female.groupby(["Year"]).sum().reset_index()

suicide_age_female.head(20)
y2=["Female upto 14 years", "Female 15-29 years", "Female 30-44 years","Female 45-59 years","Female 60 years and above"]



fig = px.bar(suicide_age_female,x="Year", y=y2, title="Female Sucide Cases of All age Group",color_discrete_sequence=px.colors.qualitative.G10,labels={'variable':'Age Group',"value":"Count"})



fig.update_xaxes(

        title_text = "Years",dtick=1,)



fig.update_yaxes(

        title_text = "Count",tickformat="d")



fig.update_layout(legend_title="Female Age Group")



fig.show()

suicide_state_data=suicide_india_data[["STATE/UT","Grand Total"]]

suicide_state_data=suicide_state_data.rename(columns ={"Grand Total":"Total Suicide Count"})
suicide_state_data=suicide_state_data.groupby(["STATE/UT"]).sum().sort_values(by="Total Suicide Count",ascending=False).head(20)
suicide_state_data.head(10)
suicide_state_data=suicide_state_data.reset_index()

fig = px.bar(suicide_state_data, y='STATE/UT', x='Total Suicide Count',color="Total Suicide Count",text="Total Suicide Count",color_continuous_scale=px.colors.sequential.Plasma_r,orientation="h")

fig.update_traces( textposition='outside',cliponaxis = False)

fig.update_layout(title_text='Suicide Cases in each State From 2001-2014')

#fig.update_xaxes(dtick=1)

fig.update_yaxes(autorange="reversed")

#yaxis=dict(autorange="reversed")

fig.show()
suicide_cause_data=suicide_india_data[["CAUSE","Grand Total"]]
suicide_cause_data=suicide_cause_data.groupby(["CAUSE"]).sum().sort_values(by="Grand Total",ascending=False)
suicide_cause_data.head(10)
suicide_cause_data=suicide_cause_data.reset_index()
#fig = px.bar(suicide_cause_data, x='CAUSE', y='Grand Total',text="Grand Total")

fig = px.bar(suicide_cause_data, x='Grand Total', y='CAUSE',text="Grand Total",orientation="h")

fig.update_traces( textposition='outside',textfont_size=9.5)

fig.update_layout(title_text='Causes for suicide From 2001-2014',margin=dict(l=200,r=0))

#fig.update_xaxes(dtick=1)

fig.update_yaxes(automargin=False)

fig.update_yaxes(autorange="reversed")

fig.show()

suicide_cause_2011=suicide_india_data[["Year","CAUSE","Grand Total"]]

suicide_cause_2011=suicide_cause_2011[suicide_cause_2011["Year"]==2011]

suicide_cause_2011=suicide_cause_2011.set_index(["Year"])

suicide_cause_2011=suicide_cause_2011.groupby(["CAUSE"]).sum().sort_values(by="Grand Total",ascending=False).reset_index()





suicide_cause_2012=suicide_india_data[["Year","CAUSE","Grand Total"]]

suicide_cause_2012=suicide_cause_2012[suicide_cause_2012["Year"]==2012]

suicide_cause_2012=suicide_cause_2012.set_index(["Year"])

suicide_cause_2012=suicide_cause_2012.groupby(["CAUSE"]).sum().sort_values(by="Grand Total",ascending=False).reset_index()



suicide_cause_2013=suicide_india_data[["Year","CAUSE","Grand Total"]]

suicide_cause_2013=suicide_cause_2013[suicide_cause_2013["Year"]==2013]

suicide_cause_2013=suicide_cause_2013.set_index(["Year"])

suicide_cause_2013=suicide_cause_2013.groupby(["CAUSE"]).sum().sort_values(by="Grand Total",ascending=False).reset_index()



suicide_cause_2014=suicide_india_data[["Year","CAUSE","Grand Total"]]

suicide_cause_2014=suicide_cause_2014[suicide_cause_2014["Year"]==2014]

suicide_cause_2014=suicide_cause_2014.set_index(["Year"])

suicide_cause_2014=suicide_cause_2014.groupby(["CAUSE"]).sum().sort_values(by="Grand Total",ascending=False).reset_index()
data1=suicide_cause_2011.head(12)

data2=suicide_cause_2012.head(12)

data3=suicide_cause_2013.head(12)

data4=suicide_cause_2014.head(12)



y1=data1["CAUSE"]

y2=data2["CAUSE"]

y3=data3["CAUSE"]

y4=data4["CAUSE"]



x1=data1["Grand Total"]

x2=data2["Grand Total"]

x3=data3["Grand Total"]

x4=data4["Grand Total"]





fig = make_subplots(rows=2, cols=2, start_cell="top-left",horizontal_spacing=0.25,vertical_spacing=0.1,subplot_titles=("2011","2012", "2013","2014"))



fig.add_trace(go.Bar(name="2011",x=x1, y=y1,text=x1,orientation='h'),

              row=1, col=1)



fig.add_trace(go.Bar(name="2012",x=x2, y=y2,text=x2,orientation='h'),

              row=1, col=2)



fig.add_trace(go.Bar(name="2013",x=x3, y=y3,text=x3,orientation='h'),

              row=2, col=1)



fig.add_trace(go.Bar(name="2014",x=x4, y=y4,text=x4,orientation='h'),

              row=2, col=2)



fig.update_yaxes(automargin=True,autorange="reversed")





fig.update_traces( textposition='outside',cliponaxis = False,texttemplate='%{text:.3s}')



fig.update_layout(title_text='Causes for suicide From 2011-2014',height=800, width=1000)



fig.show()
suicide_agegroup=suicide_india_data[["CAUSE","Female 15-29 years"]]

suicide_agegroup=suicide_agegroup.rename(columns ={"Female 15-29 years":"Suicide Count"})

suicide_agegroup=suicide_agegroup.groupby(["CAUSE"]).sum().sort_values(by="Suicide Count",ascending=False)
suicide_agegroup.head(7)
suicide_agegroup=suicide_agegroup.head(11).reset_index()


fig = px.bar(suicide_agegroup, x="Suicide Count", y="CAUSE",text="Suicide Count" ,color="Suicide Count",color_continuous_scale=px.colors.sequential.Plasma_r,orientation='h' )



fig.update_layout(title_text='Main Causes of suicide of Females in the age group of 15-29 years')



fig.update_yaxes(autorange="reversed")



fig.show()



suicide_cause_gender=suicide_india_data[["CAUSE","Total Male","Total Female"]]

#suicide_cause_female=suicide_india_data[["CAUSE","Total Female"]]
suicide_cause_gender=suicide_cause_gender.groupby(["CAUSE"]).sum().sort_values(by="Total Male",ascending=False)

suicide_cause_gender
suicide_cause_gender1=suicide_cause_gender.reset_index().head(11)

suicide_cause_gender2=suicide_cause_gender.reset_index().tail(10)
x1=suicide_cause_gender1["CAUSE"]

y1=suicide_cause_gender1["Total Male"]

y11=suicide_cause_gender1["Total Female"]

x2=suicide_cause_gender2["CAUSE"]

y2=suicide_cause_gender2["Total Male"]

y22=suicide_cause_gender2["Total Female"]





fig = make_subplots(rows=2, cols=1,vertical_spacing=0.25)



#fig = go.Figure()



fig.add_trace(go.Bar(

    x=x1,

    y=y1,

    text=y1,

    name='Male',

    marker_color='indianred'

),row=1, col=1)



fig.add_trace(go.Bar(

    x=x1,

    y=y11,

    name='Female',

    text=y11,

    marker_color='lightsalmon'

),row=1, col=1)



fig.add_trace(go.Bar(

    x=x2,

    y=y2,

    text=y2,

    marker_color='indianred',

    showlegend=False

),row=2, col=1)



fig.add_trace(go.Bar(

    x=x2,

    y=y22,

    text=y22,

    marker_color='lightsalmon',

    showlegend=False

),row=2, col=1)



fig.update_traces( textposition='outside',cliponaxis = False,texttemplate='%{text:.2s}')



fig.update_xaxes(

        title_text = "Causes")



fig.update_yaxes(

        title_text = "Count")







# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group',height=800,title_text="Suicide deaths due to various Causes(Male Vs Female)")





fig.show()
suicide_india_data_2018_new=suicide_india_data_2018_new.groupby(["CAUSE"]).sum().sort_values(by="COUNT",ascending=False)
suicide_india_data_2018_new.head(30)
suicide_india_data_2018_new=suicide_india_data_2018_new.reset_index()
x1=suicide_india_data_2018_new["COUNT"].head(10)

y1=suicide_india_data_2018_new["CAUSE"].head(10)

y2=suicide_india_data_2018_new["CAUSE"].tail(10)

x2=suicide_india_data_2018_new["COUNT"].tail(10)



fig = make_subplots(rows=2, cols=1,vertical_spacing=0.15)



#fig = go.Figure()



fig.add_trace(go.Bar(

    x=x1,

    y=y1,

    text=x1,

    orientation='h',

    showlegend=False

),row=1, col=1)



fig.add_trace(go.Bar(

    x=x2,

    y=y2,

    text=x2,

    orientation='h',

    showlegend=False

),row=2, col=1)





fig.update_traces( textposition='outside',cliponaxis = False,texttemplate='%{text:.2s}')



fig.update_xaxes(

        title_text = "Causes")



fig.update_yaxes(

        title_text = "Count")



fig.update_yaxes(autorange="reversed")









# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group',height=800,title_text="YEAR 2018")





fig.show()