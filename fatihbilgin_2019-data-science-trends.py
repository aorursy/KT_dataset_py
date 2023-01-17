import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')
df_responses = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv", skiprows=(1,1))

df_wdidata = pd.read_csv("../input/world-development-indicators/wdi-csv-zip-57-mb-/WDIData.csv")

df_country = pd.read_csv("../input/world-development-indicators/wdi-csv-zip-57-mb-/WDICountry.csv")
df_data = df_responses.loc[:,["Q1", "Q2", "Q3", "Q4", "Q5", "Q10", "Q11"]]

df_data.columns = ["Age", "Gender", "Country", "Education", "Title", "Salary", "Spending"]

df_data["Country"] = df_data["Country"].str.replace("United Kingdom of Great Britain and Northern Ireland","United Kingdom", regex=False)

df_data["Country"] = df_data["Country"].str.replace("United States of America","United States", regex=False)

df_data["Country"] = df_data["Country"].str.replace("Iran, Islamic Republic of...","Iran", regex=False)

df_data["Country"] = df_data["Country"].str.replace("Republic of Korea","South Korea", regex=False)

df_data["Country"] = df_data["Country"].str.replace("Viet Nam","Vietnam", regex=False)

df_data_ds = df_data[df_data.Title == "Data Scientist"]
df_country_count = pd.DataFrame({'Country':df_data.Country.value_counts().index, 

                             'Count':df_data.Country.value_counts().values})



data = [ dict(

        type = 'choropleth',

        locations = df_country_count['Country'],

        locationmode = 'country names',

        z = df_country_count['Count'],

        colorscale=

            [[0.0, "rgb(251, 237, 235)"],

            [0.09, "rgb(245, 211, 206)"],

            [0.12, "rgb(239, 179, 171)"],

            [0.15, "rgb(236, 148, 136)"],

            [0.22, "rgb(239, 117, 100)"],

            [0.35, "rgb(235, 90, 70)"],

            [0.45, "rgb(207, 81, 61)"],

            [0.65, "rgb(176, 70, 50)"],

            [0.85, "rgb(147, 59, 39)"],

            [1.00, "rgb(110, 47, 26)"]],

        autocolorscale = False,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) 

        ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Participant'),

      ) ]



layout = dict(

    title = "Country Distribution of All Respondents",

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(type = 'Mercator'),

        width=500,height=400)

)



w_map = dict( data=data, layout=layout)

iplot( w_map, validate=False)
df_ds_country = df_data_ds[df_data_ds.Country.isin(df_data_ds.Country.value_counts()[:25].index)].loc[:,["Country", "Gender"]].groupby(["Country", "Gender"]).size().reset_index()

df_ds_country.columns = ["Country", "Gender", "Count"]
fig = px.bar(df_ds_country, x="Country", y="Count", color='Gender')



fig.update_layout(

    title_text='Country Distribution of Data Scientist by Gender',

    height=500, width=1000,

    xaxis={'categoryorder':'total descending'}

)



fig.show()
df_ages_count = pd.DataFrame({'Age Group':df_data["Age"].value_counts().index, 

                               'Count':df_data["Age"].value_counts().values}).sort_values("Age Group",ascending=False)



df_ds_ages_count = pd.DataFrame({'Age Group':df_data_ds["Age"].value_counts().index, 

                               'Count':df_data_ds["Age"].value_counts().values}).sort_values("Age Group",ascending=False)





fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('Age Group of All Respondents', 'Age Group of Data Scientist'))



fig.add_trace(

    go.Bar(x=df_ages_count["Count"],

                     y=df_ages_count["Age Group"],

                     marker_color='darkorange',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6,

                     orientation='h'),

    row=1, col=1)



fig.add_trace(

    go.Bar(x=df_ds_ages_count["Count"],

                     y=df_ds_ages_count["Age Group"],

                     marker_color='darkorange',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6,

                     orientation='h'),

    row=1, col=2)





fig.update_layout(

    title_text="Age Groups",

    height=500, width=800, showlegend=False)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()
df_ages_gender = df_data.loc[:,["Age", "Gender"]].groupby(["Age", "Gender"]).size().reset_index()

df_ages_gender.columns = ["Age Group", "Gender", "Count"]



fig = px.bar(df_ages_gender, x='Age Group', y='Count', color="Gender", 

             barmode='group', title ="Age Distribution by Gender", 

             height=450, width=800)



fig.update_traces( marker_line_color='rgb(8,48,107)',

                  marker_line_width=2, opacity=0.7)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()
df_age_ind_usa = df_data[(df_data["Country"].isin(["India", "United States"])) & (df_data["Gender"].isin(["Female", "Male"]))].groupby(["Age", "Gender", "Country"]).size().reset_index()

df_age_ind_usa.columns = ["Age Group", "Gender", "Country", "Count"]



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('United States of America', 'India'))



fig.add_trace(

    go.Bar(y=df_age_ind_usa[(df_age_ind_usa["Country"]=="United States")&(df_age_ind_usa["Gender"]=="Male")]["Count"].tolist(), 

           x=df_age_ind_usa[(df_age_ind_usa["Country"]=="United States")&(df_age_ind_usa["Gender"]=="Male")]["Age Group"].tolist(), 

           name="US-Male", marker_color='mediumpurple'),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=df_age_ind_usa[(df_age_ind_usa["Country"]=="United States")&(df_age_ind_usa["Gender"]=="Female")]["Count"].tolist(), 

           x=df_age_ind_usa[(df_age_ind_usa["Country"]=="United States")&(df_age_ind_usa["Gender"]=="Female")]["Age Group"].tolist(), 

           name="US-Female", marker_color='mediumvioletred'),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=df_age_ind_usa[(df_age_ind_usa["Country"]=="India")&(df_age_ind_usa["Gender"]=="Male")]["Count"].tolist(), 

           x=df_age_ind_usa[(df_age_ind_usa["Country"]=="India")&(df_age_ind_usa["Gender"]=="Male")]["Age Group"].tolist(), 

           name="Ind-Male", marker_color='mediumpurple'),

    row=1, col=2)



fig.add_trace(

    go.Bar(y=df_age_ind_usa[(df_age_ind_usa["Country"]=="India")&(df_age_ind_usa["Gender"]=="Female")]["Count"].tolist(), 

           x=df_age_ind_usa[(df_age_ind_usa["Country"]=="India")&(df_age_ind_usa["Gender"]=="Female")]["Age Group"].tolist(), 

           name="Ind-Female", marker_color='mediumvioletred'),

row=1, col=2)





fig.update_layout(

    title_text="Age Group (India - U.S. Comparison)",

    height=450, width=800, showlegend=True)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()
df_edu_ind_usa = df_data[(df_data["Country"].isin(["India", "United States"])) & (df_data["Gender"].isin(["Female", "Male"]))].groupby(["Education", "Gender", "Country"]).size().reset_index()

df_edu_ind_usa.columns = ["Education", "Gender", "Country", "Count"]
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'tan']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=('All Respondents', 'Data Scientist'))



fig.add_trace(

    go.Pie(labels=df_data.Education.value_counts().index.tolist(), 

           values=df_data.Education.value_counts().values.tolist()),1, 1)



fig.add_trace(

    go.Pie(labels=df_data_ds.Education.value_counts().index.tolist(), 

           values=df_data_ds.Education.value_counts().values.tolist()),1, 2)



fig.update_traces(hoverinfo="label+value")



fig.update_layout(

    title_text="Education Level Comparison (Around the World)",

    height=400, width=1000)



fig.update_traces(hoverinfo='label+value+percent', textinfo='percent', textfont_size=14,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(

    go.Pie(labels=df_edu_ind_usa[df_edu_ind_usa.Country == "United States"].groupby(['Education']).agg('sum').reset_index()["Education"], 

           values=df_edu_ind_usa[df_edu_ind_usa.Country == "United States"].groupby(['Education']).agg('sum').reset_index()["Count"], 

           name="U.S."),1, 1)



fig.add_trace(

    go.Pie(labels=df_edu_ind_usa[df_edu_ind_usa.Country == "India"].groupby(['Education']).agg('sum').reset_index()["Education"], 

           values=df_edu_ind_usa[df_edu_ind_usa.Country == "India"].groupby(['Education']).agg('sum').reset_index()["Count"], 

           name="India"),1, 2)



fig.update_traces(hole=.4, hoverinfo="label+value")



fig.update_layout(

    title_text="Education Level Comparison (All Respondents)",

    height=400, width=1100,

    annotations=[dict(text='U.S.', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='India', x=0.82, y=0.5, font_size=20, showarrow=False)])



fig.update_traces(hoverinfo='label+value+percent', textinfo='percent', textfont_size=15,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.show()
df_title_ind_usa = df_data[(df_data["Country"].isin(["India", "United States"])) & (df_data["Gender"].isin(["Female", "Male"]))].groupby(["Title", "Gender", "Country"]).size().reset_index()

df_title_ind_usa.columns = ["Title", "Gender", "Country", "Count"]

df_title_ind_usa = df_title_ind_usa.sort_values("Count", ascending=False)
df_title_count= pd.DataFrame({'Title':df_data["Title"].value_counts().index, 'Count':df_data["Title"].value_counts().values}).sort_values("Count")



fig = go.Figure()

fig.add_trace(go.Bar(x=df_title_count["Count"],

                y=df_title_count["Title"],

                marker_color='indianred',

                marker_line_color='rgb(8,48,107)',

                marker_line_width=1.5, 

                opacity=0.6,

                orientation='h'))



fig.update_layout(

    title_text='Titles of Respondents',

    height=550, width=700,

    showlegend=False)



fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('United States of America', 'India'))



fig.add_trace(

    go.Bar(y=df_title_ind_usa[(df_title_ind_usa["Country"]=="United States")&(df_title_ind_usa["Gender"]=="Male")]["Count"].tolist(), 

           x=df_title_ind_usa[(df_title_ind_usa["Country"]=="United States")&(df_title_ind_usa["Gender"]=="Male")]["Title"].tolist(), 

           name="US-Male", marker_color='slateblue'),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=df_title_ind_usa[(df_title_ind_usa["Country"]=="United States")&(df_title_ind_usa["Gender"]=="Female")]["Count"].tolist(), 

           x=df_title_ind_usa[(df_title_ind_usa["Country"]=="United States")&(df_title_ind_usa["Gender"]=="Female")]["Title"].tolist(), 

           name="US-Female", marker_color='crimson'),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=df_title_ind_usa[(df_title_ind_usa["Country"]=="India")&(df_title_ind_usa["Gender"]=="Male")]["Count"].tolist(), 

           x=df_title_ind_usa[(df_title_ind_usa["Country"]=="India")&(df_title_ind_usa["Gender"]=="Male")]["Title"].tolist(), 

           name="Ind-Male", marker_color='slateblue'),

    row=1, col=2)



fig.add_trace(

    go.Bar(y=df_title_ind_usa[(df_title_ind_usa["Country"]=="India")&(df_title_ind_usa["Gender"]=="Female")]["Count"].tolist(), 

           x=df_title_ind_usa[(df_title_ind_usa["Country"]=="India")&(df_title_ind_usa["Gender"]=="Female")]["Title"].tolist(), 

           name="Ind-Female", marker_color='crimson'),

    row=1, col=2)



fig.update_layout(

    title_text="Titles of Respondents (India - USA Comparison)",

    height=550, width=1000, showlegend=True)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()
df_salary = df_data.dropna()

df_salary["SalaryTemp"] = df_salary["Salary"].str.replace("$0-999","0-999", regex=False)

df_salary["SalaryTemp"] = df_salary["SalaryTemp"].str.replace("> $500,000","500,000-499,999", regex=False)

df_salary["SalaryTemp"] = df_salary["SalaryTemp"].str.replace("300,000-500,000","300,000-499,999", regex=False)

df_salary["SalaryTemp"] = df_salary["SalaryTemp"].str.replace(",","", regex=False)

df_salary["SalaryLowLimit"] = df_salary["SalaryTemp"].str.split('-', expand=True)[0]

df_salary["SalaryLowLimit"] = pd.to_numeric(df_salary["SalaryLowLimit"], errors='coerce')

df_salary["SalaryUpLimit"] = df_salary["SalaryTemp"].str.split('-', expand=True)[1]

df_salary["SalaryUpLimit"] = pd.to_numeric(df_salary["SalaryUpLimit"], errors='coerce')+1

df_salary["SalaryMean"] = (df_salary["SalaryLowLimit"]+df_salary["SalaryUpLimit"])/2



df_ds_salary = df_salary[df_salary.Title == "Data Scientist"]



df_ds_salary_by_country  = df_ds_salary.loc[:,["SalaryMean","Country"]].groupby(["Country"]).mean().reset_index()

df_ds_salary_by_country["SalaryMean"] = df_ds_salary_by_country["SalaryMean"].round(2)



df_countrygdp = df_wdidata[df_wdidata["Indicator Name"]=="GDP per capita (current US$)"].loc[:,["Country Name", "Country Code", "2017"]]

df_countrygdp.columns = ["Country", "CountryCode", "GDP"]

df_countrygdp["GDP"] = df_countrygdp["GDP"].round(2)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Egypt, Arab Rep.","Egypt", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Hong Kong SAR, China","Hong Kong (S.A.R.)", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Iran, Islamic Rep.","Iran", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Korea, Rep.","South Korea", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Russian Federation","Russia", regex=False)

df_ds_salary_by_country = pd.merge(df_ds_salary_by_country, df_countrygdp, on='Country')
df_salary_spent = df_salary.loc[:,["Spending", "Salary", "SalaryMean"]].groupby(["Spending", "Salary", "SalaryMean"]).size().reset_index().sort_values(["Spending","SalaryMean"]).dropna()

df_salary_spent.columns = ["Spending", "Salary", "SalaryMean", "Count"]



df_salary_spent["Spent"] = df_salary_spent["Spending"]

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$0 (USD)","$ 0", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$1-$99","$ 1-99", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$100-$999","$ 100-999", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$1000-$9,999","$ 1000-9999", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$10,000-$99,999","$ 10000-99999", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("> $100,000 ($USD)","$ 100000 and more", regex=False)
fig = px.scatter_geo(df_ds_salary_by_country, locations="CountryCode", color="GDP",

                     hover_name="Country", size="SalaryMean", 

                     projection="natural earth") 



fig.update_layout(title="GDP Correlation with Average Data Scientists Salaries Around the World")



fig.show()
df_ds_salary_by_country_top25 = df_ds_salary_by_country.sort_values("SalaryMean", ascending=False)[:25]





fig = go.Figure()

fig.add_trace(go.Bar(x=df_ds_salary_by_country_top25["Country"],

                y=df_ds_salary_by_country_top25["SalaryMean"],

                marker_color=df_ds_salary_by_country_top25["SalaryMean"],

                marker_line_color='rgb(8,48,107)',

                marker_line_width=1.5, 

                opacity=0.8))



fig.update_layout(

    title_text='Top 25 Countries with the Highest Average Data Scientist Salary',

    height=550, width=700,

    showlegend=False)



fig.show()
df_s_usa = df_salary[df_salary["Country"]=="United States"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_india = df_salary[df_salary["Country"]=="India"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_brazil = df_salary[df_salary["Country"]=="Brazil"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_russia = df_salary[df_salary["Country"]=="Russia"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_germany = df_salary[df_salary["Country"]=="Germany"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_uk = df_salary[df_salary["Country"]=="United Kingdom"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)



mc_c1 = ["gold",]*10

mc_c2 = ["gold",]*10

mc_c3 = ["gold",]*10

mc_c4 = ["gold",]*10

mc_c1[0] = mc_c2[1] = mc_c3[6] = mc_c4[4] ="lime"





fig = make_subplots(rows=3, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}],[{'type':'xy'}, {'type':'xy'}],[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('United States of America', 'India','Brazil', 'Russia','Germany', 'United Kingdom'))



fig.add_trace(

    go.Bar(x=df_s_usa["Title"], y=df_s_usa["SalaryMean"], name="United States of America", marker_color=mc_c1),

    row=1, col=1)



fig.add_trace(

    go.Bar(x=df_s_india["Title"], y=df_s_india["SalaryMean"], name="India", marker_color=mc_c2),

    row=1, col=2)



fig.add_trace(

    go.Bar(x=df_s_brazil["Title"], y=df_s_brazil["SalaryMean"], name="Brazil", marker_color=mc_c3),

    row=2, col=1)



fig.add_trace(

    go.Bar(x=df_s_russia["Title"], y=df_s_russia["SalaryMean"], name="Russia", marker_color=mc_c1),

    row=2, col=2)



fig.add_trace(

    go.Bar(x=df_s_germany["Title"], y=df_s_germany["SalaryMean"], name="Germany", marker_color=mc_c4),

    row=3, col=1)



fig.add_trace(

    go.Bar(x=df_s_uk["Title"], y=df_s_uk["SalaryMean"], name="United Kingdom", marker_color=mc_c4),

    row=3, col=2)



fig.update_layout(

    title_text="Data Scientist Salary (Comparison of Top 6 Countries With More Participants )",

    height=1500, width=850, showlegend=False)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6) 

                 

fig.show()
df_salary_spent_sorted = df_salary_spent.sort_values(['SalaryMean', 'Spent'])



fig = px.bar(df_salary_spent_sorted, 

             x="Salary", y="Count", color='Spent')



fig.update_layout(

    title_text='Salary and Spent on ML/Cloud Products Correlation',

    height=500, width=1000)



fig.show()
df_ds_salary_ind_usa  = df_ds_salary[(df_ds_salary["Country"].isin(["India", "United States"])) & (df_ds_salary["Gender"].isin(["Female", "Male"]))].groupby(["Salary", "SalaryMean", "Gender", "Country"]).size().reset_index().sort_values("SalaryMean")

df_ds_salary_ind_usa.columns = ["Salary", "SalaryMean", "Gender", "Country", "Count"]



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('United States of America', 'India'))



fig.add_trace(

    go.Bar(y=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="United States")&(df_ds_salary_ind_usa ["Gender"]=="Male")]["Count"].tolist(), 

           x=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="United States")&(df_ds_salary_ind_usa ["Gender"]=="Male")]["Salary"].tolist(), 

           name="US-Male", marker_color='skyblue'),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="United States")&(df_ds_salary_ind_usa ["Gender"]=="Female")]["Count"].tolist(), 

           x=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="United States")&(df_ds_salary_ind_usa ["Gender"]=="Female")]["Salary"].tolist(), 

           name="US-Female", marker_color='tomato'),

    row=1, col=1)



fig.add_trace(

    go.Bar(y=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="India")&(df_ds_salary_ind_usa ["Gender"]=="Male")]["Count"].tolist(), 

           x=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="India")&(df_ds_salary_ind_usa ["Gender"]=="Male")]["Salary"].tolist(), 

           name="Ind-Male", marker_color='skyblue'),

    row=1, col=2)



fig.add_trace(

    go.Bar(y=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="India")&(df_ds_salary_ind_usa ["Gender"]=="Female")]["Count"].tolist(), 

           x=df_ds_salary_ind_usa [(df_ds_salary_ind_usa ["Country"]=="India")&(df_ds_salary_ind_usa ["Gender"]=="Female")]["Salary"].tolist(), 

           name="Ind-Female", marker_color='tomato'),

    row=1, col=2)



fig.update_layout(

    title_text="Number of Data Scientist Salary Group (India - U.S. Comparison)",

    height=500, width=1300, showlegend=True)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6) 



fig.show()
df_sal = df_ds_salary[df_ds_salary["Country"]=="United States"].loc[:,["Education","Salary", "SalaryMean"]].sort_values("SalaryMean")



sal_list = df_sal["Salary"].drop_duplicates().values.tolist()

noanswer = []

highschool = []

somecollege = []

professional = []

bachelors = []

masters =  []

doctoral = []



for i in sal_list:

    noanswer.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="I prefer not to answer")])/len(df_sal[(df_sal["Salary"]==i)]),2))

    highschool.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="No formal education past high school")])/len(df_sal[(df_sal["Salary"]==i)]),2))

    somecollege.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="Some college/university study without earning a bachelor’s degree")])/len(df_sal[(df_sal["Salary"]==i)]),2))

    professional.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="Professional degree")])/len(df_sal[(df_sal["Salary"]==i)]),2))

    bachelors.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="Bachelor’s degree")])/len(df_sal[(df_sal["Salary"]==i)]),2))

    masters.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="Master’s degree")])/len(df_sal[(df_sal["Salary"]==i)]),2))

    doctoral.append(round(100*len(df_sal[(df_sal["Salary"]==i) & (df_sal["Education"]=="Doctoral degree")])/len(df_sal[(df_sal["Salary"]==i)]),2))
fig = go.Figure(go.Bar(x=sal_list, y=noanswer, name='I prefer not to answer'))

fig.add_trace(go.Bar(x=sal_list, y=highschool, name='No formal education past high school'))

fig.add_trace(go.Bar(x=sal_list, y=somecollege, name='Some uni. stu. without earn a bach deg.'))

fig.add_trace(go.Bar(x=sal_list, y=professional, name='Professional degree'))

fig.add_trace(go.Bar(x=sal_list, y=bachelors, name='Bachelor’s degree'))

fig.add_trace(go.Bar(x=sal_list, y=masters, name='Master’s degree'))

fig.add_trace(go.Bar(x=sal_list, y=doctoral, name='Doctoral degree'))



fig.update_layout(barmode='stack',height=450, width=1100, 

                 title_text='Data Scientist Salary by Percentile Distribution of Education Level in the U.S.')



fig.show()
source_list = ["Twitter", "Hacker News", "Reddit", "Kaggle", "Course Forums (fast.ai etc)", "YouTube (AI Adventures etc)", "Podcasts(Chai Time DS etc)",  "Blogs (Medium, Towards DS)", "Journal Publications", "Slack Communities", "None", "Other"]

df_sources_ds = df_responses[df_responses.Q5 == "Data Scientist"].iloc[:,22:34]

df_sources = df_responses[df_responses.Q5 != "Data Scientist"].iloc[:,22:34]

df_sources_ds.columns = source_list

df_sources.columns = source_list



source_ds_counts = []

source_counts = []



for i in source_list:

    source_ds_counts.append(len(df_sources_ds.loc[:,i:i].dropna()))

    source_counts.append(len(df_sources.loc[:,i:i].dropna()))
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('Data Scientists', 'Others'))



fig.add_trace(

    go.Bar(x=source_list,

                     y=source_ds_counts,

                     marker_color='darksalmon',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6),

    row=1, col=1)



fig.add_trace(

    go.Bar(x=source_list,

                     y=source_counts,

                     marker_color='bisque',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6),

    row=1, col=2)



fig.update_layout(

    title_text="Favorite Media Sources for Data Science",

    height=600, width=800, showlegend=False)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()
course_list = ["Udacity", "Coursera", "edX", "DataCamp", "DataQuest", "Kaggle Courses", "Fast.ai",  "Udemy", "Linkedin Learning", "University Courses", "None", "Other"]

df_courses_ds = df_responses[df_responses.Q5 == "Data Scientist"].iloc[:,35:47]

df_courses = df_responses[df_responses.Q5 != "Data Scientist"].iloc[:,35:47]

df_courses_ds.columns = course_list

df_courses.columns = course_list



course_ds_counts = []

course_counts = []



for i in course_list:

    course_ds_counts.append(len(df_courses_ds.loc[:,i:i].dropna()))

    course_counts.append(len(df_courses.loc[:,i:i].dropna()))
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('Data Scientists', 'Others'))



fig.add_trace(

    go.Bar(x=course_list,

                     y=course_ds_counts,

                     marker_color='indianred',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6),

    row=1, col=1)



fig.add_trace(

    go.Bar(x=course_list,

                     y=course_counts,

                     marker_color='lightpink',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6),

    row=1, col=2)



fig.update_layout(

    title_text="Platforms where Data Science Courses Have Begun or Completed",

    height=550, width=800, showlegend=False

)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()
ide_list = ["Jupyter", "RStudio", "PyCharm", "Atom", "MATLAB", "Visual Studio", "Spyder",  "Vim / Emacs", "Notepad++", "Sublime Text", "None", "Other"]

df_ide_ds = df_responses[df_responses.Q5 == "Data Scientist"].iloc[:,56:68]

df_ide = df_responses[df_responses.Q5 != "Data Scientist"].iloc[:,56:68]

df_ide_ds.columns = ide_list

df_ide.columns = ide_list



ide_ds_counts = []

ide_counts = []





for i in ide_list:

    ide_ds_counts.append(len(df_ide_ds.loc[:,i:i].dropna()))

    ide_counts.append(len(df_ide.loc[:,i:i].dropna()))
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('Data Scientists', 'Others'))



fig.add_trace(

    go.Bar(x=ide_list,

                     y=ide_ds_counts,

                     marker_color='violet',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6),

    row=1, col=1)



fig.add_trace(

    go.Bar(x=ide_list,

                     y=ide_counts,

                     marker_color='lavender',

                     marker_line_color='rgb(8,48,107)',

                     marker_line_width=1.5, 

                     opacity=0.6),

    row=1, col=2)



fig.update_layout(

    title_text="Development Environments Used Regularly",

    height=500, width=800, showlegend=False)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)



fig.show()