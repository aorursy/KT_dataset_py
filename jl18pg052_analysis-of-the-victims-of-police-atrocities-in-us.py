import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.io as pio

from wordcloud import WordCloud
data=pd.read_csv("../input/police-fatalities-in-the-us-from-2000-to-2020/police_fatalities.csv")

data.head()
data.shape
data.isnull().sum()
data.info()
data.drop(columns=["Subject's race with imputations","Imputation probability","URL of image of deceased","Video",

                   "Unique ID formula","Unique identifier (redundant)"],inplace=True)
data.drop(columns=["Full Address","Link to news article or photo of official document","Unique ID","Date&Description"],inplace=True)
data.drop(columns=["Date (Year)"],inplace=True)
data.drop(columns=["Location of injury (address)"],inplace=True)
data.info()
data["Subject's age"].replace(to_replace="20s",value=28,inplace=True)
data["Subject's age"].replace(to_replace="60s",value=65,inplace=True)
data["Subject's age"].replace(to_replace="18-25",value=23,inplace=True)

data["Subject's age"].replace(to_replace="18 months",value=18,inplace=True)

data["Subject's age"].replace(to_replace="46/53",value=53,inplace=True)

data["Subject's age"].replace(to_replace="3 months",value=3,inplace=True)

data["Subject's age"].replace(to_replace="40s",value=45,inplace=True)

data["Subject's age"].replace(to_replace="30s",value=35,inplace=True)

data["Subject's age"].replace(to_replace="50s",value=55,inplace=True)

data["Subject's age"].replace(to_replace="6 months",value=6,inplace=True)

data["Subject's age"].replace(to_replace="9 months",value=9,inplace=True)

data["Subject's age"].replace(to_replace="10 months",value=10,inplace=True)

data["Subject's age"].replace(to_replace="2 months",value=2,inplace=True)

data["Subject's age"].replace(to_replace="7 months",value=7,inplace=True)

data["Subject's age"].replace(to_replace="8 months",value=8,inplace=True)

data["Subject's age"].replace(to_replace="3 days",value=3,inplace=True)

data["Subject's age"].replace(to_replace="20s-30s",value=25,inplace=True)

data["Subject's age"].replace(to_replace="40-50",value=45,inplace=True)

data["Subject's age"].replace(to_replace="4 months",value=4,inplace=True)

data["Subject's age"].replace(to_replace="70s",value=75,inplace=True)
data["Subject's age"]=pd.to_numeric(data["Subject's age"])
data["Date of injury resulting in death (month/day/year)"]=pd.to_datetime(data["Date of injury resulting in death (month/day/year)"])

data["Year"]=data["Date of injury resulting in death (month/day/year)"].dt.year
data["Year"]=data["Year"].astype("str")
data["Year"].replace(to_replace="2100",value="2001",inplace=True)
data.isnull().sum()
data["Location of death (state)"].fillna(value="CA",inplace=True)
data["Subject's gender"].fillna(value="Female",inplace=True)

data["Cause of death"].fillna(value="Gunshot",inplace=True)

data["Subject's race"].fillna(value="European-American/White",inplace=True)

data["Agency responsible for death"].fillna(value="Los Angeles Police Department",inplace=True)

data["Location of death (county)"].fillna(value="Los Angeles",inplace=True)

data["Location of death (city)"].fillna(value="Chicago",inplace=True)

data["Symptoms of mental illness? INTERNAL USE, NOT FOR ANALYSIS"].fillna(value="Unknown",inplace=True)
data["Subject's age"].fillna(value=28,inplace=True)
data.isnull().sum()
data["Location of death (city)"].nunique()
data["Subject's race"].value_counts()
data["Subject's race"].replace(to_replace="",value="European-American/White",inplace=True)

data["Subject's race"].replace(to_replace="HIspanic/Latino",value="Hispanic/Latino",inplace=True)
data["Subject's gender"].replace(to_replace="",value="Female",inplace=True)
fig4=px.histogram(data,x="Subject's gender",title="Gender distribution",width=700,height=500,color="Subject's gender")

fig4.update_layout(xaxis={'categoryorder':'total descending'})
px.histogram(data,x="Subject's gender",title="Number of deaths for each race gender wise",width=960,height=700,color="Subject's gender",facet_col="Subject's race",

            facet_col_wrap=3)
px.box(data,x="Subject's race",y="Subject's age",color="Subject's race",facet_col="Subject's gender",

       facet_col_wrap=2,width=900,height=600,title="Age distribution of all the victims racial and gender wise")
fig3=px.histogram(data,x="Subject's race",title="How many racial categories are there in our dataset who suffered police attrocities",

                  width=800,height=600,color="Subject's race")

fig3.update_layout(xaxis={'categoryorder':"total descending"})
fig=px.histogram(data,x="Location of death (state)",title="Number of deaths for each state",color="Location of death (state)",

                 width=900,height=600)

fig.update_layout(xaxis={'categoryorder':'total descending'})
px.histogram(data,x="Location of death (state)",title="No of deaths in each state for different race",

                 width=1060,height=800,facet_col="Subject's race",facet_col_wrap=3,color="Subject's race")
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

data["Year"].value_counts().plot(kind="bar",color="red")

plt.xlabel("Year",size=15,color="black")

plt.ylabel("No of people died year wise",size=15,color="black")

plt.title("Number of people died yearly because of police atrocities",size=18)
px.histogram(data,x="Location of death (state)",facet_col="Year",facet_col_wrap=5,

             title="Number of deaths in different states for different years",

            color="Year",width=1150,height=1000,template="plotly_dark")
px.histogram(data,x="Subject's race",facet_col="Year",facet_col_wrap=5,title="Distribution of deaths due to police attrocities yearly and racial wise",

            color="Year",width=1100,height=900)
data["Location of death (city)"].value_counts()[:30].plot(kind="bar",figsize=(12,6),color="darkorchid")

plt.title("Top 30 cities in terms of number of deaths for different racial people",size=18)
px.histogram(data[data["Location of death (city)"]=="Chicago"],x="Subject's race",title="Number of people from different race died in Chicago in different years",

            width=1100,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data[data["Location of death (city)"]=="Houston"],x="Subject's race",title="Number of people from different race died in Houston in different years",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
fig6=px.histogram(data,x="Cause of death",title="Number of people died because of different reasons",

            width=900,height=600,color="Cause of death")

fig6.update_layout(xaxis={'categoryorder':'total descending'})
px.histogram(data,x="Cause of death",title="Number of people died because of different reasons in different years",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data,x="Cause of death",title="Number of people from different race who died because of different reasons",

            width=1100,height=850,color="Subject's race",facet_col="Subject's race",facet_col_wrap=3)
px.histogram(data[data["Year"]=="2019"],x="Cause of death",title="Number of people from different race who died because of different reasons in the year 2019",

            width=1050,height=800,color="Subject's race",facet_col="Subject's race",facet_col_wrap=3)
px.histogram(data[data["Year"]=="2018"],x="Cause of death",title="Number of people from different race who died because of different reasons in the year 2018",

            width=1100,height=850,color="Subject's race",facet_col="Subject's race",facet_col_wrap=3)
px.histogram(data[data["Year"]=="2000"],x="Cause of death",title="Number of people from different race who died because of different reasons in the year 2000",

            width=1050,height=750,color="Subject's race",facet_col="Subject's race",facet_col_wrap=3)
px.histogram(data[data["Year"]=="2002"],x="Cause of death",title="Number of people from different race who died because of different reasons in the year 2002",

            width=1050,height=850,color="Subject's race",facet_col="Subject's race",facet_col_wrap=3)
cloud=WordCloud(colormap="autumn",width=800,height=400).generate(str(data["A brief description of the circumstances surrounding the death"]))

fig=plt.figure(figsize=(14,10))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')
data["Agency responsible for death"].nunique()
gradient = data["Agency responsible for death"].value_counts()[:30]

data2 = pd.DataFrame(gradient)

data2.style.background_gradient(cmap="Reds")
px.histogram(data[data["Agency responsible for death"]=="Los Angeles Police Department"],x="Cause of death",

             title="Number of people died because of different reasons in different years by Los Angeles Police Department",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data[data["Agency responsible for death"]=="Chicago Police Department"],x="Cause of death",

             title="Number of people died because of different reasons in different years by Chicago Police Department",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data[data["Agency responsible for death"]=="Texas Department of Public Safety"],x="Cause of death",

             title="Number of people died because of different reasons in different years by Texas Department of Public Safety",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data[data["Agency responsible for death"]=="Los Angeles Police Department"],x="Subject's race",

             title="Number of people from different race who died in different years by Los Angeles Police Department",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data[data["Agency responsible for death"]=="Chicago Police Department"],x="Subject's race",

             title="Number of people from different race who died in different years by Chicago Police Department",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)
px.histogram(data[data["Agency responsible for death"]=="Texas Department of Public Safety"],x="Subject's race",

             title="Number of people from different race who died in different years by Texas Department of Public Safety",

            width=1150,height=1000,color="Year",facet_col="Year",facet_col_wrap=5)