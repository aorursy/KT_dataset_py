import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
df=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

data=pd.DataFrame(df)

data.head(6)
data.shape
data.info()
data.isnull().sum()
data_copy=data.copy()
data_copy.describe()
data_copy["year"]=data_copy["year"].astype("str")
data_copy.info()
data_copy.drop(columns=["country-year","HDI for year"],inplace=True)
px.histogram(data_copy,x="country",title="No. of times each country appear in the dataset",color="country",width=1200,height=600)
descending_order=data_copy["year"].value_counts().sort_values(ascending=False).index

sns.set(style="whitegrid")

graph=sns.catplot(x="year",data=data_copy,kind="count",height=5,aspect=3.2,order=descending_order,palette="Blues_d")

graph.set_xticklabels(rotation=90)
px.histogram(data_copy,x="sex",title="Frequency of male and Female in the whole dataset",color="sex",width=1000)
px.histogram(data_copy,x="age",title="Frequency of each age category in the dataset",color="age",width=1000)
fig=px.histogram(data_copy,x="suicides_no",nbins=100,width=1000,marginal="box",title="Distribution of suicides numbers using histogram and box plot")

fig.update_xaxes(range=[0,5000])

fig2=px.histogram(data_copy,x="population",title="Distribution of Population in the whole dataset",width=1000,marginal="box")

fig2.update_xaxes(range=[0,6000000])
fig3=px.histogram(data_copy,x="suicides/100k pop",title="Distribution of the varaible suicides/100k pop",width=1000,marginal="box")

fig3.update_xaxes(range=[0,100])
px.histogram(data_copy,x="generation",title="Frequency of the variable generation in the dataset", color="generation",width=1000)
px.bar(data_copy.query("country=='Japan'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,

      title="Population change over the years for different age category in Japan")
px.bar(data_copy.query("country=='Italy'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,

      title="Population change over the years for different age category in Italy")
px.bar(data_copy.query("country=='United States'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,

      title="Population change over the years for different age category in United States")
px.bar(data_copy.query("country=='Spain'"),x="year",y="population",color="sex",facet_col="age",barmode="group",facet_col_wrap=3,width=1000,height=650,

      title="Population change over the years for different age category in Spain")
px.line(data_copy.query("country=='Japan'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="Total number of suicides in Japan for different age category over the years")
px.line(data_copy.query("country=='Italy'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="Total number of suicides in Italy for different age category over the years")
px.line(data_copy.query("country=='United States'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="Total number of suicides in United States for different age category over the years")
px.line(data_copy.query("country=='Spain'"),x="year",y="suicides_no",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="Total number of suicides in Spain for different age category over the years")
px.line(data_copy.query("country=='Japan'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="'Suicides per 100k population' in Japan for different age group over the years")
px.line(data_copy.query("country=='Italy'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="'Suicides per 100k population' in Italy for different age group over the years")
px.line(data_copy.query("country=='United States'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="'Suicides per 100k population' in United States for different age group over the years")
px.line(data_copy.query("country=='Spain'"),x="year",y="suicides/100k pop",color="sex",facet_col="age",facet_col_wrap=3,width=1000,height=650,

        title="'Suicides per 100k population' in Spain for different age group over the years")
px.line(data_copy.query("country==['Japan','Italy','United States','Spain','Colombia']"),x="year",y="gdp_per_capita ($)",width=1000,height=560,color="country",

       title="Comparison of 'GDP per capita' of Colombia, Italy, Japan, Spain and United States for different years")
px.line(data_copy.query("country==['Japan','Italy','United States','Spain','Colombia']"),x="year",y=" gdp_for_year ($) ",width=1000,height=560,color="country",

       title="Comparison of 'GDP' of Colombia, Italy, Japan, Spain and United States for different years")