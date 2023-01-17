import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import plotly.offline as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import random
from plotly import tools
from plotly.tools import FigureFactory as ff

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

so_data=pd.read_csv('../input/survey_results_public.csv')
print("size of the data",so_data.shape)

so_data.head()
total=so_data.isnull().sum().sort_values(ascending=False)
percent=(so_data.isnull().sum()/so_data.isnull().count()*100).sort_values(ascending=False)
missing_data=pd.concat([total,percent] ,axis=1 ,keys=['Total','Percent'])
#missing_data




hb=so_data['Hobby'].value_counts()
data=[go.Pie(labels=hb.index,
             values=hb.values,
             marker=dict(colors=random_colors(2)),
             
             )]
layout=go.Layout(title="% of developers coding as hobby")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
open=so_data['OpenSource'].value_counts()
data=[go.Pie(labels=open.index,
             values=open.values,
             marker=dict(colors=random_colors(2)),
            
             )]
layout=go.Layout(title="% of developers contributing to open source")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
temp=so_data['Country'].dropna().value_counts().sort_values(ascending=False).head(10)


x=["united states","india","Germany","Uk","Canada","russia","France","Brazil","Poland","Australia"]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
y=so_data['Country'].dropna().value_counts().sort_values(ascending=False).head(10)


width=1/1.5
plt.bar(x,y,width,color='blue')
xticks=x
plt.xticks(rotation=90)
plt.grid()
plt.xlabel='country'
plt.ylabel='count'
plt.title='country with high no of developers'
plt.show()

#plt.bar(xlabel,ylabel ,title="country with highest no of developers",color=["#080808"])
temp=so_data['Country'].dropna().value_counts().sort_values(ascending=False).head(10)
x=["united states","india","Germany","Uk","Canada","russia","France","Brazil","Poland","Australia"]

data=[go.Pie(labels=x,
             values=temp.values,
             marker=dict(colors=random_colors(6)),
            
             )]
layout=go.Layout(title="% country with greater # of developers")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
temp=so_data["Student"].value_counts()
len(so_data["Country"])



print(temp)
y=so_data.Student.value_counts()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
x=["Not_student","Full_Time_Student","Part_Time_student"]

xticks=x
xlabel=x
ylabel=y
plt.grid()
title="Number of developers who are currently enrolled in formal education"
width=1/1.5
plt.bar(x,y,width,color=random_colors(3))
plt.show()
std=so_data["Student"].value_counts()
data=[go.Pie(labels=std.index,
            values=std.values,
            marker=dict(colors=random_colors(3)),
            )]
layout=go.Layout(title="No of developers currently enrolled")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
temp=so_data.Employment.isnull().value_counts()
print("Mensioned:",temp[False],"Missing_values:",temp[True])
percent_employed=(temp[False])/(temp[True]+temp[False])*100
print(" %age of developers mensioned :",percent_employed)
print("%age of developers not mensioned",temp[True]/98855*100)

employed=so_data.Employment.value_counts()
x=["full-time","independent","Not employed","part-time","not-interested","retired"]
y=employed
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
xticks=x
plt.xticks(rotation=90)
yticks=y
xlabel=x
ylabel=y
title="%age of employement categories"
width=1/1.5
plt.bar(x,y,width,color=random_colors(6))

y=so_data.FormalEducation.value_counts()

x=["Bechlor","Master","degreeless-study","secondary_school","Associate","doctoral","primar/elemantry","Professional","No_formal_education"]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
width=1/1.5
xticks=x
plt.xticks(rotation=90)
yticks=y

xlabel=x
ylabel="developers with formal education"
plt.bar(x,y,width,color=random_colors(9))


x=["Bechlor","Master","degreeless-study","secondary_school","Associate","doctoral","primar/elemantry","Professional","No_formal_education"]

fe=so_data["FormalEducation"].value_counts()
data=[go.Pie(labels=x,
            values=fe,
            marker=dict(colors=random_colors(9)),
            )]
layout=go.Layout(title="%of developers with formal education")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)

so_data.head(4)
major=so_data.UndergradMajor.value_counts()
print(major)
major=so_data.UndergradMajor.value_counts()

data=[go.Pie(labels=major.index,
            values=major.values,
            marker=dict(colors=random_colors(12)),
            )]
layout=go.Layout(title="% of developers with undergraduate major")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
so_data.CompanySize.value_counts()
y=so_data.CompanySize.value_counts()
x=y.index
fig=plt.Figure()
ax=fig.add_subplot(1,1,1)
xlabel=x
ylabel=y
xticks=x
plt.xticks(rotation=90)
yticks=y
plt.grid()
width=1/1.5
plt.bar(x,y,width,color=random_colors(8))
y=so_data.CompanySize.value_counts()

data=[go.Pie(labels=y.index,
             values=y,
             marker=dict(colors=random_colors(8)),
            )]
layout=go.Layout(title="%age of developer by compay size")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
temp=pd.DataFrame(so_data.DevType.dropna().str.split(";").tolist()).stack()
grouped=temp.value_counts()

x=grouped.index
y=(grouped/grouped.sum())*100

fig=plt.Figure()
ax=fig.add_subplot(1,1,1)
xlabel=x
xticks=x
plt.grid()
plt.xticks(rotation=90)
ylabel=y
width=1/1.5
plt.bar(x,y,width,color=random_colors(20))
plt.show()
y=so_data.JobSatisfaction.value_counts()
data=[go.Pie(labels=y.index,
            values=y.values,
            marker=dict(colors=random_colors(7)),
            )]
layout=go.Layout(title="% of developers satisfied with current job")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
y=so_data.CareerSatisfaction.value_counts()
data=[go.Pie(labels=y.index,
            values=y.values,
            marker=dict(colors=random_colors(7)),
            )]
layout=go.Layout(title="% of developers satisfied with their career")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
temp=pd.DataFrame(so_data.LanguageWorkedWith.dropna().str.split(";").tolist()).stack()
lang=temp.unique()
print("Total language are:",len(lang))
print("which are :",lang)
print(temp.value_counts())



y=temp.value_counts().head(10)
fig=plt.figure()
ax=plt.axes()
x=y.index
plt.xticks=x
plt.yticks=y

plt.plot(x,y,color="red")





temp=pd.DataFrame(so_data.LanguageWorkedWith.dropna().str.split(";").tolist()).stack()
y=temp.value_counts().head(10)
fig=plt.figure()
ax=plt.axes()
x=y.index.unique()


plt.scatter(x,y,color="red",alpha=0.8)

dbs=pd.DataFrame(so_data.DatabaseWorkedWith.dropna().str.split(";").tolist()).stack()
top_db=dbs.value_counts().head(10)
data=[go.Bar(
    x=top_db.index,
    y=top_db.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="top 10 databases")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
nextdb=pd.DataFrame(so_data.DatabaseDesireNextYear.dropna().str.split(";").tolist()).stack()
temp=nextdb.value_counts().head(10)
data=[go.Bar(
    x=temp.index,
    y=temp.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="Popular databases of next year")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)

db_2018=pd.DataFrame(so_data.DatabaseWorkedWith.dropna().str.split(";").tolist()).stack()
db_2019=pd.DataFrame(so_data.DatabaseDesireNextYear.dropna().str.split(";").tolist()).stack()
top_2018=db_2018.value_counts().head(10)
top_2019=db_2019.value_counts().head(10)

trace1=go.Bar(
    x=top_2018.index,
    y=top_2018.values,
  
    name="Top 10 databases of 2018",
)
trace2=go.Bar(
    x=top_2019.index,
    y=top_2019.values,
  
    name="Top 10 databases of 2019",
)
data=[trace1,trace2]
layout=go.Layout(title="databases analysis 2018 vs 2019")

fig=go.Figure(data=data,layout=layout)
py.iplot(fig)

plate_form=pd.DataFrame(so_data.PlatformWorkedWith.dropna().str.split(";").tolist()).stack()
top_pf18=plate_form.value_counts().head(10)
data=[go.Bar(
    x=top_pf18.index,
    y=top_pf18.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="top 10 plateforms of 2018")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
next_pf=pd.DataFrame(so_data.PlatformDesireNextYear.dropna().str.split(";").tolist()).stack()
top_pf19=next_pf.value_counts().head(10)
data=[go.Bar(
    x=top_pf19.index,
    y=top_pf19.values,
    marker=dict(color=random_colors(10)),
)]
layout=go.Layout(title="top 10 plateforms of 2019")
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


