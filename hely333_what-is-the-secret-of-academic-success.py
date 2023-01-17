import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20,.2f}'.format


data = pd.read_csv('../input/student-mat.csv')
data.head()
sns.catplot(x="sex", kind="count",palette="magma", data=data, height = 6)
plt.title("Gender of students : F - female,M - male")
ages = data["age"].value_counts()
labels = (np.array(ages.index))
sizes = (np.array((ages / ages.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="Аge of students")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.iplot(fig, filename="age")
data['st_time'] = np.nan
df = [data]

for col in df:
    col.loc[col['studytime'] == 1 , 'st_time'] = '< 2 hours'
    col.loc[col['studytime'] == 2 , 'st_time'] = '2 to 5 hours'
    col.loc[col['studytime'] == 3, 'st_time'] = '5 to 10 hours'
    col.loc[col['studytime'] == 4, 'st_time'] = '> 10 hours'  
 
labels = data["st_time"].unique().tolist()
amount = data["st_time"].value_counts().tolist()

colors = ["pink", "cyan", "green", "yellow"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
dt = [trace]
layout = go.Layout(title="Study time")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename='pie')
plt.figure(figsize=(18,7))
plt.title("Box plot for final grades,depending on the study time")
sns.boxplot(y="st_time", x="G3", data = data , orient="h", palette = 'rainbow')
sns.catplot(x="address", kind="count",hue = "traveltime",palette="brg", data=data, height = 6)
plt.title("Students address: U - urban, R - rural")
f= plt.figure(figsize=(18,7))

ax=f.add_subplot(121)
sns.distplot(data[(data.address == 'U')]["absences"],color='orange',ax=ax)
ax.set_title('Distribution of absences for students who live is city')

ax=f.add_subplot(122)
sns.distplot(data[(data.address == 'R')]['absences'],color='gray',ax=ax)
ax.set_title('Distribution of absences for students who live in village')
sns.lmplot(x="absences", y="G3",hue = 'address',data=data, palette = 'inferno_r', size = 7)

f= plt.figure(figsize=(17,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.romantic == 'no')]["absences"],color='coral',ax=ax)
ax.set_title('Distribution of absences for classes by single people')

ax=f.add_subplot(122)
sns.distplot(data[(data.romantic == 'yes')]['absences'],color='purple',ax=ax)
ax.set_title('Distribution of absences for classes by people in love')
f= plt.figure(figsize=(17,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.romantic == 'no')]["G3"],color='cyan',ax=ax)
ax.set_title('Distribution of grades in  single people')

ax=f.add_subplot(122)
sns.distplot(data[(data.romantic == 'yes')]['G3'],color='blue',ax=ax)
ax.set_title('Distribution of grades in people in love')
sns.catplot(x="romantic", kind="count",palette="pink", data=data, height = 7)
plt.title("How many students are in a romantic relationship?")
labels = data["health"].unique().tolist()
amount = data["health"].value_counts().tolist()

colors = ["coral","lightgreen","pink","cyan","white"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Current health status (numeric: from 1 - very bad to 5 - very good)")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 'h_chart')
plt.figure(figsize=(16,5))
plt.title("Box plot for final grades,depending on current health")
sns.boxplot(y="health", x="G3", data = data , orient="h", palette = 'winter')
plt.figure(figsize=(16,5))
plt.title("Box plot for absences,depending on current health")
sns.boxplot(y="health", x="absences", data = data , orient="h", palette = 'summer')
labels = data["Dalc"].unique().tolist()
amount = data["Dalc"].value_counts().tolist()

colors = ["pink","lightgreen","white","cyan","gray"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 'rt')
f= plt.figure(figsize=(17,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Dalc == 5)]["absences"],color='green',ax=ax)
ax.set_title('Distribution of absences for people who consume a lot of alcohol on weekdays')

ax=f.add_subplot(122)
sns.distplot(data[(data.Dalc == 1)]['absences'],color='blue',ax=ax)
ax.set_title('Distribution of absences for people who consume little alcohol on weekdays')
f= plt.figure(figsize=(17,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Dalc == 5)]["G3"],color='red',ax=ax)
ax.set_title('Distribution of grades for people who consume a lot of alcohol on weekdays')

ax=f.add_subplot(122)
sns.distplot(data[(data.Dalc == 1)]['G3'],color='gray',ax=ax)
ax.set_title('Distribution of grades for people who consume little alcohol on weekdays')
labels = data["Walc"].unique().tolist()
amount = data["Walc"].value_counts().tolist()

colors = ["yellow","cyan","green","orange","gray"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 't')
f= plt.figure(figsize=(17,5))

ax=f.add_subplot(121)
sns.distplot(data[(data.Walc == 5)]["G3"],color='lightgreen',ax=ax)
ax.set_title('Distribution of grades for people who consume a lot of alcohol on weekend')

ax=f.add_subplot(122)
sns.distplot(data[(data.Walc == 1)]['G3'],color='black',ax=ax)
ax.set_title('Distribution of grades for people who consume little alcohol on weekend')
g = sns.jointplot(x="age", y="G3", data = data[(data.paid == 'yes')],kind="kde", color="cyan")
g.plot_joint(plt.scatter, c="black", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
ax.set_title('Distribution of final grades and age for students who have additional paid classes')
g = sns.jointplot(x="age", y="G3", data = data[(data.paid == 'no')],kind="kde", color="pink")
g.plot_joint(plt.scatter, c="black", s=30, linewidth=1, marker="*")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")
ax.set_title('Distribution of final grades and age for students who do not have additional paid classes')
f= plt.figure(figsize=(18,6))

ax=f.add_subplot(121)
sns.distplot(data[(data.paid == 'yes')]["G3"],color='lightgreen',ax=ax)
ax.set_title('Distribution of grades for students who have additional paid classes')

ax=f.add_subplot(122)
sns.distplot(data[(data.paid == 'no')]['G3'],color='coral',ax=ax)
ax.set_title('Distribution of grades for students who do not have additional paid classes')
sns.catplot(x="higher", kind="count",palette="rocket", data=data, height = 6)
plt.title("How many students want to ger higher education?")
plt.figure(figsize=(17,5))
plt.title("Box plot for final grades,depending on the desire to have higher education")
sns.boxplot(y="higher", x="G3", data = data , orient="h", palette = 'tab10')

f= plt.figure(figsize=(17,5))
ax=f.add_subplot(121)
sns.distplot(data[(data.higher == 'yes')]["G3"],color='orange',ax=ax)
ax.set_title('Distribution of grades for students who wants to get higher education')

ax=f.add_subplot(122)
sns.distplot(data[(data.higher == 'no')]['G3'],color='red',ax=ax)
ax.set_title('Distribution of grades for students who does not want to get higher education')
sns.catplot(x="internet", kind="count",palette="autumn", data=data, height = 6)
plt.title("How many students have not Internet (yes, i am shocked too)?")
time1 =data[(data.internet == 'no')].st_time.value_counts()
labels = (np.array(time1.index))
sizes = (np.array((time1 / time1.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="How many hours do students without access to the Internet spend on studies?")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.iplot(fig, filename="time1")
time2 =data[(data.internet == 'yes')].st_time.value_counts()
labels = (np.array(time2.index))
sizes = (np.array((time2 / time2.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title="How many hours do students with access to the Internet spend on studies?")
dat = [trace]
fig = go.Figure(data=dat, layout=layout)
py.iplot(fig, filename="time2")
plt.figure(figsize=(17,5))
plt.title("Box plot for final grades,depending on the access to the Internet")
sns.boxplot(y="internet", x="G3", data = data , orient="h", palette = 'pink')

sns.catplot(x="famsize", kind="count",hue = "Pstatus",palette="spring", data=data, height = 7)
plt.title("Number of people in the family: GT3 - more than 3, LE3 - less than 3")
labels = data["Mjob"].unique().tolist()
amount = data["Mjob"].value_counts().tolist()

colors = ["orange", "green", "yellow", "white",'cyan']

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Mother's job")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 'pi_chart')
labels = data["Fjob"].unique().tolist()
amount = data["Fjob"].value_counts().tolist()

colors = ["coral","lightgreen","gray","cyan","white"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Father's job")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 'pg_chart')
plt.figure(figsize=(16,5))
plt.title("Box plot for final grades,depending on mothers profession")
sns.boxplot(y="Mjob", x="G3", data = data , orient="h", palette = 'winter')

plt.figure(figsize=(16,5))
plt.title("Box plot for final grades,depending on fathers profession")
sns.boxplot(y="Fjob", x="G3", data = data , orient="h", palette = 'summer')
labels = data["famrel"].unique().tolist()
amount = data["famrel"].value_counts().tolist()

colors = ["pink","cyan","coral","orange","white"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Quality of family relationships(numeric: from 1 - very bad to 5 - excellent)")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 'pf_chart')
labels = data[(data.Pstatus == 'T')].famrel.unique().tolist()
amount = data[(data.Pstatus == 'T')].famrel.value_counts().tolist()

colors = ["yellow","cyan","pink","orange","white"]

trace = go.Pie(labels=labels, values=amount,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

dt = [trace]
layout = go.Layout(title="Quality of relationships in families where parents live together (numeric: from 1 - very bad to 5 - excellent)")

fig = go.Figure(data=dt, layout=layout)
iplot(fig, filename = 'pf1_chart')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#sex
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
#address
le.fit(data.address.drop_duplicates()) 
data.address = le.transform(data.address)
#famsize
le.fit(data.famsize.drop_duplicates()) 
data.famsize = le.transform(data.famsize)
#Pstatus
le.fit(data.Pstatus.drop_duplicates()) 
data.Pstatus = le.transform(data.Pstatus)
#schoolsup
le.fit(data.schoolsup.drop_duplicates()) 
data.schoolsup = le.transform(data.schoolsup)
#famsup
le.fit(data.famsup.drop_duplicates()) 
data.famsup = le.transform(data.famsup)
#paid
le.fit(data.paid.drop_duplicates()) 
data.paid = le.transform(data.paid)
#activities
le.fit(data.activities.drop_duplicates()) 
data.activities = le.transform(data.activities)
#nursery
le.fit(data.nursery.drop_duplicates()) 
data.nursery = le.transform(data.nursery)
#higher
le.fit(data.higher.drop_duplicates()) 
data.higher = le.transform(data.higher)
#romantic
le.fit(data.romantic.drop_duplicates()) 
data.romantic = le.transform(data.romantic)
#internet
le.fit(data.internet.drop_duplicates()) 
data.internet = le.transform(data.internet)
#not binary features
data = data.drop(["st_time"], axis = 1) #I created this column for one of the graphs

#not binary features
data= pd.get_dummies(data)
data.head()
plt.figure(figsize=(20,9))
cols = ['Dalc','Walc','G3','G2', 'G1','studytime','romantic','internet','age','sex','Pstatus']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.2)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)