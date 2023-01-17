# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")
student=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df=student.copy()
df.head()
df.info()
df.describe().T
df.corr()
df.rename(columns = {'race/ethnicity': 'race'}, inplace = True)
ax = sns.barplot(x="race", y="math score", data=df)
plt.ylabel('Math Score')
plt.xlabel('Race')
plt.title('Math Score With Race');
ax = sns.barplot(x="race", y="math score", hue="gender", data=df,palette='autumn')
plt.ylabel('Math Score')
plt.xlabel('Race')
plt.title('Math Score With Race,Gender');
ax = sns.barplot(x="race", y="math score", hue="parental level of education", data=df,palette='rainbow')
plt.ylabel('Math Score')
plt.xlabel('Race')
plt.title('Math Score With Race,Gender,Parental Level Of Education');
ax = sns.barplot(x="race", y="reading score", data=df,
                 linewidth=2.5, facecolor=(1, 1, 1, 0),
                 errcolor=".2", edgecolor=".2")
plt.ylabel('Reading Score')
plt.xlabel('Race')
plt.title('Reading Score With Race');
df_rece_math=df.groupby(by =['race'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

fig = go.Figure(go.Bar(
    y=df_rece_math['Race'],x=df_rece_math['Reading Score'],orientation="h",
    marker={'color': df_rece_math['Reading Score'], 
    'colorscale': 'sunsetdark'},  
    text=df_rece_math['Reading Score'],
    textposition = "outside",
))
fig.update_layout(title_text='Mean Reading Score',xaxis_title="Race",yaxis_title="Reading Score",title_x=0.5)
fig.show()
df_rece_math=df.groupby(by =['race'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

fig = go.Figure(go.Bar(
    x=df_rece_math['Race'],y=df_rece_math['Reading Score'],
    marker={'color': df_rece_math['Reading Score'], 
    'colorscale': 'Viridis'},  
    text=df_rece_math['Reading Score'],
    textposition = "outside",
))
fig.update_layout(title_text='Mean Reading Score',xaxis_title="Race",yaxis_title="Reading Score",title_x=0.5)
fig.show()
df_rece_gender_math=df.groupby(by =['race','gender'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

fig = px.bar(df_rece_gender_math, x="Race", y="Reading Score",color="gender",barmode="group",
             
             )
fig.update_layout(title_text='Mean Reading Score with Gender',title_x=0.5)
fig.show()
df_rece_gender_math=df.groupby(by =['race','parental level of education'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

fig = px.bar(df_rece_gender_math, x="Race", y="Reading Score",color="parental level of education",barmode="group",
             
             )
fig.update_layout(title_text='Mean Reading Score with Gender',title_x=0.5)
fig.show()
sns.scatterplot(data=df, x="math score", y="reading score")
plt.ylabel('Reading Score')
plt.xlabel('Math Score')
plt.title('Reading Score With Math Score');
sns.scatterplot(data=df, x="writing score", y="math score", hue="gender", style="gender")
plt.ylabel('Reading Score')
plt.xlabel('Writing Score')
plt.title('Reading Score Vs Writing Score With Gender');
sns.scatterplot(data=df, x="writing score", y="math score", hue="parental level of education")
plt.ylabel('Math Score')
plt.xlabel('Writing Score')
plt.title('Math Score Vs Writing Score With Parental Level Of Education');
fig = px.scatter(df, x='math score', y='writing score')
fig.update_layout(title='Math Score Vs Writing Score',xaxis_title="Math Score",yaxis_title="Writing Score",title_x=0.5)
fig.show()
fig = px.scatter(df, x='math score', y='writing score', color='gender')
fig.update_layout(title='Math Score Vs Writing Score With Gender',xaxis_title="Math Score",yaxis_title="Writing Score",title_x=0.5)
fig.show()
fig = go.Figure(data=go.Scatter(x=df['parental level of education'],
                                y=df['math score'],
                                mode='markers',
                                marker_color=df['math score'],
                                text=df['parental level of education'])) # hover text goes here

fig.update_layout(title='Math Score With Parental Level Of Education',title_x=0.5)
fig.show()
fig = px.scatter(df, x='math score', y='writing score', color='parental level of education')
fig.update_layout(title='Math Score Vs Writing Score With Parental Level Of Education',xaxis_title="Math Score",yaxis_title="Writing Score",title_x=0.5)
fig.show()
sns.lineplot(data=df, x="math score", y="reading score")
plt.ylabel('Reading Score')
plt.xlabel('Math Score')
plt.title('Reading Score Vs Math Score ');
sns.lineplot(data=df, x="math score", y="reading score",hue='gender')
plt.ylabel('Reading Score')
plt.xlabel('Math Score')
plt.title('Reading Score Vs Math Score With Gender');
sns.lineplot(data=df, x="math score", y="reading score", hue="race")
plt.ylabel('Reading Score')
plt.xlabel('Math Score')
plt.title('Reading Score Vs Math Score With Race');
df_rece_gender_reading=df.groupby(by =['race','parental level of education'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

df_groupa=df_rece_gender_reading[df_rece_gender_reading['Race']=='group A']

fig = go.Figure(go.Scatter(x=df_groupa['parental level of education'], y=df_groupa['Reading Score']))
fig.show()
df_rece_gender_reading=df.groupby(by =['race','parental level of education'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

df_groupa=df_rece_gender_reading[df_rece_gender_reading['Race']=='group A']
df_groupb=df_rece_gender_reading[df_rece_gender_reading['Race']=='group B']
df_groupc=df_rece_gender_reading[df_rece_gender_reading['Race']=='group C']

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_groupa['parental level of education'],
                         y=df_groupa['Reading Score'],
                    mode='lines+markers',
                    name='group A'))
fig.add_trace(go.Scatter(x=df_groupb['parental level of education'],
                         y=df_groupb['Reading Score'],
                    mode='lines',
                    name='group B'))
fig.add_trace(go.Scatter(x=df_groupc['parental level of education'],
                         y=df_groupc['Reading Score'],
                    mode='markers', name='group C'))

fig.show()
df_rece_gender_reading=df.groupby(by =['race','parental level of education'])['reading score'].mean().to_frame().reset_index().rename(columns={'race':'Race','reading score':'Reading Score'})

df_groupd=df_rece_gender_reading[df_rece_gender_reading['Race']=='group D']
df_groupe=df_rece_gender_reading[df_rece_gender_reading['Race']=='group E']


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_groupd['parental level of education'],
                         y=df_groupd['Reading Score'],
                         name='group D',
                         line=dict(color='brown', width=4,dash="dash")))
fig.add_trace(go.Scatter(x=df_groupe['parental level of education'],
                         y=df_groupe['Reading Score'],
                         name='group E',
                         line=dict(color='green', width=4,dash="dashdot")))


fig.show()
# Race rates according in data 

labels = df.race.value_counts().index
colors = ['grey','blue','red','yellow','green']
explode = [0.1,0,0,0,0]
sizes = df.race.value_counts().values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Race ',color = 'blue',fontsize = 15);

df_category=df['gender'].value_counts().to_frame().reset_index().rename(columns={'index':'gender','gender':'count'})


fig = go.Figure([go.Pie(labels=df_category['gender'], values=df_category['count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title="Gender Count",title_x=0.5)
fig.show()

df_category=df['parental level of education'].value_counts().to_frame().reset_index().rename(columns={'index':'parental level of education','parental level of education':'count'})


fig = go.Figure([go.Pie(labels=df_category['parental level of education'], values=df_category['count'],hole=0.3)]) # can change the size of hole 

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title="Parental Level Of Education Students Count",title_x=0.5)
fig.show()
df_race=df['race'].value_counts().to_frame().reset_index().rename(columns={'index':'race','race':'count'})

colors=['cyan','royalblue','blue','darkblue',"darkcyan"]
fig = go.Figure([go.Pie(labels=df_race['race'], values=df_race['count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Race Students Count",title_x=0.5)
fig.show()
df_parental=df['parental level of education'].value_counts().to_frame().reset_index().rename(columns={'index':'parental level of education','parental level of education':'Count'})
df_parental

fig = go.Figure(data=[go.Scatter(
    x=df_parental['parental level of education'], y=df_parental['Count'],
    mode='markers',
    marker=dict(
        color=df_parental['Count'],
        size=df_parental['Count']*0.3, # Multiplying by 0.3 to reduce size and stay uniform accross all points
        showscale=True
    ))])

fig.update_layout(title='Parental Level Of Education',xaxis_title="Level Of Education",yaxis_title="Number Of Student",title_x=0.5)
fig.show()
ax = sns.countplot(x="gender", data=df)
plt.ylabel('Count')
plt.xlabel('Gender')
plt.title('Gender Count');
ax = sns.countplot(x="gender", hue='race',data=df,palette='bone')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.title('Gender Count With Race');
g = sns.catplot(x="gender", hue="race", col="lunch",
                data=df, kind="count",
                height=4, aspect=.7);

ax = sns.swarmplot(x=df["math score"])
plt.xlabel('Math Score')
plt.title('Math Score Distribution');
ax = sns.swarmplot(y="math score",x='race',data=df)
plt.ylabel('Math Score')
plt.title('Math Score Distribution With Race');
ax = sns.swarmplot(x="race", y="math score", hue="gender", data=df)
plt.ylabel('Math Score');
plt.title('Math Score Distribution With Gender');
sns.pairplot(df);

sns.pairplot(df, hue="gender");
sns.pairplot(df, hue="gender", diag_kind="hist");
sns.jointplot(data=df, x="math score", y="reading score");

sns.jointplot(data=df, x="math score", y="reading score", kind="reg");
sns.jointplot(data=df, x="math score", y="reading score", kind="hex");
ax = sns.pointplot(y="math score", x="gender", data=df,hue='lunch',palette="gnuplot")
plt.xlabel('Gender')
plt.ylabel('Math Score');

ax = sns.pointplot(x="math score", y="gender", data=df,hue='race',palette="ocean")
plt.ylabel('Gender')
plt.xlabel('Math Score');
g = sns.catplot(x="gender", y="math score",
                hue="race", col="lunch",
                data=df, kind="point",
                dodge=True,
                height=4, aspect=.7);

sns.kdeplot(df['math score'],shade=True,color='LightSeaGreen');
plt.ylabel('Possibility')
plt.xlabel('Math Score');
sns.kdeplot(df['reading score'],color='Indigo')
sns.kdeplot(df['writing score'],color='IndianRed')
plt.ylabel('Possibility')
plt.xlabel('Score');
sns.kdeplot(df['math score'],df['writing score'])
plt.ylabel('Writing Score')
plt.xlabel('Math Score')
plt.show()
ax = sns.boxplot(x=df["math score"])

plt.xlabel('Math Score Distributions ')
plt.show()
ax = sns.boxplot(y="math score", x="gender", data=df)
plt.ylabel('Math Score ')
plt.xlabel('Gender ')
plt.title('Math Score Distributions With Gender')
plt.show()

ax = sns.boxplot(x="gender", y="math score", hue="lunch",
                 data=df, linewidth=2.5)
plt.ylabel('Math Score ')
plt.xlabel('Gender ')
plt.title('Math Score Distributions With Gender,Lunch')
plt.show()
ax = sns.boxplot(data=df, orient="h", palette="gist_rainbow")
plt.xlabel(' Score ')
plt.title('Math,Reading And Writing Score Distributions ')
plt.show()
ax = sns.boxplot(x="race", y="math score", data=df)
ax = sns.swarmplot(x="race", y="math score", data=df, color=".25")
plt.xlabel(' Race ')
plt.ylabel('Math Score ')
plt.title('Math Score Distributions With Box And Swarm plot ')
plt.show()
ax = sns.violinplot(x="gender", y="math score", data=df)
plt.xlabel(' Gender ')
plt.ylabel('Math Score ')
plt.title('Math Score Distributions With Gender ')
plt.show()
ax = sns.violinplot(x="gender", y="math score", hue="race",
                    data=df, palette="cool")
plt.xlabel(' Gender ')
plt.ylabel('Math Score ')
plt.title('Math Score Distributions With Gender,Race ')
plt.show()
ax = sns.violinplot(data=df, palette="coolwarm")
plt.ylabel(' Score ')
plt.title('Math,Reading And Writing Score Distributions')
plt.show()
ax=sns.distplot(df["math score"])
plt.ylabel(' Possibility ')
plt.xlabel(' Score ')
plt.title('Math Score Distributions')
plt.show()

sns.distplot(df['reading score'], bins = 10, kde = False);
plt.ylabel(' Count ')
plt.xlabel(' Score ')
plt.title('Reading Score Distributions')
plt.show()

sns.heatmap(df.corr(),annot=True, linewidths=.5,fmt="f", cmap="inferno")
plt.show()
fig = go.Figure(data=[go.Histogram(x=df['math score'],  # To get Horizontal plot ,change axis 
                                  marker_color="DarkOrange",
                      xbins=dict(
                      start=0, #start range of bin
                      end=100,  #end range of bin
                      size=10    #size of bin
                      ))])
fig.update_layout(title="Distribution Of Math Score",xaxis_title="Points",yaxis_title="Counts",title_x=0.5)
fig.show()
fig = go.Figure(data=[go.Histogram(x=df['math score'],histnorm='probability',
                                  marker_color="orange")]) # To get Horizontal plot ,change axis 
fig.update_layout(title="Distribution of Math Score",xaxis_title="Score",yaxis_title="Possibility")
fig.show()
fig = go.Figure()
fig.add_trace(go.Histogram(x=df['math score'],marker_color="green",name="Math"))
fig.add_trace(go.Histogram(x=df['reading score'],marker_color="orange",name="Reading"))
#fig.add_trace(go.Histogram(x=df['writing score'],marker_color="SaddleBrown",name="Writing"))
# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.55)
fig.update_layout(title="Distribution Of Math &  Reading Score",xaxis_title="Score",title_x=0.5,yaxis_title="Counts")
fig.show()
fig = go.Figure(go.Box(y=df['reading score'],name="Reading Score")) # to get Horizonal plot change axis 
fig.update_layout(title="Distribution of Reading Score",title_x=0.5)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(y=df['math score'],
                     marker_color="Maroon",
                     boxmean='sd',
                     name="Math Score"))
fig.add_trace(go.Box(y=df['reading score'],
                     boxmean=True,
                     marker_color="Plum",
                     name="Reading Score"))
fig.add_trace(go.Box(y=df['writing score'],
                     boxpoints='all',
                     marker_color="SandyBrown",
                     name="Writing Score"))
fig.update_layout(title="Distribution of Math Reading Writing Score",title_x=0.5)
fig.show()
fig = go.Figure(data=go.Violin(y=df['math score'],
                               marker_color="Sienna",
                               x0='Math score'))

fig.update_layout(title="Distribution of Math Score",title_x=0.5)
fig.show()
fig = go.Figure()
fig.add_trace(go.Violin(y=df['math score'],
                     marker_color="Tomato",
                     box_visible=True, 
                     name="Math Score"))

fig.add_trace(go.Violin(y=df['reading score'],
                     marker_color="darkcyan",
                     meanline_visible=True,
                     name="Reading Score"))

fig.add_trace(go.Violin(y=df['writing score'],
                     fillcolor="darkblue",
                     line_color='black',
                     opacity=0.6,
                     name="Writing Score"))

fig.update_layout(title="Distribution Of Score",title_x=0.5)
fig.show()

fig = px.sunburst(df, path=['gender', 'race','parental level of education',], values='math score',
                   color=df['math score'],
                  color_continuous_scale='electric')
fig.show()
df_all_100=df[(df['math score']==100)&(df['reading score']==100)&(df['writing score']==100)]

colors=['lightblue','lightpink','lightgreen','yellow','DarkSalmon','Khaki','LightCoral']
    
fig = go.Figure(data=[go.Table(header=dict(values=['Gender', 'Race','Parental Level Of Education','Test Preparation Course','Math Score','Reading Score','Writing Score'],
                                          line_color='white', fill_color='LightSlateGray',
                                  align='center',font=dict(color='white', size=12)
                                          ),
                               
                 cells=dict( values=[df_all_100['gender'], df_all_100['race'],df_all_100['parental level of education'],df_all_100['test preparation course'],df_all_100['math score'],df_all_100['reading score'],df_all_100['writing score']],
                           line_color=colors, fill_color=colors,
                           align='center', font=dict(color='#660033', size=11))
                              )])
                      
fig.show()