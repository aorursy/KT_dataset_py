import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
plt.style.use('ggplot')
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode(connected=True)
from plotly import tools
from wordcloud import  WordCloud

df=pd.read_csv("../input/train/train.csv")
df=df.copy()
print('The shape of dataframe',df.shape)
df.head(2)
df.info()
plt.figure(figsize=(6,6))
missing=df.isnull().sum()
sns.barplot(y=missing.index,x=missing)
plt.show()
bar=pd.DataFrame(df['AdoptionSpeed'].value_counts())
sns.barplot(x=[4,3,2,1,0],y=bar['AdoptionSpeed'])
plt.show()

plt.figure(figsize=(14,8))

plt.subplot(1,5,1)
sex=df[df["AdoptionSpeed"]==4]["Gender"].value_counts()
labels = ['Female','Male','Mixed']
sizes = sex
colors = [ 'lightcoral', 'lightskyblue','yellowgreen']
explode = (0.1, 0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.gca().set_title("AdoptionSpeed 4")

plt.subplot(1,5,2)
sex=df[df["AdoptionSpeed"]==3]["Gender"].value_counts()
labels = ['Female','Male','Mixed']
sizes = sex
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.gca().set_title("AdoptionSpeed 3")

plt.subplot(1,5,3)
sex=df[df["AdoptionSpeed"]==2]["Gender"].value_counts()
labels = ['Female','Male','Mixed']
sizes = sex
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.gca().set_title(" AdoptionSpeed 2")

plt.subplot(1,5,4)
sex=df[df["AdoptionSpeed"]==1]["Gender"].value_counts()
labels = ['Female','Male','Mixed']
sizes = sex
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.gca().set_title(" AdoptionSpeed 1")

plt.subplot(1,5,5)
sex=df[df["AdoptionSpeed"]==0]["Gender"].value_counts()
labels = ['Female','Male','Mixed']
sizes = sex
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.gca().set_title(" AdoptionSpeed 0")


plt.show()
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.boxplot(df['Fee'])
plt.gca().set_title("Fee distribution")
plt.subplot(1,2,2)
sns.violinplot(x="AdoptionSpeed",y="Fee",hue="Type",data=df)
plt.gca().set_title("Fee distribution by AdoptionSpeed and Type")
plt.show()
fig,((ax1,ax2))=plt.subplots(1,2)
fig.set_figwidth(15)
age=df.groupby('Type')['Age'].agg('mean')
sns.barplot(['Dog',"Cat"],age,ax=ax1)
ax1.set_ylabel("Age in months")
sns.boxplot(x='Type',y="Age",data=df,ax=ax2)
plt.gca().set_ylabel('Age in months')
ax1.set_title("mean age of adoption")
ax2.set_title(" age distribution By Types")
ax2.set_xticklabels(['Dog',"Cat"])
plt.show()
plt.figure(figsize=(10, 6));
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=df);
plt.title('AdoptionSpeed by Type and age');
plt.gca().set_ylabel("Age in months")
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(x="Type",hue='AdoptionSpeed',data=df)
plt.gca().set_xticklabels(['Dog',"Cat"])
plt.show()
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.countplot(x="FurLength",hue="AdoptionSpeed",data=df)
plt.gca().set_title("Furlength type by AdoptionSpeed")

plt.subplot(2,2,2)
total=df['FurLength'].value_counts()
sns.barplot(total.index,total)

plt.subplot(2,2,3)
sns.countplot(x="FurLength",hue="AdoptionSpeed",data=df[df['Type']==1])
plt.gca().set_title("Dog")

plt.subplot(2,2,4)
sns.countplot(x="FurLength",hue="AdoptionSpeed",data=df[df['Type']==2])
plt.gca().set_title("Cat")


def compare_plot(typeo,col,title):
    tr1=df[(df['Type']==typeo) & (df[col]==1)]['AdoptionSpeed'].value_counts()
    tr2=df[(df['Type']==typeo) & (df[col]==2)]['AdoptionSpeed'].value_counts()
    tr3=df[(df['Type']==typeo) & (df[col]==3)]['AdoptionSpeed'].value_counts()
    
    xx=[4,3,2,1,0]
    trace1=go.Bar(y=tr1,x=xx,name='Yes')
    trace2=go.Bar(y=tr2,x=xx,name='No')
    trace3=go.Bar(y=tr3,x=xx,name='Not Sure')
    
    return (trace1,trace2,trace3)




        


dog=df[df['Type']==1]['Vaccinated'].value_counts()
cat=df[df['Type']==2]['Vaccinated'].value_counts()

trace11=go.Bar(x=dog.index,y=dog,name='dog')
trace22=go.Bar(x=cat.index,y=cat,name="cat")

trace1,trace2,trace3=compare_plot(1,'Vaccinated','')
trace4,trace5,trace6=compare_plot(2,"Vaccinated",'')


fig=tools.make_subplots(rows=2,cols=2,subplot_titles=['Number of pets','','Dogs',"Cats"])

fig.append_trace(trace11,1,1)
fig.append_trace(trace22,1,1)

fig.append_trace(trace1,2,1)
fig.append_trace(trace2,2,1)
fig.append_trace(trace3,2,1)

fig.append_trace(trace4,2,2)
fig.append_trace(trace5,2,2)
fig.append_trace(trace6,2,2)
fig['layout'].update(height=1000,width=800, title="AdoptionSpeed by Type and Vaccination", barmode="stack", showlegend=False)

plotly.offline.iplot(fig)
trace1,trace2,trace3=compare_plot(1,'Dewormed','')
trace4,trace5,trace6=compare_plot(2,"Dewormed",'')


fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Dogs',"Cats"])


fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,1)
fig.append_trace(trace3,1,1)

fig.append_trace(trace4,1,2)
fig.append_trace(trace5,1,2)
fig.append_trace(trace6,1,2)
fig['layout'].update(height=500,width=800, title="AdoptionSpeed by Type and Dewormed", barmode="stack", showlegend=False,xaxis=dict(title="AdoptionSpeed"))

plotly.offline.iplot(fig)
trace1,trace2,trace3=compare_plot(1,'Sterilized','')
trace4,trace5,trace6=compare_plot(2,"Sterilized",'')


fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Dogs',"Cats"])


fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,1)
fig.append_trace(trace3,1,1)

fig.append_trace(trace4,1,2)
fig.append_trace(trace5,1,2)
fig.append_trace(trace6,1,2)
fig['layout'].update(height=500,width=800, title="AdoptionSpeed by Type and Sterilized", barmode="stack", showlegend=False,xaxis=dict(title="AdoptionSpeed"))

plotly.offline.iplot(fig)
def compare_plot2(typeo):
    tr1=df[(df['Type']==typeo) & (df["MaturitySize"]==1)]['AdoptionSpeed'].value_counts()
    tr2=df[(df['Type']==typeo) & (df["MaturitySize"]==2)]['AdoptionSpeed'].value_counts()
    tr3=df[(df['Type']==typeo) & (df["MaturitySize"]==3)]['AdoptionSpeed'].value_counts()
    tr4=df[(df['Type']==typeo) & (df["MaturitySize"]==4)]['AdoptionSpeed'].value_counts()
    tr5=df[(df['Type']==typeo) & (df["MaturitySize"]==0)]['AdoptionSpeed'].value_counts()
    xx=[4,3,2,1,0]
    trace1=go.Bar(y=tr1,x=xx,name='Small')
    trace2=go.Bar(y=tr2,x=xx,name='Medium')
    trace3=go.Bar(y=tr3,x=xx,name='large')
    trace4=go.Bar(y=tr4,x=xx,name='Extralarge')
    trace5=go.Bar(y=tr5,x=xx,name='Not Sprecified')
    return(trace1,trace2,trace3,trace4,trace5)

tr1,tr2,tr3,tr4,tr5=compare_plot2(1)

tr11,tr22,tr33,tr44,tr55=compare_plot2(2)

fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Dogs',"Cats"])


fig.append_trace(tr1,1,1)
fig.append_trace(tr2,1,1)
fig.append_trace(tr3,1,1)
fig.append_trace(tr4,1,1)
fig.append_trace(tr5,1,1)

fig.append_trace(tr11,1,2)
fig.append_trace(tr22,1,2)
fig.append_trace(tr33,1,2)
fig.append_trace(tr44,1,2)
fig.append_trace(tr55,1,2)

fig['layout'].update(height=500,width=800, title="AdoptionSpeed by Type and MaturitySize", barmode="stack", showlegend=False,xaxis=dict(title="AdoptionSpeed"))

plotly.offline.iplot(fig)





    


def compare_plot3(typeo):
    tr1=df[(df['Type']==typeo) & (df["Health"]==1)]['AdoptionSpeed'].value_counts()
    tr2=df[(df['Type']==typeo) & (df["Health"]==2)]['AdoptionSpeed'].value_counts()
    tr3=df[(df['Type']==typeo) & (df["Health"]==3)]['AdoptionSpeed'].value_counts()
    tr4=df[(df['Type']==typeo) & (df["Health"]==0)]['AdoptionSpeed'].value_counts()
    xx=[4,3,2,1,0]
    trace1=go.Bar(y=tr1,x=xx,name='Healthy')
    trace2=go.Bar(y=tr2,x=xx,name='Minor injury')
    trace3=go.Bar(y=tr3,x=xx,name='Serious injury')
    trace4=go.Bar(y=tr4,x=xx,name='Not Sprecified')
    return(trace1,trace2,trace3,trace4)

tr1,tr2,tr3,tr4=compare_plot3(1)

tr11,tr22,tr33,tr44=compare_plot3(2)

fig=tools.make_subplots(rows=1,cols=2,subplot_titles=['Dogs',"Cats"])


fig.append_trace(tr1,1,1)
fig.append_trace(tr2,1,1)
fig.append_trace(tr3,1,1)
fig.append_trace(tr4,1,1)


fig.append_trace(tr11,1,2)
fig.append_trace(tr22,1,2)
fig.append_trace(tr33,1,2)
fig.append_trace(tr44,1,2)


fig['layout'].update(height=500,width=700, title="AdoptionSpeed by Type and Health", barmode="stack", showlegend=False,xaxis=dict(title="AdoptionSpeed"))

plotly.offline.iplot(fig)

pure=df[(df["Breed2"]==0)]["Type"].value_counts()
mixed=df[(df["Breed2"]!=0)]["Type"].value_counts()
dogs_pure=df[(df["Breed2"]==0) & (df["Type"]==1)]['AdoptionSpeed'].value_counts()
dogs_mixed=df[(df["Breed2"]!=0) & (df["Type"]==1)]['AdoptionSpeed'].value_counts()
cats_pure=df[(df["Breed2"]==0) & (df["Type"]==2)]['AdoptionSpeed'].value_counts()
cats_mixed=df[(df["Breed2"]!=0) & (df["Type"]==2)]['AdoptionSpeed'].value_counts()

trace1=go.Bar(x=['Dogs',"Cats"],y=pure,name="pure")
trace2=go.Bar(x=['Dogs','Cats'],y=mixed,name='mixed')
xx=[4,3,2,1,0]
trace3=go.Bar(x=xx,y=dogs_pure,name="dogs pure breed")

trace4=go.Bar(x=xx,y=dogs_mixed,name="dogs mixed breed")


trace5=go.Bar(x=xx,y=cats_pure,name="cats pure breed")

trace6=go.Bar(x=xx,y=cats_mixed,name="cats mixed breed")


fig=tools.make_subplots(rows=2,cols=2,subplot_titles=["","","Dogs","Cats"])

fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,1)


fig.append_trace(trace3,2,1)
fig.append_trace(trace4,2,1)


fig.append_trace(trace5,2,2)
fig.append_trace(trace6,2,2)

fig['layout'].update(height=1000,width=800,barmode="stack",showlegend=False,title="AdoptionSpeed by Breed and Type")
plotly.offline.iplot(fig)






fig, ax = plt.subplots(figsize = (12, 8))
text_cat = ' '.join(df[df["Type"]==1]['Description'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top words in description');
plt.axis("off");


fig, ax = plt.subplots(figsize = (12, 8))
text_cat = ' '.join(df[df["Type"]==2]['Description'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top words in description');
plt.axis("off");



