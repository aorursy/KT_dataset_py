import pandas as pd

import seaborn as sns

import numpy as np

import plotly 

from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot

import plotly.graph_objs as go

import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/udemy-courses/udemy_courses.csv')
#Creating a new column which consists of year only

date=dataset.published_timestamp.copy()



dataset['Published_year']=0

dataset=dataset.drop(index=2066)

dataset=dataset.reset_index(drop=True)

dataset.Published_year=dataset.Published_year.astype(int)

for i in range(len(date)):

    date[i]=date[i].split('-')

    dataset['Published_year'][i]=date[i][0]
#Creating a new column for title length

dataset['title_length']=[len(i) for i in dataset.course_title]
dataset=dataset.drop(columns=['course_id','url'])
#Making the is_paid column binary

dataset.is_paid=dataset.is_paid.map({'True':1,'False':0})
#Seperating the hours from content duration

"""for i in range(len(dataset.content_duration)):

    dataset.content_duration[i]=dataset.content_duration[i].split(' ')[0]

dataset.content_duration=dataset.content_duration.astype(float)"""
#Filling the missing values in is_paid

dataset=dataset.replace("Free",0)

dataset.price=dataset.price.astype(int)

check=dataset.is_paid.isna()

for i in range(len(dataset)):

    if(check[i]==True and dataset.price[i]!=0):

        dataset.is_paid[i]=0

    elif(check[i]==True and dataset.price[i]==0):

        dataset.is_paid[i]=1 
dataset.head()
sns.countplot(data=dataset,x='Published_year')
dataset.Published_year=dataset.Published_year.astype(int)
sns.countplot(data=dataset,x='subject')
dataset.subject.value_counts()
#unique_subject=list(dataset.subject.unique())

pd.pivot_table(dataset,values='num_subscribers',index='subject',aggfunc=np.sum)
unique_subject=list(dataset.subject.unique())



val=[1870747,1063148,846689,7980572]

trace=go.Pie(labels=unique_subject,values=val,

            hoverinfo='label+value',textinfo='percent',

            textfont=dict(size=25),

            marker=dict(line=dict(width=1)),

            title="Subscribers Vs Subject")

iplot([trace])
y=dataset.level.value_counts().rename_axis('unique_values').reset_index(name='counts')

y.head()

fig = go.Figure(data=go.Scatter(x=y.unique_values, y=y.counts))

fig.show()
unique_subject=list(dataset.subject.unique())

l1=[]

for i in unique_subject:

    l2=[]

    for j in dataset.level.unique():

        sum1=0

        for k in range(len(dataset.num_subscribers)):

            if(dataset.level[k]==j and dataset.subject[k]==i):

                sum1+=1

        l2.append(sum1)

    l1.append(l2)
fig, axes = plt.subplots(2, 2,figsize=(14,8))



ax = sns.barplot(x=dataset.level.unique(), y=l1[0],ax=axes[0, 0]).set_title(unique_subject[0])

ax = sns.barplot(x=dataset.level.unique(), y=l1[1],ax=axes[0, 1]).set_title(unique_subject[1])

ax = sns.barplot(x=dataset.level.unique(), y=l1[2],ax=axes[1, 0]).set_title(unique_subject[2])

ax = sns.barplot(x=dataset.level.unique(), y=l1[3],ax=axes[1, 1]).set_title(unique_subject[3])
temp_df=pd.concat([dataset.num_subscribers,dataset.num_reviews,dataset.num_lectures,dataset.content_duration],axis=1)
sns.heatmap(temp_df.corr(),cmap="Greens")
pay_df=pd.concat([dataset.is_paid,dataset.price,dataset.num_subscribers],axis=1)

sns.heatmap(pay_df.corr(),cmap="Blues")
d=pd.pivot_table(pay_df,values='num_subscribers',index='price',aggfunc=np.sum)

d=d.reset_index()

d.price=d.price.astype(int)

d=d.sort_values(by='price')
d['num_ranks']=d['num_subscribers'].rank()
sizes=[]

colours=[]

for i in d.num_ranks:

    sizes.append(i*2)

    colours.append(120+(2*i))
x=d.iloc[:, 0].values

y=d.iloc[:, 1].values
fig = go.Figure(data=[go.Scatter(

    x=x,

    y=y,

    mode='markers',

    marker=dict(

        color=colours,

        size=sizes,

        showscale=True

        )

    )])

fig.show()
extension=dataset.loc[dataset['price']==200]

extension=extension.reset_index(drop=True)
subject_extension=list(extension.subject.unique())

l1=[]

for i in subject_extension:

    l2=[]

    for j in extension.level.unique():

        sum1=0

        for k in range(len(extension.num_subscribers)):

            if(extension.level[k]==j and extension.subject[k]==i):

                sum1+=1

        l2.append(sum1)

    l1.append(l2)

fig, axes = plt.subplots(2, 2,figsize=(14,8))



ax = sns.barplot(x=extension.level.unique(), y=l1[0],ax=axes[0, 0]).set_title(subject_extension[0])

ax = sns.barplot(x=extension.level.unique(), y=l1[1],ax=axes[0, 1]).set_title(subject_extension[1])

ax = sns.barplot(x=extension.level.unique(), y=l1[2],ax=axes[1, 0]).set_title(subject_extension[2])

ax = sns.barplot(x=extension.level.unique(), y=l1[3],ax=axes[1, 1]).set_title(subject_extension[3])
expert=dataset.loc[dataset['level']=='Expert Level']

print(" Expert price median: " +str(expert.price.median()),"  Median number of reviews expert:" + str(expert.num_reviews.median()))

print("Overall dataset review number median: "+str(dataset.num_reviews.median()))

print("Expensive review number median: "+str(extension.num_reviews.median()))
title=dataset.iloc[:, [3,11]]

title=title.sort_values(by=['title_length'])

title

x_title=title.title_length.values

y_title=title.num_subscribers.values
fig = go.Figure(data=go.Scatter(x=x_title, y=y_title))

fig.show()