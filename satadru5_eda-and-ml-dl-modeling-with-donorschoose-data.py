#import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from IPython.display import display_html
import os as os
import matplotlib.pyplot as plt
import matplotlib
import squarify
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import missingno as msno
import xgboost

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
#load data
train=pd.read_csv("../input/train.csv")
resources=pd.read_csv("../input/resources.csv")
test=pd.read_csv("../input/test.csv")
def display_side_by_side(*args):
    html_str=' '
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
col_trn=pd.DataFrame(train.columns.values.tolist()).reset_index()
col_trn.columns=['index','Train_Column_Name']

col_tst=pd.DataFrame(test.columns.values.tolist()).reset_index()
col_tst.columns=['index','Test_Column_Name']

res=pd.DataFrame(resources.columns.values.tolist()).reset_index()
res.columns=['index','Res_Column_Name']


display_side_by_side(col_trn,col_tst,res)
resources.head(3)
#some part borrowed from https://www.kaggle.com/jagangupta/understanding-approval-donorschoose-eda-fe-eli5

resources['total_cost']=resources['quantity']*resources['price']
resources['description']=resources['description'].astype(str)
des=resources.groupby('id')['description'].apply(lambda des: "%s" % ', '.join(des))


resources_gp=resources.groupby('id')['quantity','price','total_cost'].agg({'quantity':['sum','count'],'price':['mean'],'total_cost':['sum']})
resources_gp.columns=['item_quantity_sum','unique_items','avg_price_per_item','total_cost']
resources_gp['collated_description']=des
resources_gp=resources_gp.reset_index()
resources_gp['total_word']=resources_gp['collated_description'].apply(lambda x: len(str(x).split()))
resources_gp['count_unique_word']=resources_gp['collated_description'].apply(lambda x: len(set(str(x).split())))
resources_gp.head()
#check null data and visualize..
msno.bar(train)
data = [go.Bar(
            x=train.apply(lambda x : len(x.unique())).index.values,
            y=train.apply(lambda x : len(x.unique())).values,
    marker=dict(
        color='rgba(50, 171, 96, 0.7)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=2
        ))
    )]

py.iplot(data, filename='basic-bar')
#check approved or target column..
plt.figure(figsize=(8,5))
data = [go.Bar(
            x=train.project_is_approved.value_counts().index.values,
            y=train.project_is_approved.value_counts().values
    )]

py.iplot(data, filename='basic-bar')
#adding  year month as a seperate column test and train..
train['project_submitted_yearmonth'] = train['project_submitted_datetime'].apply(lambda x: x[:4]+x[5:7])
#test['project_submitted_yearmonth'] = test['project_submitted_datetime'].apply(lambda x: x[:4]+x[5:7])
train['date'] = pd.to_datetime(train.project_submitted_datetime).dt.date
train['weekday'] = pd.to_datetime(train.project_submitted_datetime).dt.weekday
train['day'] = pd.to_datetime(train.project_submitted_datetime).dt.day
train['quater']=pd.to_datetime(train['project_submitted_datetime']).dt.quarter
train['week']=pd.to_datetime(train['project_submitted_datetime']).dt.week
train['month']=pd.to_datetime(train['project_submitted_datetime']).dt.month
train['hour']=pd.to_datetime(train['project_submitted_datetime']).dt.hour
count_by_date = train.groupby('date')['project_is_approved'].count()
mean_by_date = train.groupby('date')['project_is_approved'].mean()

train_df=train.groupby(['project_submitted_yearmonth'])['id'].count().reset_index()
train_df.columns = ['submitted_yearmonth', 'count']
train_df = pd.DataFrame(train_df.sort_values(by='count',ascending=False))

train_app=train.groupby(['project_submitted_yearmonth'])['project_is_approved'].sum().reset_index()
train_app.columns = ['submitted_yearmonth', 'count']
train_app = pd.DataFrame(train_app.sort_values(by='count',ascending=False))
sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(15, 5))
sns.despine(left=True, bottom=True)
sns.set_color_codes("pastel")
sns.barplot(x='submitted_yearmonth',y='count',data=train_df, color="g")

sns.set_color_codes("muted")
sns.barplot(x="submitted_yearmonth", y="count", data=train_app,
            label="Alcohol-involved", color="g")

plt.show()
fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of approval rates and number of projects")
count_by_date.rolling(window=12,center=False).mean().plot(ax=ax1, legend=False)
ax1.set_ylabel('Projects count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
mean_by_date.rolling(window=12,center=False).mean().plot(ax=ax2, color='g', legend=False)
ax2.set_ylabel('Approval rate', color='g')
plt.legend(['Approval rate'], loc=(0.875, 0.9))
plt.grid(False)
plt.figure(figsize=(18,4))
grouped_df = train.groupby(['project_submitted_yearmonth','project_is_approved'])['id'].count().reset_index()
sns.pointplot(x=grouped_df.project_submitted_yearmonth.values, y=grouped_df.id.values, hue=grouped_df.project_is_approved.values, alpha=0.8)
grouped_df = train.groupby('project_submitted_yearmonth')['id'].count().reset_index()
approve_df=train.groupby(['project_submitted_yearmonth'])['project_is_approved'].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 4))
sns.pointplot(ax=ax,x=grouped_df.project_submitted_yearmonth.values, y=grouped_df.id.values, alpha=0.8,label="Submitted project")
plt.legend(['Submitted project'])
sns.pointplot(ax=ax,x=approve_df.project_submitted_yearmonth.values, y=approve_df.project_is_approved.values, alpha=0.8,color='g',label="Approved project")
#plt.legend(['Approved project'])

plt.ylabel('No of project', fontsize=12)
plt.xlabel('yearmonth', fontsize=12)
plt.xticks(rotation='vertical')
#plt.legend(['Submitted project'], loc=(0.875, 0.9))
plt.show()
fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Project count and approval rate by day of week.")
sns.countplot(x='weekday', data=train, ax=ax1)
ax1.set_ylabel('Projects count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
sns.pointplot(x="weekday", y="project_is_approved", data=train, ci=99, ax=ax2, color='black')
ax2.set_ylabel('Approval rate', color='g')
plt.legend(['Approval rate'], loc=(0.875, 0.9))
plt.grid(False)
plt.figure(figsize=(18,4))
grouped_df = train.groupby(['weekday','project_is_approved'])['id'].count().reset_index()
sns.barplot(x=grouped_df.weekday.values, y=grouped_df.id.values, hue=grouped_df.project_is_approved.values, alpha=0.8)
plt.figure(figsize=(18,4))
grouped_df = train.groupby(['month','teacher_prefix'])['id'].count().reset_index()
sns.pointplot(x=grouped_df.month.values, y=grouped_df.id.values, hue=grouped_df.teacher_prefix.values, alpha=0.8)
plt.figure(figsize=(18,4))
grouped_df = train.groupby(['hour','project_is_approved'])['id'].count().reset_index()
sns.pointplot(x=grouped_df.hour.values, y=grouped_df.id.values, hue=grouped_df.project_is_approved.values, alpha=0.8)
grouped_df = train.groupby('month')['id'].count().reset_index()
approve_df=train.groupby(['month'])['project_is_approved'].sum().reset_index()
plt.figure(figsize=(18,4))
sns.pointplot(grouped_df.month.values, grouped_df.id.values, alpha=0.8)
sns.pointplot(approve_df.month.values, approve_df.project_is_approved.values, alpha=0.8,color='g')
plt.ylabel('No of project', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()
labels =train["teacher_prefix"].value_counts().index.values
values = train["teacher_prefix"].value_counts().values

data = [go.Pie(
            
           labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=10),
    )]

py.iplot(data, filename='basic-Pie')
pd.crosstab(train.teacher_prefix, train.project_is_approved, dropna=False, normalize='index')
missing_df = train.groupby(['school_state'])['id'].count().reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df=missing_df.sort_values(by='missing_count')
#missing_df=missing_df.head(15)
ind = np.arange(missing_df.shape[0])

width = 0.9
fig, ax = plt.subplots(figsize=(12,12))
rects = ax.barh(ind, missing_df.missing_count.values, color='g')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of projects")
ax.set_title("State wise distribution of project proposal")
plt.show()


missing_df['missing_count'].head(3)
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
data = [ dict(
        type='choropleth',
        colorscale ='Electric',
        autocolorscale = False,
        locations = missing_df['column_name'],
        z = missing_df['missing_count'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 4
            )
        ),
        colorbar = dict(
            title = "Millions USD"
        )
    ) ]

layout = dict(
        title = 'Number of Submitted Project Proposals per US state<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    
    )
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


my_values=[i**3 for i in range(1,10)]
 
#tree map staet wise..
cmap = matplotlib.cm.RdBu_r
mini=min(my_values)
maxi=max(my_values)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in my_values]
plt.figure(figsize=(20,9))
squarify.plot(sizes=train['school_state'].value_counts().values, label=train['school_state'].value_counts().index ,  alpha=.8, color=colors )
plt.title("State wise project distribution -> Tree map")
plt.axis('off')
plt.show()
teacher = train.groupby(['school_state'])['teacher_id'].nunique().reset_index()
teacher.columns = ['column_name', 'missing_count']
teacher = teacher.ix[teacher['missing_count']>0]
teacher=teacher.sort_values(by='missing_count')
#missing_df=missing_df.head(15)
ind = np.arange(missing_df.shape[0])

width = 0.9
fig, ax = plt.subplots(figsize=(12,12))
rects = ax.barh(ind, teacher.missing_count.values, color='r')
ax.set_yticks(ind)
ax.set_yticklabels(teacher.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of teacher")
ax.set_title("State wise distribution of teacher")
plt.show()
teacher.head()
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
data = [ dict(
        type='choropleth',
        colorscale =scl,
        autocolorscale = False,
        locations = teacher['column_name'],
        z = teacher['missing_count'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 4
            )
        ),
        colorbar = dict(
            title = "Millions USD"
        )
    ) ]

layout = dict(
        title = 'US State wise teacher distribution<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    
    )
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d4-cloropleth-map' )


squarify.plot(sizes=train['project_grade_category'].value_counts().values, label=train['project_grade_category'].value_counts().index , alpha=.8)
plt.axis('off')
plt.title("Grade wise project distribution  -> Tree map")
plt.show()
grade_ts=train.groupby(['project_grade_category','teacher_prefix'])['id'].count().reset_index()
grade_ts=grade_ts.sort_values(['id','project_grade_category'],ascending=False)
df=pd.crosstab(grade_ts.project_grade_category,grade_ts.teacher_prefix,values=grade_ts.id,aggfunc=np.sum)
plt.figure(figsize=(12,4))
sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5)
plt.title("Distribution of teacher grade wise -> Heatmap")
#BuPu
colorscale = [[0, '#edf8fb'], [.3, '#b3cde3'],  [.6, '#8856a7'],  [1, '#810f7c']]

heatmap = go.Heatmap(z=df.as_matrix(), x=df.columns, y=df.index, colorscale=colorscale)
data = [heatmap]
py.iplot(data, filename='basic-heatmap')
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=2000,
                          max_font_size=40, 
    
                          random_state=42
                         ).generate(str(train['project_title']))

print(wordcloud)
fig = plt.figure(2)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word cloud ->project title")
plt.show()
fig.savefig("word1.png", dpi=900)

wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=2000,
                          max_font_size=40, 
    
                          random_state=42
                         ).generate(str(train['project_essay_2']))

print(wordcloud)
fig = plt.figure(2)
#plt.title("Word cloud ->project_essay_2")
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word2.png", dpi=900)
#check unique and shape
resources_gp.head()
plt.figure(figsize=(10,4))
sns.distplot(resources_gp['total_cost'])
#top 
top=resources_gp[['id','total_cost','collated_description']].sort_values(by='total_cost',ascending=False).head(100)
bot=resources_gp[['id','total_cost','collated_description']].sort_values(by='total_cost').head(10)

display_side_by_side(top.head(5))
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=2000,
                          max_font_size=40, 
    
                          random_state=42
                         ).generate(str(top['collated_description']))

print(wordcloud)
fig = plt.figure(2)
#plt.title("Word cloud ->project_essay_2")
plt.figure(figsize=(12,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word2.png", dpi=900)
#merge train data with resources

train_data=pd.merge(train,resources_gp,on='id')
train_data.head(2)
#Unique value check..
plt.figure(figsize=(10,5))
data = [go.Bar(
            x=train_data.groupby(['project_grade_category'])['total_cost'].sum().index.values,
            y=train_data.groupby(['project_grade_category'])['total_cost'].sum().values
    )]

py.iplot(data, filename='basic-bar')
plt.figure(figsize=(10,5))
data = [go.Bar(
            x=train_data.groupby(['project_subject_categories'])['total_cost'].sum().index.values,
            y=train_data.groupby(['project_subject_categories'])['total_cost'].sum().values
    )]

py.iplot(data, filename='basic-bar')
train_data['project_submitted_year'] = train_data['project_submitted_yearmonth'].apply(lambda x: x[:4])
df1=train_data.groupby(['project_grade_category','project_submitted_year'])['total_cost'].sum().reset_index()
plt.figure(figsize=(15,6))
sns.barplot(x='project_grade_category',y='total_cost',hue='project_submitted_year',data=df1)
res_geo=train_data.groupby('school_state')['total_cost'].sum().reset_index()
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
data = [ dict(
        type='choropleth',
        colorscale =scl,
        autocolorscale = False,
        locations = res_geo['school_state'],
        z = res_geo['total_cost'],
        locationmode = 'USA-states',
        #text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 4
            )
        ),
        colorbar = dict(
            title = "Millions USD"
        )
    ) ]

layout = dict(
        title = 'US state wise donation distribution<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    
    )
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d4-cloropleth-map' )


del train_data
#include resource with train and test data
#merge resources with training data..
#test['project_submitted_yearmonth'] = test['project_submitted_datetime'].apply(lambda x: x[:4]+x[5:7])


#Add datetime related feature to train and test..
#train['project_submitted_day']=pd.to_datetime(train['project_submitted_datetime']).dt.day
#train['project_submitted_month']=pd.to_datetime(train['project_submitted_datetime']).dt.month
#train['project_submitted_year']=pd.to_datetime(train['project_submitted_datetime']).dt.year
#train['project_submitted_week']=pd.to_datetime(train['project_submitted_datetime']).dt.week
#train['project_submitted_quater']=pd.to_datetime(train['project_submitted_datetime']).dt.quarter

#test['project_submitted_day']=pd.to_datetime(test['project_submitted_datetime']).dt.day
#test['project_submitted_month']=pd.to_datetime(test['project_submitted_datetime']).dt.month
#test['project_submitted_year']=pd.to_datetime(test['project_submitted_datetime']).dt.year
#test['project_submitted_week']=pd.to_datetime(test['project_submitted_datetime']).dt.week
#test['project_submitted_quater']=pd.to_datetime(test['project_submitted_datetime']).dt.quarter
#delete id and teacher_id from test and train 
#test[test['teacher_id']=='484aaf11257089a66cfedc9461c6bd0a'].head(3)
test.tail(3)
test.head(3)
