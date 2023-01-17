import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(True)
plt.style.use('ggplot')
pd.options.display.max_rows =300
plt.rc('ytick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=12)
#path ='file/'
path = '../input/'

hacker_numeric = pd.read_csv(path+'HackerRank-Developer-Survey-2018-Numeric.csv',na_values='#NULL!',low_memory=False)
#hacker_map = pd.read_csv('file/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv')
hacker_qna = pd.read_csv(path+'HackerRank-Developer-Survey-2018-Codebook.csv')
df = pd.read_csv(path+ 'HackerRank-Developer-Survey-2018-Values.csv',na_values='#NULL!',low_memory=False)
print('Number of rows and columns in Hacker value data set',df.shape)
hacker_qna = hacker_qna.set_index('Data Field')
hacker_qna.head()
df.head()
def basic_details(df):
    print('Number of rows {} and columns {}'.format(df.shape[0],df.shape[1]))
    k = pd.DataFrame()
    k['dtype'] = df.dtypes
    k['Number of unique value'] = df.nunique()
    k['Missing_value'] = df.isnull().sum()
    k['% missing_value'] = df.isnull().sum()/df.shape[0]
    return k
basic_details(df).T
df.tail()
# Conver to date time
df['StartDate'] = pd.to_datetime(df['StartDate'])
df['EndDate'] = pd.to_datetime(df['EndDate'])
f,ax = plt.subplots(2,1, figsize=(16,6))
df.set_index('StartDate').resample('D')['q2Age'].count().plot(ax=ax[0])
ax[0].set_title('Response count per day')
ax[0].set_ylabel('Count')
df.set_index('StartDate').resample('H')['q2Age'].count().plot(ax=ax[1],color='b')
ax[1].set_title('Response count per hour in perticular')
ax[1].set_ylabel('Count')
plt.subplots_adjust(hspace=0.3)
f,ax = plt.subplots(1,2,figsize=(16,4))
duration = pd.to_datetime(df['EndDate'] - df['StartDate']).dt.hour * 60 +\
            pd.to_datetime(df['EndDate'] - df['StartDate']).dt.minute

sns.distplot(duration.dropna().values, color = 'r',ax=ax[0],)
ax[0].set_title('Distribution of survey time')
ax[0].set_xlabel('Time in minute')

poo = df['StartDate'].dt.hour
sns.countplot(y = poo.values, palette='cool',ax=ax[1])
ax[1].set_title('Distribution of survey time by Hour')
ax[1].set_ylabel('Time in Hour')
ax[1].set_xlabel('Count');
# Count by country

poo = df['CountryNumeric'].value_counts()

# plotly 
data = [dict(
    type ='choropleth',
    locations = poo.index,
    locationmode ='country names',
    z = poo.values,
    text = ('Count'+'<br>'),
    colorscale='Jet',
    reversescale=False,
    marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
    
    colorbar = dict(title ='Response count')
    )]
layout = dict(title = 'Number of response by country',
             geo = dict( showframe= False,
                       showcoastlines =True,
                       projection = dict(type = 'Mercator')))
fig = dict(data=data, layout=layout)
py.iplot(fig)

# count female bt country

poo = df[df['q3Gender'] == 'Female']['CountryNumeric'].value_counts()

# plotly 
data = [dict(
    type ='choropleth',
    locations = poo.index,
    locationmode ='country names',
    z = poo.values,
    text = 'Female count',
    colorscale='YlOrRd',
    reversescale=False,
    marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
    colorbar = dict(title ='Response count')
    )]
layout = dict(title = 'Number of Female response by country',
             geo = dict( showframe= False,
                       showcoastlines =True,
                       projection = dict(type = 'Mercator')))
fig = dict(data=data, layout=layout)
py.iplot(fig,filename='map')

print('Q?:',hacker_qna.loc['q3Gender']['Survey Question'])
plt.figure(figsize=(16,2))
poo = df['q3Gender'].value_counts()
sns.barplot(poo.values,poo.index, palette='Wistia')
for i, v in enumerate(poo.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.title('Distribution of Gender')
plt.xlabel('Count');
print('Q?:',hacker_qna.loc['q1AgeBeginCoding']['Survey Question'])

f,ax = plt.subplots(1,2, figsize=(16,6))
st_age = df['q1AgeBeginCoding'].value_counts()
sns.barplot(st_age.values, st_age.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Begin coding')
ax[0].set_xlabel('Count')
for i, v in enumerate(st_age.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)


foo = df.groupby(['q1AgeBeginCoding','q3Gender'])['q3Gender'].count().reset_index(name='count')
foo = foo.pivot(columns='q3Gender', index='q1AgeBeginCoding', values ='count')
#foo.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(foo,annot=True, fmt='.0f', linewidths=.01, cmap='cool', cbar=False, ax=ax[1])
ax[1].set_title('Distribution of AgeBeginCoding by Grender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('')
#plt.savefig('age.png')
plt.subplots_adjust(wspace=0.4);
print('Q?:',hacker_qna.loc['q2Age']['Survey Question'])

f,ax = plt.subplots(1,2,figsize=(16,6))
age = df['q2Age'].value_counts()
sns.barplot(age.values, age.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Current Age')
ax[0].set_xlabel('Count')
for i, v in enumerate(age.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)

poo = df.groupby(['q2Age','q3Gender'])['q3Gender'].count().reset_index(name='count')
poo = poo.pivot(columns='q3Gender',index='q2Age',values='count')
#poo.plot(kind='barh',ax=ax[0],colormap='tab10')
sns.heatmap(poo,annot=True, fmt='.0f', linewidths=.01, cmap='spring', cbar=False, ax=ax[1])
ax[1].set_title('Distribution of age by Gender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Age')
plt.subplots_adjust(wspace=0.5);
print('Q?:',hacker_qna.loc['q4Education']['Survey Question'])

f,ax = plt.subplots(1,2, figsize=(16,6))
st_age = df['q4Education'].value_counts()
sns.barplot(st_age.values, st_age.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Education')
ax[0].set_xlabel('Count')
for i, v in enumerate(st_age.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)

poo = df.groupby(['q4Education','q3Gender'])['q3Gender'].count().reset_index(name='count')
poo = poo.pivot(columns='q3Gender',index='q4Education',values='count')
#poo.plot(kind='barh',ax=ax[0],colormap='tab10')
sns.heatmap(poo, annot=True, fmt='.0f',cmap='cool',linewidths=0.01, cbar=False, ax=ax[1],)
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('')

plt.subplots_adjust(wspace=0.7)
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='cool',background_color='White'\
              ).generate(' '.join(df['q0004_other'].dropna().astype(str)))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
#plt.savefig('cloud.png')
plt.title('Wordcloud of Education');
print('Q?:',hacker_qna.loc['q5DegreeFocus']['Survey Question'])

f,ax = plt.subplots(1,2, figsize=(16,2))
deg = df['q5DegreeFocus'].value_counts()
sns.barplot(deg.values, deg.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Degree')
ax[0].set_xlabel('Count')
for i, v in enumerate(deg.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)

foo = df.groupby(['q5DegreeFocus','q3Gender'])['q3Gender'].count().reset_index(name='count')
foo = foo.pivot(columns='q3Gender', index='q5DegreeFocus', values ='count')
#foo.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(foo, annot=True, linewidths=0.1, fmt='.0f',cmap='cool', cbar=False, ax=ax[1])
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('')

plt.subplots_adjust(wspace=0.4);
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=2000,stopwords=STOPWORDS,colormap='spring',background_color='White'\
              ).generate(' '.join(df['q0004_other'].dropna().astype(str)))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Distribution of Degree Focus')
columns = df.columns[df.columns.str.startswith('q6')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

code = pd.DataFrame() 
for c in columns:
    agg = df.groupby([c,'q3Gender'])['q3Gender'].count().reset_index(name='count')
    agg = agg.pivot(columns='q3Gender',index=c, values='count')
    code = pd.concat([code,agg])
code.style.background_gradient(cmap='cool')
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='cool',background_color='White'\
              ).generate(' '.join(df['q0006_other'].dropna().astype(str)))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcloud of learn Code')
print('Q?:',hacker_qna.loc['q7Level1']['Survey Question'])

f,ax = plt.subplots( figsize=(16,2))
pro = df['q7Level1'].value_counts()
sns.barplot(pro.values, pro.index, palette='Wistia',ax=ax)
ax.set_xlabel('Count')
ax.set_title('Level 1 to Level2 Unlocking Question')
for i, v in enumerate(pro.values):
    ax.text(0.8,i,v,color='k',fontsize=12)
print('Q?:',hacker_qna.loc['q8JobLevel']['Survey Question'])

f,ax = plt.subplots(1,2,figsize=(16,6))
job = df['q8JobLevel'].value_counts()
sns.barplot(job.values, job.index, palette='Wistia',ax=ax[0])
ax[0].set_xlabel('Count')
ax[0].set_title('Distribution of Job Level')
for i, v in enumerate(job.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)
    
agg = df.groupby(['q8JobLevel','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg = agg.pivot(columns='q3Gender',index='q8JobLevel',values='count')
#agg.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(agg,cmap='cool',annot=True,linewidths=0.01,fmt='.0f',cbar=False, ax=ax[1])
ax[1].set_title('Distribution of Job Level by Gender')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.4);
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='cool',background_color='White'\
              ).generate(' '.join(df['q0008_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcloud of Job Level')
print('Q?:',hacker_qna.loc['q9CurrentRole']['Survey Question'])

f,ax = plt.subplots(1,2,figsize=(16,10))
role = df['q9CurrentRole'].value_counts()
sns.barplot(role.values, role.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Current Role')
ax[0].set_xlabel('Count')
for i, v in enumerate(role.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)
    
agg = df.groupby(['q9CurrentRole','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg = agg.pivot(columns='q3Gender', values='count', index='q9CurrentRole')
#agg.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(agg,cmap='cool',annot=True,linewidths=0.01,fmt='.0f',cbar=False, ax=ax[1])
ax[1].set_title('Distribution of Current Role by Gender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Current Role')
plt.subplots_adjust(wspace=0.7);
print('Q?:',hacker_qna.loc['q10Industry']['Survey Question'])

f,ax = plt.subplots(1,2,figsize=(16,10))
ind = df['q10Industry'].value_counts()
sns.barplot(ind.values, ind.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Indusry')
ax[0].set_xlabel('Count')
for i, v in enumerate(ind.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)
    
agg = df.groupby(['q10Industry','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg = agg.pivot(columns='q3Gender', values='count', index='q10Industry')
#agg.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(agg,cmap='cool',annot=True,linewidths=0.01,fmt='.0f',cbar=False, ax=ax[1])
ax[1].set_title('Distribution of Industry by Gender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.7);
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='magma',background_color='White'\
              ).generate(' '.join(df['q0010_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcloud of Industry');
columns = df.columns[df.columns.str.startswith('q12')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

interview = pd.DataFrame() 
for c in columns:
    agg = df.groupby([c,'q3Gender'])['q3Gender'].count().reset_index(name='count')
    agg = agg.pivot(columns='q3Gender',index=c, values='count')
    interview = pd.concat([interview,agg])
interview.style.background_gradient(cmap='cool')
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='magma',background_color='White'\
              ).generate(' '.join(df['q0012_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcloud of Employe Expectation');
columns = df.columns[df.columns.str.startswith('q13')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(14,8))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax)
ax.set_xlabel('Count')
ax.set_title('Distribution of Employer Skill of candidates')
for i,v in enumerate(skill.values):
    ax.text(0.8,i,v[0], fontsize=10,color='k')
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='spring',background_color='White'\
              ).generate(' '.join(df['q0013_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcloud of Employer Skill measure technique');
print('Q?:',hacker_qna.loc['q14GoodReflecAbilities']['Survey Question'])

f,ax = plt.subplots(figsize=(16,4))
ind = df['q14GoodReflecAbilities'].value_counts()
sns.barplot(ind.values, ind.index, palette='Wistia',ax=ax)
ax.set_title('Distribution of Ability')
ax.set_xlabel('Count')
for i, v in enumerate(ind.values):
    ax.text(0.8,i,v,color='k',fontsize=12)
print('Q?:',hacker_qna.loc['q15Level2']['Survey Question'])

f,ax = plt.subplots(figsize=(16,4))
l2 = df['q15Level2'].value_counts()
sns.barplot(l2.values, l2.index, palette='Wistia',ax=ax)
ax.set_title('Distribution of Level2')
ax.set_xlabel('Count')
for i, v in enumerate(l2.values):
    ax.text(0.8,i,v,color='k',fontsize=12)
    
print('Q?:',hacker_qna.loc['q16HiringManager']['Survey Question'])

f,ax = plt.subplots(1,2,figsize=(16,2))
hm = df['q16HiringManager'].value_counts()
sns.barplot(hm.values, hm.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Hirring Manger')
ax[0].set_xlabel('Count')

for i, v in enumerate(hm.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)
    
agg = df.groupby(['q16HiringManager','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg = agg.pivot(columns='q3Gender', values='count', index='q16HiringManager')
#agg.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(agg,cmap='cool',annot=True,linewidths=0.01,fmt='.0f',cbar=False, ax=ax[1])
ax[1].set_title('Distribution of Hirring Manger by Gender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.7);
columns = df.columns[df.columns.str.startswith('q17')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,8))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax)
ax.set_xlabel('Count')
ax.set_title('Distribution of Hiring Challenges')
for i,v in enumerate(skill.values):
    ax.text(0.8,i,v[0], fontsize=12,color='k')
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='magma',background_color='White'\
              ).generate(' '.join(df['q0017_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcloud of Industry');
print('Q?:',hacker_qna.loc['q18NumDevelopHireWithinNextYear']['Survey Question'])

f,ax = plt.subplots(figsize=(16,4))
plan = df['q18NumDevelopHireWithinNextYear'].value_counts()
sns.barplot(plan.values, plan.index, palette='Wistia',ax=ax)
ax.set_title('Deleloper hiring target for coming year')
ax.set_xlabel('Count')
for i, v in enumerate(plan.values):
    ax.text(0.8,i,v,color='k',fontsize=12)
columns = df.columns[df.columns.str.startswith('q19')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,8))
tool = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    tool = pd.concat([tool,value])

tool = tool.rename(columns={0:'Count'})
tool = tool.sort_values(by='Count')
tool.plot(kind='barh',ax=ax,colormap='spring')
ax.set_xlabel('Count')
ax.set_title('Distribution of Interview process')
for i,v in enumerate(tool.Count):
    ax.text(0.8,i,v, fontsize=10,color='k')
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='cool',background_color='White'\
              ).generate(' '.join(df['q0019_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Wordcolud of  Talent assessment tools');
columns = df.columns[df.columns.str.startswith('q20')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,10))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax,colormap='rainbow')
ax.set_xlabel('Count')
ax.set_title('Distribution of important qualifications of candidate')

for i,v in enumerate(skill.Count):
    ax.text(0.8,i,v, fontsize=10,color='k')
columns = df.columns[df.columns.str.startswith('q21')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,8))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax,colormap='Set1')
ax.set_xlabel('Count')
ax.set_title('Distribution of important qualifications of candidate')

for i,v in enumerate(skill.Count):
    ax.text(0.8,i,v, fontsize=10,color='k')
columns = df.columns[df.columns.str.startswith('q22')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,12))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax,colormap='cool')
ax.set_xlabel('Count')
ax.set_title('Distribution of core competencies in software developer candidates')

for i,v in enumerate(skill.Count):
    ax.text(0.8,i,v, fontsize=10,color='k')
# other than above type mentioned

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='magma',background_color='White'\
              ).generate(' '.join(df['q0022_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.savefig('cloud.png')

plt.title('Wordcloud of Industry');
columns = df.columns[df.columns.str.startswith('q23')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,12))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax,colormap='Wistia')
ax.set_xlabel('Count')
ax.set_title('Distribution of Frame work')

for i,v in enumerate(skill.Count):
    ax.text(0.8,i,v, fontsize=10,color='k')
print('Q?:',hacker_qna.loc['q24VimorEmacs']['Survey Question'])

f,ax = plt.subplots(1,2,figsize=(16,2))
hm = df['q24VimorEmacs'].value_counts()
sns.barplot(hm.values, hm.index, palette='Wistia',ax=ax[0])
ax[0].set_title('Distribution of Vim or Emacs')
ax[0].set_xlabel('Count')
for i, v in enumerate(hm.values):
    ax[0].text(0.8,i,v,color='k',fontsize=12)
    
agg = df.groupby(['q24VimorEmacs','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg = agg.pivot(columns='q3Gender', values='count', index='q24VimorEmacs')
#agg.plot(kind='barh',ax=ax[1],colormap='cool')
sns.heatmap(agg,cmap='cool',annot=True,linewidths=0.01,fmt='.0f',cbar=False, ax=ax[1])
ax[1].set_title('Distribution of Vim or Emacs')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.7);
columns = df.columns[df.columns.str.startswith('q25')]
#print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])
length = len(columns)
#print(length)
columns = columns.drop('q25LangOther')
f,ax= plt.subplots(4,6,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    sns.countplot(df[c],ax=axs[i],palette='magma')
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')
    axs[i].set_title(hacker_qna.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle(' Programming language known or willing to know',fontsize=14);
columns = df.columns[df.columns.str.startswith('q26')]
#print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])
length = len(columns)
#print(length)

f,ax= plt.subplots(4,5,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    if df[c].nunique()>1: 
        sns.countplot(df[c],ax=axs[i],palette='magma')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(hacker_qna.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle('Frame work known or willing to know',fontsize=14);
print('Q?:',hacker_qna.loc['q27EmergingTechSkill']['Survey Question'])

f,ax = plt.subplots(figsize=(16,6))
ind = df['q27EmergingTechSkill'].value_counts()
sns.barplot(ind.values, ind.index, palette='Wistia',ax=ax)
ax.set_title('Distribution of Emerging Tech skill')
ax.set_xlabel('Count')
for i, v in enumerate(ind.values):
    ax.text(0.8,i,v,color='k',fontsize=12)
columns = df.columns[df.columns.str.startswith('q28')]
#print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])
length = len(columns)
#print(length)
columns = columns.drop('q28LoveOther')
f,ax= plt.subplots(4,6,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    if df[c].nunique()>1: 
        sns.countplot(df[c],ax=axs[i],palette='magma')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(hacker_qna.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle('Love or Hate programming language',fontsize=14);
columns = df.columns[df.columns.str.startswith('q29')]
#print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])
length = len(columns)
#print(length)
#columns = columns.drop('q28LoveOther')
f,ax= plt.subplots(4,5,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    if df[c].nunique()>1: 
        sns.countplot(df[c],ax=axs[i],palette='ocean')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(hacker_qna.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle('Love or Hate Frame work',fontsize=14);
columns = df.columns[df.columns.str.startswith('q30')]
print('Q?:',hacker_qna.loc[columns[0]]['Survey Question'])

f,ax= plt.subplots(figsize=(16,10))
skill = pd.DataFrame()
for c in columns:
    value = df[c].value_counts()
    skill = pd.concat([skill,value])

skill = skill.rename(columns={0:'Count'})
skill = skill.sort_values(by='Count')
skill.plot(kind='barh',ax=ax,colormap='summer')
ax.set_title('The popular Resources learn and practice')
ax.set_xlabel('Count')
for i,v in enumerate(skill.Count):
    ax.text(0.8,i,v, fontsize=10,color='k')
print('Q?:',hacker_qna.loc['q31Level3']['Survey Question'])

f,ax = plt.subplots(figsize=(16,4))
l2 = df['q31Level3'].value_counts()
sns.barplot(l2.values, l2.index, palette='Wistia',ax=ax)
ax.set_title('Distribution of Level2')
ax.set_xlabel('Count')
for i, v in enumerate(l2.values):
    ax.text(0.8,i,v,color='k',fontsize=12)
    
import matplotlib.gridspec as gridspec
f,ax = plt.subplots(figsize=(16,12))
gridspec.GridSpec(3,3)

plt.subplot2grid((3,3),(0,0),colspan=3, rowspan=2)
rec = df['q32RecommendHackerRank'].value_counts()
plt.pie(rec.values,labels=rec.index,autopct='%1.1f%%',colors=sns.color_palette('binary'))
plt.title('Q?: {}'.format(hacker_qna.loc['q32RecommendHackerRank']['Survey Question']))

plt.subplot2grid((3,3),(2,0))
job = df['q33HackerRankChallforJob'].value_counts()
plt.pie(job.values,labels=job.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))
#plt.title('Q?: {}'.format(hacker_qna.loc['q33HackerRankChallforJob']['Survey Question']))
plt.title('Hacker Rank Challege for Job')

plt.subplot2grid((3,3),(2,1))
test = df['q34IdealLengHackerRankTest'].value_counts()
plt.pie(test.values,labels=test.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))
#plt.title('Q?: {}'.format(hacker_qna.loc['q34IdealLengHackerRankTest']['Survey Question']))
plt.title('Ideal Length Hacker Rank Test')

plt.subplot2grid((3,3),(2,2))
test = df['q34PositiveExp'].value_counts()
plt.pie(test.values,labels=test.index,autopct='%1.1f%%',colors=sns.color_palette('cool'))
plt.title('Q?: {}'.format(hacker_qna.loc['q34PositiveExp']['Survey Question']))
plt.savefig('hacker_review.png');
# other than feedback

wc = WordCloud(height=600,width=1600,max_words=1000,stopwords=STOPWORDS,colormap='cool',background_color='White'\
              ).generate(' '.join(df['q0035_other'].dropna().str.lower()))
plt.figure(figsize=(16,6))
plt.imshow(wc)
plt.axis('off')
plt.title('Feed back');
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import  normalize
seed = 532018
level1_columns = ['CountryNumeric',
       'q1AgeBeginCoding', 'q2Age', 'q3Gender', 'q4Education',
       'q0004_other', 'q5DegreeFocus', 'q0005_other', 'q6LearnCodeUni',
       'q6LearnCodeSelfTaught', 'q6LearnCodeAccelTrain',
       'q6LearnCodeDontKnowHowToYet', 'q6LearnCodeOther', 'q0006_other']
level1 = hacker_numeric[level1_columns]
level1_df = df[level1_columns]

#missing value
#ob_col = level1.select_dtypes(include='object').columns
ob_col = ['q0004_other', 'q0005_other', 'q0006_other']
level1.loc[:,ob_col] = level1.loc[:,ob_col].fillna('entered',axis=0)
level1 = level1.fillna(0,axis=0)

# encoder
le = LabelEncoder()
for c  in ob_col:
    level1[c] = le.fit_transform(level1[c])

# Heatmap
plt.figure(figsize=(16,6))
sns.heatmap(level1.corr(),cmap='cool',annot=True)
plt.title('Heatmap of Level1 QnA')
# determine value of k using elbow method
def cluster_elbow(df,n_clusters):
    wcss = []
    for i in range(1,n_clusters):
        cluster = KMeans(n_clusters=i,random_state=seed)
        cluster.fit(df)
        wcss.append(cluster.inertia_) # Sum of square distance P,C
    plt.figure(figsize=(16,4))
    plt.plot(range(1,n_clusters),wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of cluster')
    plt.ylabel('wcss')
    
# Elbow on level1 QnA
cluster_elbow(level1,n_clusters=10)
# Kmeans cluster
cluster = KMeans(n_clusters=4,random_state=seed)
pred = cluster.fit_predict(level1)
# Cluster 1
print('Shape',level1_df[pred==0].shape)
level1_df[pred==0].describe()
# Cluster 2
print('Shape',level1_df[pred==1].shape)
level1_df[pred==1].describe()
# Cluster 3
print('Shape',level1_df[pred==3].shape)
level1_df[pred==2].describe()
# Cluster 4
print('Shape',level1_df[pred==3].shape)
level1_df[pred==3].describe()