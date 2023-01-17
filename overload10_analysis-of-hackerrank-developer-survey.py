import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
survey = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv', low_memory=False)
survey_codebook =pd.read_csv('../input/HackerRank-Developer-Survey-2018-Codebook.csv')
survey_codebook=survey_codebook.set_index('Data Field')
print(survey.head(5))
print('-'*50)
print(survey.info())

print(survey.shape)

survey = survey.replace('#NULL!',np.nan)

print(survey.head(5))
survey.drop(['StartDate','EndDate'], axis=1, inplace=True)

print(survey.shape)

survey.isnull().values.any(),
survey.isnull().sum()
print(survey.shape)
survey.dropna(axis=0,how='all',inplace=True)
print(survey.shape)
plt.figure(figsize=(16,2))
count =  survey['q3Gender'].value_counts()
print(count)
sns.barplot(count.values,count.index,palette = 'YlGn')
for i,v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
    
plt.title('Gender Distribution')
plt.xlabel('Count')
plt.show()
count = survey['q2Age'].value_counts()
print(count)
plt.figure(figsize=(12,5))
sns.barplot(count.values,count.index,palette='pink')
for i,v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
    
plt.title('Age Distribution')
plt.show()

count = survey['q1AgeBeginCoding'].value_counts()
print (count)
plt.figure(figsize=(10,6))
sns.barplot(count.values,count.index,palette='terrain')
for i,v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12,va='center')

plt.title('Distribution of Age begin coding')
plt.xlabel('count')
plt.show()


count = survey.groupby(['q1AgeBeginCoding','q3Gender'])['q3Gender'].count().reset_index(name = 'count')
#print(count)
count = count.pivot(columns='q3Gender', index= 'q1AgeBeginCoding',values='count')
print(count)
plt.figure(figsize= (16,5))
sns.heatmap(count,fmt='.0f',cmap='Wistia',linewidths=.01,annot=True)

    
plt.title('Age Begin Coding based on Gender')
plt.xlabel('Gender')
plt.ylabel('Age Band')
plt.show()
count = survey['q4Education'].value_counts()
print(count)
plt.figure(figsize=(16,4))
sns.barplot(count.values,count.index,palette='coolwarm')
for i,v in enumerate(count):
    plt.text(1,i,v,fontsize=12,va='center')
plt.xlabel('count')
plt.title('Education Quallfication')
plt.show()
from wordcloud import WordCloud, STOPWORDS
wc = WordCloud(height=600,width=1400,max_words=1000,stopwords=STOPWORDS,colormap='coolwarm',background_color='Cyan').generate(' '.join(survey['q0004_other'].dropna().astype(str)))
plt.figure(figsize = (16,16))
plt.imshow(wc)
plt.title('Education Wordcloud')
plt.axis('off')
plt.show()
col = survey.columns[survey.columns.str.startswith('q6')]
#print(col)

codeLearn = pd.DataFrame()
for c in col:
    agg = survey.groupby([c,'q3Gender'])['q3Gender'].count().reset_index(name='count')
    agg = agg.pivot(columns='q3Gender',index=c,values='count')
    codeLearn = pd.concat([codeLearn,agg])
    

plt.figure(figsize=(10,4))
sns.heatmap(codeLearn,fmt='.0f',cmap='YlOrBr',annot=True)
plt.xlabel('Gender')
plt.ylabel('Type of Learning')
plt.show()
#codeLearn


wc = WordCloud(height=600,width=1400,max_words=1000,stopwords=STOPWORDS,colormap='terrain',background_color='white').generate(' '.join(survey['q0006_other'].dropna().astype(str)))
plt.figure(figsize=(16,16))
plt.imshow(wc)
plt.axis('off')
plt.title('WordCloud of Learn to code')
plt.show()
count = survey['q8JobLevel'].value_counts()
sns.barplot(x=count.values,y=count.index,palette='cool')
for i,v in enumerate(count.values):
    plt.text(1,i,v,fontsize=12,va='center')
plt.show()    
count = survey.groupby(['q8JobLevel','q3Gender'])['q3Gender'].count().reset_index(name='count')
count = count.pivot(columns='q3Gender', index='q8JobLevel',values='count')
#print (count)

plt.figure(figsize=(10,4))
sns.heatmap(count,fmt='.0f',cmap='YlGn',annot=True)
plt.xlabel('Gender')
plt.ylabel('Job Level')
plt.title('Gender Distribution in IT')
plt.show()

wc = WordCloud(height=600,width=1400,max_words=1000,stopwords=STOPWORDS,colormap='jet',background_color='grey').generate(' '.join(survey['q0008_other'].dropna().astype(str)))
plt.figure(figsize=(16,16))
plt.imshow(wc)
plt.axis('off')
plt.title('Famous Job Title')
plt.show()
f, ax = plt.subplots(1,2,figsize=(16,10))
rolecount = survey['q9CurrentRole'].value_counts()
sns.barplot(rolecount.values,rolecount.index,palette='hsv',ax=ax[0],)
ax[0].set_title('Current Job Role')
ax[0].set_xlabel('count')
for i,v in enumerate(rolecount.values):
    ax[0].text(10,i,v,fontsize='12',va='center')
    
agg = survey.groupby(['q9CurrentRole','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg = agg.pivot(columns='q3Gender',index='q9CurrentRole',values='count')
sns.heatmap(agg,cmap='Pastel1',annot=True,fmt='.0f',ax=ax[1])
ax[1].set_title('Current role grouped be gender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Current role')
plt.subplots_adjust(wspace=0.7)

f,ax = plt.subplots(1,2,figsize=(16,10))
industry =  survey['q10Industry'].value_counts()
sns.barplot(industry.values,industry.index,ax=ax[0],palette='Pastel1')
for i,v in enumerate(industry.values):
    ax[0].text(10,i,v,fontsize=12,va='center')
ax[0].set_title('Industry distribution')
ax[0].set_xlabel('count')

agg = survey.groupby(['q10Industry','q3Gender'])['q3Gender'].count().reset_index(name='count')
agg =agg.pivot(columns='q3Gender',index='q10Industry',values='count')
sns.heatmap(agg,cmap='Pastel2',ax=ax[1],fmt='.0f',annot=True)
ax[1].set_title('Industry Distribution grouped by gender')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Industry')
plt.subplots_adjust(wspace=0.8)
cols =  survey.columns[survey.columns.str.startswith('q12')]
#cols
newJob = pd.DataFrame()

for i in cols:
    inter = survey[i].value_counts().reset_index(name='count')
    newJob=pd.concat([newJob,inter])
    #agg = survey.groupby([i,'q3Gender'])['q3Gender'].count().reset_index(name='count')
    #agg = agg.pivot(columns='q3Gender', index=i,values='count')
    #newJob = pd.concat([newJob,agg])

newJob=newJob.sort_values(by='count',ascending=False)
#newJob

plt.figure(figsize=(16,10))
sns.barplot(newJob['count'],newJob['index'],palette='Wistia')
for i,v in enumerate(newJob['count']):
    plt.text(0.8,i,v,fontsize=12,va='center')
plt.xlabel('Count')
plt.title('Top most qualities to be looked before taking a job')
plt.show()
cols = survey.columns[survey.columns.str.startswith('q13')]
#cols
skills = pd.DataFrame()

for i in cols:
    agg = survey[i].value_counts().reset_index(name='count')    
    skills = pd.concat([skills,agg])
    
#print(skills)
skills.sort_values(by='count',ascending=False,inplace=True)

plt.figure(figsize=(16,6))
sns.barplot(skills['count'],skills['index'],palette='spring')

for i,v in enumerate(skills['count']):
    plt.text(1,i,v,fontsize=20,verticalalignment='center')
                     
plt.title('Employers top most ways of mesuring skills')
plt.xlabel('Count')
plt.show()
cols = survey.columns[survey.columns.str.startswith('q17')]
#cols
challenges = pd.DataFrame()

for i in cols:
    agg = survey[i].value_counts().reset_index(name='count')
    challenges = pd.concat([challenges,agg])

challenges.sort_values(by='count',ascending=False,inplace=True)

plt.figure(figsize=(16,6))
sns.barplot(challenges['count'],challenges['index'],palette='winter')

for i,v in enumerate(challenges['count']):
    plt.text(1,i,v,fontsize=20,verticalalignment='center')
    
plt.title('Challenges Faced by Interviewers')
plt.xlabel('Count')
plt.show()
cols =  survey.columns[survey.columns.str.startswith('q19')]
#cols

tools =pd.DataFrame()
for i in cols:
    agg = survey[i].value_counts().reset_index(name='count')
    tools = pd.concat([tools,agg])
    
tools.sort_values(by='count',ascending=False,inplace=True)
plt.figure(figsize=(16,6))
sns.barplot(tools['count'],tools['index'],palette='PuBu')
for i,txt in enumerate(tools['count']):
    plt.text(1,i,txt,fontsize=20,verticalalignment='center')
    
plt.title('Popular assesing tools for interviews')
plt.xlabel('count')
plt.show()

cols = survey.columns[survey.columns.str.startswith('q20')]
#cols

qual = pd.DataFrame()
for i in cols:
    agg = survey[i].value_counts().reset_index(name='count')
    qual=pd.concat([qual,agg])

qual.sort_values(by='count',ascending=False,inplace=True)
qual

plt.figure(figsize=(16,10))
sns.barplot(qual['count'],qual['index'],palette='hsv')
for i,v in enumerate(qual['count']):
    plt.text(0.5,i,v,fontsize=20,verticalalignment='center')
plt.xlabel('count')
plt.title('Qualifications required for the onsite')
plt.show()
col1 = survey.columns[survey.columns.str.startswith('q21')]
col2 = survey.columns[survey.columns.str.startswith('q22')]
skills = pd.DataFrame()
col1

for i in col1:
    agg = survey[i].value_counts().reset_index(name='count')
    skills = pd.concat([skills,agg],axis=0,ignore_index=True)

lang = pd.DataFrame()
for i in col2:
    agg2 = survey[i].value_counts().reset_index(name='count')
    lang = pd.concat([lang,agg2])
    
lang.sort_values(by='count',ascending=False,inplace=True)

skills.loc[len(skills)]=['Language Proficiency',lang['count'].sum()]
skills.sort_values(by='count',ascending=False,inplace=True)
#skills

f,ax =plt.subplots(1,2,figsize=(15,10))
sns.barplot(skills['count'], skills['index'], palette='GnBu',ax=ax[0])
for i,v in enumerate(skills['count']):
    ax[0].text(100,i,v,fontsize=18,verticalalignment='center')
ax[0].set_xlabel('Count')
ax[0].set_ylabel('Skills')
ax[0].set_title('Desired Skill required')

sns.barplot(lang['count'],lang['index'],palette='YlGn',ax=ax[1])
for i,v in enumerate(lang['count']):
    ax[1].text(100,i,v,fontsize=12)
ax[1].set_xlabel("Count")
ax[1].set_ylabel('Languages')
ax[1].set_title('Language Proficiency')
plt.subplots_adjust(wspace=0.5)
plt.show()
cols =  survey.columns[survey.columns.str.startswith('q23')]
frame=pd.DataFrame()

for i in cols:
    agg = survey[i].value_counts().reset_index(name='count')
    frame=pd.concat([frame,agg],axis=0,ignore_index=True)
#frame
frame.sort_values(by='count',ascending=False,inplace=True)
plt.figure(figsize=(16,8))
sns.barplot(frame['count'],frame['index'],palette='OrRd')
for i,v in enumerate(frame['count']):
    plt.text(10,i,v,fontsize=12,verticalalignment='center')
    
plt.xlabel('Count')
plt.title('Frameworks in Demand')
plt.show()
wc=WordCloud(height=600,width=1400,colormap='hsv',stopwords=STOPWORDS,max_words=1000,background_color='white').generate(' '.join(survey['q0023_other'].dropna().astype(str)))
plt.figure(figsize=(16,10))
plt.imshow(wc)
plt.title('Favourite Frameworks')
plt.axis('off')
plt.show()
cols= survey.columns[survey.columns.str.startswith('q25')]
cols = cols.drop('q25LangOther')

f,ax = plt.subplots(4,6,figsize=(16,25))
axs=ax.ravel()

for i,c in enumerate(cols):
    sns.countplot(survey[c],ax=axs[i],palette='pink')
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')
    axs[i].set_title(survey_codebook.loc[c]['Survey Question'])

plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.suptitle('Programming language know or will learn',fontsize=14)
plt.show()
cols = survey.columns[survey.columns.str.startswith('q26')]
#print(len(cols))


f,ax = plt.subplots(4,5,figsize=(16,25))
axs=ax.ravel()

for i,c in enumerate(cols):
    if survey[c].nunique() > 1:
        sns.countplot(survey[c],ax=axs[i],palette='cool')
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        axs[i].set_title(survey_codebook.loc[c]['Survey Question'])
    
plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.suptitle('Framework know or will learn',fontsize=14)
plt.show()

res=survey['q27EmergingTechSkill'].value_counts()

fig = plt.figure(figsize=(16,10))
sns.barplot(x=res.values,y=res.index,palette='Wistia')
for i,v in enumerate(res.values):
    plt.text(10,i,v,fontsize=20,va='center')
plt.xlabel('Count',fontsize=12)
plt.title('Emerging Technology')
plt.show()

cols = survey.columns[survey.columns.str.startswith('q28')]
#cols
cols = cols.drop('q28LoveOther')

fig, ax = plt.subplots(6,4,figsize=(16,25))
axs=ax.ravel()

for i,c in enumerate(cols):
    sns.countplot(survey[c],palette='plasma',ax=axs[i])
    axs[i].set_xlabel("")
    axs[i].set_ylabel('')
    axs[i].set_title(survey_codebook.loc[c]['Survey Question'])
    
plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.suptitle('Programming Language -Love or Hate?',fontsize=20)
plt.show()

cols = survey.columns[survey.columns.str.startswith('q29')]
#len(cols)

f,ax = plt.subplots(5,4,figsize=(16,25))
axs = ax.ravel()

for i,c in enumerate(cols):
    sns.countplot(survey[c],palette='spring',ax=axs[i])
    axs[i].set_title(survey_codebook.loc[c]['Survey Question'])
    axs[i].set_xlabel('')
    axs[i].set_ylabel('')
    
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.suptitle('Programming Framework -Love or Hate?',fontsize=20)
plt.show()
cols = survey.columns[survey.columns.str.startswith('q30')]
learn =pd.DataFrame()

for i in cols:
    agg = survey[i].value_counts().reset_index(name='count')
    learn = pd.concat([learn,agg])

learn.sort_values(by='count',ascending=False,inplace=True)
#print(learn)

plt.figure(figsize=(16,10))
sns.barplot(learn['count'],learn['index'],palette='cool')
for i,v in enumerate(learn['count']):
    plt.text(10,i,v,fontsize=12,va='center')
    
plt.xlabel('Count')
plt.ylabel('')
plt.title('Source of learning')
plt.show()
agg = survey['q32RecommendHackerRank'].value_counts().reset_index(name='count')
#print (agg)

plt.figure(figsize=(10,2))
#sns.countplot(survey['q32RecommendHackerRank'],palette='cool')
sns.barplot(agg['count'],agg['index'],palette='cool')
for i,v in enumerate(agg['count']):
    plt.text(10,i,v,fontsize='12',va='center')
plt.xlabel('Count')
plt.ylabel('')
plt.title('Recommend Hackerrank?')
plt.show()
agg = survey['q34PositiveExp'].value_counts().reset_index(name='count')
#agg['index'] = agg['index'].astype(str)
#agg.sort_values(by='count',ascending=False)
#print (agg)
#print(agg.info())

plt.figure(figsize=(15,5))
sns.barplot(agg['count'], agg['index'], palette='terrain', orient='h', order=agg['index'])
for i,v in enumerate(agg['count']):
    plt.text(10,i,v,fontsize=12,va='center')
    
plt.xlabel('Count')
plt.ylabel('Rating')
plt.title('Rate your experience in Hackerrank')
plt.show()
