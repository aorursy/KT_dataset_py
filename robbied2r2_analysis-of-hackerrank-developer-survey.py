import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
survey = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv', low_memory=False)

# Select a subset that only includes student responses
survey = survey[survey['q8Student'] == 'Students']

survey_codebook =pd.read_csv('../input/HackerRank-Developer-Survey-2018-Codebook.csv')
survey_codebook=survey_codebook.set_index('Data Field')
print("Total number of students surveyed:",len(survey))
# Select a subset of students from the United States only
survey = survey[survey['CountryNumeric2'] == 'United States']

# Select yet another subset of FEMALE students from the United States 
surveyF = survey[survey['q3Gender'] == 'Female']
print("Number of U.S.-based students:",len(survey))
survey = survey.replace('#NULL!',np.nan)

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

    
plt.title('Age Began Coding based on Gender')
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
plt.title('Education Qualification')
plt.show()
from wordcloud import WordCloud, STOPWORDS
wc = WordCloud(height=600,width=1400,max_words=1000,stopwords=STOPWORDS,colormap='coolwarm',background_color='Cyan').generate(' '.join(survey['q0004_other'].dropna().astype(str)))
plt.figure(figsize = (16,16))
plt.imshow(wc)
plt.title('Wordcloud for responses of \"OTHER\"')
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
plt.title('WordCloud for responses of \"OTHER\"')
plt.show()
res=surveyF['q27EmergingTechSkill'].value_counts()

fig = plt.figure(figsize=(16,10))
sns.barplot(x=res.values,y=res.index,palette='Wistia')
for i,v in enumerate(res.values):
    plt.text(10,i,v,fontsize=20,va='center')
plt.xlabel('Count',fontsize=12)
plt.title('Emerging Technology')
plt.show()

cols = surveyF.columns[surveyF.columns.str.startswith('q30')]
learn =pd.DataFrame()

for i in cols:
    agg = surveyF[i].value_counts().reset_index(name='count')
    learn = pd.concat([learn,agg])

learn.sort_values(by='count',ascending=False,inplace=True)

plt.figure(figsize=(16,10))
sns.barplot(learn['count'],learn['index'],palette='cool')
for i,v in enumerate(learn['count']):
    plt.text(10,i,v,fontsize=12,va='center')
    
plt.xlabel('Count')
plt.ylabel('')
plt.title('Source of learning')
plt.show()
agg = surveyF['q32RecommendHackerRank'].value_counts().reset_index(name='count')

plt.figure(figsize=(10,2))
sns.barplot(agg['count'],agg['index'],palette='cool')
for i,v in enumerate(agg['count']):
    plt.text(10,i,v,fontsize='12',va='center')
plt.xlabel('Count')
plt.ylabel('')
plt.title('Recommend Hackerrank?')
plt.show()
