import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/medium.csv')
data.head(3)
tag = data.groupby('1.Tag').size()
top = list(tag.index)
value = list(tag.values)
colors = ['gold', 'lightcoral', 'lightskyblue']
plt.pie(value, labels=top, colors=colors, autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
len(data)
len(data['3.Title'].unique())
len(data['2.Name'].unique())
name = data.groupby('2.Name').size()
name = name[(name.values>2)]
plt.figure(figsize=(15,5))
plt.xticks(rotation='vertical')
sns.barplot(name.index, name.values, alpha=0.8)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Number of Articles', fontsize=14)
plt.title("Authors with Respect to the number of Articles tagged under AI,ML & DL", fontsize=20)
plt.show()
from wordcloud import WordCloud
from wordcloud import STOPWORDS
text = ""
for ind, row in data.iterrows():
    text += row["3.Title"] + " "
text = text.strip()
plt.figure(figsize=(10,8))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=100, max_words=40).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
for i in range(0,len(data)):
    data["4.Body"][i] = str(data["4.Body"][i])
text = ""
for ind, row in data.iterrows():
    text += row["4.Body"] + " "
text = text.strip()
plt.figure(figsize=(10,8))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=100, max_words=40).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
for i in range(0,len(data)):
    if 'K' in data["5.Upvotes"][i]:
        data["5.Upvotes"][i]= int(float(data["5.Upvotes"][i].replace("K",""))*1000)
    else:
        data["5.Upvotes"][i] = int(data["5.Upvotes"][i])
data.head(2)
data['5.Upvotes'].mean()
data = data.drop("Unnamed: 0",axis=1)
upvote=data.groupby("2.Name").sum()
upvote.head(2)
len(upvote.index)
upvote=(upvote[(upvote['5.Upvotes'] > 7000)])
colorr =           ['#78C850',  
                    '#F08030',  
                    '#6890F0',  
                    '#A8B820',  
                    '#A8A878',  
                    '#A040A0',  
                    '#F8D030',  
                    '#E0C068',  
                    '#EE99AC',  
                    '#C03028',  
                    '#F85888',  
                    '#B8A038',  
                    '#705898',  
                    '#98D8D8',  
                    '#7038F8',  
                   ]
plt.figure(figsize=(15,6))
plt.xticks(rotation='vertical')
sns.barplot(upvote.index, upvote["5.Upvotes"], alpha=0.8,palette=colorr)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Number of Upvotes', fontsize=14)
plt.title("Authors with Respect to the number of Upvotes wrt articles tagged under AI,ML & DL", fontsize=20)
plt.show()
plt.figure(figsize=(15,6))
sns.swarmplot(x='1.Tag', y='5.Upvotes', data=data)
plt.xlabel('Tag', fontsize=14)
plt.ylabel('Number of Upvotes', fontsize=14)
plt.title("Number of Upvotes with respect to Tags", fontsize=20)
plt.show()
ML = data[(data['1.Tag']=='Machine Learning')]
ML.head(1)
AI = data[(data['1.Tag']=='Artificial Intelligence')]
AI.head(1)
DL = data[(data['1.Tag']=='Deep Learning')]
DL.head(1)
from functools import reduce
ML_DL = list(reduce(set.intersection, map(set, [ML['3.Title'], DL['3.Title']])))
ML_AI = list(reduce(set.intersection, map(set, [ML['3.Title'], AI['3.Title']])))
AI_DL = list(reduce(set.intersection, map(set, [AI['3.Title'], DL['3.Title']])))
tag = [len(ML_DL),len(ML_AI),len(AI_DL)]
top = ['ML and DL','ML and AI','AI and DL']
colors = ['gold', 'lightcoral', 'lightskyblue']
plt.pie(tag, labels=top, colors=colors, autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
ML_DL_AI = list(reduce(set.intersection, map(set, [ML['3.Title'], DL['3.Title'],AI['3.Title']])))
len(ML_DL_AI)
data.head(3)
for i in range(0,len(data)):
    if (pd.isnull(data['7.Comments'][i])) is True:  #enter null comments as 0
        data['7.Comments'][i] = 0    
    else:    
        data['7.Comments'][i] = str(data['7.Comments'][i])
        data['7.Comments'][i]= data['7.Comments'][i].replace(' responses','')
        data['7.Comments'][i]= data['7.Comments'][i].replace(' response','')
        data['7.Comments'][i]= int(data['7.Comments'][i])
data.head(1)
np.mean(data['7.Comments'])
plt.figure(figsize=(15,6))
sns.swarmplot(x='1.Tag', y='7.Comments', data=data)
plt.xlabel('Tag', fontsize=14)
plt.ylabel('Number of Comments', fontsize=14)
plt.title("Number of Comments with respect to Tags", fontsize=20)
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(data=data)
plt.show()
comments=data.groupby("2.Name").sum()
comments=(comments[(comments['7.Comments'] > 19)])
len(comments)
plt.figure(figsize=(15,6))
plt.xticks(rotation='vertical')
sns.barplot(comments.index, comments["7.Comments"], alpha=1)
plt.xlabel('Name', fontsize=14)
plt.ylabel('Number of Comments', fontsize=14)
plt.title("Authors with Respect to the number of Comments wrt articles tagged under AI,ML & DL", fontsize=20)
plt.show()
for i in range(0,len(data)):
    data['6.Date'][i] =''.join([i for i in data['6.Date'][i] if not i.isdigit()])
month = data.groupby('6.Date').size()
plt.figure(figsize=(15,5))
plt.xticks(rotation='vertical')
sns.barplot(month.index, month.values)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Articles', fontsize=14)
plt.title("Articles with respect to the Month it was written", fontsize=20)
plt.show()
