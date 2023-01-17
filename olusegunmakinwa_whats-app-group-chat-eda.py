import numpy as np
import pandas as pd 
import re
import emoji 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
##reading in the already data
df=pd.read_csv(r"C:\Users\User\Downloads\Discussing Football.csv",index_col=None)
##checking the data again, just to be sure 
df
#converting date and time to timsstamp
df['DateTime']=pd.to_datetime(df['Date']+ ' '+ df["Time"], dayfirst=True)
#indexing with the timestamp
df.index=df['DateTime']
member_activity=df.Sender.value_counts()
plt.figure(figsize=[15,5])
plt.xlabel('Group Members')
plt.ylabel('No. of Messages')
member_activity.plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.xticks(rotation=50)
plt.title('Members of the Group and their Relative Activity')
plt.show()
df["Sender"].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('Top 10 Most Active Members of the Group')
plt.xlabel('No. of messages')
plt.ylabel('Senders')
plt.show()
df["Sender"].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title(' 10 Least Active Members of the Group')
plt.xlabel('No. of messages')
plt.ylabel('Senders')
plt.show()
#Top 5 Active Hours of the Day
plt.figure(figsize=[15,5])
plt.xlabel('Hours of the day')
plt.ylabel('No. of media messages')
df.Hour.value_counts().head().sort_values().plot.bar(color={'r', 'b','g','y','b'})
plt.xticks(rotation=50)
plt.title('Top 5 Active Hours of the Day')
plt.show()
#5 Least Active Hours of the Day
plt.figure(figsize=[15,5])
plt.xlabel('Hours of the day')
plt.ylabel('No. of media messages')
df.Hour.value_counts().tail().sort_values(ascending=False).plot.bar(color={'r', 'b','g','y','b'})
plt.xticks(rotation=50)
plt.title('Least 5 Active Hours of the Day')
plt.show()
nocturnals=df[df['Hour']<6]
nocturnals.Sender.value_counts().head(10).plot.bar(color={'r', 'b','g','y','c'}, figsize=[15,5])
plt.xlabel('Names')
plt.ylabel('No. of Messages')
plt.title('Messengers at night')
plt.show()
top_10_active_days_of_the_group = df['Date'].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('top 10 active days of the group')
plt.xlabel('no. of messages')
plt.ylabel('dates')
plt.show()
least_10_active_days_of_the_group = df['Date'].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5] )
DHM=df[df['Date']=='24/02/2020']
DHM.Hour.value_counts().plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.xlabel('Hours of the Day')
plt.ylabel('No of messages')
plt.show()
DHM=df[df['Date']=='24/02/2020']
DHM.Sender.value_counts().plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.xlabel('Names')
plt.ylabel('No of messages')
plt.show()
#nUMBER OF MEDIA MESSAGES 
media=df[df["Message"]=='<Media omitted>']
len(media)
plt.figure(figsize=[15,5])
plt.xlabel('Group Members')
plt.ylabel('No. of media messages')
media.Sender.value_counts().head().plot.bar(color={'r', 'b','g','y','b'})
plt.xticks(rotation=50)
plt.title('Top 5 media senders')
plt.show()
#Top 5 longest sentences and their senders 
long_sents=df.sort_values(by=['Word_Count'],ascending=False)[1:7]
long_sents=long_sents[['Sender', "Word_Count"]]
plt.figure(figsize=[15,5])
sns.barplot( x="Sender",
    y="Word_Count",
    hue="Sender",
    data=long_sents)
plt.title('Longest Sentences Ever and their Senders')
plt.show()
word_count=df[['Sender', 'Letter_Count','Word_Count','Avg_Word_length']]
word_count.index=word_count['Sender']
#To check the those instances where group members sent sentences with long words 
word_count.sort_values(by=['Letter_Count'],ascending=False).head(5).plot.bar(figsize=[15,5])
plt.title('Comparing number of letters, word count and Average Word Length')
plt.xlabel('Names')
plt.ylabel('No of messages')
plt.show()
#However, let us check those who had very lenghty words... 
word_count2=df[['Sender', 'Letter_Count','Word_Count','Avg_Word_length']]
word_count2.groupby(df["Sender"]).sum().sort_values(by=['Letter_Count'],ascending=False).head(5).plot.bar(figsize=[15,5])
plt.title('Comparing number of letters, word count and Average Word Length')
plt.xlabel('Names')
plt.show()
word_count2.groupby(df["Sender"]).mean().sort_values(by=['Word_Count'],ascending=False).head().plot.bar(figsize=[15,5])
plt.title('Comparing number of letters, word count and Average Word Length')
plt.xlabel('Names')
plt.show()
plt.figure(figsize=[15,5])
plt.xlabel('Year')
plt.ylabel('No. of Messages')
df.index.year.value_counts().plot.bar(color={'r', 'b'})
plt.title('Number of Messages Sent in 2019 and 2020')
plt.show()
#spliting the dataset into 2019 and 2020 groups 
dateG2019=df[df.index.year==2019]
dateG2020=df[df.index.year==2020]

##checking if the split is correct, just to be sure 
len(dateG2019)+len(dateG2020)==len(df)
dateG2020.describe()
dateG2019.describe()
dateG2019['Date'].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of highest activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')
plt.show()
dateG2019['Date'].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of least activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')
plt.show()
dateG2020['Date'].value_counts().head(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of highest activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')
plt.show()
dateG2020['Date'].value_counts().tail(10).plot.bar(color={'r', 'b','g','y','b'},figsize=[15,5])
plt.title('10 days of least activity in 2019')
plt.xlabel('No. of messages')
plt.ylabel('Dates')

plt.show()
print('Date with the lowest activity in 2019: {}' .format(dateG2019['Date'].value_counts().tail(1)))
print('Date with the highest activity in 2019: {}' .format(dateG2019['Date'].value_counts().head(1)))
print('Date with the lowest activity in 2020: {}' .format(dateG2020['Date'].value_counts().tail(1)))
print('Date with the highest activity in 2020: {}' .format(dateG2020['Date'].value_counts().head(1)))
dateG2019.groupby(dateG2019.index.month)["Message"].count().plot.line(color={'b'},figsize=[15,5])
dateG2020.groupby(dateG2020.index.month)["Message"].count().plot.line(color={'r'})
plt.title('Line Graph Cmparing The Activity In the Group for 2019 and 2020')
plt.xlabel('Number of Messages')
plt.ylabel('Months of the year')
plt.show()
###2019 is in blue 
dateG2019.groupby(dateG2019.index.month).sum().plot.line()
dateG2020.groupby(dateG2020.index.month).sum().plot.line()
plt.title('Line Graph Cmparing The Activity In the Group for 2019 and 2020')
plt.xlabel('no. of messages')
plt.ylabel('Months of the year')
plt.show()
df
def gen_text(col):
    col=col.dropna()
    txt=" ". join(message for message in col)
    txt=re.sub('..... omitted', '', txt)
    return txt

text2019=gen_text(dateG2019['Message'])
text2020=gen_text(dateG2020['Message'])
stopwords = set(STOPWORDS)
wordcloud19=WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords).generate(text=text2019)
wordcloud20=WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=stopwords).generate(text=text2020)
plt.figure(figsize=[15,5])
plt.imshow(wordcloud19, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.figure(figsize=[15,5])
plt.imshow(wordcloud20, interpolation='bilinear')
plt.axis("off")
plt.show()
