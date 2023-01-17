import numpy as np
import pandas as pd
# For visualizations
import matplotlib.pyplot as plt
# For regular expressions
import re
# For handling string
import string
# For performing mathematical operations
import math

# Importing dataset
df=pd.read_csv('../input/review/review.csv') 
print("Shape of data=>",df.shape)
df.head()
df.isnull().sum()
df['rating'] = df['rating'].str[2].astype(int)
df['helpful']=df['helpful'].str.extract('(\d+)').astype(float)
df['helpful']=df['helpful'].round().astype('Int64')
date=[]
data=df['date'].str.split(" ")
month_year=[]
for i in range(0,len(data)):
    lis=data[i][5:7]
    month_year.append(lis)
tmp=[]
for i in range(0,len(data)):
    month_year[i][1]=month_year[i][1].strip("\']")
    month_year[i]=" ".join(month_year[i])
r=df['rating']
month_year[0][-4:]
r
year=[]
for i in range(0,len(data)):
    year.append(month_year[i][-4:])
    
print(set(year))
avg_2018=0
avg_2019=0
avg_2020=0
s_2018=0
s_2019=0
s_2020=0
for i in range(0,len(year)):
    if(year[i]=="2018"):
        avg_2018=avg_2018+r[i]
        s_2018=s_2018+1
    if(year[i]=="2019"):
        avg_2019=avg_2019+r[i]
        s_2019=s_2019+1
    if(year[i]=="2020"):
        avg_2020=avg_2020+r[i]
        s_2020=s_2020+1
avg_2018=avg_2018/s_2018
avg_2019=avg_2019/s_2019
avg_2020=avg_2020/s_2020
x=['2018','2019','2020']
y=[avg_2018,avg_2019,avg_2020]
plt.plot(x,y)
h=df['helpful']
avg_1=0
avg_2=0
avg_3=0
avg_4=0
avg_5=0
s_1=0
s_2=0
s_3=0
s_4=0
s_5=0
for i in range(0,len(year)):
    if(r[i]==1):
        if(h[i]!=0):
            avg_1=avg_1+h[i]
            s_1=s_1+1
    if(r[i]==2):
        if(h[i]!=0):
            avg_2=avg_2+h[i]
            s_2=s_2+1
    if(r[i]==3):
        if(h[i]!=0):    
            avg_3=avg_3+h[i]
            s_3=s_3+1
    if(r[i]==4):
        if(h[i]!=0):    
            avg_4=avg_4+h[i]
            s_4=s_4+1
    if(r[i]==5):
        if(h[i]!=0):    
            avg_5=avg_5+h[i]
            s_5=s_5+1
avg_1=avg_1/s_1
avg_2=avg_2/s_2
avg_3=avg_3/s_3
avg_4=avg_4/s_4
avg_5=avg_5/s_5
x=[1,2,3,4,5]
y=[avg_1,avg_2,avg_3,avg_4,avg_5]
plt.bar(x,y)
from datetime import datetime
month_year.sort(key=lambda date: datetime.strptime(date, '%B %Y'))
price=259.00
reviews=df["name"].shape[0]
x=["price","reviews"]
y=[price,reviews]
plt.bar(x,y)

x=[]
y=[]
for m in [ele for ind, ele in enumerate(month_year,1) if ele not in month_year[ind:]]:
    x.append(m)
    y.append(month_year.count(m))
x
import matplotlib 
from matplotlib import pyplot as plt
plt.bar(x,y)
plt.xticks(x, rotation=90)
plt.show()
df.head()
df['helpful']
df['helpful']=df['helpful'].fillna(value=0)
df.isnull().sum()
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, text)

# Expanding Contractions in the reviews
df['body']=df['body'].apply(lambda x:expand_contractions(x))
df['cleaned']=df['body'].apply(lambda x: x.lower())
df['cleaned']=df['cleaned'].apply(lambda x: re.sub('\w*\d\w*','', x))
df['cleaned']=df['cleaned'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
df['cleaned']=df['cleaned'].apply(lambda x: re.sub(' +',' ',x))


for index,text in enumerate(df['cleaned'][35:40]):
  print('Review %d:\n'%(index+1),text)
df['name']='AmazonBasics AAA Performance Alkaline Non-Rechargeable Batteries (8-Pack)'
import spacy

# Loading model
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

# Lemmatization with stopwords removal
df['lemmatized']=df['cleaned'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))
df_grouped=df[['name','lemmatized']].groupby(by='name').agg(lambda x:' '.join(x))
df_grouped.head()
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_grouped['lemmatized'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
df_dtm.index=df_grouped.index
df_dtm.head(3)
df_dtm - 1

from wordcloud import WordCloud
from textwrap import wrap

# Function for generating word clouds
def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()
  
# Transposing document term matrix
df_dtm=df_dtm.transpose()

# Plotting word cloud for each product
for index,product in enumerate(df_dtm.columns):
  generate_wordcloud(df_dtm[product].sort_values(ascending=False),product)
from textblob import TextBlob
df['polarity']=df['lemmatized'].apply(lambda x:TextBlob(x).sentiment.polarity)
df['name']=df['name'].apply(lambda x: x.split('\'\'\'')[0])
print("3 Random Reviews with Highest Polarity:")
for index,review in enumerate(df.iloc[df['polarity'].sort_values(ascending=False)[:3].index]['body']):
  print('Review {}:\n'.format(index+1),review)
print("3 Random Reviews with Lowest Polarity:")
for index,review in enumerate(df.iloc[df['polarity'].sort_values(ascending=True)[:3].index]['body']):
  print('Review {}:\n'.format(index+1),review)
product_polarity_sorted=pd.DataFrame(df.groupby('name')['polarity'].mean().sort_values(ascending=True))

plt.figure(figsize=(16,1))
plt.xlabel('Polarity')
plt.ylabel('Products')
plt.title('Polarity of Amazon Product Reviews')
polarity_graph=plt.barh(np.arange(len(product_polarity_sorted.index)),product_polarity_sorted['polarity'],color='purple',)

# Writing product names on bar
for bar,product in zip(polarity_graph,product_polarity_sorted.index):
  plt.text(0.005,bar.get_y()+bar.get_width(),'{}'.format(product),va='center',fontsize=11,color='white')

# Writing polarity values on graph
for bar,polarity in zip(polarity_graph,product_polarity_sorted['polarity']):
  plt.text(bar.get_width()+0.001,bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=11,color='black')
  
plt.yticks([])
plt.show()
!pip install textstat
import textstat
df['dale_chall_score']=df['body'].apply(lambda x: textstat.dale_chall_readability_score(x))
df['flesh_reading_ease']=df['body'].apply(lambda x: textstat.flesch_reading_ease(x))
df['gunning_fog']=df['body'].apply(lambda x: textstat.gunning_fog(x))

print('Dale Chall Score of upvoted reviews=>',df[df['helpful']>1]['dale_chall_score'].mean())
print('Dale Chall Score of not upvoted reviews=>',df[df['helpful']<=1]['dale_chall_score'].mean())

print('Flesch Reading Score of upvoted reviews=>',df[df['helpful']>1]['flesh_reading_ease'].mean())
print('Flesch Reading Score of not upvoted reviews=>',df[df['helpful']<=1]['flesh_reading_ease'].mean())

print('Gunning Fog Index of upvoted reviews=>',df[df['helpful']>1]['gunning_fog'].mean())
print('Gunning Fog Index of not upvoted reviews=>',df[df['helpful']<=1]['gunning_fog'].mean())
df['text_standard']=df['body'].apply(lambda x: textstat.text_standard(x))

print('Text Standard of upvoted reviews=>',df[df['helpful']>1]['text_standard'].mode())
print('Text Standard of not upvoted reviews=>',df[df['helpful']<=1]['text_standard'].mode())
df['reading_time']=df['body'].apply(lambda x: textstat.reading_time(x))

print('Reading Time of upvoted reviews=>',df[df['helpful']>1]['reading_time'].mean())
print('Reading Time of not upvoted reviews=>',df[df['helpful']<=1]['reading_time'].mean())
reviews=df["name"].shape[0]
y=[reviews,78768]
x=["reviews","ratings"]
plt.bar(x,y)
review_list=[]
df.head()
k=df["cleaned"]
p=[]
for i in range(0,len(k)):
    p.append(len(k[i]))
    
len(p)
for m in [ele for ind, ele in enumerate(p,1) if ele not in p[ind:]]:
    x.append(m)
    y.append(p.count(m))
x=x[2:]
y=y[2:]
fig= plt.figure(figsize=(30,30))
plt.bar(x,y)
plt.show()
p=df['helpful']
count=0
for i in range(0,len(p)):
    if(p[i]!=0):
        count=count+1
        
x=["helpful votes",'ratings']
y=[count,78768]
plt.bar(x,y)
plt.yticks(y)
plt.show()
x=["helpful votes",'reviews']
y=[count,reviews]
plt.bar(x,y)
bad=[]
good=[]
for i in range(0,2600):
    if(df["polarity"][i]<0):
        bad.append(df['cleaned'][i])
    else:
        good.append(df['cleaned'][i])
bad[0]
reader_contents = bad
  
# empty string is declare 
text = "" 
  
# iterating through list of rows 
for word in reader_contents : 
        text = text + " " + word 
wordcloud = WordCloud(width=1000, height=800, max_words=150,colormap="Dark2").generate(text) 
  
# plot the WordCloud image  
plt.figure() 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.margins(x=0, y=0) 
plt.show() 
reader_contents = good
  
# empty string is declare 
text = "" 
  
# iterating through list of rows 
for word in reader_contents : 
        text = text + " " + word 
wordcloud = WordCloud(width=1000, height=800, max_words=150,colormap="Dark2").generate(text) 
  
# plot the WordCloud image  
plt.figure() 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.margins(x=0, y=0) 
plt.show() 
