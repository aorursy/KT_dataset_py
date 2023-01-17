# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
#reading in the test and training data
traindata=pd.read_csv( "../input/train.tsv",sep='\t')

traindata.head(5)
traindata.shape
item_desc=traindata['item_description']
traindata['item_condition_id'].unique()
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(item_desc))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
ones=traindata[traindata['item_condition_id']==1]['item_description']
ones.head()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(ones))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
twos=traindata[traindata['item_condition_id']==2]['item_description']
twos.head()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(twos))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
threes=traindata[traindata['item_condition_id']==2]['item_description']
threes.head()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(threes))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
fours=traindata[traindata['item_condition_id']==4]['item_description']
fours.head()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(fours))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
fives=traindata[traindata['item_condition_id']==5]['item_description']
fives.head()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(fives))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
len(item_desc)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
#TESING THE ANALYSER FOR A SAMPLE OUTPUT
result = analyser.polarity_scores("hi i am good")
result
item_senti=[]
#NOT GOING THROUGHT THE ENTIRE DATA TO FINISH EXECUTION FASTER
for desc in range(int(len(item_desc)/3)):
    senti = analyser.polarity_scores(str(item_desc[desc]))
    item_senti.append(senti['compound'])
    
    
    
item_senti[:5]
len(item_senti)
df=pd.DataFrame({'item_description':item_desc[:len(item_senti)],'sentiment':item_senti})
df.head()
from textblob import TextBlob
text="tHIS IS the best SHIRT"
blob = TextBlob(text)
blob.sentiment.polarity
textblob_senti=[]
for desc in range(int(len(item_desc)/3)):
    senti = TextBlob(str(item_desc[desc]))
    textblob_senti.append(senti.sentiment.polarity)
textblob_senti[:5]
df2=pd.DataFrame({'item_description':item_desc[:len(textblob_senti)],'sentiment':textblob_senti})
df2.head()
df2[df2.sentiment<-0.5].head()
NEG_SEG=-0.3   #anything less than this value is negative
POS_SEG=0.4    #anyting greater than this value is positive
#here the values in the range of  -0.3  - 0.4 will be neutral 
for i in range(int(len(item_desc)/3)):
    if item_senti[i]<=POS_SEG and item_senti[i]>=NEG_SEG:
        item_senti[i]=0
    elif item_senti[i]<NEG_SEG:
        item_senti[i]=-1
    elif item_senti[i]>POS_SEG:
        item_senti[i]=1
        
        
        
    