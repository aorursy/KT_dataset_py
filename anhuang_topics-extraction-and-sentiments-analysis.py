from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import random 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

text = pd.read_csv('../input/abcnews-date-text.csv')
text['publish_date'] = text['publish_date']/10000
text['publish_date'] = text['publish_date'].astype(int)
text.head()

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
result = []
start = 0
end = 0
for i in range(2003,2018):
    word_list = {}
    temp = text.loc[text['publish_date']==i]
    temp = temp['headline_text']
    start = end
    lenth = len(temp)
    end = end + lenth
    for j in range(start,end):
        token = temp[j].split()
        for w in token:
            if w not in stop_words:
                w = lemmatizer.lemmatize(w)
                if w not in word_list:
                    word_list[w]=1
                else:
                    word_list[w]+=1
    count_list = sorted(word_list.items(),key = lambda x:x[1],reverse = True)
    temp_list = list(zip(*count_list[0:20]))
    result.append(list(temp_list[0]))
    print(i)
    print(count_list[0:20])
se = SentimentIntensityAnalyzer()
neg_change = []
neu_change = []
pos_change = []
compound_change = []
start = 0
end = 0
for i in range(2003,2018):
    temp = text.loc[text['publish_date']==i]
    temp = temp['headline_text']
    start = end
    lenth = len(temp)
    end = end + lenth
    neg = 0.0
    pos = 0.0
    neu = 0.0
    compound = 0.0
    for j in range(start,end):
        Sentiment = se.polarity_scores(temp[j])
        neg = neg + Sentiment['neg']
        neu = neu + Sentiment['neu']
        pos = pos + Sentiment['pos']
        compound = compound + Sentiment['compound']
    neg_change.append(neg/lenth)
    pos_change.append(pos/lenth)
    neu_change.append(neu/lenth)
    compound_change.append(compound/lenth)
    print(i)
    print('neg:%-6.3f,neu:%-6.3f,pos:%-6.3f,compound:%-6.3f'%(neg/lenth,neu/lenth,pos/lenth,compound/lenth))
year = [i for i in range(2003,2018)]

stack_bottom = []
for i in range(0,len(neg_change)):
    stack_bottom.append(neg_change[i] + neu_change[i])
b1 = plt.bar(year, neg_change)
b2 = plt.bar(year, neu_change, bottom = neg_change)
b3 = plt.bar(year, pos_change, bottom = stack_bottom)

for i in year:
    k = i-2003
    for j in range(0,20):
        plt.text(i-0.3,0.85-0.03*(j+1) ,result[k][j])
plt.title('Sentiment Change Bars')
plt.xlabel('years')
plt.ylabel('sentiment rate')
plt.legend([b1,b2,b3],['neg','neu','pos'])
plt.gcf().set_size_inches(18,10)
plt.show()


year = [i for i in range(2003,2018)]
l1 = plt.plot(year,neg_change,label='neg')
l2 = plt.plot(year,neu_change,label='neu')
l3 = plt.plot(year,pos_change,label='pos')
for i in year:
    k = i-2003
    for j in range(0,20):
        plt.text(i-0.2,0.85-0.03*(j+1) ,result[k][j])
plt.title('Sentiment Change Curves')
plt.xlabel('years')
plt.ylabel('sentiment rate')
plt.legend([b1,b2,b3],['neg','neu','pos'],loc='lower left')
plt.gcf().set_size_inches(18,10)
plt.show()