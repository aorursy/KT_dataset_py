import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input")[1], os.listdir("../input")[7])
pathList = [os.listdir("../input")[1], os.listdir("../input")[7]]
pathList
dfList = []
for i in pathList:
    path = "../input/%s" % i
    print(path)
    df = pd.read_json(path, lines=2)
    dfList.append(df)
df = pd.concat(dfList)
group = df.groupby('asin')
a = group.count()[(group.count().reviewText >= 100)]
asin = a.index
newDF = []
for i in asin[0:5]:
    newDF.append(df[(df.asin == i)])
df = pd.concat(newDF)
df.head()
import matplotlib.pyplot as plt
count = a['reviewText'].values[0:5]
index = np.arange(len(asin[0:5]))
plt.bar(index, count)
plt.xlabel('Products', fontsize=8)
plt.ylabel('Number of Products', fontsize=8)
plt.xticks(index, asin[0:5], fontsize=8)
from pylab import rcParams
plt.title("Review's Product")
plt.show()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
def getSentiment(sentences):
    neg_count = 0
    pos_count = 0
    neu_count = 0
    sid = SentimentIntensityAnalyzer()
    scores = []
    for i in sentences:
        score = sid.polarity_scores(i)
        scores.append(score)
    for i in scores:
        if i['compound'] < 0:
            neg_count += 1
        elif i['compound'] > 0:
            pos_count += 1
        else:
            neu_count += 1
    if pos_count >= neu_count >= neg_count:
        return "positive"
    elif neg_count >= neu_count >= pos_count:
        return "negative"
    else:
        return "neutral"
        
def splitSentences(reviewText):
    return tokenize.sent_tokenize(reviewText)

def sentimentCount(sentiments):
    pos = 0
    neg = 0
    neutral = 0
    for i in sentiments:
        if i == 'positive':
            pos += 1
        elif i == 'negative':
            neg += 1
        else:
            neutral += 1
    return [pos, neg, neutral]

def ratingCount(ratings):
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    for i in ratings:
        if i == 1:
            one += 1
        elif i == 2:
            two += 1
        elif i == 3:
            three += 1
        elif i == 4:
            four += 1
        else:
            five += 5
             
    return [one, two, three, four, five]
                
def calculateRating(rating):
    result = 0
    for i in range(1, len(rating) + 1):
        result += i * rating[i - 1]
    return result/ sum(rating)
    
def calculateRatioOfPositiveSentiment(p, n):
    if p == 0 and n == 0:
         return 0
    else:
        return (p/ (p+n)) * 5
    
def calculateTrustWeight(p_1, n_1, p_2, n_2):
    dimension = p_1 + n_1
    all_dimensions = dimension + p_2 + n_2
    
    if all_dimensions == 0:
        return 0
    else:
        return (p_1 + n_1)/ (p_1 + n_1 + p_2 + n_2)

def calculateTrustScoreOfSeller(t_1, w_1, t_2, w_2):
    return (t_1 * w_1) + (t_2 * w_2)
    
    
df = df[['asin', 'overall', 'reviewText', 'summary']]

#predict the sentiment of reviewText
reviewTextList = df['reviewText'].values
sentimentList = []
for i in reviewTextList:
    sentenceList = []
    sentenceList.extend(splitSentences(i))
    sentimentList.append(getSentiment(sentenceList))
df['reviewText_sentiment'] = sentimentList
print("review text done")

#predict the sentiment of summary
summaryList = df['summary'].values
sentimentList = []
for i in summaryList:
    sentenceList = []
    sentenceList.extend(splitSentences(i))
    sentimentList.append(getSentiment(sentenceList))
df['summary_sentiment'] = sentimentList
print("summary done")
#group by asin
data = df.groupby('asin')
asinList = []
rsList = []
t1List = []
t2List = []
w1List = []
w2List = []
TList = []
for i, j in data:
    #calculate rating
    ratings = ratingCount(j['overall'].values)
    rs = calculateRating(ratings)
    
    reviewTextSentiment = sentimentCount(j['reviewText_sentiment'].values)
    summarySentiment = sentimentCount(j['summary_sentiment'].values)
    
    #number of negative and positive of reviewText
    reviewTextSentiment_p = reviewTextSentiment[0]
    reviewTextSentiment_n = reviewTextSentiment[1]
    #calculate ratio of positive sentiment "t" of reviewText
    reviewTextSentiment_t = calculateRatioOfPositiveSentiment(reviewTextSentiment_p , reviewTextSentiment_n)
    
    #number of negative and positive of summary
    summarySentiment_p = summarySentiment[0]
    summarySentiment_n = summarySentiment[1]
    #calculate ratio of positive sentiment "t" of summary
    summarySentiment_t = calculateRatioOfPositiveSentiment(summarySentiment_p, summarySentiment_n)
    
    #calculate trust weight "w" of reviewText and summary
    reviewText_w = calculateTrustWeight(reviewTextSentiment_p, reviewTextSentiment_n, summarySentiment_p, summarySentiment_n)
    summary_w = calculateTrustWeight(summarySentiment_p, summarySentiment_n, reviewTextSentiment_p, reviewTextSentiment_n)
    
    #caluate trust score of seller "T" of reviewText and summary
    T = calculateTrustScoreOfSeller(reviewTextSentiment_t, reviewText_w, summarySentiment_t, summary_w)
    
    asinList.append(i)
    rsList.append(rs)
    t1List.append(reviewTextSentiment_t)
    t2List.append(summarySentiment_t)
    w1List.append(reviewText_w)
    w2List.append(summary_w)
    TList.append(T)
resultDF = pd.DataFrame({'asin':asinList, 'rating': rsList, 
                         'reviewText_t': t1List,
                         'summary_t': t2List,
                         'reviewText_w': w1List,
                         'summary_w': w2List,
                         'trust_of_seller': TList})
resultDF[['asin', 'rating', 'trust_of_seller']]
import matplotlib.pyplot as plt
ratingList = resultDF['rating'].values
trustList = resultDF['trust_of_seller'].values
labelList = resultDF['asin'].values
index = np.arange(len(labelList))
plt.xlabel('Products', fontsize=8)
plt.ylabel('Number of Products', fontsize=8)
plt.xticks(index, labelList, fontsize=8)
plt.title("Review's Product")
plt.bar(index-0.2, ratingList, width=0.4, label="rating")
plt.bar(index+0.2, trustList, width=0.4, label="trust of seller")
plt.legend()