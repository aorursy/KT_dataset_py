import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
enviroment = twosigmanews.make_env()
(market_train_df, news_train_df) = enviroment.get_training_data()
market_train_df.head()
# market_train_df.describe()
# min(market_train_df['time']) 2007-02-01
# max(market_train_df['time']) 2016-12-30
# len(set(market_train_df['assetCode'])) 3780
# len(set(market_train_df['assetName'])) 3511
boxplot=market_train_df.iloc[:,4:-1].boxplot()
plt.xticks(rotation='vertical')
# check the number of assetcodes that an assetName includes
df=market_train_df.groupby('assetName')['assetCode'].nunique()
print(np.unique(df,return_counts=True)) # there are 269 assetNames with 0 assetcodes,.... 
df[df==110] # The assetname is unknown
# if each code corresponds to only one name
df=market_train_df.groupby('assetCode')['assetName'].nunique()
print(np.unique(df,return_counts=True)) 
# the result below shows that every code corresponds to only one name.
# check universe for each day
df=market_train_df.groupby('time')['universe'].agg(['sum','count'])
df['proportion']=df['sum']/df['count']
plt.hist(df['proportion'])
plt.show()
np.mean(df['proportion'])
# Generally speaking, the proportion of instruments that are avilable for trading is around 0.6
# check missing data
# check columns with missing data
missing_columns=market_train_df.columns[market_train_df.isnull().any()]
print(missing_columns) # only adjusted return columns have missing data
# check proportion of missing data in each column
for i in missing_columns:
    total_observation=market_train_df.shape[0]
    print(i, ' :', sum(market_train_df['returnsClosePrevMktres1'].isnull())/total_observation)
import statsmodels.api as sm
df=market_train_df.iloc[:,3:-1]
for i in range(df.shape[1]-1):
    for j in range(i+1,df.shape[1]):
        trainset=df.iloc[:,[i,j]].dropna()
        model=sm.OLS(trainset.iloc[:,0],trainset.iloc[:,1]).fit()
        if abs(model.rsquared_adj)>0.3:
            print(list(trainset),':',model.rsquared_adj)
# The correlation between returnsClosePrevMktres10 and returnsClosePrevRaw10, but let me check:
plt.scatter(market_train_df['returnsClosePrevRaw10'],market_train_df['returnsClosePrevMktres10'])
plt.show()
# The plot still shows high correlation.
news_train_df.head()
news_train_df.describe()
news_train_df.boxplot(column=['sentimentNegative','sentimentNeutral','sentimentPositive'])
print(np.unique(news_train_df['sentimentClass'],return_counts=True))
# Therefore, the distribution of sentiments among assets is kind of even.
df=news_train_df['relevance'].to_frame()
df['sentiment_wordcount_proportion']=news_train_df.sentimentWordCount/news_train_df.wordCount
# plt.scatter(df.relevance,df.sentiment_wordcount_proportion)
# plt.show()
# There is no correlation between the two features.
df.boxplot()
news_train_df.boxplot(column=['companyCount'])
plt.scatter(news_train_df.wordCount,news_train_df.companyCount)
plt.show()
import statsmodels.api as sm
model=sm.OLS(news_train_df.wordCount,news_train_df.companyCount).fit()
model.summary()
# Double checking the plot and the model, I find no relationship between wordcount and companycount.
df=news_train_df.iloc[:,-10:]
# x=list(range(df.shape[0]))
# for i in range(df.shape[1]):
#     plt.figure()
#     plt.plot(x,df.iloc[:,i])
# plt.show()
# By ploting, I find that the novelty count and volum count increase with time, to check my hypothesis:
for i in range(df.shape[1]-1):
    df['diff_'+str(i)]=df.iloc[:,i+1]-df.iloc[:,i]
    print(i,'：' , sum(df['diff_'+str(i)]<0))
# Therefore, the hypothesis is true.
# To check if the increase only affected by time
df=news_train_df.iloc[:,-10:-5]
for i in range(df.shape[1]-1):
    df['diff_'+str(i)]=df.iloc[:,i+1]-df.iloc[:,i]
df.iloc[:,-4:].boxplot()
df.iloc[:,-4:].describe()
# Therefore, only minority of assets increase their novelty count with time. I will check if the increase contribute to the prediction
df=news_train_df.iloc[:,-5:]
for i in range(df.shape[1]-1):
    df['diff_'+str(i)]=df.iloc[:,i+1]-df.iloc[:,i]
df.iloc[:,-4:].boxplot()
df.iloc[:,-4:].describe()
# Therefore, only minority of assets increase their volumn count with time. I will check if the increase contribute to the prediction
# news_train_df.groupby(['urgency','takeSequence'])['takeSequence'].count()
# print(min(news_train_df.takeSequence[news_train_df.urgency==1]))
# print(max(news_train_df.takeSequence[news_train_df.urgency==1]))
# print(min(news_train_df.takeSequence[news_train_df.urgency==3]))
# print(max(news_train_df.takeSequence[news_train_df.urgency==3]))
# There is no correlation between urgency and takeSequence
# check if there are 3780 unique assetcode in news data
assetcode_list=[]
for i in range(news_train_df.shape[0]):
    assetcode_list.append(list(eval(news_train_df.assetCodes[i])))
    if i%100==0:
        if len(set([j for i in assetcode_list for j in i]))>3780:
            print('Unique number of assetcode in news data becomes larger than 3780 before the ', i, ' row in the news data set. The total number of rows in the data is ',news_train_df.shape[0],'.')
            break
        else: 
            continue
# there are much more asset codes in the news data set 
# check the number of unique assetcode in news data
d={'codes':[i for i in news_train_df.assetCodes]}
df=pd.DataFrame(data=d)
codes=df.codes.apply(lambda x: list(eval(x))).tolist()
codes_news_unique=set([j for i in codes for j in i])
len(codes_news_unique) # There are 14293 unique asset codes in the news data set
# check the overlap of assetcodes in both data sets
ac_list_market=set(market_train_df.assetCode)
len(list(codes_news_unique & ac_list_market)) # 3663
# check assetNames
print('Number of unique assetName in market data:',len(set(market_train_df.assetName.tolist())))
print('Number of unique assetName in news data:',len(set(news_train_df.assetName.tolist())))
print('Number of unique assetName in both data sets:',len(set(market_train_df.assetName.tolist()) & set(news_train_df.assetName.tolist())))
# Check if the mapping between assetcode and assetname is the same for those in the both data sets:
df_news=news_train_df[['assetName','assetCodes']].sort_values('assetName').drop_duplicates()
df_news=df_news.groupby('assetName')['assetCodes'].apply(lambda x: ', '.join(x)).reset_index()
df_market=market_train_df[['assetName','assetCode']].sort_values('assetName').drop_duplicates()
df_market=df_market.groupby('assetName')['assetCode'].apply(lambda x: ', '.join(x)).reset_index()
df_market.columns=['assetName','assetCodeM']
df=df_market.join(df_news.set_index('assetName'),on='assetName')
df
len(set(news_train_df.sourceId))/len(news_train_df.sourceId)
# SouceID is not a unique identifier
print(len(set(news_train_df.headlineTag))) # 163 different headlineTag
print(len(set(news_train_df.subjects))) # 1733963 different combination of subjects -- too many to use
print(len(set(news_train_df.sourceId))) # 6340206 different combination of subjects -- too many to use
def prepare_data(market_df, news_df,key='name'):
    # market data
    market_df['time'] = market_df.time.dt.date
    market_df['price_diff']=market_df['close']-market_df['open']
    market_df=market_df.drop(['returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevRaw10',
                    'returnsOpenPrevRaw10','open','close'],axis=1)
    # news data
    counts=news_df.columns[25:35] # get the names of the novelty and volumn count columns
    news_df['firstCreated']=news_df.firstCreated.dt.date
    news_df['increase_novelty_count']=news_df['noveltyCount7D']-news_df['noveltyCount12H']
    news_df['increase_volume_count']=news_df['noveltyCount7D']-news_df['noveltyCount12H']
    news_df['sentiment_wordcount/wordcount']=news_df['sentimentWordCount']/news_df['wordCount']
    news_df=news_df.drop(['time','sourceTimestamp','sourceId','urgency','takeSequence','provider','subjects',
                 'audiences','marketCommentary','sentenceCount','sentenceCount','wordCount',
                 'firstMentionSentence','sentimentWordCount'],axis=1)
    news_df=news_df.drop(counts,axis=1)
    # left join using time and assetname
    if key=='name':
        data=pd.merge(market_df,news_df,how='left',left_on=['time','assetName'],right_on=['firstCreated','assetName'])
    if key=='code':
        # since every code has only one name, we just need to drop where assetcode in market does not exist in the news data
        data=pd.merge(market_df,news_df,how='left',left_on=['time','assetName'],right_on=['firstCreated','assetName']) 
        # remove where assetcodes is na in the news data set
        data=data.dropna(subset=['assetCodes'])
        def tell(x,y):
            result=x in y
            return result
        index=data[['assetCode','assetCodes']].apply(lambda x:tell(x['assetCode'],x['assetCodes']),axis=1)
        data=data[index]
    return data
market_df=market_train_df.copy()
news_df=news_train_df.copy()
df=prepare_data(market_df,news_df,key='name')
print(df.shape)
df.head()
# This method takes too long to run
# market_df=market_train_df.copy()
# news_df=news_train_df.copy()
# df2=prepare_data(market_df,news_df,key='code')
# print(df2.shape)
# df2.head()
# combine with assetcode step by step
data=df.dropna(subset=['assetCodes'])
data.shape