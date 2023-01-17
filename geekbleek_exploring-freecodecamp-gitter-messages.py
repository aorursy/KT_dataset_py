import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
chat = pd.read_csv("../input/freecodecamp_casual_chatroom.csv", index_col=0)
fig_kwargs = {'figsize': (12, 6), 'fontsize': 16}
chat = chat.iloc[:, 1:]
chat.head(3)
chat.sent.duplicated().sum() / len(chat)
chat = chat.drop_duplicates()
chat = chat.loc[~chat.sent.duplicated()]
import re
regExPattern = re.compile(r'.*\bcisco\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
pd.set_option('display.max_colwidth', -1)
searchFrame[['fromUser.displayName','sent','text']].tail(10)
len(searchFrame)
len(searchFrame) / len(chat) * 100
len(chat)
(pd.to_datetime(searchFrame.sent)
     .to_frame()
     .set_index('sent')
     .assign(n=0)
     .resample('M')
     .count()
     .plot.line(**fig_kwargs, title="Postings About Cisco over Time"))
(pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(n=0)
     .resample('M')
     .count()
     .plot.line(**fig_kwargs, title="Number of freeCodeCamp Messages over Time"))
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
combinedTimeSeries.head(10)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"Cisco\" over Time"))
regExPattern = re.compile(r'.*\baws\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"AWS\" over Time"))
searchFrame = searchFrame.assign(timestamp = lambda x: pd.to_datetime(x.sent))
filteredFrames = searchFrame[(searchFrame.timestamp >= '2017-08-01') & (searchFrame.timestamp <= '2017-09-01')]
filteredFrames[['fromUser.displayName','timestamp','text']].head(30)
#Number of records in our filtered data
len(filteredFrames)
#Confirm matches number of records in our graphed time-series data
combinedTimeSeries.loc['2017-08-31'].n
regExPattern = re.compile(r'.*\bjavascript\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
searchFrame = searchFrame.assign(timestamp = lambda x: pd.to_datetime(x.sent))
filteredFrames = searchFrame[(searchFrame.timestamp >= '2017-08-01') & (searchFrame.timestamp <= '2017-09-01')]
filteredFrames[['fromUser.displayName','timestamp','text']].tail(5)
len(filteredFrames)
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"Javascript\" over Time"))
regExPattern = re.compile(r'.*\bpython\b.*', flags=re.IGNORECASE|re.MULTILINE|re.S)
searchFrame = chat[chat['text'].str.match(regExPattern,na = False)]
searchTimeSeries = (pd.to_datetime(searchFrame.sent)
         .to_frame()
         .set_index('sent')
         .assign(n=0)
         .resample('M')
        .count()
        )

chatTimeSeries = (pd.to_datetime(chat.sent)
     .to_frame()
     .set_index('sent')
     .assign(p=0)
     .resample('M')
     .count())
                  
combinedTimeSeries = searchTimeSeries.merge(chatTimeSeries, left_index=True, right_index=True, how='inner')
combinedTimeSeries = combinedTimeSeries.assign(pcnt = lambda x: x.n/x.p * 100)
(combinedTimeSeries['pcnt']
    .plot.line(**fig_kwargs, title="Percentage of Messages Containing \"Python\" over Time"))