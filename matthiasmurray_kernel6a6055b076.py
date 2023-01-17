# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from langdetect import detect



from sklearn.feature_extraction.text import CountVectorizer

from nltk import ngrams

from nltk.corpus import stopwords



plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tweets=pd.read_csv("../input/financial-tweets/stocks_cleaned.csv")

stockerbot=pd.read_csv("../input/financial-tweets/stockerbot-export.csv",error_bad_lines=False) #Some lines prevent the file from being loaded, we'll just skip them here
tweets.head()
stockerbot.head()
!pip install yfinance

import yfinance as yf
#It will be helpful to distinguish tweets that happen during market hours from those that do not

#This way we can identify the time period of trading data most immediately affected by the tweet

#Using a simplification of normal trading hours (930AM-4PM weekdays)



type(stockerbot.timestamp[0])

time=stockerbot.timestamp[0].split()

time[3].split(':')[0]

def in_sess(x):

    time=x.split()

    mins=time[3].split(':')[1]

    hour=time[3].split(':')[0]

    day=time[0]

    if day in ['Sat','Sun']:

        return False

    else:

        if int(hour)<16:

            if int(hour)>9:

                return True

            elif int(hour)==9 and int(mins)>=30:

                return True

            else:

                return False

        else:

            return False
stockerbot['insess']=stockerbot.timestamp.apply(lambda x:in_sess(x))
in_sess=stockerbot.loc[stockerbot['insess']==True,]

no_sess=stockerbot.loc[stockerbot['insess']==False,]

in_sess.head()
print("The number of in-session tweets is: "+str(len(in_sess)))

print("The number of verified in-session tweets is: "+str(len(in_sess.loc[in_sess['verified']==True,])))

print("The number of out-of-session tweets is: "+str(len(no_sess)))

print("The number of verified out-of-session tweets is: "+str(len(no_sess.loc[no_sess['verified']==True,])))
#Verified in_session tweets

ver_in=in_sess.loc[in_sess['verified']==True,]

ver_in.head()
#pull AZO (AutoZone) ticker data and info as an example

#First head entry is of TheStreet tweeting at Mon Jul 09 12:17:00 2018

azo=yf.Ticker("AZO")

azots=ver_in.iloc[0].timestamp

azo1st_1h=yf.download("AZO",start=pd.to_datetime(str(pd.to_datetime(azots)).split()[0]),end=pd.to_datetime(azots)+pd.Timedelta('1 day'),interval="1h")

azo1st_1h.head()

azoall=yf.download("AZO",period='1h')
#it appears the yf library publishes hourly for 930-10AM,10-11AM,11-12,12-1PM,1-2PM,2-3PM,3-4PM.

def hridx(ts):

    return int(ts.split()[3].split(':')[0])-9

    #return int((str(ts).split()[1]).split(':')[0])-9

    #return ts.hour-9

    #return int(str(ts).split)

#startprice=azo1st_1h.iloc[hridx(azots)]['Open']

#closeprice=azo1st_1h.iloc[-1]['Close']

#closeprice/startprice

def hourlystd(ticker,ts):

    return np.std(yf.download(ticker,start=ts,end=pd.to_datetime(str(ts.year+1)),interval='1h')['Close'])

#hourlystd('AZO',pd.to_datetime(azots))
[hridx(t) for t in list(in_sess.iloc[:5].timestamp)]
#our formula will be (1/hourlystd)*(closeprice/startprice)
ver_in['std_1hr']=ver_in[['symbols','timestamp']].apply(lambda x:hourlystd(x.symbols,pd.to_datetime(x.timestamp)),axis=1)
#drop the assorted errors causing NAs

ver_in=ver_in.dropna()
len(ver_in)
def eventprices_1h(ticker,ts):

    return yf.download(ticker,start=pd.to_datetime(str(pd.to_datetime(ts)).split()[0]),end=pd.to_datetime(str(pd.to_datetime(ts)).split()[0])+pd.Timedelta('1 day'),interval="1h")

def eventprices_1d(ticker,ts):

    return yf.download(ticker,start=pd.to_datetime(str(pd.to_datetime(ts)).split()[0]),end=pd.to_datetime(str(pd.to_datetime(ts)).split()[0])+pd.Timedelta('1 day'),interval='1d')

azo1st_1h=yf.download("AZO",start=pd.to_datetime(str(pd.to_datetime(azots)).split()[0]),end=pd.to_datetime(azots)+pd.Timedelta('1 day'),interval="1h")

eventprices_1h('AZO',azots)

azo1st_1d=yf.download("AZO",start=pd.to_datetime(str(pd.to_datetime(azots)).split()[0]),end=pd.to_datetime(str(pd.to_datetime(azots)).split()[0])+pd.Timedelta('1 day'),interval='1d')
azo1st_1d.head()
def chgratio(ticker,ts):

    try:

        try:

            df=eventprices_1h(ticker,ts)

            #startprice=df.iloc[hridx(ts)]['Open']

            startprice=df.iloc[0]['Open']

            closeprice=df.iloc[-1]['Close']

            return closeprice/startprice

        except:

            df=eventprices_1d(ticker,ts)

            startprice=df.iloc[0]['Open']

            closeprice=df.iloc[-1]['Close']

            return closeprice/startprice

    except:

        return ("ERROR",{'ts':ts,'ticker':ticker,'df':df})

    #except:

    #    return np.nan

#ver_in['chgratio']=ver_in[['symbols','timestamp']].apply(lambda x:chgratio(x.symbols,x.timestamp),axis=1)
#ver_in['move_score']=ver_in['chgratio']/ver_in['std_1hr']
ver_in=ver_in.dropna()

ver_in.head()
import matplotlib.pyplot as plt

#plt.hist(ver_in.move_score)
#movers=ver_in.loc[ver_in['move_score']>np.median(ver_in.move_score),]
from functools import reduce

#[(x,(reduce(lambda z,y:int(z)+int(y),movers.source.apply(lambda y:y==x)))) for x in set(movers.source)]
#movers.loc[movers.source=='Benzinga',:]
#bigmovers=movers.sort_values('move_score',ascending=False).head(10)

#bigmovers
#grpn_tfidf=mover_companies_df['GRPN']/mover_companies_all_df['GRPN'].loc[mover_companies_df['GRPN'].index,:]
def movegrams(biglist,movelist):

    #sb_ngr=stockerbot.copy()

    biglist['timestamp']=pd.to_datetime(biglist['timestamp'])

    biglist['text']=biglist['text'].astype(str)

    biglist['url']=biglist['url'].astype(str)

    biglist['company_names']=biglist['company_names'].astype('category')

    biglist['symbols']=biglist['symbols'].astype('category')

    biglist['source']=biglist['source'].astype('category')

        

    stop = stopwords.words('english')

    stop.append("RT")

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    handle_regex= "^@?(\w){1,15}$"

    

    biglist['text']=biglist['text'].str.replace(url_regex,'')

    biglist['text']=biglist['text'].str.replace(handle_regex,'')

    

    biglist['text']=biglist['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    

    biglist=biglist[biglist['source']!='test5f1798']

    

    unique_companies=np.array(biglist["symbols"].unique())



    companies_all_df={}

    for company in unique_companies:

        word_vectorizer = CountVectorizer(ngram_range=(2,3), analyzer='word')

        sparse_matrix = word_vectorizer.fit_transform(biglist.loc[biglist["symbols"] == company,['text']]['text'])

        frequencies = sum(sparse_matrix).toarray()[0]



        ### Now let's run bi/tri gram analysis (for each company and keep the 10 most relevant for each.)

        companies_all_df[company]=pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

    #return companies_all_df

    mover_companies_df={}

    for company in movelist.symbols.unique():

        word_vectorizer=CountVectorizer(ngram_range=(2,3),analyzer='word')

        sparse_matrix=word_vectorizer.fit_transform(movelist.loc[movelist['symbols']==company,['text']]['text'])

        frequencies=sum(sparse_matrix).toarray()[0]

        mover_companies_df[company]=pd.DataFrame(frequencies,index=word_vectorizer.get_feature_names(),columns=['frequency']).sort_values(by=['frequency'],ascending=False)

    #return mover_companies_df

    mover_tfidf={}

    for company in movelist.symbols.unique():

        co_tfidf=mover_companies_df[company]/companies_all_df[company].loc[mover_companies_df[company].index]

        mover_tfidf[company]=co_tfidf.dropna().sort_values('frequency',ascending=False)

    

    return mover_tfidf
def get_movelist(df,pct):

    df['std_1hr']=df[['symbols','timestamp']].apply(lambda x:hourlystd(x.symbols,pd.to_datetime(x.timestamp)),axis=1)

    df=df.dropna()

    df['chgratio']=df[['symbols','timestamp']].apply(lambda x:chgratio(x.symbols,x.timestamp),axis=1)

    df['move_score']=df['chgratio']/df['std_1hr']

    df=df.dropna()

    movers=df.sort_values('move_score',ascending=False).iloc[:int(pct*len(df))]

    return movers
!pip install findspark

!pip install pyspark

import findspark

#findspark.init()

import pyspark

try:

    sc.stop()

except:

    pass

sc=pyspark.SparkContext("local[*]","MyApp")
def addstd(row):

    row.append(hourlystd(row[4],pd.to_datetime(row[2])))

    #row['std_1hr']=row[['symbols','timestamp']].apply(lambda x:hourlystd(x.symbols,pd.to_datetime(x.timestamp)),axis=1)

    #row=row.dropna()

    return row

def addchg(row):

    row.append(chgratio(row[4],row[2]))

    #row['chgratio']=row[['symbols','timestamp']].apply(lambda x:chgratio(x.symbols,x.timestamp),axis=1)

    #row=row.dropna()

    return row

def addmov(row):

    row.append(row[-1]/row[-2])

    #row['move_score']=row['chgratio']/row['std_1hr']

    #row=row.dropna()

    return row

def get_movelist(df,pct):

    df['std_1hr']=df[['symbols','timestamp']].apply(lambda x:hourlystd(x.symbols,pd.to_datetime(x.timestamp)),axis=1)

    df=df.dropna()

    df['chgratio']=df[['symbols','timestamp']].apply(lambda x:chgratio(x.symbols,x.timestamp),axis=1)

    df['move_score']=df['chgratio']/df['std_1hr']

    df=df.dropna()

    movers=df.sort_values('move_score',ascending=False).iloc[:int(pct*len(df))]

    return movers



def get_movelist_wspark(df,pct):

    cols=list(df.columns)+['std_1hr','chgratio','move_score']

    #rdd=sc.parallelize(range(int(np.floor(len(df)/100))+int(not (len(df)%100==0))))

    rdd=sc.parallelize(range(len(df)))

    #def chunky(x):

    #    return df.iloc[x*100:min(len(df),(x+1)*100)]

    #rows=rdd.map(lambda x:chunky(x))

    #return rows.collect()

    rows=rdd.map(lambda x:list(df.iloc[x]))

    rows_w_std=rows.map(lambda x:addstd(x))

    rows_w_chg=rows_w_std.map(lambda x:addchg(x))

    rows_w_chg=rows_w_chg.filter(lambda x:type(x[-1])!=tuple)

    rows_w_mov=rows_w_chg.map(lambda x:addmov(x))

    #return rows_w_mov.take(100)

    data=rows_w_mov.collect()

    data=pd.DataFrame(data=data,columns=cols)

    return data.head().dropna().sort_values('move_score',ascending=False).iloc[:int(pct*len(df))]
#get_movelist_wspark(in_sess.iloc[:1000,],.5)
in_session_movers=get_movelist_wspark(in_sess,.1)
ingrams=movegrams(in_sess,in_session_movers)
ingrams
in_session_movers.head()