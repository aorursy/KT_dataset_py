import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import collections
import warnings
from kaggle.competitions import twosigmanews
from datetime import datetime
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
#getting environment for accessing full data
env = twosigmanews.make_env()
market_train_full_df = env.get_training_data()[0]
sample_market_df = pd.read_csv("../input/marketdata_sample.csv")
sample_news_df = pd.read_csv("../input/news_sample.csv")
"market_train_full_df dimention:{}".format(market_train_full_df.shape)
market_train_full_df.head(5)
market_train_full_df.columns
market_train_full_df.time.describe()
# pd.to_datetime(market_train_full_df.time).apply(lambda x: pd.Series({"daily":datetime.date(x)}))["daily"].value_counts()
fig,axes = plt.subplots(1,1,figsize=(15,10))
axes.set_title("Time Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(market_train_full_df.time.dt.date.value_counts().sort_index().index, market_train_full_df.time.dt.date.value_counts().sort_index().values)
market_train_full_df.assetCode.describe()
market_train_full_df.assetCode.value_counts().describe()
list(market_train_full_df.assetCode)[:5]
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Daily assetCodes Violin")
axes.set_ylabel("Repetition")
axes.violinplot(list(market_train_full_df.assetCode.value_counts().values),showmeans=False,showmedians=True)
market_train_full_df.assetName.describe()
list(market_train_full_df.assetName)[:10]
from wordcloud import WordCloud
# Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(" ".join(market_train_full_df.assetName))
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(20,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
market_train_full_df.universe.value_counts()
univers_df_dict = dict(collections.Counter(list(market_train_full_df.universe)))
percent_univers_df_dict = {k: v / total for total in (sum(univers_df_dict.values()),) for k, v in univers_df_dict.items()}
explode=(0,0.1)
labels ='notUniverse','isUniverse'
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Univere Status")
ax.pie(list(percent_univers_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

market_train_full_df.universe.value_counts()
market_train_full_df.volume.describe()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Volume Violin")
axes.set_ylabel("Volume")
axes.violinplot(list(market_train_full_df["volume"].values),showmeans=False,showmedians=True)
fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Volume")
axes.set_ylabel("volume")
axes.set_xlabel("records")
axes.plot(market_train_full_df["volume"])
market_train_full_df.close.describe()
fig, axes = plt.subplots(figsize=(20,10))

axes.set_title("Close Price")
axes.set_ylabel("close price")
axes.set_xlabel("records")
axes.plot(market_train_full_df["close"])
fig, axes = plt.subplots(figsize=(20,10))
axes.set_title("Open Price")
axes.set_ylabel("open price")
axes.set_xlabel("records")
axes.plot(market_train_full_df["open"])
market_returns_df = pd.concat(
    [
        market_train_full_df["returnsClosePrevRaw1"].describe(),
        market_train_full_df["returnsOpenPrevRaw1"].describe(),
        market_train_full_df["returnsClosePrevMktres1"].describe(),
        market_train_full_df["returnsOpenPrevMktres1"].describe(),
        market_train_full_df["returnsClosePrevRaw10"].describe(),
        market_train_full_df["returnsOpenPrevRaw10"].describe(),
        market_train_full_df["returnsClosePrevMktres10"].describe(),
        market_train_full_df["returnsOpenPrevMktres10"].describe(),
        market_train_full_df["returnsOpenNextMktres10"].describe()
        ],
        axis=1
    )
market_returns_df
market_returns_df.drop(["returnsClosePrevMktres1","returnsOpenPrevMktres1","returnsClosePrevMktres10","returnsOpenPrevMktres10"],axis=1,inplace=True)
market_returns_df
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
axes.set_title("Box Plot")
axes.set_ylabel("Returns")
market_train_full_df.boxplot(column=['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', "returnsClosePrevRaw10","returnsOpenPrevRaw10","returnsOpenNextMktres10"])    
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8),sharey=True)
axes[0].set_title("One-day difference in close and open returns")
axes[0].set_ylabel("difference")
axes[0].violinplot(list((market_train_full_df["returnsClosePrevRaw1"] - market_train_full_df["returnsOpenPrevRaw1"]).values),showmeans=False,showmedians=True,widths=0.9, showextrema=True)
axes[1].set_title("10-day difference in close and open returns")
axes[1].set_ylabel("difference")
axes[1].violinplot(list((market_train_full_df["returnsClosePrevRaw10"] - market_train_full_df["returnsOpenPrevRaw10"]).values),showmeans=False,showmedians=True,widths=0.9, showextrema=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9),sharey=True)
axes.set_title("returnsOpenNextMktres10 violon")
axes.set_ylabel("")
axes.violinplot(list((market_train_full_df["returnsOpenNextMktres10"]).values),showmeans=False,showmedians=True,widths=0.9, showextrema=True)
del axes
del market_train_full_df
news_train_full_df = env.get_training_data()[1]
news_train_full_df.head()
news_train_full_df.columns
news_train_full_df.time.describe()
fig,axes = plt.subplots(1,1,figsize=(20,10))
axes.set_title("Time Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(news_train_full_df.time.dt.date.value_counts().sort_index().index, news_train_full_df.time.dt.date.value_counts().sort_index().values)
news_train_full_df.sourceTimestamp.describe()
fig,axes = plt.subplots(1,1,figsize=(20,10))
axes.set_title("sourceTimestamp Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(news_train_full_df.sourceTimestamp.dt.date.value_counts().sort_index().index, news_train_full_df.sourceTimestamp.dt.date.value_counts().sort_index().values)
news_train_full_df.firstCreated.describe()
fig,axes = plt.subplots(1,1,figsize=(20,10))
axes.set_title("firstCreated Distro")
axes.set_ylabel("# of records")
axes.set_xlabel("date")
axes.plot(news_train_full_df.firstCreated.dt.date.value_counts().sort_index().index, news_train_full_df.firstCreated.dt.date.value_counts().sort_index().values)
news_train_full_df.sourceId.describe()
news_train_full_df.sourceId.value_counts().describe()
news_train_full_df.headline.describe()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 

# Create a list of word
# text=("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")
 
# Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(( " ".join(list(news_train_full_df.head(200000).headline))))
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
news_train_full_df.urgency.value_counts()
urgency_df_dict = dict(collections.Counter(list(news_train_full_df.urgency)))
percent_urgency_df_dict = {k: v / total for total in (sum(urgency_df_dict.values()),) for k, v in urgency_df_dict.items()}
explode=(0,0.1,0.1)
labels ="article","alert", "unknown"
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Urgency Status")
ax.pie(list(percent_urgency_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
news_train_full_df.takeSequence.value_counts().head(20)
fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.set_xlabel("name")
ax.set_ylabel("#")
news_train_full_df.provider.value_counts().plot(kind="bar",legend="provider",color="tan")
news_train_full_df.head(5).subjects
from collections import Counter
tmp_list = []
for i in news_train_full_df.head(200000).subjects:
    tmp_list += i.replace("{","").replace("}","").replace(" ","").split(",")
# Counter(tmp_list)
# fig,ax = plt.subplots(1,1,figsize=(30,10))
# ax.set_xticklabels(dict(Counter(tmp_list)).keys(),rotation=90)
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 10}
# plt.rc('font', **font)
# ax.bar(dict(Counter(tmp_list)).keys(),dict(Counter(tmp_list)).values())
text =" ".join(tmp_list).replace("'","")
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
news_train_full_df.head(5).audiences
from collections import Counter
tmp_list = []
for i in news_train_full_df.head(200000).audiences:
    tmp_list += i.replace("{","").replace("}","").replace(" ","").split(",")
# Counter(tmp_list)

# fig,ax = plt.subplots(1,1,figsize=(30,10))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
# plt.rc('font', **font)
# ax.set_xticklabels(dict(Counter(tmp_list)).keys(),rotation=90)
# ax.bar(dict(Counter(tmp_list)).keys(),dict(Counter(tmp_list)).values())
text =" ".join(tmp_list).replace("'","")
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
pd.concat([news_train_full_df.bodySize.describe(),news_train_full_df.companyCount.describe(),news_train_full_df.sentenceCount.describe(),news_train_full_df.wordCount.describe()],axis=1)
pd.concat([news_train_full_df.sentimentNegative.describe(),news_train_full_df.sentimentNeutral.describe(),news_train_full_df.sentimentPositive.describe()],axis=1)
fig , axes = plt.subplots(1,1,figsize=(20,8))
news_train_full_df.sentimentNegative.head(100).plot(kind="bar",legend="Negative",colormap="brg")
news_train_full_df.sentimentPositive.head(100).plot(colormap="Set2",linewidth=2,legend="Postive")
news_train_full_df.sentimentNeutral.head(100).plot(colormap="RdGy",linewidth=0.8,linestyle='dashed',legend="Neutral")
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
axes.set_xticklabels(news_train_full_df.head(100).index,rotation=90)
legend = axes.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.rc('font', **font)
pd.concat([news_train_full_df.noveltyCount12H.describe(),news_train_full_df.noveltyCount24H.describe(), news_train_full_df.noveltyCount3D.describe(),
          news_train_full_df.noveltyCount5D.describe(),news_train_full_df.noveltyCount7D.describe()],axis=1)
fig,axes = plt.subplots(3,2,figsize=(10,15))

noveltyCount12H_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount12H)))
percent_noveltyCount12H_dict = {k: v / total for total in (sum(noveltyCount12H_dict.values()),) for k, v in noveltyCount12H_dict.items()}
sizes = list(percent_noveltyCount12H_dict.values())
axes[0][0].set_title("noveltyCount12H",loc="left")
axes[0][0].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)


noveltyCount24H_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount24H)))
percent_noveltyCount24H_dict = {k: v / total for total in (sum(noveltyCount24H_dict.values()),) for k, v in noveltyCount24H_dict.items()}
sizes = list(percent_noveltyCount24H_dict.values())
axes[0][1].set_title("noveltyCount24H",loc="left")
axes[0][1].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)

noveltyCount3D_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount3D)))
percent_noveltyCount3D_dict = {k: v / total for total in (sum(noveltyCount3D_dict.values()),) for k, v in noveltyCount3D_dict.items()}
sizes = list(percent_noveltyCount3D_dict.values())
axes[1][0].set_title("noveltyCount3D",loc="left")
axes[1][0].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)


noveltyCount5D_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount5D)))
percent_noveltyCount5D_dict = {k: v / total for total in (sum(noveltyCount5D_dict.values()),) for k, v in noveltyCount5D_dict.items()}
sizes = list(percent_noveltyCount5D_dict.values())
axes[1][1].set_title("noveltyCount5D",loc="left")
axes[1][1].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)

noveltyCount7D_dict = dict(collections.Counter(list(news_train_full_df.noveltyCount7D)))
percent_noveltyCount7D_dict = {k: v / total for total in (sum(noveltyCount7D_dict.values()),) for k, v in noveltyCount7D_dict.items()}
sizes = list(percent_noveltyCount7D_dict.values())
axes[2][0].set_title("noveltyCount7D",loc="left")
axes[2][0].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)


overal_dict = pd.concat([news_train_full_df.noveltyCount12H,news_train_full_df.noveltyCount24H, news_train_full_df.noveltyCount3D,
          news_train_full_df.noveltyCount5D,news_train_full_df.noveltyCount7D],axis=0)
noveltyOveral_dict = dict(collections.Counter(list(overal_dict)))
percent_overal_dict = {k: v / total for total in (sum(noveltyOveral_dict.values()),) for k, v in noveltyOveral_dict.items()}
sizes = list(percent_overal_dict.values())
axes[2][1].set_title("overalNovelty",loc="left")
axes[2][1].pie(sizes,  autopct='%1.1f%%',shadow=False, startangle=90)
print()
pd.concat([news_train_full_df.volumeCounts12H.describe(),news_train_full_df.volumeCounts24H.describe(), news_train_full_df.volumeCounts3D.describe(),
          news_train_full_df.volumeCounts5D.describe(),news_train_full_df.volumeCounts7D.describe()],axis=1)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15),squeeze=False)
axes[0][0].set_title("volumeCounts12H")
axes[0][0].violinplot(list(news_train_full_df["volumeCounts12H"].values))

axes[0][1].set_title("volumeCounts24H")
axes[0][1].violinplot(list(news_train_full_df["volumeCounts24H"].values))

axes[1][0].set_title("volumeCounts3D")
axes[1][0].violinplot(list(news_train_full_df["volumeCounts3D"].values))

axes[1][1].set_title("volumeCounts5D")
axes[1][1].violinplot(list(news_train_full_df["volumeCounts5D"].values))

axes[2][0].set_title("volumeCounts7D")
axes[2][0].violinplot(list(news_train_full_df["volumeCounts7D"].values))

fig.delaxes(axes[2][1])
text=""
text =" ".join(list(news_train_full_df.headlineTag))
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
news_train_full_df.marketCommentary.value_counts()
market_commentary_df_dict = dict(collections.Counter(list(news_train_full_df.marketCommentary)))
percent_commentary_df_dict = {k: v / total for total in (sum(market_commentary_df_dict.values()),) for k, v in market_commentary_df_dict.items()}
explode=(0,0.1)
labels ='False','True'
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Representing Genral Market Conditions")
ax.pie(list(percent_commentary_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
news_train_full_df.sentimentWordCount.describe()
fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title("Hist(log sentimentWordCount)")
ax.set_xlabel("Log(sentimentWordCount)")
np.log10(news_train_full_df.sentimentWordCount).hist(ax=ax,)
news_train_full_df.assetName.head(5)
tmp_list = []
for i in news_train_full_df.head(200000).assetName:
    tmp_list += i.replace("{","").replace("}","").replace(" ","").split(",")
text =" ".join(tmp_list).replace("'","")
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
news_train_full_df.assetCodes.head(10)
news_train_full_df.sentimentClass.value_counts()
news_train_full_df.sentimentClass.value_counts()
sentiment_df_dict = dict(collections.Counter(list(news_train_full_df.sentimentClass)))
percent_univers_df_dict = {k: v / total for total in (sum(sentiment_df_dict.values()),) for k, v in sentiment_df_dict.items()}
explode=(0.0,0.025,0.05)
labels ='1','0',"-1"
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("sentimentClass")
ax.pie(list(percent_univers_df_dict.values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
news_train_full_df.relevance.describe()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
axes.set_title("Volume Violin")
axes.set_ylabel("Volume")
axes.violinplot(list(news_train_full_df["relevance"].values),showmeans=False,showmedians=True)
news_train_full_df.firstMentionSentence.describe()