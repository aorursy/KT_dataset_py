#internet to be turned on

#import nltk

#nltk.download('punkt')

#nltk.download('stopwords')

#nltk.download('vader_lexicon')

#pip install emoji

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy  as np

import nltk

from datetime import datetime, timedelta

import string

import emoji

import re

import seaborn as sns

sns.set(style='darkgrid')

from wordcloud import WordCloud

from matplotlib import pyplot as plt

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer
data = pd.read_csv("../input/newnlp/newnlp.csv")

data = data[data['Name']!='Chayadeep']

data['Gender'][data['Gender']=='f'] = 0

data['Gender'][data['Gender']=='m'] = 1  

data['Gender'] = data['Gender'].astype(int)



dname = data.groupby(['Name'])['DaysTexted','DaysBeen','ActiveDays','Gender'].mean()

dname['Consistency'] = 100*dname['DaysTexted']/dname['DaysBeen']

dname['TimesTexted'] = data.groupby(['Name'])['Name'].count()

dname['Frequentness'] = 10*dname['TimesTexted']/dname['DaysBeen']

dname['Agressiveness'] = 10*dname['TimesTexted']/dname['ActiveDays']

#dname.sort_values(by='Consistency',ascending=False)

#dname.sort_values(by='TimesTexted',ascending=False)

print('😎 '+'Aman Jain')
dname
print('Top 10 Popular folks \n(TimesTexted)')

dname.sort_values(by='TimesTexted',ascending=False).head(10).iloc[:,5:6]
print('Top 10 Consistent folks \n(DaysTexted/DaysBeen)')

dname.sort_values(by='Consistency',ascending=False).head(10).iloc[:,4:5]
print('Top 10 most Frequently texting folks \n(TimesTexted/DaysBeen)')

dname.sort_values(by='Frequentness',ascending=False).head(10).iloc[:,6:7]
names = pd.DataFrame(data.groupby(['Name']).size(),columns={'TextTimes'}).reset_index()

names['avgWordspertext'] = 0

names['minWordspertext'] = 0

names['maxWordspertext'] = 0

names['evocab'] = 0

names['totalemojis'] = 0

names['top5emojis'] = 0

names['vocab'] = 0

names['top5words'] = 0

def nltk_sentiment(sentence):

      nltk_sentiment = SentimentIntensityAnalyzer()

      score = nltk_sentiment.polarity_scores(sentence)

      return score

def getResult(pos, neu, neg):

    if (pos > neu and pos > neg):

        return ("Positive")

    elif (neg > neu and neg > pos):

        return ("Negative")

    else:

        return('Neutral')



dfHFreqs = pd.DataFrame(data.groupby(['Hour'])['Hour'].count())

dfHFreqs.columns = ['Group']

dfHFreqs = dfHFreqs.reset_index()

dfHFreqs.columns = ['Hour','Group']

s = dfHFreqs['Group'].sum()

dfHFreqs['Group'] = 100*dfHFreqs['Group']/s



names['Pos'] = 0

names['Neu'] = 0

names['Neg'] = 0

names['avgTime'] = 0

stop = stopwords.words('english')

vStopWords = ["thats","dont","also","like","https","from","all","also","and","any","are","but","can","cant","cry","due","etc","few","for","get","had","has","hasnt","have","her","here","hers","herself","him","himself","his","how","inc","into","its","ltd","may","nor","not","now","off","once","one","only","onto","our","ours","out","over","own","part","per","put","see","seem","she","than","that","the","their","them","then","thence","there","these","they","this","those","though","thus","too","top","upon","very","via","was","were","what","when","which","while","who","whoever","whom","whose","why","will","with","within","without","would","yet","you","your","yours","the"]

stop = stop + vStopWords

ylabel='% of total time spent'

xlabel='Time clock in 24 hours'
for name in names['Name']:

  data1 = data[data['Name']==name]

  dstr = ' '.join(data1['Text'])

  dlist = data1['Text'].to_list()

  

  #names['avgTime'][names['Name']==name] = data1['Time'].mean().strftime("%I:%M %p")



  L1 = []

  for l in dlist:

    L1.append(len(l.split()))

  names['avgWordspertext'][names['Name']==name] = np.mean(L1)

  names['minWordspertext'][names['Name']==name] = np.min(L1)

  names['maxWordspertext'][names['Name']==name] = np.max(L1)

  

  LE = []

  LE = [c for c in dstr if c in emoji.UNICODE_EMOJI]

  dfE = pd.DataFrame({'Emoji':LE})

  dfEFreqs = pd.DataFrame(dfE.groupby(['Emoji'])['Emoji'].count())

  dfEFreqs.columns = ['Freq']

  dfEFreqs = dfEFreqs.reset_index()

  dfEFreqs.columns = ['Emoji','Freq']

  names['evocab'][names['Name']==name] = len(dfEFreqs)

  names['totalemojis'][names['Name']==name] = dfEFreqs['Freq'].sum()

  dfEFreqs = dfEFreqs.sort_values('Freq',ascending=False)

  names['top5emojis'][names['Name']==name] = ' '.join(dfEFreqs['Emoji'][0:5])



  demoji = dstr.encode('ascii', 'ignore').decode('ascii')

  demoji = re.sub(r'[`!?~@#$%^&*()_+-=<>,.:;]', '', demoji)

  demoji = re.sub(r'[–]', '', demoji)

  demoji = re.sub(r'[\[\]\(\)\{\}]', '', demoji)

  demoji = re.sub(r'[\t\"\'\/\\]', '', demoji)

  lstAllWords = demoji.split()



  lstTmpWords=[]

  for strWord in lstAllWords:

      if len(strWord)>3:

          lstTmpWords.append(strWord)

  lstAllWords = lstTmpWords

  del lstTmpWords



  for i in range(0,len(lstAllWords)):

      lstAllWords[i] = str.lower(lstAllWords[i])



  dfWords = pd.DataFrame({'Words':lstAllWords})

  dfWords = dfWords[-dfWords['Words'].isin(stop)]

  dfWords = dfWords[-dfWords['Words'].isin(emoji.UNICODE_EMOJI.keys())]



  dfFreqs = pd.DataFrame(dfWords.groupby(['Words'])['Words'].count())

  dfFreqs.columns = ['Freq']

  dfFreqs = dfFreqs.reset_index()

  dfFreqs.columns = ['Word','Freq']

  names['vocab'][names['Name']==name] = len(dfFreqs)

  dfFreqs = dfFreqs.sort_values('Freq',ascending=False)

  names['top5words'][names['Name']==name] = ' '.join(dfFreqs['Word'][0:5])



  print('😎 '+name)

  d = {}

  for a, x in dfFreqs[0:10].values:

      d[a] = x 

  wordcloud = WordCloud(background_color="white")

  wordcloud.generate_from_frequencies(frequencies=d)

  plt.figure()

  plt.imshow(wordcloud, interpolation="bilinear")

  plt.axis("off")

  plt.show()



  lstLines = sent_tokenize(dstr)

  lstLines = [t.lower() for t in lstLines]

  lstLines = [t.translate(str.maketrans('','',string.punctuation)) for t in lstLines]

  saResults = [nltk_sentiment(t) for t in lstLines]

  # create dataframe

  df = pd.DataFrame(lstLines, columns=['Lines'])

  df['Pos']=[t['pos'] for t in saResults]

  df['Neu']=[t['neu'] for t in saResults]

  df['Neg']=[t['neg'] for t in saResults]

  #df['Result']= [getResult(t['pos'],t['neu'],t['neg']) for t in saResults]

  names['Pos'][names['Name']==name] = df['Pos'].mean()

  names['Neu'][names['Name']==name] = df['Neu'].mean()

  names['Neg'][names['Name']==name] = df['Neg'].mean()  

    

names = names.set_index('Name')
print('Top 10 Emoji-using folks')

names.sort_values(by='totalemojis',ascending=False).head(10).iloc[:,5:6]
print('Top 10 follks with hishest Emoji vocab')

names.sort_values(by='evocab',ascending=False).head(10).iloc[:,4:5]
print('Top 10 folks who write LONG texts')

pcs = names[names['TextTimes']>40]

pcs.sort_values(by='avgWordspertext',ascending=False).head(10).iloc[:,1:2]
print('Top 10 folks who write SHORT texts')

pcs = names[names['TextTimes']>40]

pcs.sort_values(by='avgWordspertext',ascending=True).head(10).iloc[:,1:2]
print('Top 10 Positively texting folks')

pcs = names[names['TextTimes']>30]

pcs.sort_values(by='Pos',ascending=False).head(10).iloc[:,9:10]
print('Top 10 Neutraly texting folks')

pcs = names[names['TextTimes']>30]

pcs.sort_values(by='Neu',ascending=False).head(10).iloc[:,10:11]
print('Top 10 folks with diverse vocab \n(maximum different words)')

names.sort_values(by='vocab',ascending=False).head(10).iloc[:,7:8]
print('Top 5 emojis most used \nby Top 10 popular folks')

names.sort_values(by='TextTimes',ascending=False).head(10).iloc[:,6:7]
names['Frequentness'] = dname['Frequentness']

print('Top 5 most used words by Top 10 frequently texting folks')

names.sort_values(by='Frequentness',ascending=False).head(10).iloc[:,8:9]
names['Gender'] = dname['Gender']
print('Number of Men and Women in the group')

pcs = names.groupby(['Gender']).size()

print('Women:'+str(pcs[0])+'\nMen:'+str(pcs[1]))
print('Top 10 Popular Women (TimesTexted)')

pcs = dname[dname['Gender']==0]

pcs.sort_values(by='TimesTexted',ascending=False).head(10).iloc[:,5:6]
print('Top 10 Consistent Women \n(DaysTexted/DaysBeen)')

pcs = dname[dname['Gender']==0]

pcs.sort_values(by='Consistency',ascending=False).head(10).iloc[:,4:5]

print('Top 10 most Frequently texting Women\n (TimesTexted/DaysBeen)')

pcs = dname[dname['Gender']==0]

pcs.sort_values(by='Frequentness',ascending=False).head(10).iloc[:,6:7]

print('Top 10 most Emoji-using Women')

pcs = names[names['Gender']==0]

pcs.sort_values(by='totalemojis',ascending=False).head(10).iloc[:,5:6]
print('Top 10 Women with hishest Emoji vocab')

pcs = names[names['Gender']==0]

pcs.sort_values(by='evocab',ascending=False).head(10).iloc[:,4:5]

print('Top 10 Women who write LONG texts')

pcs = names[names['Gender']==0]

pcs = pcs[pcs['TextTimes']>40]

pcs.sort_values(by='avgWordspertext',ascending=False).head(10).iloc[:,1:2]

print('Top 10 Women who write SHORT texts')

pcs = names[names['Gender']==0]

pcs = pcs[pcs['TextTimes']>40]

pcs.sort_values(by='avgWordspertext',ascending=True).head(10).iloc[:,1:2]

print('Top 10 Positively texting women')

pcs = names[names['Gender']==0]

pcs = pcs[pcs['TextTimes']>30]

pcs.sort_values(by='Pos',ascending=False).head(10).iloc[:,9:10]
print('Top 10 Neutraly texting Women')

pcs = names[names['Gender']==0]

pcs = pcs[pcs['TextTimes']>30]

pcs.sort_values(by='Neu',ascending=False).head(10).iloc[:,10:11]

print('Top 10 Women with diverse vocab \n(maximum different words)')

pcs = names[names['Gender']==0]

pcs.sort_values(by='vocab',ascending=False).head(10).iloc[:,7:8]
print('Top 5 emojis most used by Top 10 popular Women')

pcs = names[names['Gender']==0]

pcs.sort_values(by='TextTimes',ascending=False).head(10).iloc[:,6:7]
print('Top 5 most used words by Top 10 frequently texting Women')

pcs = names[names['Gender']==0]

pcs.sort_values(by='Frequentness',ascending=False).head(10).iloc[:,8:9]
for name in names.index:

    data1 = data[data['Name']==name]

    dstr = ' '.join(data1['Text'])

    dlist = data1['Text'].to_list()



    dfHPFreqs = pd.DataFrame(data[data['Name']==name].groupby(['Hour'])['Hour'].count())

    dfHPFreqs.columns = [name]

    dfHPFreqs = dfHPFreqs.reset_index()

    dfHPFreqs.columns = ['Hour',name]

    s = dfHPFreqs[name].sum()

    dfHPFreqs[name] = 100*dfHPFreqs[name]/s

    dfC = pd.merge(dfHFreqs,dfHPFreqs,how='left')

    dfC = dfC.fillna(0)

    title = 'Texting Pattern: 😎'+name

    plt.figure()

    ax = dfC.iloc[:, 2].plot(legend=True,figsize=(12,6),title=title,color='r')

    dfC.iloc[:,1].plot(legend=True,figsize=(12,6),color='y')

    ax.autoscale(axis='x',tight=True)

    ax.set(xlabel=xlabel, ylabel=ylabel);
dstr = ' '.join(data['Text'])

dlist = data['Text'].to_list()

L1 = []

for l in dlist:

  L1.append(len(l.split()))

LE = []

LE = [c for c in dstr if c in emoji.UNICODE_EMOJI]

dfE = pd.DataFrame({'Emoji':LE})

dfEFreqs = pd.DataFrame(dfE.groupby(['Emoji'])['Emoji'].count())

dfEFreqs.columns = ['Freq']

dfEFreqs = dfEFreqs.reset_index()

dfEFreqs.columns = ['Emoji','Freq']

evocab = len(dfEFreqs)

totalemojis = dfEFreqs['Freq'].sum()

dfEFreqs = dfEFreqs.sort_values('Freq',ascending=False)

top5emojis =  ' '.join(dfEFreqs['Emoji'][0:5])
print('Group Emoji Vocab: '+str(evocab)+'\n(different emojis used)') 
print('Total Emojis used in the group: '+str(totalemojis)) 
print('Top 10 emojis used in the group')

dfEFreqs.set_index('Emoji').head(10)
demoji = dstr.encode('ascii', 'ignore').decode('ascii')

demoji = re.sub(r'[`!?~@#$%^&*()_+-=<>,.:;]', '', demoji)

demoji = re.sub(r'[–]', '', demoji)

demoji = re.sub(r'[\[\]\(\)\{\}]', '', demoji)

demoji = re.sub(r'[\t\"\'\/\\]', '', demoji)

lstAllWords = demoji.split()

totalwords = len(lstAllWords)

lstTmpWords=[]

for strWord in lstAllWords:

    if len(strWord)>3:

        lstTmpWords.append(strWord)

lstAllWords = lstTmpWords

del lstTmpWords



for i in range(0,len(lstAllWords)):

    lstAllWords[i] = str.lower(lstAllWords[i])



dfWords = pd.DataFrame({'Words':lstAllWords})

dfWords = dfWords[-dfWords['Words'].isin(stop)]

dfWords = dfWords[-dfWords['Words'].isin(emoji.UNICODE_EMOJI.keys())]



dfFreqs = pd.DataFrame(dfWords.groupby(['Words'])['Words'].count())

dfFreqs.columns = ['Freq']

dfFreqs = dfFreqs.reset_index()

dfFreqs.columns = ['Word','Freq']

vocab = len(dfFreqs)

#totalwords = dfFreqs['Freq'].sum()

dfFreqs = dfFreqs.sort_values('Freq',ascending=False)

#names['top5words'][names['Name']==name] = ' '.join(dfFreqs['Word'][0:10])
print('Group Vocab: '+str(vocab)+'\n(different words used)') 
print('Total Words used in the group: '+str(totalwords)) 
#100*evocab/vocab

print(str(int(np.round(100*totalemojis/totalwords)))+'% of words used were emojis')
print('Top 10 words used in the group')

dfFreqs.set_index('Word').head(10)
d = {}

for a, x in dfFreqs[0:10].values:

    d[a] = x 

wordcloud = WordCloud(background_color="white")

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

lstLines = sent_tokenize(dstr)

lstLines = [t.lower() for t in lstLines]

lstLines = [t.translate(str.maketrans('','',string.punctuation)) for t in lstLines]

saResults = [nltk_sentiment(t) for t in lstLines]

# create dataframe

df = pd.DataFrame(lstLines, columns=['Lines'])

df['Pos']=[t['pos'] for t in saResults]

df['Neu']=[t['neu'] for t in saResults]

df['Neg']=[t['neg'] for t in saResults]

#df['Result']= [getResult(t['pos'],t['neu'],t['neg']) for t in saResults]

pos = df['Pos'].mean()

neu = df['Neu'].mean()

neg = df['Neg'].mean()  
print('Sentiments of the group')

print(str(round(pos*100,2))+'% Positive')

print(str(round(neu*100,2))+'% Neutral')

print(str(round(neg*100,2))+'% Negative')
dfHFreqs = dfHFreqs.set_index('Hour')

title='Group Texting Pattern'

plt.figure()

ax = dfHFreqs.plot(legend=True,figsize=(12,6),title=title,color='r')

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel);