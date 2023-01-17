from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
%matplotlib inline



df = pd.read_csv('../input/17072018.csv', encoding='utf-8-sig')
print(len(df))
df['social_media'].value_counts()
df['country'].value_counts()[:15]
unique_users = df.groupby('user')
print(len(unique_users))
df['published_on'] = df['published_on'].map(lambda x:pd.to_datetime(str(x).split()[0]))
tweets_by_day = df.groupby('published_on').count().reset_index()
tweets_by_day = tweets_by_day[tweets_by_day['published_on'] > '2018-05-19']
len(tweets_by_day)
tweets_by_day = tweets_by_day.sort_values(by='published_on')
x = tweets_by_day['published_on']
y = tweets_by_day['text']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
top_10_max_tweets_days = tweets_by_day.sort_values(by='user').tail(10)
x = top_10_max_tweets_days['published_on']
y = top_10_max_tweets_days['text']
plt.xlabel('Data')
plt.ylabel('Número de Ocorrências')
plt.title('Maior Ocorrências em 1 dia')
plt.xticks(range(10),x,rotation=45)
plt.bar(range(10), y, label='Maior ocorrências em 1 dia')
plt.show

tweeteries = df.groupby(['user']).count().reset_index()
tweeteries = tweeteries.sort_values(by='text').tail(10)
x = tweeteries['user']
y = tweeteries['text']
plt.xlabel('Usuário')
plt.ylabel("Número de Tweets")
plt.title("Maior número de posts por usuário")
plt.xticks(range(10),x,rotation=45)
plt.bar(range(10), y, label='Maior posts por usuário')
plt.show
most_followed_users = df.drop_duplicates('user', keep='last')
most_followed_users_top_10 = most_followed_users.sort_values(by='reach').tail(10)
x = most_followed_users_top_10['user']
y = most_followed_users_top_10['reach']
plt.xlabel('Usuário')
plt.ylabel('Seguidores')
plt.title('Usuários mais seguidos')
plt.xticks(range(10), x, rotation=60)
plt.bar(range(10), y, label='Usuários mais seguidos')
plt.show()
tweets = df['text'].astype('str').dropna()
tweets = ''.join(tweets)
tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)
hashtags_list = re.findall('#[a-z]+', tweets)
fd = nltk.FreqDist(hashtags_list)
for x in fd.most_common(200): 
    print(x)
fd.plot(10, cumulative=False)
df_plasticfreejuly = df[df['text'].str.contains('#plasticfreejuly',na=False)]
tweets = df_plasticfreejuly["text"].astype("str").dropna()
tweets =''.join(tweets)
tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)
tweets = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',tweets)
tweets = re.sub(r'[\W]+',' ',tweets)
tweets_plasticfreejuly = tweets
tokens = word_tokenize(tweets)
tokens = [w.lower() for w in tokens if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
tokens = [w for w in tokens if w.lower() not in stopwords]
tokens = [w for w in tokens if len(w) > 3]
fd = nltk.FreqDist(tokens)
print('25 palavras mais comum para #plasticfreejuly:')
print(fd.most_common(25))
df_beatplasticpollution = df[df['text'].str.contains('#beatplasticpollution',na=False)]
tweets = df_beatplasticpollution["text"].astype("str").dropna()
tweets =''.join(tweets)
tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)
tweets = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',tweets)
tweets = re.sub(r'[\W]+',' ',tweets)
tweets_beatplasticpollution = tweets
tokens = word_tokenize(tweets)
tokens = [w.lower() for w in tokens if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
tokens = [w for w in tokens if w.lower() not in stopwords]
tokens = [w for w in tokens if len(w) > 3]
fd = nltk.FreqDist(tokens)
print('25 palavras mais comum para #beatplasticpollution:')
print(fd.most_common(25))
df_plasticfreejuly = df_plasticfreejuly.groupby('published_on').count().reset_index()
df_plasticfreejuly = df_plasticfreejuly[df_plasticfreejuly['published_on'] > '2018-05-19']
df_plasticfreejuly = df_plasticfreejuly.sort_values(by='published_on')
x = df_plasticfreejuly['published_on']
y = df_plasticfreejuly['text']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - #Plastic Free July')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
df_beatplasticpollution = df_beatplasticpollution.groupby('published_on').count().reset_index()
df_beatplasticpollution = df_beatplasticpollution[df_beatplasticpollution['published_on'] > '2018-05-19']
df_beatplasticpollution = df_beatplasticpollution.sort_values(by='published_on')
x = df_beatplasticpollution['published_on']
y = df_beatplasticpollution['text']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - #Beat Plastic Pollution')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
df_bagfree = df[df['text'].str.contains('#plasticbagfreeday',na=False)]
df_bagfree = df_bagfree.groupby('published_on').count().reset_index()
df_bagfree = df_bagfree[df_bagfree['published_on'] > '2018-05-19']
df_bagfree = df_bagfree.sort_values(by='published_on')
x = df_bagfree['published_on']
y = df_bagfree['text']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - #Plastic Bag Free Day')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
df_stopsucking = df[df['text'].str.contains('#stopsucking',na=False)]
df_stopsucking = df_stopsucking.groupby('published_on').count().reset_index()
df_stopsucking = df_stopsucking[df_stopsucking['published_on'] > '2018-05-19']
df_stopsucking = df_stopsucking.sort_values(by='published_on')
x = df_stopsucking['published_on']
y = df_stopsucking['text']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - #Stop Sucking')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
rts = df['text'].astype('str').dropna()
rts = ''.join(rts)
rts = re.sub(r'[^\x00-\x7F]+',' ', rts)
rts_list = re.findall('(@\S+)', rts)
fd = nltk.FreqDist(rts_list)
print(fd.most_common(20))
fd.plot(20, cumulative=False)

rts = df['text'].value_counts()[:20].to_frame()
rts.reset_index(level=0, inplace=True)
pd.set_option('display.max_colwidth', -1)
rts.style.set_properties(**{'text-align': 'left'})
top_posts = df.sort_values(by="reach", ascending=False)
pd.set_option('display.max_colwidth', -1)
top_posts[['user','text','reach']][:20].style.set_properties(**{'text-align': 'left'})
pfreejuly = df[df['text'].str.contains('#plasticfreejuly',na=False)]
pfreejuly = pfreejuly.sort_values(by="reach", ascending=False)
pd.set_option('display.max_colwidth', -1)
pfreejuly[['user','text','reach']][:20].style.set_properties(**{'text-align': 'left'})
ssucking = df[df['text'].str.contains('#stopsucking',na=False)]
ssucking = ssucking.sort_values(by="reach", ascending=False)
pd.set_option('display.max_colwidth', -1)
ssucking[['user','text','reach']][:20].style.set_properties(**{'text-align': 'left'})
empresas = ["Starbucks","McDonalds","Adama","AkzoNobel","Ambev","Arysta Lifescience","Basf","Bayer","Cibrafértil","DuPont",
"Duratex","FMC","General Eletric","Ihara","LyondellBasell","M&G Poliéster","Nestlé","Nufarm","O Boticario","P&G","Produquímica",
"Rhodia/Solvay","SIEMENS","SUEZ","SYNGENTA","Syngenta","Tetrapack","Unigel Participações","Unilever","Unipar","WhiteMartins",
"Yara Brasil Fertilizantes","Dow Chemical","Borealis","Formosa Plastics","Dell","Renault","Michelin ","Coca-Cola","WalMart",
"Evian","Samsung","Cisco",'Sinopec','CNPC','ExxonMobil', 'Reliance','INEOS', 'NPC','Repsol','Petrobras', 'Petroken','Eco Petrol'
'ENAP', 'Pampa Energía', 'Koch Industries', 'PDVSA','Nova Chemicals', 'Chevron Phillips']
content = []
for x in empresas:
    temp_dataframe = df.loc[df['text'].str.contains(x)]
    tweets = temp_dataframe["text"].astype("str").dropna()
    tweets =''.join(tweets)
    tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)
    tweets = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',tweets)
    tweets = re.sub(r'[\W]+',' ',tweets)
    tokens = word_tokenize(tweets)
    tokens = [w.lower() for w in tokens if w.isalpha()]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [w for w in tokens if w.lower() not in stopwords]
    tokens = [w for w in tokens if len(w) > 3]
    fd = nltk.FreqDist(tokens)
    info = {
        'Empresa': x,
        'Ocorrências': len(temp_dataframe),
        'Palavras Frequentes': fd.most_common(25)
    }
    content.append(info)
empresas_count = pd.DataFrame(content)
empresas_count[empresas_count['Ocorrências'] > 0]
df_noticias = pd.read_excel('../input/noticiasbeatplastic.xlsx', encoding='utf-8-sig')
len(df_noticias)
import datetime
df_noticias['DATA DE PUBLICAÇÃO'] = df_noticias['DATA DE PUBLICAÇÃO'].map(lambda x:datetime.datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d"))
df_noticias['DATA DE PUBLICAÇÃO'] = pd.to_datetime(df_noticias['DATA DE PUBLICAÇÃO'])
df_nottempo = df_noticias
df_nottempo = df_nottempo.groupby('DATA DE PUBLICAÇÃO').count().reset_index()
df_nottempo = df_nottempo.sort_values(by='DATA DE PUBLICAÇÃO')
x = df_nottempo['DATA DE PUBLICAÇÃO']
y = df_nottempo['URL']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - Notícias #Beat Plastic Pollution')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
#df_noticias['DATA DE PUBLICAÇÃO']
palavras = df_noticias["TÍTULO"].astype("str").dropna()
palavras =''.join(palavras)
palavras = re.sub(r'[^\x00-\x7F]+',' ', palavras)
palavras = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',palavras)
palavras = re.sub(r'[\W]+',' ',palavras)
tokens = word_tokenize(palavras)
tokens = [w.lower() for w in tokens if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
tokens = [w for w in tokens if w.lower() not in stopwords]
tokens = [w for w in tokens if len(w) > 3]
fd = nltk.FreqDist(tokens)
print('25 palavras mais comum nos títulos:')
print(fd.most_common(25))
df_noticias['FONTE'].value_counts()[:15]
df_noticias['LOCALIDADE'].value_counts()[:20]
df_noticias.groupby('FONTE').first().reset_index()[['FONTE', 'AUDIÊNCIA']].sort_values(by='AUDIÊNCIA', ascending=False).head(15)
df_noticiasplastic = pd.read_excel('../input/noticiasplasticfreejuly.xlsx', encoding='utf-8-sig')
len(df_noticiasplastic)
import datetime
df_noticiasplastic['DATA DE PUBLICAÇÃO'] = df_noticiasplastic['DATA DE PUBLICAÇÃO'].map(lambda x:datetime.datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d"))
df_noticiasplastic['DATA DE PUBLICAÇÃO'] = pd.to_datetime(df_noticiasplastic['DATA DE PUBLICAÇÃO'])
df_nottempo2 = df_noticiasplastic
df_nottempo2 = df_nottempo2.groupby('DATA DE PUBLICAÇÃO').count().reset_index()
df_nottempo2 = df_nottempo2.sort_values(by='DATA DE PUBLICAÇÃO')
x = df_nottempo2['DATA DE PUBLICAÇÃO']
y = df_nottempo2['URL']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - Notícias #Plastic Free July')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
#df_noticias['DATA DE PUBLICAÇÃO']
palavras = df_noticiasplastic["TÍTULO"].astype("str").dropna()
palavras =''.join(palavras)
palavras = re.sub(r'[^\x00-\x7F]+',' ', palavras)
palavras = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',palavras)
palavras = re.sub(r'[\W]+',' ',palavras)
tokens = word_tokenize(palavras)
tokens = [w.lower() for w in tokens if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('english')
tokens = [w for w in tokens if w.lower() not in stopwords]
tokens = [w for w in tokens if len(w) > 3]
fd = nltk.FreqDist(tokens)
print('25 palavras mais comum nos títulos das notícias de Plastic Free July:')
print(fd.most_common(25))
df_noticiasplastic['FONTE'].value_counts()[:15]
df_noticiasplastic['LOCALIDADE'].value_counts()[:20]
df_noticiasplastic.groupby('FONTE').first().reset_index()[['FONTE', 'AUDIÊNCIA']].sort_values(by='AUDIÊNCIA', ascending=False).head(15)
df_noticiasacabe = pd.read_excel('../input/noticiasacabe.xlsx', encoding='utf-8-sig')
len(df_noticiasacabe)
import datetime
df_noticiasacabe['DATA DE PUBLICAÇÃO'] = df_noticiasacabe['DATA DE PUBLICAÇÃO'].map(lambda x:datetime.datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d"))
df_noticiasacabe['DATA DE PUBLICAÇÃO'] = pd.to_datetime(df_noticiasacabe['DATA DE PUBLICAÇÃO'])
df_nottempo3 = df_noticiasacabe
df_nottempo3 = df_nottempo3.groupby('DATA DE PUBLICAÇÃO').count().reset_index()
df_nottempo3 = df_nottempo3.sort_values(by='DATA DE PUBLICAÇÃO')
x = df_nottempo3['DATA DE PUBLICAÇÃO']
y = df_nottempo3['URL']
plt.xlabel('Data')
plt.ylabel('Quantidade de Ocorrências')
plt.xticks(rotation=45)
plt.title('Número de Ocorrências por dia - Notícias #Plastic Free July')
plt.plot(x,y, label='Tendencia Diária')
plt.show()
#df_noticias['DATA DE PUBLICAÇÃO']
palavras = df_noticiasacabe["TÍTULO"].astype("str").dropna()
palavras =''.join(palavras)
palavras = re.sub(r'[^\x00-\x7F]+',' ', palavras)
palavras = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',palavras)
palavras = re.sub(r'[\W]+',' ',palavras)
tokens = word_tokenize(palavras)
tokens = [w.lower() for w in tokens if w.isalpha()]
stopwords = nltk.corpus.stopwords.words('portuguese')
tokens = [w for w in tokens if w.lower() not in stopwords]
tokens = [w for w in tokens if len(w) > 3]
fd = nltk.FreqDist(tokens)
print('25 palavras mais comum nos títulos das notícias de Plastic Free July:')
print(fd.most_common(25))
df_noticiasacabe['FONTE'].value_counts()[:15]
df_noticiasacabe['LOCALIDADE'].value_counts()[:15]
df_noticiasacabe.groupby('FONTE').first().reset_index()[['FONTE', 'AUDIÊNCIA']].sort_values(by='AUDIÊNCIA', ascending=False).head(15)
