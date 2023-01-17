import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df1 = pd.read_csv('../input/Comcast_telecom_complaints_data.csv',index_col=0)
df1.head()
df1[df1.isnull()].count()
#No Nulls
df1.describe(include='all')
df1.info()
df1['Date_month_year'] = pd.to_datetime(df1['Date_month_year'])
df1['Created_Month'] =  df1['Date_month_year'].apply(lambda x: x.month)
df1['Created_Day'] = df1['Date_month_year'].apply(lambda x: x.day)
df1['Created_Day of Week'] = df1['Date_month_year'].apply(lambda x: x.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
df1['Created_Day of Week']=df1['Created_Day of Week'].map(dmap)
df1.head(5)
#number of complaints monthly
plt.figure(figsize=(8,4))
bymonth = df1.groupby('Created_Month').count().reset_index()
lp = sns.lineplot(x='Created_Month', y= 'Customer Complaint', data = bymonth, sort=False,markers = "o")
ax = lp.axes
ax.set_xlim(0,13)
ax.annotate('Max complaints in Jun', color='red',
            xy=(6, 1060), xycoords='data',
            xytext=(0.8, 0.85), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.1),
            horizontalalignment='right', verticalalignment='top')
#number of complaints Daily
plt.figure(figsize=(18,6))
byday = df1.groupby('Created_Day').count().reset_index()
lp = sns.lineplot(x='Created_Day', y= 'Customer Complaint', data = byday, sort=False, color = 'red',markers = "o", )
ax = lp.axes
ax.set_xlim(0,32)
#number of complaints based on created day of the week
sns.countplot(x='Created_Day of Week', data = df1, order=df1['Created_Day of Week'].value_counts().index, palette ="Reds_d")
#More number of complaints on Tuesday and wednesday
df1['Customer Complaint'] = df1['Customer Complaint'].str.title() 
CT_freq = df1['Customer Complaint'].value_counts()
CT_freq
import nltk
%pip install wordcloud
from wordcloud import WordCloud, STOPWORDS
common_complaints = df1['Customer Complaint'].dropna().tolist()
common_complaints =''.join(common_complaints).lower()

list_stops = ('Comcast','Now','Company','Day','Someone','Thing','Also','Got','Way','Call','Called','One','Said','Tell')

for word in list_stops:
    STOPWORDS.add(word)
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='white',
                      width=1200,
                      height=1000).generate(common_complaints)

plt.figure( figsize=(10,12) )
plt.imshow(wordcloud)
plt.title('Frequent words for customer complaints')
plt.axis('off')
plt.show()
#Internet complaints are Maximum
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
nltk.download('wordnet')
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
doc_complete = df1['Customer Complaint'].tolist()
doc_clean = [clean(doc).split() for doc in doc_complete]
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
dictionary
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
doc_term_matrix
from gensim.models import LdaModel
num_topic = 9
ldamodel = LdaModel(doc_term_matrix,num_topics=num_topic,id2word = dictionary,passes=10)
topics = ldamodel.show_topics()
for topic in topics:
    print(topic)
    print()
word_dict = {}
for i in range(num_topic):
    words = ldamodel.show_topic(i,topn = 20)
    word_dict['Topic '+"{}".format(i)]=[i[0] for i in words]
pd.DataFrame(word_dict)
import pyLDAvis.gensim
Lda_display = pyLDAvis.gensim.prepare(ldamodel,doc_term_matrix,dictionary,sort_topics=False)
pyLDAvis.display(Lda_display)
df1['Highlevel_Status'] = ["Open" if Status=="Open" or Status=="Pending" else "Closed" for Status in df1["Status"]]
df1['Highlevel_Status'].unique()
df1['State'] = df1['State'].str.title() 
st_cmp = df1.groupby(['State','Highlevel_Status']).size().unstack().fillna(0)
st_cmp
st_cmp.sort_values('Closed',axis = 0,ascending=True).plot(kind="barh", figsize=(10,8), stacked=True)
df1.groupby(["State"]).size().sort_values(ascending=False).to_frame().rename({0: "Complaint count"}, axis=1)[:1]
#Georgia has highest complaints
CT = df1.groupby(["State","Highlevel_Status"]).size().unstack().fillna(0)
CT.sort_values('Closed',axis = 0,ascending=False)[:1]
#highest percentage of unresolved complaints
CT['Resolved_cmp_prct'] = CT['Closed']/CT['Closed'].sum()*100
CT['Unresolved_cmp_prct'] = CT['Open']/CT['Open'].sum()*100
CT.sort_values('Unresolved_cmp_prct',axis = 0,ascending=False)[:1]
#Georgia state has highest Unresolved complaints when compared to other states 
cr = df1.groupby(['Received Via','Highlevel_Status']).size().unstack().fillna(0)
cr['resolved'] = cr['Closed']/cr['Closed'].sum()*100
cr['resolved']
#df["item"].value_counts().nlargest(n=1).values[0]