# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

        

        

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
d1=pd.read_csv('/kaggle/input/ted-talks/ted_main.csv')

d1.head(5)

# d1.columns



#keep relavant data only

cols=['comments','duration','event','languages', 'main_speaker','speaker_occupation','title','url', 'views']

d1=d1[[columns for columns in cols]]

d1.head(5)

# print(d1.shape)
len(d1.speaker_occupation.unique())
d11=d1.groupby(['event']).agg({'main_speaker':'nunique',

                          'event':'count',

                              'views':'sum',

                              'comments':'sum'})

d11.columns=['ct_events','ct_speakers','net_views','net_comments']

# d11.reset_index(inplace=True)

d11.sort_values('ct_speakers',ascending=False,inplace=True)



# obtain top 10 events 

d12=d11.nlargest(10,'ct_speakers')

d12.reset_index(inplace=True)

# d12.head(5)



#obtain top 10 talks by View counts

d13=d11.nlargest(10,'net_views')



#obtain top 10 talks by Comment counts

d14=d11.nlargest(10,'net_comments')

d14.reset_index(inplace=True)

d13.reset_index(inplace=True)



# Lets Plot



fig1=px.bar(d12,x="event",y="ct_speakers",width=800,height=400)

fig1.update_layout(margin=dict(l=10,r=10,t=30,b=10),title="Top 10 Ted Events",

                   xaxis_title="Event",yaxis_title="Speakers Participated")

fig1.update_traces(marker_color='Turquoise')

fig1.show()
fig2=make_subplots(rows=1,cols=2,subplot_titles=("Total Views","Total Comments"))



trace_1=go.Bar(x=d13.event,y=d13.net_views,name='views')

trace_2=go.Bar(x=d14.event,y=d14.net_comments,name='comments')



fig2.add_trace(trace_1, 1, 1)

fig2.add_trace(trace_2, 1, 2)



fig2.update_layout(showlegend=False, title_text="Ted Event Popularity",height=450)

fig2.show()
# Lets focus on top 25 talks by view count for now

d4=d1[['title','views','duration','comments']].nlargest(25,'views')

d11.ct_speakers.mean()

d4.sort_values('views',ascending=True,inplace=True)

d4['duration']=d4['duration'].apply(lambda x:(x/60))

d4['duration']=d4['duration'].round(1)

d4.head(5)
fig5=go.Figure(go.Bar(y=d4.title,x=d4.views,orientation='h',marker=dict(color='rgba(246, 78, 139, 0.6)'),name='Views'))

fig5.update_layout(height=600,width=900,autosize=False,title="Popular Ted Talks (View Count)")

fig5.show()
fig6=px.scatter(d4,x='duration',y='comments',size='views',color='title',width=1000,height=500)

fig6.update_layout(showlegend=False,xaxis_title="Duration (min)",yaxis_title="Total Comments",title="Ted Talks Engagement")

fig6.show()
d2=pd.read_csv('/kaggle/input/ted-talks/transcripts.csv')



# lets merge the transcript data with above one based on 'url' as common key

d3=pd.merge(d1,d2,on='url',how='inner')

d3.head(5)



# print(d1.shape)

# print(d2.shape)

# print(d3.shape)
# we will use nltk library for our purpose and visualize results using genesim since it is quite interactive 



from nltk.tokenize import RegexpTokenizer

from stop_words import get_stop_words

from nltk.stem.porter import PorterStemmer

from gensim import corpora, models

from gensim.models import CoherenceModel

import gensim



tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')

p_stemmer = PorterStemmer()



t1=d3['transcript']

texts = []



#lets clean up our raw text such as remove common words such as articles, prespositions and convert to lower case. 

#In addition we are stemming words as well. for example :'playing', 'played' to 'play' etc



for i in t1:

    

    raw = i.lower()

    tokens = tokenizer.tokenize(raw)



    stopped_tokens = [i for i in tokens if not i in en_stop]

#     texts.append(stopped_tokens)

#     stem tokens

    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    

    # add tokens to list

    texts.append(stemmed_tokens)



# turn our tokenized documents into a id <-> term dictionary

dictionary = corpora.Dictionary(texts)

    

# convert tokenized documents into a document-term matrix

corpus = [dictionary.doc2bow(text) for text in texts]
%time

import pyLDAvis.gensim

# pyLDAvis.enable_notebook()

# topics = pyLDAvis.gensim.prepare(lda_model,corpus, dictionary)



# Build LDA model

def model(n):

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,

                                                 num_topics=n,

                                                 random_state=100,

                                                 update_every=1,

                                                 chunksize=500,

                                                 passes=10,alpha='auto',per_word_topics=True)

    lda_model.print_topics()

    

    pyLDAvis.enable_notebook()

    topics = pyLDAvis.gensim.prepare(lda_model,corpus, dictionary)

    # compute perplexity

#     perplexity=lda_model.log_perplexity(corpus)

    

    # compute coherence

#     coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')

#     coherence_lda = coherence_model_lda.get_coherence()

#     print(perplexity)

#     print(coherence_lda)

    return(topics)
# Model with 8 topics

model(8)
#model with 6 topics

model(6)
#model with 5 topics

model(5)
#5 topics

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,

                                                 num_topics=5,

                                                 random_state=100,

                                                 update_every=1,

                                                 chunksize=1000,

                                                 passes=10,alpha='auto',per_word_topics=True)

pyLDAvis.enable_notebook()

topics = pyLDAvis.gensim.prepare(lda_model,corpus, dictionary)

topics
# Lets check Perplexity

print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 



# Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Merge results back with original dataset to obtain high level summaries



def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=d3['title']):

# Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0: # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)



df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=d3['title'])

#df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'title']



# Show

df_dominant_topic.head(10)
print(lda_model.print_topics())
df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==0.0,'Topic Name']='Economic Affairs'

df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==1.0,'Topic Name']='Arts and Culture'

df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==2.0,'Topic Name']='Planet and Diseases'

df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==3.0,'Topic Name']='Relations and Conflicts'

df_dominant_topic.loc[df_dominant_topic['Dominant_Topic']==4.0,'Topic Name']='Productivity'

t2=d3[['views','comments','title','duration']]

t3=pd.concat([df_dominant_topic,t2],axis=1)

t3.head(5)

# df_topic_sents_keywords
#Lets check the overall topic distribution in dataset



t5=df_dominant_topic['Topic Name'].value_counts().to_frame().reset_index()

t5.columns=['Topic','Count']

t5['Topic_Perc']=(t5['Count'])/(t5['Count'].sum())*100

t5
#Decile Analysis: Find out topic distribution among top 20% of Ted Talks based on view count



t3['Decile']=pd.qcut(t3['views'],10,labels=np.arange(10,0,-1))



t4=t3[(t3['Decile']==1)|(t3['Decile']==2)]

t41=t4['Topic Name'].value_counts().sort_values(ascending=False).to_frame().reset_index()

t41.columns=['Topic','Number of Ted Talks']

t41['Topic_Perc']=(t41['Number of Ted Talks']/t41['Number of Ted Talks'].sum()*100).round(2)

t41


fig10=make_subplots(rows=1, cols=2,subplot_titles=("Overall Topic Distribution (%)","Topic Distribution among top 20% Ted Talks (%)"))



trace_1=go.Bar(x=t41.Topic,y=t41.Topic_Perc)

trace_2=go.Bar(x=t5.Topic,y=t5.Topic_Perc)



fig10.add_trace(trace_2,1,1)

fig10.add_trace(trace_1,1,2)



fig10.update_layout(showlegend=False,title="Popular Ted Topics",height=500,width=1200,yaxis_title='Percent')

fig10.show()
# Obtain top 10 Ted Talks related to each Topic



t51=pd.DataFrame([])

for topic in ['Productivity','Economic Affairs','Arts and Culture','Planet and Diseases','Relations and Conflicts']:

    x1=t3[t3['Topic Name']==topic]

    x2=x1.nlargest(10,'views')

    t51=t51.append(x2,ignore_index=True)   

t51=t51[['title','views','Topic Name']]

t51.columns=['title','title2','views','Topic']

t51.drop(columns=['title2'],inplace=True)



t51.sort_values(['Topic','views'],ascending=[True,True],inplace=True)



# fig5=go.Figure(go.Bar(y=d4.title,x=d4.views,orientation='h',marker=dict(color='rgba(246, 78, 139, 0.6)'),name='Views'))

# fig5.update_layout(height=600,width=900,autosize=False,title="Popular Ted Talks")

# fig5.show()





fig12=make_subplots(rows=5,cols=1,subplot_titles=('Productivity','Economic Affairs','Arts and Culture','Planet and Diseases','Relations and Conflicts'))



t52=t51[t51['Topic']=='Productivity']

trace_11=go.Bar(x=t52.views,y=t52.title,orientation='h')



t53=t51[t51['Topic']=='Economic Affairs']

trace_22=go.Bar(x=t53.views,y=t53.title,orientation='h')



t54=t51[t51['Topic']=='Arts and Culture']

trace_33=go.Bar(x=t54.views,y=t54.title,orientation='h')



t55=t51[t51['Topic']=='Planet and Diseases']

trace_44=go.Bar(x=t55.views,y=t55.title,orientation='h')



t56=t51[t51['Topic']=='Relations and Conflicts']

trace_55=go.Bar(x=t56.views,y=t56.title,orientation='h')



fig12.add_trace(trace_11,1,1)

fig12.add_trace(trace_22,2,1)

fig12.add_trace(trace_33,3,1)

fig12.add_trace(trace_44,4,1)

fig12.add_trace(trace_55,5,1)



fig12.update_layout(height=1500,width=1000,showlegend=False,title="Top Ted Talks by Topics (View count)")

fig12.show()
# obtain all relavant data in one dataset

cols1=['title','transcript']

d33=d3[['title','transcript','views']]

z1=pd.merge(t3,d33, how='inner',left_on=['views'],right_on=['views'])





# we wish to look at differences between top 50 and bottom 50 talks by views 



z1_top=z1.nlargest(50,"views")

z1_bottom=z1.nsmallest(50,"views")

z12=z1_top.append(z1_bottom)

z12.head(5)

# first lets break the transcript into three equal parts



l11=[]

l22=[]

l33=[]

i=0

x=z12['transcript'].str.len().reset_index()

for line in z12['transcript']:

    i=i+i

    y=x.iloc[i,1]//3

    l1=line[:y]

    l2=line[y:(y+y)]

    l3=line[(y+y):]

    

    l11.append(l1)

    l22.append(l2)

    l33.append(l3)

    

dc5=pd.concat([pd.DataFrame(l11),pd.DataFrame(l22),pd.DataFrame(l33)],axis=1,ignore_index=True)

dc5.columns=['opening','middle','closing']

z13=pd.concat([z12,dc5.set_index(z12.index)],axis=1)
# we will use polarity scores from Vadar. 

# we obtain positive, negative and neutral in each topic part. In addition we also have 'Compound' metric which gives an overall aggregate. The compound ratings are between -1 and 1 with former indicating negative. Anything close to '0' would be neutral



from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA



sia = SIA()

results = []

L1=z13['opening']

for line in L1:

    pol_score = sia.polarity_scores(line)

    pol_score['L1'] = line

    results.append(pol_score)

dc1 = pd.DataFrame.from_records(results)



results=[]

L2=z13['middle']

for line in L2:

    pol_score = sia.polarity_scores(line)

    pol_score['L2'] = line

    results.append(pol_score)

dc2 = pd.DataFrame.from_records(results)



results=[]

L3=z13['closing']

for line in L3:

    pol_score = sia.polarity_scores(line)

    pol_score['L3'] = line

    results.append(pol_score)

dc3 = pd.DataFrame.from_records(results)



results=[]

L=z13['transcript']

for line in L:

    pol_score = sia.polarity_scores(line)

    pol_score['L'] = line

    results.append(pol_score)

dc4 = pd.DataFrame.from_records(results)



%time
# merge all individual datasets into one

dc6=pd.concat([dc1,dc2,dc3,dc4],axis=1)

dc6.columns=['negative_open','neutral_open','positive_open','compound_open','opening',

            'negative_middle','neutral_middle','positive_middle','compound_middle','middle',

            'negative_closing','neutral_closing','positive_closing','compound_closing','closing',

            'negative','neutral','positive','compound','transcript']

dc6.head(5)
# lets add topic number to above, index are not same so adjusting them accordingly



z14=pd.concat([z13,dc6.set_index(z13.index)],axis=1)

z14.drop(columns=['title_y','Document_No'],inplace=True)

z14.head(5)
# let us now summarize the above dataset. Out dataset consists of top 50 and bottom 50 Ted Talks only (filtered at the begining of the analysis). D1 is top 50 while D10 is bottom 50 



z15=z14.groupby(['Decile']).agg({'negative_open':'mean',

                             'neutral_open':'mean',

                             'positive_open':'mean',

                             

                            'negative_middle':'mean',

                             'neutral_middle':'mean',

                             'positive_middle':'mean',

                    

                             'negative_closing':'mean',

                             'neutral_closing':'mean',

                             'positive_closing':'mean',

                             

                             'negative':'mean',

                             'neutral':'mean',

                             'positive':'mean',

                            }).dropna().reset_index()

z15
z16=z15[['negative_open', 'neutral_open', 'positive_open',

       'negative_middle', 'neutral_middle', 'positive_middle',

       'negative_closing', 'neutral_closing', 'positive_closing', 'negative',

       'neutral', 'positive']].apply(lambda x: x*100).round(1)



z16['Decile']=np.where(z15.index==0,"D10","NA")

z16['Decile']=np.where(z15.index==1,"D1",z16['Decile'])

z16

#lets plot to visualize the results



titles=["Negative (Overall)","Neutral (Overall)","Positive (Overall)",

        "Negative (Opening)","Neutral (Opening)","Positive (Opening)",

       "Negative (Middle)","Neutral (Middle)","Positive (Middle)",

       "Negative (Closing)","Neutral (Closing)","Positive (Closing)"]



fig21=make_subplots(rows=4,cols=3,subplot_titles=titles)



t1=go.Bar(x=z16.Decile,y=z16.negative,text=z16.negative,textposition='auto')

t2=go.Bar(x=z16.Decile,y=z16.neutral,text=z16.neutral,textposition='auto')

t3=go.Bar(x=z16.Decile,y=z16.positive,text=z16.positive,textposition='auto')



t4=go.Bar(x=z16.Decile,y=z16.negative_open,text=z16.negative_open,textposition='auto')

t5=go.Bar(x=z16.Decile,y=z16.neutral_open,text=z16.neutral_open,textposition='auto')

t6=go.Bar(x=z16.Decile,y=z16.positive_open,text=z16.positive_open,textposition='auto')



t7=go.Bar(x=z16.Decile,y=z16.negative_middle,text=z16.negative_middle,textposition='auto')

t8=go.Bar(x=z16.Decile,y=z16.neutral_middle,text=z16.neutral_middle,textposition='auto')

t9=go.Bar(x=z16.Decile,y=z16.positive_middle,text=z16.positive_middle,textposition='auto')



t10=go.Bar(x=z16.Decile,y=z16.negative_closing,text=z16.negative_closing,textposition='auto')

t11=go.Bar(x=z16.Decile,y=z16.neutral_closing,text=z16.neutral_closing,textposition='auto')

t12=go.Bar(x=z16.Decile,y=z16.positive_closing,text=z16.positive_closing,textposition='auto')



fig21.add_trace(t1,1,1)

fig21.add_trace(t2,1,2)

fig21.add_trace(t3,1,3)

fig21.add_trace(t4,2,1)

fig21.add_trace(t5,2,2)

fig21.add_trace(t6,2,3)

fig21.add_trace(t7,3,1)

fig21.add_trace(t8,3,2)

fig21.add_trace(t9,3,3)

fig21.add_trace(t10,4,1)

fig21.add_trace(t11,4,2)

fig21.add_trace(t12,4,3)



# fig21.update_layout(xaxis={'type':'category'})

# fig21.update_layout(yaxis=dict(tickformat="%"))



fig21.update_layout(height=1200,showlegend=False,yaxis=dict(range=[0,100]),title="Breakdown of Ted Talk by Sentiment(%) and format")

fig21.update_yaxes(range=[0, 100])



fig21.show()