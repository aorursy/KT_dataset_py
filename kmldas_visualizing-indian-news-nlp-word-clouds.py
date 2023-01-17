import numpy as np 

import pandas as pd 



import sklearn.feature_extraction.text as text



import spacy

from wordcloud import WordCloud,STOPWORDS

nlp=spacy.load("en_core_web_lg")



import textblob





import PIL



import matplotlib.pyplot as plt

%matplotlib inline


data=pd.read_csv("../input/indian-financial-news-articles-20032020/IndianFinancialNews.csv")



data
data=data[['Date','Title']].drop_duplicates()

data


data['Date']=pd.to_datetime(data['Date'], infer_datetime_format=True)

data['Year']=data['Date'].dt.year

data
### Get imp words by year



def get_imp(bow,mf,ngram):

    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')

    matrix=tfidf.fit_transform(bow)

    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)



bow=data['Title'].tolist()




total_data=get_imp(bow,mf=5000,ngram=1)



### Yearly trends

imp_terms_unigram={}

for y in data['Year'].unique():

    bow=data[data['Year']==y]['Title'].tolist()

    imp_terms_unigram[y]=get_imp(bow,mf=5000,ngram=1)

    

common_unigram={}

for y in np.arange(2003,2020,1):

    if y==2003:       

        common_unigram[y]=set(imp_terms_unigram[y].index).intersection(set(imp_terms_unigram[y+1].index))

    else:

        common_unigram[y]=common_unigram[y-1].intersection(set(imp_terms_unigram[y+1].index))    





total_data_bigram=get_imp(bow=bow,mf=5000,ngram=2)



imp_terms_bigram={}

for y in data['Year'].unique():

    bow=data[data['Year']==y]['Title'].tolist()

    imp_terms_bigram[y]=get_imp(bow,mf=5000,ngram=2)

    

### Common bigrams across all the years

common_bigram={}

for y in np.arange(2003,2020,1):

    if y==2003:

         common_bigram[y]=set(imp_terms_bigram[y].index).intersection(set(imp_terms_bigram[y+1].index))

    else:

        common_bigram[y]=common_bigram[y-1].intersection(set(imp_terms_bigram[y+1].index))

### Common bigrams across all the years

common_bigram={}

for y in np.arange(2003,2020,1):

    if y==2003:

         common_bigram[y]=set(imp_terms_bigram[y].index).intersection(set(imp_terms_bigram[y+1].index))

    else:

        common_bigram[y]=common_bigram[y-1].intersection(set(imp_terms_bigram[y+1].index))



    


total_data_trigram=get_imp(bow=bow,mf=5000,ngram=3)



imp_terms_trigram={}

for y in data['Year'].unique():

    bow=data[data['Year']==y]['Title'].tolist()

    imp_terms_trigram[y]=get_imp(bow,mf=5000,ngram=3)



    

### Common trigrams, 1 year window

common_trigram_1yr={}

for y in np.arange(2003,2020,1):

    common_trigram_1yr[str(y)+"-"+str(y+1)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index))

### Commin trigrams, 2 year window

common_trigram_2yr={}

for y in np.arange(2003,2018,3):

    if y==2003:

        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))

    else:

        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))

plt.subplot(1,3,1)

total_data.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')

plt.title("Unigrams",fontsize=30)

plt.yticks([])

plt.xticks(size=20)

plt.subplot(1,3,2)

total_data_bigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')

plt.title("Bigrams",fontsize=30)

plt.yticks([])

plt.xticks(size=20)

plt.subplot(1,3,3)

total_data_trigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')

plt.title("Trigrams",fontsize=30)

plt.yticks([])

plt.xticks(size=20)



    

for i in range(1,19,1):

    plt.subplot(9,2,i)

    imp_terms_bigram[2002+i].head(5).plot(kind="barh",figsize=(20,35),colormap='Set2')

    plt.title(2002+i,fontsize=20)

    plt.xticks([])

    plt.yticks(size=20,rotation=5)
for i in range(1,19,1):

    plt.subplot(9,2,i)

    imp_terms_trigram[2002+i].head(5).plot(kind="barh",figsize=(20,30),colormap="Set2")

    plt.title(2002+i,fontsize=20)

    plt.xticks([])

    plt.yticks(size=15,rotation=5)
index_yes=data['Title'].str.match(r'(?=.*\byes\b)(?=.*\bbank\b).*$',case=False)

data_yes=data.loc[index_yes].copy()

data_yes['polarity']=data_yes['Title'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)
pos=data_yes.query("polarity>0")['Title']

neg=data_yes.query("polarity<0")['Title']

print("The number of positve headlines were {} times the negative headlines".format(round(len(pos)/len(neg),2)))
plt.figure(figsize=(8,8))

plt.bar(["Positive","Negative"],[len(pos),len(neg)])

plt.title("Frequency of Positive and Negative News about Yes Bank",fontsize=20)
bow=data_yes['Title'].str.replace(r'yes|bank',"",case=False).tolist()

yes_uni=get_imp(bow,mf=5000,ngram=1)

yes_bi=get_imp(bow,mf=5000,ngram=2)

yes_tri=get_imp(bow,mf=5000,ngram=3)
plt.subplot(1,3,1)

yes_uni.head(10).plot(kind="barh",figsize=(24,6),colormap="Set2")

plt.title("Unigrams",fontsize=30)

plt.yticks(size=20)

plt.xticks([])

plt.subplot(1,3,2)

yes_bi.head(10).plot(kind="barh",figsize=(24,6),colormap="Set1")

plt.title("Bigrams",fontsize=30)

plt.yticks(size=20)

plt.xticks([])

plt.subplot(1,3,3)

yes_tri.head(10).plot(kind="barh",figsize=(24,6),colormap="Set3")

plt.title("Trigrams",fontsize=30)

plt.yticks(size=20)

plt.xticks([])
#url="https://upload.wikimedia.org/wikipedia/en/thumb/8/85/Yes_Bank_logo.svg/1200px-Yes_Bank_logo.svg.png"



yes_text=" ".join(bow)



con_mask=np.array(PIL.Image.open('../input/word-cloud/YesBank.png'))



wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color="Black",stopwords=STOPWORDS).generate(yes_text)

plt.figure( figsize=(30,15))

plt.imshow(wc)

plt.axis("off")

plt.yticks([])

plt.xticks([])

plt.savefig('./yes.png', dpi=50)

plt.show()