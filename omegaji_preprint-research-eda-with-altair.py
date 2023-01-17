import pandas as pd

df=pd.read_csv("/kaggle/input/covid19-research-preprint-data/COVID-19-Preprint-Data_ver2.csv")


print(df.columns)

df.head()
from datetime import datetime

df["day"]=df["Date of Upload"].apply(lambda x: int(datetime.strptime(x,'%Y-%m-%d').day))

df["month"]=df["Date of Upload"].apply(lambda x:int( datetime.strptime(x,'%Y-%m-%d').month))

df["year"]=df["Date of Upload"].apply(lambda x: int( datetime.strptime(x,'%Y-%m-%d').year))

df["day_in_year"]=df["Date of Upload"].apply(lambda x:int( datetime.strptime(x,'%Y-%m-%d').timetuple().tm_yday))

df.drop(["Preprint Link","DOI","Date of Upload"],axis=1,inplace=True)
import altair_render_script

import altair as alt

alt.Chart(df.groupby(["day_in_year"]).count().reset_index()).mark_point().encode(

    x='day_in_year',

    y='Abstract',

    tooltip=["Abstract","day_in_year"],

    size="Abstract"

).interactive()



df[df["day_in_year"]==137]


alt.Chart(df[df["year"]==2020].groupby(["month"]).count().reset_index()).mark_area(

    line={'color':'darkblue'},

    color=alt.Gradient(

        gradient='linear',

        stops=[alt.GradientStop(color='white', offset=0),

               alt.GradientStop(color='blue', offset=1)],

        x1=1,

        x2=1,

        y1=1,

        y2=0



    )

).encode(

    alt.X('month'),

    alt.Y('Abstract',title="Abstract Count published"),

    tooltip=["month","Abstract"]

).interactive()

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

stop_words=set(stopwords.words('english'))

def removeSW(x):

    x=x.lower()

    word_tokens = word_tokenize(x) 

    

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    return " ".join(filtered_sentence)





df["Abstract"]=df["Abstract"].apply(removeSW)

from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None):

  

    vec = CountVectorizer(stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
unigrams=get_top_n_words(df["Abstract"],20)

unigrams_title=get_top_n_words(df["Title of preprint"],20)
unigrams
unigrams_title
d={"word":[],"count":[],"type":[]}

for i in unigrams:

    d["word"].append(i[0])

    d["count"].append(i[1])

    d["type"].append("Abstract")

for i in unigrams_title:

    d["word"].append(i[0])

    d["count"].append(i[1])

    d["type"].append("Title")

count_df=pd.DataFrame(d)



source = count_df



alt.Chart(source).mark_bar().encode(

    tooltip=["word","count"],

    column='type',

    x='word',

    y='count',

    color='type'

)
def get_top_gram(corpus,grams, n=None):

  

    vec = CountVectorizer(ngram_range=grams,stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
bigrams=get_top_gram(df["Abstract"],(2,2),20)

bigrams_title=get_top_gram(df["Title of preprint"],(2,2),20)

d={"word":[],"count":[],"type":[]}

for i in bigrams:

    d["word"].append(i[0])

    d["count"].append(i[1])

    d["type"].append("Abstract")

for i in bigrams_title:

    d["word"].append(i[0])

    d["count"].append(i[1])

    d["type"].append("Title")

count_df=pd.DataFrame(d)



source = count_df



alt.Chart(source).mark_bar().encode(

    tooltip=["word","count"],

    column='type',

    x='word',

    y='count',

    color='type'

)

fivegrams=get_top_gram(df["Abstract"],(5,5),20)

fivegrams_title=get_top_gram(df["Title of preprint"],(5,5),20)

d={"word":[],"count":[],"type":[]}

for i in fivegrams:

    d["word"].append(i[0])

    d["count"].append(i[1])

    d["type"].append("Abstract")

for i in fivegrams_title:

    d["word"].append(i[0])

    d["count"].append(i[1])

    d["type"].append("Title")

count_df=pd.DataFrame(d)



source = count_df



alt.Chart(source).mark_bar().encode(

    tooltip=["word","count"],

    column='type',

    x='word',

    y='count',

    color='type'

)

import json

myset=set()

mylist=[]

for i in df["Author(s) Institutions"].index:

    l=set(json.loads(df.loc[i,"Author(s) Institutions"]).keys())

    for j in l:

        myset.add(j)

        mylist.append(j)

len(myset)
len(mylist)
def CountFrequency(my_list): 

      

  

   count = {} 

   for i in my_list: 

       count[i] = count.get(i, 0) + 1

       

   

    

   return count 

frequency=CountFrequency(mylist)

  

d={"UNI":[],"Publishes":[]}

for i in frequency.keys():

    d["UNI"].append(i)

    d["Publishes"].append(frequency[i])



uni_df=pd.DataFrame(d)



uni_df=uni_df.sort_values(by=["Publishes"],ascending=False)

for i in uni_df.index:

    if len(uni_df.loc[i].UNI)<4:

        uni_df.drop(i,axis=0,inplace=True)
source = uni_df.iloc[:50,:]



alt.Chart(source).mark_bar().encode(

    x='Publishes',

    y="UNI",

    tooltip=["Publishes"]

).properties(height=700)
for i in frequency.keys():

   frequency[i]=[frequency[i],0]

df["Author(s) Institutions"].head()


mylist=[]

for i in df["Author(s) Institutions"].index:

    l=set(json.loads(df.loc[i,"Author(s) Institutions"]).keys())

    for j in l:

        frequency[j][1]=frequency[j][1]+json.loads(df.loc[i,"Author(s) Institutions"])[j]
 

d={"UNI":[],"AuthorCount":[]}

for i in frequency.keys():

    d["UNI"].append(i)

    d["AuthorCount"].append(frequency[i][1])



uni_df=pd.DataFrame(d)



uni_df=uni_df.sort_values(by=["AuthorCount"],ascending=False)

for i in uni_df.index:

    if len(uni_df.loc[i].UNI)<4:

        uni_df.drop(i,axis=0,inplace=True)
uni_df
source = uni_df.iloc[:50,:]



alt.Chart(source).mark_bar().encode(

    x='AuthorCount',

    y="UNI",

    tooltip=["AuthorCount","UNI"]

).properties(height=700)