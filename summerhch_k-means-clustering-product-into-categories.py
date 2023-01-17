import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from random import randint

from wordcloud import WordCloud

from collections import Counter
df_raw = pd.read_csv('../input/data.csv',encoding="ISO-8859-1")

print(df_raw.shape)
df_raw.dropna(axis = 0, subset = ['Description'], inplace = True)

print('Dataframe dimensions:', df_raw.shape)

print(df_raw['Description'].head())

print(df_raw.shape)
df_desc = pd.DataFrame(df_raw['Description'].unique()).rename(columns = {0:'Description'})

print(df_desc.shape)

print(df_desc)
# explore the punctuations in the data

find_dict={}

newp='"#$%&\'()*+-/:;<=>@[\]^_`{|}~.?'

for i, row in df_desc.iterrows():

    for c in newp:

        #print(c)

        #print(row)

        if c in row['Description']:

            val=find_dict.get(c,'')

            find_dict[c]=val+'|'+row['Description']

            #print('find:',c)

           # print(row['Description'])

#print(list(find_dict))
def punc_processing(st):

    for i,c in enumerate(list(st)):

        if c == '\'':

            #clean example like 'n'

            if i==0:

                #print(st)

                st_clean=st[1:(len(st)-1)]

                #print(st_clean)

                return st_clean

            #clean example like b'fly

            if i==1:

                #print(st)

                st_clean=st[2:]

                #print(st_clean)

                return st_clean

            #clean example like mother's

            if i!=0 and st[i-1]!=' ':

                #print(st)

                st_clean=st[:(i)]+st[(i+2):]

                #print(st_clean)

                return st_clean

        if c == '"':

            #clean example like "glamorous"

            if i==0:

                #print(st)

                st_clean=st[1:(len(st)-1)]

                #print(st_clean)

                return st_clean        

    return st
newp='["&\'()+-/[\].]'

front_quotation='[\"\'(]'

end_quotation='[\"\')]'

connect_quotation='&/+,'

toker = RegexpTokenizer('[a-z]+'+newp+'[a-z]+|[a-z]+|'+front_quotation+'[a-z]+'+end_quotation)

wordnet_lemmatizer=WordNetLemmatizer()

stopWords = set(stopwords.words('english'))

stopWords.update(['small', 'large', 'jumbo', 'set', 'pink', 'blue', 'tag', 'red', 'white'])

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")



df_desc['token']=None

df_desc['token_list']=None



for i, row in df_desc.iterrows():

    descp_st=row['Description'].lower()

    for quot in connect_quotation:

        descp_st=descp_st.replace(quot, ' '+quot+' ')

    #Tokenization

    descp_l=toker.tokenize(descp_st)

    #Punctuation pre-processing

    descp_l=[punc_processing(x) for x in descp_l]

    #Stop words removal

    descp_l2=[x for x in descp_l if x not in stopWords ]

    

    #test if use steming to pre-process

    #descp_l2=[stemmer.stem(x) for x in descp_l2]

    

    #Lematization

    descp_l2=[wordnet_lemmatizer.lemmatize(x) for x in descp_l2]

    df_desc.loc[i,'token_list']=descp_l2

    df_desc.loc[i, 'token']=' '.join(descp_l2)
#test if stop word or words mislead clustering correctly

print('Number of common stop words:', len(stopwords.words('english')))

print('Example of common stop words(not all stop words listed):', stopwords.words('english')[:30], '\n')

print('Number of unwanted words in these sample:', len(['small', 'large', 'jumbo', 'set', 'pink', 'blue', 'tag', 'red', 'white']))

print('Example of unwanted words in these sample:', ['small', 'large', 'jumbo', 'set', 'pink', 'blue', 'tag', 'red', 'white'])
#test if punctation are filttered correctly

find_dict={}

newp='"#$%&\'()*+-/:;<=>@[\]^_`{|}~.?'

for i, row in df_desc.iterrows():

    for c in newp:

        if c in row['Description']:

            #print('Punctuation:',c)

            #print('Original description:',row['Description'])

            #print('After pre-processing:',row['token_list'], '\n')

            newp.replace(c, '')

            break
#test if stopwords filttered correctly

# col='OF'

# for i, row in df_desc.iterrows():

#     if col in row['Description']:

#         print('Remove the word:',col)

#         print(row['Description'])

#         print(row['token_list'])

#         print('\n')
#test if lematize correctly

# col='AGED'

# for i, row in df_desc.iterrows():

#     if col in row['Description']:

#         print('Lematize the word:',col)

#         print(row['Description'])

#         print(row['token_list'])

#         print('\n')
print(df_desc[['Description','token_list']])

vectorizer = CountVectorizer(min_df=1)



data_desc_doc = vectorizer.fit_transform(df_desc['token'])

feature_name = vectorizer.get_feature_names()



print('Number of words appeared in corpus:', len(feature_name))

print('Example of words appeared in corpus(not all listed):',feature_name[:30], '\n')

print(df_desc.head(1))

print(data_desc_doc)
print(data_desc_doc.toarray())
def findBestN(matrix):

    for n in range(3,15):

        kmeans = KMeans(n_clusters = n, n_init=20, random_state=0 )

        kmeans.fit(matrix)

        clusters = kmeans.predict(matrix)

        silhouette_avg = silhouette_score(matrix, clusters)

        print("For n_clusters =", n, "The average silhouette_score is :", silhouette_avg)

        

findBestN(data_desc_doc)
best_no_of_cluster= 12

#k means clustering

kmeans = KMeans(n_clusters = best_no_of_cluster, n_init=20, random_state=0 )

kmeans.fit(data_desc_doc)

km_result=kmeans.predict(data_desc_doc)

df_desc['cluster_group']=pd.Series(km_result)



def desinate_color(word=None, position=None,orientation=None,font_size=None,  font_path=None, random_state=0):

    h = randint(rand_tone*3,rand_tone*3)

    s = randint(90,100)

    l = randint(40,60)

    return "hsl({}, {}%, {}%)".format(h, s, l)



fig = plt.figure(1, figsize=(20,20))

for a in range(0,best_no_of_cluster,1):    

    df_temp=df_desc[df_desc['cluster_group']==a]

    #print(a,' size:', df_temp.shape)

    #print(df_temp)

    

    c = Counter()

    rand_tone=a

    for i, row in df_temp.iterrows():

        c.update(row['token_list'])       

    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', color_func = desinate_color, relative_scaling=0.15, random_state=0)

    wordcloud.generate_from_frequencies(c)

    axis_1 = fig.add_subplot(4,3,(a+1))

    axis_1.imshow(wordcloud)

    axis_1.axis('off')

    plt.title('Cluster n={}'.format(a+1))