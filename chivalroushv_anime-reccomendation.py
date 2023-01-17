import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_csv("../input/anime-recommendations-database/Anime_data.csv")
print("Dataset shape:", df.shape)
df.head()
df[df.Genre.isnull()]
df.info()
df.columns
features=['Anime_id', 'Title', 'Genre', 'Type',
       'Rating', 'ScoredBy', 'Popularity', 'Members', 'Source']
df1=df[features]
df1.info()
df1.dropna(inplace=True)
df1.info()
df1.Type.unique()
#from sklearn.preprocessing import OneHotEncoder
#ohe=OneHotEncoder(drop='first')
#ohe.fit(df1[['Type']])
tmp=pd.get_dummies(df1[['Type']],drop_first=True)
#tmp=pd.DataFrame(tmp)
tmp
df1=df1.merge(tmp,right_index=True,left_index=True)
df1.tail()
df1.columns
df1.Genre.value_counts()
df1.head()
df1=df1.reset_index()
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
sparse_matrix = count_vectorizer.fit_transform(df1.Genre)
doc_term_matrix = sparse_matrix.todense()
tmp = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names())
df2=df1.merge(tmp,right_index=True,left_index=True)
df2.info()
similarity = cosine_similarity(df2.drop(["index","Anime_id","Title","Genre","Type","Source"],axis=1))
print(df2,similarity)
print(df2.shape)
print(similarity.shape)
m="Ake-Vono"
i = df1.loc[df1['Title']==m].index[0]
print(i)
lst = list(enumerate(similarity[i]))
lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
lst = lst[1:11] # excluding first item since it is the requested movie itself
l = []
for i in range(len(lst)):
    a = lst[i][0]
    l.append(df1['Title'][a])
print(l)
df1.loc[df1['Title']==m]
df1.loc[df1['Title']=="Haita"]
