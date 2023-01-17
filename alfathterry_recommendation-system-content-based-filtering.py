import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/content_by_synopsis.csv")
df.head()
bow = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
bank = bow.fit_transform(df['overview'])
idx = 0 # Toy Story
content = df.loc[idx, "overview"]
content
code = bow.transform([content])
code
pd.DataFrame(code.toarray())
from sklearn.metrics.pairwise import cosine_distances
distance = cosine_distances(code, bank)
pd.DataFrame(distance)
rec_idx = distance.argsort()[0, 1:11]
rec_idx
df.loc[rec_idx]
from sklearn.metrics.pairwise import cosine_distances

class RecommenderSystem:
    def __init__(self, data, content_col):
        self.df = pd.read_csv(data)
        self.content_col = content_col
        self.encoder = None
        self.bank = None
        
    def fit(self):
        self.encoder = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])
        
    def recommend(self, idx, topk=10):
        content = df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rec_idx] 
recsys = RecommenderSystem("/kaggle/input/content_by_synopsis.csv", content_col="overview")
recsys.fit()
recsys.recommend(0) # Toy Story
recsys.recommend(1) # Jumanji
recsys.recommend(579) # Home Alone
df = pd.read_csv("/kaggle/input/content_by_multiple.csv")
df.head()
recsys = RecommenderSystem("/kaggle/input/content_by_multiple.csv", content_col='metadata')
recsys.fit()
recsys.recommend(0) # Toy Story
recsys.recommend(1) # Jumanji
recsys.recommend(579) # Home Alone
df = pd.read_csv("/kaggle/input/content_by_synopsis.csv")
df.head()
df1 = pd.read_csv("/kaggle/input/content_by_multiple.csv")
#df1 = df1[['title','metadata']]
df1.head()
df = df1.set_index('title').join(df.set_index('title'))
df['join'] = df['overview'] + df['metadata']
df.reset_index(inplace=True)
df.head()
df.fillna('', inplace=True)
from sklearn.metrics.pairwise import cosine_distances

class RecommenderSystem_df:
    def __init__(self, data, content_col):
        self.df = pd.DataFrame(data) #sebelumnya csv, sekarang ubah jadi dataframe
        self.content_col = content_col
        self.encoder = None
        self.bank = None
        
    def fit(self):
        self.encoder = CountVectorizer(stop_words="english", tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])
        
    def recommend(self, idx, topk=10):
        content = df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rec_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rec_idx] 
recsys = RecommenderSystem_df(df, content_col='join')
recsys.fit()
recsys.recommend(0) # Toy Story
recsys.recommend(1) # Jumanji
recsys.recommend(579) # Home Alone