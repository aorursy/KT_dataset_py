import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial import distance
import os
import _pickle as cPickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# generate random integer values
from random import seed
from random import randint
seed(42)
from scipy.spatial import distance
from scipy import sparse


li = []
for dirname, _, filenames in os.walk('/kaggle/input/news-portal-user-interactions-by-globocom/clicks'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        df = pd.read_csv(os.path.join(dirname, filename), index_col=None, header=0)
        li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.head(10)


users=pd.DataFrame({"user_id":frame['user_id']})
sessions=pd.DataFrame({"session_id":frame['session_id']})
articles = pd.read_csv('/kaggle/input/news-portal-user-interactions-by-globocom/articles_metadata.csv', index_col=None, header=0)
articles.head()
df['session_size'].value_counts().plot(kind = 'bar', title="session unique values count")
df.groupby(by="session_id")['click_article_id'].nunique().value_counts().plot(kind = 'bar', title="articles nb per session") 
df.groupby(by="user_id")['session_id'].nunique().value_counts().plot(kind = 'bar', title="sessions nb per user")
df.groupby(by="user_id")['click_article_id'].nunique().value_counts().plot(kind = 'bar', title="articles nb per user")

with open(r"/kaggle/input/news-portal-user-interactions-by-globocom/articles_embeddings.pickle", "rb") as input_file:
    e = cPickle.load(input_file)
def getFiveArticles(e, userId):
    
    ee=e
    #get all articles read by user
    var= frame.loc[frame['user_id']==userId]['click_article_id'].tolist()
    #chose randomly one
    value = randint(0, len(var))
    #delete all read articles except the selected one( we do not want to offer user to read something he already read)
    for i in range(0, len(var)):
        if i != value:
            ee=np.delete(ee,[i],0)
    arr=[]
    
    #delecte selected article in the new matrix
    f=np.delete(ee,[value],0)
    #get 5 articles the most similar to the selected one
    for i in range(0,5):
        distances = distance.cdist([ee[value]], f, "cosine")[0]
        min_index = np.argmin(distances)
        f=np.delete(f,[min_index],0)
        #find corresponding matrix in original martix
        result = np.where(e == f[min_index])
        arr.append(result[0][0])
        
    return arr
        
print(getFiveArticles(e, 92059))
d=e
max=articles['words_count'].max()
articles['words_count']= articles['words_count'].apply(lambda x: x/max)

d=np.append(d,np.reshape(articles['words_count'].to_numpy(), newshape=(articles['words_count'].shape[0],1)),axis=1)

d.shape

print(getFiveArticles(d, 92059))

n_users = frame['user_id'].values.ravel()
n_users=pd.unique(n_users)
n_sessions = frame['session_id'].values.ravel()
n_sessions=pd.unique(n_sessions)
#Target Variable: articles
y= np.zeros((frame.shape[0], 250))

for i in range(0,frame.shape[0]):
    y[i]=e[[frame.iloc[i,: ]['click_article_id']]]
    


'''
#processed separately in batches
articles_prepared= pd.DataFrame(columns=["article_id","category_id","created_at_ts","publisher_id","words_count"])

for index, row in frame.iterrows():
    articles_prepared= articles_prepared.append(articles.loc[row['click_article_id']])
'''

li = []
for dirname, _, filenames in os.walk('/kaggle/input/reorderedarticles'):
    for filename in sorted(filenames):
      
        
        df = pd.read_csv(os.path.join(dirname, filename), index_col=None, header=0)
        li.append(df)

reordered = pd.concat(li, axis=0, ignore_index=True)
reordered.head()


n_cats = reordered['category_id'].values.ravel()
n_cats=pd.unique(reordered['category_id'])
n_users = users.values.ravel()
n_users=pd.unique(n_users)

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Multiply, LSTM, Dropout
from keras.models import Model
from keras.optimizers import SGD

user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users.shape[0],250, name="Users-Embedding")(user_input)
user_vec = Dense(250, activation='relu')(user_embedding)

cat_input = Input(shape=[1], name="Category-Input")
cat_embedding = Embedding(reordered['category_id'].max(), 250, name="Catgory-Embedding")(cat_input)
cat_vec = Dense(250, activation='relu')(cat_embedding)

words_input = Input(shape=[1], name="Words-Input")
words_vec=Dense(250, activation='relu')(words_input)

date_input= Input(shape=(1,1),name="Date_Created-Input")
date_vec= LSTM(500, input_shape=(1,1))(date_input)
date_drop=Dropout(0.2)(date_vec)
date_dense= Dense(250, activation="relu")(date_drop)

prod = Multiply()([user_vec,cat_vec,words_vec,date_dense])

dense2= Dense(250, activation ="relu")(prod)
fin= Flatten()(dense2)
model = Model([user_input, cat_input,words_input, date_input], fin)
opt = SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', 'mse'] )
model.summary()
import multiprocessing
_max=reordered['words_count'].max()
model.fit([users,reordered['category_id'],reordered['words_count'].apply(lambda x: x/_max),reordered['created_at_ts'].to_numpy().reshape( -1,1,1)] ,y, epochs=10, batch_size=128,  verbose=1,validation_split=0.2, workers=multiprocessing.cpu_count())