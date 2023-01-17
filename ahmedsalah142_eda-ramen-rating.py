import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import PCA
%matplotlib inline
ramen_df = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')
print(ramen_df.shape)
ramen_df.head()
print(ramen_df['Review #'].nunique())
ramen_df.info()

ramen_df[ramen_df['Top Ten'].isnull() == False]['Top Ten'].unique()
ramen_df[ramen_df['Top Ten'].isnull() == False][['Review #','Top Ten']].sort_values(by='Review #')
ramen_df['Top Ten'].fillna('not ranked',inplace=True)
ramen_df.loc[ramen_df["Top Ten"] =="\n",'Top Ten'] = 'not ranked'

top_ramen = pd.DataFrame()
top_ramen = ramen_df[ramen_df["Top Ten"] != "not ranked"]
top_ramen['year'] = ramen_df[ramen_df["Top Ten"] != "not ranked"]["Top Ten"].str.extract(r'([0-9]+)')
top_ramen['rank'] = ramen_df[ramen_df["Top Ten"] != "not ranked"]["Top Ten"].str.extract(r'(#[0-9]+)').replace('#',' ',regex=True)
top_ramen.drop('Top Ten',inplace =True,axis=1)
top_ramen.head()
ramen_df['Stars'].unique()
ramen_df.dtypes
row = ramen_df.query('Stars != "Unrated"')['Stars'].astype('float')
plt.hist(row,bins=20);
ramen_df.query('Stars == "Unrated"').shape[0]
ramen_df.drop(ramen_df[ramen_df['Stars']=="Unrated"].index,inplace=True)
ramen_df['Stars'] = ramen_df['Stars'].astype('float')
category_cols = ['Brand','Variety','Style','Country']

for i,col in enumerate(category_cols):
    print(ramen_df[col].nunique())
color = sns.color_palette()[0]
sns.countplot(data=ramen_df,x='Style',color=color,order = ramen_df['Style'].value_counts().index);
plt.figure(figsize=(12,12))
sns.countplot(data=ramen_df,y='Country',color=color,order = ramen_df['Country'].value_counts().index);

ramen_df['Brand'].value_counts()[ramen_df['Brand'].value_counts() >30]
plt.figure(figsize=(12,12))
B_count = ramen_df['Brand'].value_counts()
B_count = B_count[B_count>10]
sns.barplot(x=B_count,y=B_count.index,color=color);

cols = ['Country', 'Brand', 'Style']
fig,axs = plt.subplots(3,1,figsize=(10,10))
for i,col in enumerate(cols):
    top = list(ramen_df[cols[i]].value_counts()[:10].index)
    
    df= ramen_df[ramen_df[col].isin(top)]
    sns.violinplot(data=df,x=cols[i],y='Stars',ax=axs[i],order=top);
    

cols = ['Country', 'Brand', 'Style']
fig,axs = plt.subplots(3,1,figsize=(10,10))
for i,col in enumerate(cols):
    top = list(ramen_df[cols[i]].value_counts()[:10].index)
    
    df= ramen_df[ramen_df[col].isin(top)]
    sns.boxplot(data=df,x=cols[i],y='Stars',ax=axs[i],order=top);
    

k = ramen_df.loc[:,'Stars'].groupby(np.arange(len(ramen_df))//30).mean()
plt.plot(k)
plt.ylim(0);
df = ramen_df
TopRamenCountris = ramen_df['Country'].value_counts()[:11]
df = ramen_df.query('Country in @TopRamenCountris.index')
df = df.groupby(['Country','Brand']).size().reset_index()
df = df.pivot('Brand','Country',0).fillna(0)
brands = df.sum(axis=1)[df.sum(axis=1)>50].index
sns.clustermap(df.loc[brands,:])
variety_words =set()
for index,row in ramen_df.iterrows():
    #print(row)
    word_list = row['Variety'].split()
    variety_words.update(word_list)
len(variety_words)

text = ramen_df['Variety'].unique()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(text)]
model = Doc2Vec(documents, vector_size=6, window=2, min_count=1, workers=4)
#Persist a model to disk:


fname = get_tmpfile("my_doc2vec_model")

model.save(fname)



fname = get_tmpfile("my_doc2vec_model")
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!
#If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
#Infer vector for a new document:

for index,row in ramen_df.iterrows(): 
    ramen_df.loc[index,['1','2','3','4','5','6']] = model.infer_vector(row['Variety'].split()) 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(ramen_df[['1','2','3','4','5','6']])
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf['Stars'] = ramen_df['Stars']
plt.scatter(data=principalDf,x='principal component 1',y='principal component 2',c=principalDf['Stars'],alpha=5/10)
plt.colorbar()
TopRamenCountris = ramen_df['Country'].value_counts()[:11]
TopRamenCountris
df = ramen_df.query('Country in @TopRamenCountris.index')
k = df.loc[:,['Country','Stars']].groupby([np.arange(len(df))//40,'Country']).mean()
k = k.reset_index()
k

d = k.pivot("Country", "level_0", "Stars").fillna(0)
d = d.drop(55,axis=1)
sns.clustermap(d,col_cluster=False)
TopRamenBrands = ramen_df['Brand'].value_counts()[:11]
df = ramen_df.query('Brand in @TopRamenBrands.index')
k = df.loc[:,['Brand','Stars']].groupby([np.arange(len(df))//40,'Brand']).mean()
k = k.reset_index()
d = k.pivot("Brand", "level_0", "Stars").fillna(0)
#d = d.drop(55,axis=1)
sns.clustermap(d,col_cluster=False)
TopRamenBrands = ramen_df['Brand'].value_counts()[:10]
df = ramen_df.query('Brand in @TopRamenBrands.index')
k = df.loc[:,['Brand','Stars']].groupby([np.arange(len(df))//40,'Brand']).mean()
k = k.reset_index()
g= sns.FacetGrid(data=k,col='Brand',col_order=TopRamenBrands.index,aspect=2,col_wrap=3);

g.map(plt.plot,'level_0','Stars');
