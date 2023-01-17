import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
import gc

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
f1 = '../input/data-science-for-good-kiva-crowdfunding'
f2 = '../input/currency-excahnge-rate'
f3 = '../input/countries-iso-codes'
f4 = '../input/undata-country-profiles'
#This file contains Alpha-2 and Alpha-3 codes for countries
codes = pd.read_csv(f3+'/wikipedia-iso-country-codes.csv')
locs = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
#This file contains the Currency Exchange Rates in year
cer = pd.read_csv(f2+'/currency_exchange_rate.csv')
mpi = pd.read_csv(f1+'/kiva_mpi_region_locations.csv')
cstas = pd.read_csv(f4+'/country_profile_variables.csv')
loans = pd.read_csv(f1+'/kiva_loans.csv')
loans['year'] = pd.to_datetime(loans.date).dt.year
#print(os.listdir(f1))
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1, 1, figsize=(15,6))
data={'Activities':len(loans.activity.unique()),
      'Sectors':len(loans.sector.unique()),
      'Countries':len(loans.country.unique()),
      'Currencies':len(loans.currency.unique())
      };
sns.barplot(y=list(data.keys()),x=list(data.values()),orient='h')
plt.show()
k = gc.collect()
df = pd.merge(loans,codes,left_on='country_code',right_on='Alpha-2 code',how='left')
k = gc.collect()
gdf = df.groupby(['country'],as_index=False).mean()
fig,axa = plt.subplots(1,1,figsize=(15,6))
sns.distplot(gdf[gdf['loan_amount']<100000].loan_amount)
plt.show()
k = gc.collect()
gdf = df.groupby(['country'],as_index=False).mean()

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
data = [ dict(
        type='choropleth',
        colorscale = 'Jet',
        autocolorscale = False,
        locations = gdf['country'],
        locationmode = 'country names',
        z = gdf.loan_amount,
        text = gdf['country'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Count")
        ) ]

layout = dict(
        title = 'Mean Loan Amounts in US Dollars',
        geo = dict(
            scope='world',
            showframe=False,
            projection=dict( type='Mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig)
k = gc.collect()
gdf = df.groupby(['country'],as_index=False).count()
data = [ dict(
        type='choropleth',
        colorscale = 'Jet',
        autocolorscale = False,
        locations = gdf['country'],
        locationmode = 'country names',
        z = gdf.id,
        text = gdf['country'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Count")
        ) ]

layout = dict(
        title = 'Number of Loans',
        geo = dict(
            scope='world',
            showframe=False,
            projection=dict( type='Mercator' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig)
k = gc.collect()
df2 = pd.merge(gdf[['country','id']],mpi)
df2 = df2[['MPI','id']].fillna(0)
sns.set(font_scale=1.5)
sns.jointplot(x = df2.MPI, y=df2.id, size=10)
plt.show()
k = gc.collect()
sectors = df.sector.unique();
max_loans=[]
min_loans=[]
mean_loans=[]
std_loans=[]

fig,axa = plt.subplots(2,2,figsize=(18,15))
for i in range(len(sectors)):
    max_loans.append(df[df['sector']==sectors[i]].loan_amount.max())
for i in range(len(sectors)):
    min_loans.append(df[df['sector']==sectors[i]].loan_amount.min())
for i in range(len(sectors)):
    mean_loans.append(df[df['sector']==sectors[i]].loan_amount.mean())
for i in range(len(sectors)):
    std_loans.append(df[df['sector']==sectors[i]].loan_amount.std())
    
axa[0][0].set_title('min loan amounts')
sns.barplot(y=sectors,x=min_loans,orient='h',ax=axa[0][0])
axa[0][1].set_title('max loan amounts')
sns.barplot(y=sectors,x=max_loans,orient='h',ax=axa[0][1])
axa[1][0].set_title('mean loan amounts')
sns.barplot(y=sectors,x=mean_loans,orient='h',ax=axa[1][0])
axa[1][1].set_title('std of loan amounts')
sns.barplot(y=sectors,x=std_loans,orient='h',ax=axa[1][1])
plt.show()
k = gc.collect()
coldf = df[df['Alpha-3 code']=='COL']
coldf = pd.merge(coldf,mpi,on='region')
coldf.groupby(['region'],as_index=False).mean()
k = gc.collect()
tmp = pd.concat([pd.get_dummies(df.sector),df[['region']]],axis=1)
tmp = tmp.groupby(['region'],as_index=False).sum()
tmp = pd.merge(tmp,mpi,on='region')
tmp.head()
k = gc.collect()
import math
import matplotlib
cmap = matplotlib.cm.get_cmap('jet')

data = []
layout = go.Layout(
    title = 'Sectors',
    showlegend = False,
    width=1000, height=1000,
    geo = dict(
            scope='world',
            showframe = False,
            #projection=dict( type = 'Mercator'),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),)

for i in range(len(sectors)):
    for j in range(len(tmp)):
        geo_key = 'geo'+str(i+1) if i != 0 else 'geo'
        # Year markers
        if tmp[sectors[i]][j] >0:
            data.append(
            dict(
            type = 'scattergeo',
            geo = geo_key,
            lon = [tmp['lon'][j]],
            lat = [tmp['lat'][j]],
            text = sectors[i]+str((tmp[sectors[i]][j])),
            #colorscale='Magma',
            marker = dict(
                size =math.log(tmp[sectors[i]][j] )*3,# (tmp[sectors[i]][j]/max(tmp[sectors[i]]))*9,
                color = 'rgb(0,0,200,0.5)',
                opacity = 0.5,
                line = dict(width=0.5, color='rgb(40,40,40)'),
                sizemode = 'diameter'
            ),)
            )
        
    layout[geo_key] = dict(
        scope = 'world',
        showland = True,
        showframe = False,
        landcolor = 'rgb(229, 229, 229)',
        showcountries = False,
        domain = dict( x = [], y = [] ),
        subunitcolor = "rgb(255, 255, 255)",
    )
    # Year markers
    data.append(
        dict(
            type = 'scattergeo',
            showlegend = False,
            lon = [28],
            lat = [-55],
            geo = geo_key,
            text = [sectors[i]],
            mode = 'text',
        )
    )
    
    
z = 0
COLS = 3
ROWS = 5
for y in reversed(range(ROWS)):
    for x in range(COLS):
        geo_key = 'geo'+str(z+1) if z != 0 else 'geo'
        layout[geo_key]['domain']['x'] = [float(x)/float(COLS), float(x+1)/float(COLS)]
        layout[geo_key]['domain']['y'] = [float(y)/float(ROWS), float(y+1)/float(ROWS)]
        z=z+1
        if z > 42:
            break
            
fig = { 'data':data, 'layout':layout}
iplot( fig )
k = gc.collect()
mpis = []
import matplotlib
cmap = matplotlib.cm.get_cmap('magma')
for i in range(len(mpi)):
    mpis.append(
        dict(
        type = 'scattergeo',
        #locationmode = 'world',
        lon = [mpi['lon'][i]],
        lat = [mpi['lat'][i]],
        text = mpi.country+' MPI:'+str(mpi['MPI'][i]),
        colorscale='Magma',
        marker = dict(
            size = 9,
            color = cmap(mpi['MPI'][i]),
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'diameter'
        ),)
        
    )

layout = go.Layout(
    title = 'MPI',
    showlegend = False,
    geo = dict(
            scope='world',
            #projection=dict( type = 'Mercator'),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),)

fig = dict( data=mpis, layout=layout ) #fig =  go.Figure(layout=layout, data=mpis)
iplot( fig, validate=False)
cols = list(sectors)
cols.append('MPI')
f,axa = plt.subplots(1,1,figsize=(15,10))
sns.heatmap(tmp[cols].corr().filter(['MPI']).drop(['MPI']))
plt.show()
import nltk
def get_nouns(data):
    use = data.use.isna().fillna('')
    use = [[k[0] for k in nltk.pos_tag(nltk.word_tokenize(str(data.use.iloc[i]))) if k[1] in ['NN','NNS'] ] for i in range(len(data))]
    return use
wv = "../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt"
embeddings_index = {}
f = open(wv)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
def get_vectors(data):
    embuse=[]
    for i in range(len(data)):
        emb = []
        for j in data[i]:
            if j in embeddings_index:
                emb.extend(embeddings_index[j])
        for j in range(len(data[i]),20):
            emb.extend([0]*100)
        embuse.append(emb)  
    return embuse
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
def get_clusters(data,n_comp,n_clust,c1,c2):
    pca = PCA(n_components=n_comp)
    pca.fit(np.asarray(data))
    components = pd.DataFrame(pca.transform(data))
    tocluster = pd.DataFrame(pd.concat([components.loc[:,c1], components.loc[:,c2]],axis=1))
    clusterer = KMeans(n_clusters=n_clust,random_state=42).fit(tocluster)
    centers = clusterer.cluster_centers_
    c_preds = clusterer.predict(tocluster)
    fig,axa = plt.subplots(1,2,figsize=(15,6))
    colors = ['orange','blue','purple','green','yellow','brown']
    colored = [colors[k] for k in c_preds]
    axa[0].plot(components.loc[:,c1], components.loc[:,c2], 'o', markersize=0.7, color='blue', alpha=0.5, label='class1')
    axa[1].scatter(tocluster.loc[:,c1],tocluster.loc[:,c2],  s= 2, color = colored)
    for ci,c in enumerate(centers):
        axa[1].plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))
    #plt.xlabel('x_values')
    #plt.ylabel('y_values')
    #plt.legend()
    #plt.show()
    return c_preds
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_clouds(data):
    fig,axa = plt.subplots(2,2,figsize=(15,6))
    for i in range(4):
        l = []
        for k in data.loc[data['cluster']==i].uselist:
            l.extend(k)
        wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(l))
        axa[int(np.floor(i/2))][i%2].imshow(wordcloud)
        axa[int(np.floor(i/2))][i%2].axis('off')
        axa[int(np.floor(i/2))][i%2].set_title("Cluster "+str(i))
    axa[1][1].axis('off')
    
plt.show()
def plotSunkey(data,cluster_labels,_title):
    c0 = tdf.columns[0]
    c1 = tdf.columns[1]
    c2 = tdf.columns[2]

    c2_labels = list(tdf[c2].unique())
    #c1_labels = list(tdf[c1].unique())
    c0_labels = list(tdf[c0].unique())

    c2_id = [c2_labels.index(k) for k in tdf[c2]]
    tdf.loc[:,c2]=c2_id
    c0_id = [c0_labels.index(k) for k in tdf[c0]]
    tdf.loc[:,c0]=c0_id

    hist_1 = tdf.groupby([c0,c1],as_index=False).count()

    _labels = list(c0_labels)
    _labels.extend(cluster_labels)
    _source = []
    _target = []
    _value = list(hist_1.id)

    _source = list(hist_1[c0])
    _target = list(hist_1[c1]+len(hist_1[c0].unique()) )

    hist_2 = tdf.groupby([c1,c2],as_index=False).count()
    _labels.extend(list(c2_labels))
    _source.extend(list(hist_2[c1]+len(hist_1[c0].unique())))
    _target.extend(list(hist_2[c2]+len(hist_1[c0].unique())+len(hist_2[c1].unique())))
    _value.extend(list(hist_2.id))
    
    data = dict(
    type='sankey',
    node = dict(
      #pad = 15,
      thickness = 20,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label = _labels,
      #color = ["blue"]*15
    ),
    link = dict(
      source = _source,
      target = _target,
      value = _value
  ))

    layout =  dict(
    title = _title,
    font = dict(
      #size = 20
    )
)

    fig = dict(data=[data], layout=layout)
    iplot(fig,validate=True)
west_africa = ['Niger','Mali','Maurithania','Senegal','Gambia','Burkina Faso',
               "Cote D'Ivoire",'Ghana','Benin','Liberia','Nigeria']
wadf  = df[df.country.isin(west_africa)]
genders = wadf.borrower_genders.fillna('')
def gender_check(genders_):
    if genders_ == '':
        return 0
    else:
        genders_ = genders_.split()
        male = genders_.count('male')
        female = genders_.count('female')
        if male >0 and female >0:
            return 3
        if female >0:
            return 2
        if male > 0:
            return 1
    
borrowercount = [len(genders.iloc[i].split()) for i in range(len(genders))]
_genders = [gender_check(genders.iloc[i]) for i in range(len(genders))]
wadf.loc[:,'borowers_count'] = borrowercount
wadf.loc[:,'gender_code'] = _genders
fig,axa = plt.subplots(1,2,figsize=(15,6))
ghist = wadf.groupby('gender_code',as_index=False).count()
sns.barplot(x=ghist.gender_code,y=ghist.id,ax=axa[0])
ghist = wadf[wadf.country=='Nigeria'].groupby('gender_code',as_index=False).count()
sns.barplot(x=ghist.gender_code,y=ghist.id,ax=axa[1])
axa[0].set_title('Western Africa Gender Distribution')
axa[1].set_title('Nigeria Gender Distribution')
plt.show()
use = get_nouns(wadf)
embuse= get_vectors(use)
embdf = pd.DataFrame(embuse)
embdf = embdf.fillna(0)
#get_clusters(data,n_comp,n_clust):
c_preds = get_clusters(embdf,5,4,1,3)
tmp = pd.DataFrame({'uselist':use,'cluster':c_preds})
show_clouds(tmp)

tdf = pd.DataFrame({'sector':wadf.sector,'cluster':c_preds,'country':wadf.country,'id':range(len(_genders))})
tdf = tdf[['sector','cluster','country','id']]

cluster_labels = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"]
plotSunkey(tdf,cluster_labels,'Western Africa')
northern_africa = ['Egypt','Morocco','Algeria','Libya','Tunisia']
nadf  = df[df.country.isin(northern_africa)]
use = get_nouns(nadf)
embuse= get_vectors(use)
embdf = pd.DataFrame(embuse)
embdf = embdf.fillna(0)
#get_clusters(data,n_comp,n_clust):
c_preds = get_clusters(embdf,5,4,1,3)
tmp = pd.DataFrame({'uselist':use,'cluster':c_preds})
show_clouds(tmp)
tdf = pd.DataFrame({'sector':nadf.sector,'cluster':c_preds,'country':nadf.country,'id':range(len(nadf))})
tdf = tdf[['sector','cluster','country','id']]

cluster_labels = ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"]
plotSunkey(tdf,cluster_labels,'Egypt')
eastern_africa = ['Burundi','Comoros','Djibouti','Eritrea','Ethiopia','Kenya','Madagascar','Malawi']
eadf  = df[df.country.isin(eastern_africa)]
use = get_nouns(eadf)
embuse= get_vectors(use)
embdf = pd.DataFrame(embuse)
embdf = embdf.fillna(0)
#get_clusters(data,n_comp,n_clust):
c_preds = get_clusters(embdf,5,4,1,3)
tmp = pd.DataFrame({'uselist':use,'cluster':c_preds})
show_clouds(tmp)


