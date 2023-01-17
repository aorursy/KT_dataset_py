import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import missingno as msno
from pprint import pprint
# plotly
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
py.init_notebook_mode(connected=True)
init_notebook_mode(connected=True)
import plotly.graph_objs as gobj

# word cloud library
from wordcloud import WordCloud
import math 
# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
print(os.listdir("../input"))
movie = pd.read_csv('../input/film-rate/movie_metadata.csv')
#fonction multi distrib
#fonction qui affiche, par lignes de 4, le subplot correspondant à chaque elements de la liste de label
#la fonction prend une liste de string et un Dataframe en parametre puis renvoie une Dataframe
#Fonctionnement : elle clean le data en supprimant les colonne ayant moins de 200 sample car pas exploitable
#ensuite elle supprime les Nan puis
#elle prends le premier elements de la liste Label, affiche le suplots correpondant
#puis supprime le premier element de Label, la fonction s'arrete quand Label et vide
#List(string)* pd.data -> pd.data
def multi_distrib(label,Data, typeg ='hist'):
    height = math.ceil(len(label)/4)
    fig = plt.figure(figsize=(16, height*4))
    index =0
    d= colonne_list(label,Data)
    print(d.shape[1])
    for feat_idx in range(d.shape[1]):
        data = d.iloc[:, feat_idx].dropna()
        ax = fig.add_subplot(height,4, (index+1))
        if typeg == 'hist':
            ax.hist(data, bins=50, color='steelblue',normed=True, edgecolor='none')
        if typeg == 'boxplot':
            ax.boxplot(data)
        ax.set_title(d.columns[feat_idx], fontsize=14)
        index = index + 1
def colonne_list(StringList,Data):
    indexList = []
    for label in StringList:
        indexList.append(Data.columns.get_loc(label))
    return Data.iloc[:,indexList]

#fonction pour selectionner un intervalle de colonne
#prend deux label de colonne et renvoie le data correspondant à toutes les colonnes de cet interval inclusif
#couple[string]*pd.data ->pd.data
def inter_list(couple, Data):
    #comme les label nutrifacts ne sont pas dans le bonne ordre on les compare
    a=Data.columns.get_loc(couple[0])
    b=Data.columns.get_loc(couple[1])
    if b>a:
        return Data.iloc[:,Data.columns.get_loc(couple[0]): Data.columns.get_loc(couple[1])+1 ]
    else:
        return Data.iloc[:,Data.columns.get_loc(couple[1]): Data.columns.get_loc(couple[0])+1 ]
    
import math as mt
def multi_plot(label,Data):
    height = mt.ceil(len(label)/4)
    fig = plt.figure(figsize=(16, height*4))
    index =0
    d= colonne_list(label,Data)
    print(d.shape[1])
    #for feat_idx in range(d.shape[1]):
    for feat_idx in d.columns:
        if d[d[feat_idx].notnull()].shape[0] >= 20000:
            data = d.loc[:, feat_idx].dropna()
            if 'nutrition-score' in feat_idx:
                pass
            else:
                ax = fig.add_subplot(height,4, (index+1))
                ax.hist(data, bins=20)
                ax.set_title(feat_idx, fontsize=14)
                #ax.set_title(d.columns[feat_idx], fontsize=10)
                index = index + 1
                
def plot_value_counts(df,col_name,table=False,bar=False, limit_percent=5):
    #we don't want more than 31 bars
    if len(pd.DataFrame(df[col_name].dropna().value_counts())) > 31:
        max = 31
    else:
        max = len(pd.DataFrame(df[col_name].dropna().value_counts()))
        
        
    values_count = pd.DataFrame(df[col_name].dropna().value_counts()[:max])
    values_count.columns = ['count']
    # convert the index column into a regular column.
    values_count[col_name] = [ str(i) for i in values_count.index ]
    # add a column with the percentage of each data point to the sum of all data points.
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    # change the order of the columns.
    values_count = values_count.reindex([col_name,'count','percent'],axis=1)
    values_count.reset_index(drop=True,inplace=True)
    
    if bar :
        # add a font size for annotations0 which is relevant to the length of the data points.
        font_size = 20 - (.25 * len(values_count[col_name]))
        
        trace0 = gobj.Bar( x = values_count[col_name], y = values_count['count'] )
        data_ = gobj.Data( [trace0] )
        
        #total quantity showed on top of bars
        #annotations0 = [ dict(x = xi,
        #                     y = yi, 
        #                     showarrow=False,
        #                     font={'size':font_size},
        #                     text = "{:,}".format(yi),
        #                     xanchor='center',
        #                     yanchor='bottom' )
        #               for xi,yi,_ in values_count.values ]
        
        #percentage showed inside the bars
        annotations1 = [ dict( x = xi,
                              y = yi,
                              showarrow = False,
                              text = "{}%".format(int(pi)),
                              xanchor = 'center',
                              yanchor = 'bottom',
                              font = {'color':'purple', 'size':font_size})
                         for xi,yi,pi in values_count.values if pi >= limit_percent ]
        
        annotations =  annotations1   # annotations0 +                    
        
        layout = gobj.Layout( title = col_name.replace('_',' ').capitalize(),
                             titlefont = {'size': 50},
                             yaxis = {'title':'count'},
                             xaxis = {'type':'category'},
                            annotations = annotations  )
        figure = gobj.Figure( data = data_, layout = layout )
        py.iplot(figure)
    
    if table : 
        values_count['count'] = values_count['count'].apply(lambda d : "{:,}".format(d))
        table = ff.create_table(values_count,index_title="race")
        py.iplot(table)
    

    

def keepfour(label,tag,data):
    print('label',label)
    #labelt = label[0:4] 
    #print(labelt)
    temp = data
    for el in label:
        data2=temp[temp[tag] != el]
        temp = data2
    return data2   

def sort_four( tag, data ):
    dicmax={}
    lismax=[]
    listpays=[]
    label = data[tag].unique()
    for l in label :
        total = data[data[tag]== l ].loc[:,['count']].sum()
        lismax.append(total[0])
        #listpays.append(l)
        dicmax[total[0]]= l
    lismax = sorted(lismax, reverse=True)
    #print(lismax)
    lispays = [ dicmax[e] for e in lismax]
    #print(lispays)
    lispays = lispays[0:4]
    return lispays

def graph_pie(pays,tag,data_c,title="Titre"):
    size = 15 #font size
    label = ["a","b","c","d","e"]#legend
    hole =.5 #size hole
    listData=[]
    listAnot=[]
    fig={}
    layout={}
    dict1={}
    dict2={}
    dicfont ={}
    coor = [{"x": [0, .52],'y': [.51, 1]},{'x': [.51, 1],'y': [.51, 1]},
            {'x': [0, .52],'y': [0, .48]},{'x': [.52, 1],'y': [0, .48]} ]
    coora =[[0.218, 0.8] , [0.785, 0.8] , [0.23, 0.218] , [0.80 ,0.218]]
    
    data1={}
    data2={}
    data3={}
    data4={}
    lay1={}
    lay2={}
    lay3={}
    lay4={}
    datadic =[data1,data2,data3,data4]
    layoutdic =[lay1,lay2,lay3,lay4]
    dicfont["size"] =size
    data=data_c.sort_values('nutrition_grade_fr', ascending = True)
    dicfont["family"] = "Open Sans, sans-serif"
    for i  in range(len(pays)):
        dict1 =datadic[i]
        dict2 =layoutdic[i]
        dict1["values"]= list(data[data[tag] == pays[i]]['count'])
        dict1["labels"]= label
        dict1["domain"]= coor[i]
        dict1["name"]= pays[i]
        dict1["hoverinfo"] = "label+percent+name"
        dict1["hole"] = hole
        dict1["type"] = "pie"
        #print(dict1)
        dict2["font"]= dicfont
        dict2["showarrow"]= False
        dict2["text"]= pays[i]
        dict2["x"]= coora[i][0]
        dict2["y"]= coora[i][1]
        listAnot.append(dict2)
        listData.append(dict1)
        #print("i=",i," ",listData)
    layout["title"]= title
    layout["annotations"]= listAnot
    fig["layout"]= layout
    fig["data"]= listData
    return fig

def graph_pie_one(data_c, title="Titre",texte="center"):
    size = 15 #font size
    label = ["a","b","c","d","e"]#legend
    hole =.5 #size hole
    listData=[]
    listAnot=[]
    fig={}
    layout={}
    dict1={}
    dict2={}
    dicfont ={}
    

    dicfont["size"] =size
    data=data_c.sort_values('nutrition_grade_fr', ascending = True)
    dicfont["family"] = "Open Sans, sans-serif"

    dict1["values"]= list(data['count'])
    dict1["labels"]= label
    dict1["name"]= "grade"
    dict1["hoverinfo"] = "label+percent+name"
    dict1["hole"] = hole
    dict1["type"] = "pie"
        #print(dict1)
    dict2["font"]= dicfont
    dict2["showarrow"]= False
    dict2["text"]= texte
    listAnot.append(dict2)
    listData.append(dict1)
        #print("i=",i," ",listData)
    layout["title"]= title
    layout["annotations"]= listAnot
    fig["layout"]= layout
    fig["data"]= listData
    return fig

def auto_pie(tag, data, limit=30, title ='title'):
    ct4 = data.groupby([tag,'nutrition_grade_fr']).size().reset_index(name='count')
    ct5 = ct4[ct4['count']>limit]
    lisp2 = check_grade(tag,ct5)
    data_c2 = keepfour(lisp2,tag,ct5)
    manu_pie = sort_four(tag, data_c2 )
    fig2 =graph_pie(manu_pie,tag, data_c2, title )
    py.iplot(fig2, filename='donut')
msno.dendrogram(movie)
msno.bar(movie)
float_l=[]
for l in movie.columns:
    if type(movie[l][0])==np.float64:
        float_l.append(l)
        
multi_distrib(float_l,movie, typeg ='boxplot')
movie.loc[movie['budget'].idxmax(),['budget','movie_title','title_year']]
movie.loc[2988,['budget']]=11000000
movie[movie["duration"]>=500].loc[:,['movie_imdb_link','movie_title','duration']]
#movie = movie.drop(1710,0)
for l in float_l[:-1]:
    print(movie.loc[movie[l].idxmax(),[l,'movie_title','title_year']])
    print(' ')
movie.loc[movie['actor_1_facebook_likes'].idxmax(),['actor_1_facebook_likes','actor_1_name','movie_title','title_year']]
movie.loc[movie['actor_2_facebook_likes'].idxmax(),['actor_2_facebook_likes','actor_2_name','movie_title','title_year']]
movie.loc[1902,['actor_1_facebook_likes']]= np.nan
movie.loc[1223,['actor_2_facebook_likes']]= np.nan
movie = movie.drop('budget',1)
movie = movie.drop('gross',1)
str_l=[]
for l in movie.columns:
    if type(movie[l][0])==str:
        str_l.append(l)
        movie[l]= movie[l].str.lower()
        
str_l    
for l in str_l:
    movie[l] = movie[l].str.strip()
    plot_value_counts(movie,l,table=False,bar=True, limit_percent=3)
type(movie.loc[5011,'content_rating'])
#movie[movie['content_rating']=='r'].loc[:,'content_rating']
movier1 = movie.copy()
for index , item in movier1.iterrows():
    e = movier1.loc[index, 'content_rating']
    if e == 'tv-14':
        movier1.loc[index, 'content_rating']= 'pg-13'
    elif e == 'tv-pg':
        movier1.loc[index, 'content_rating']='pg'
    elif e == 'tv-g':
        movier1.loc[index, 'content_rating']='g'
    elif e == 'tv-ma':
        movier1.loc[index, 'content_rating']='r'
    elif e == 'x':
        movier1.loc[index, 'content_rating']='nc-17'
    elif e == 'tv-y7':
        movier1.loc[index, 'content_rating']='tv-y'
    elif e not in ('nc-17','r','pg-13','pg','g','tv-y'):
        movier1.loc[index, 'content_rating']= np.nan
movier1['content_rating'].value_counts()
movier1.columns
plot_value_counts(movier1,'content_rating',table=False,bar=True, limit_percent=3)
movie[movie['actor_2_name']=='will smith']
movie.shape
movie = movier1
movie.content_rating.value_counts()
movie = movie.drop_duplicates(subset=['movie_title', 'title_year', 'movie_imdb_link'])
#movie['genres'] = movie['genres'].map(lambda x: x.split('|'))
def plotlist(label, data):
    d= dict()
    totale = 0
    x =[]
    y =[]
    tup =[]
    for liste in movie[movie[label].notnull()].loc[:,label]:
        for e in liste.split('|'):
            totale +=1
            if e in d:
                d[e] +=1
            else:
                d[e] =1
    for key, value in d.items():
        d[key]= value
        if d[key]>0.000:
            tup.append([key,d[key]])
    tup =sorted(tup, key =lambda x: x[1], reverse=True)
    for i in tup:
        x.append(i[0])
        y.append(i[1])
    return [x,y]
x,y =plotlist('genres',movie)

movie['plot_keywords'].sample(5)
for e in movie[movie['plot_keywords'].notnull()].loc[:,'plot_keywords']:
    if type(e)==float:
        print(e)
x1,y1=plotlist('plot_keywords',movie)
def plot_value_counts2(X,Y,title ='title', limit_percent=5, limbar=5):
    #we don't want more than 31 bars
    if len(Y) > 31:
        max = 31
    else:
        max = len(Y)
    Y=Y[:max]
    font_size = 20 - (.35 * len(Y))
    #percentage showed inside the bars
    #values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    P= map(lambda x : round((x/sum(Y))*(100),2), Y)

    P = [p for p in P if p >= limbar]

    Y=Y[:len(P)]
    X=X[:len(P)]
    trace0 = gobj.Bar( x = X, y = Y )
    data_ = gobj.Data( [trace0] )
    annotations1 = [ dict( x = xi,
                          y = yi,
                          showarrow = False,
                          text = "{}%".format(int(pi)),
                          xanchor = 'center',
                          yanchor = 'bottom',
                          font = {'color':'purple', 'size':font_size})
                        for xi,yi,pi in zip(X,Y,P)if pi>= limit_percent ]
        
    annotations =  annotations1   # annotations0 +                    
        
    layout = gobj.Layout( title = title.replace('_',' ').capitalize(),
                         titlefont = {'size': 50},
                         yaxis = {'title':'count'},
                         xaxis = {'type':'category'},
                         annotations = annotations1  )
    figure = gobj.Figure( data = data_, layout = layout )
    py.iplot(figure)
    
plot_value_counts2(x,y,title ='genres', limit_percent=1, limbar =1)
plot_value_counts2(x1,y1,title ='plot_keywords', limit_percent=0.2, limbar=0.2)
#movie[movie['genres'].notnull() & movie['genres'].str.contains('documentary')].loc[:,['movie_title', 'director_name','genres']].sample(10)
movie[ movie['movie_title'].str.contains('star wars')].loc[:,['movie_title', 'director_name','genres']]
#movie['genres'][4]= 'Documentary'
#movie[ movie['movie_title'].str.contains('Star Wars')].loc[:,['movie_title', 'director_name','genres']]
movie['movie_title']=movie['movie_title'].map(lambda x: x.strip())
duplicate= movie[movie.duplicated(subset='movie_title')].movie_title.values
for d in duplicate:
    #print(d)
    to_clean = movie[movie['movie_title']==d]
    #print(to_clean['movie_title'])
    for index, t in to_clean.iterrows():
        #print('title',t['movie_title'], 'year', t['title_year'])
        if pd.notnull(t['title_year']):
            movie['movie_title'][index]= t['movie_title']+' ('+str(round(t['title_year']))+')'
#movie[movie.duplicated(subset='movie_title')].movie_title.values
#movie = movie.reset_index(drop=True)
movie.shape
#movieEtape1.columns
movieEtape1 = movie.copy()
movieEtape1.content_rating.value_counts()
movie1 = movieEtape1.copy()
def one_hot(movie2):
    #movie2['plot_keywords'] = movie2['plot_keywords'].fillna('')
    #movie2['genres'] = movie2['genres'].fillna('')
    unique_words = set()
    for wordlist in movie2.plot_keywords.str.split('|').values:
        if wordlist is not np.nan:
            unique_words = unique_words.union(set(wordlist))
    plot_wordbag = list(unique_words)
    for word in plot_wordbag:
        movie2['pkw_' + word.replace(' ', '-')] = movie2.plot_keywords.str.contains(word).astype(float)
    #movie2 = movie2.drop('plot_keywords', axis=1)    


    unique_genre_labels = set()
    for genre_flags in movie2.genres.str.split('|').values:
        if wordlist is not np.nan:
            unique_genre_labels = unique_genre_labels.union(set(genre_flags))
    for label in unique_genre_labels:
        movie2['grs_'+label.replace(' ', '-')] = movie2.genres.str.contains(label).astype(int)
    #df = df.drop('genres', axis=1)

one_hot(movie1)

movie1 = movie1.reset_index(drop=True).copy()
movie1.shape
#percent = list(map(lambda x : round((x/sum(y))*(100),4), y))
#percent =[e for e in percent if e >0.0071]
#t = x[len(percent):]
#[e for e in movie1.columns if 'grs_' in e]
movie0 = movie1.copy()
movie0.shape
movie0.content_rating.value_counts()
#for e in t:
#    elem = 'grs_'+e
#    movie0 = movie0.drop(elem,1)
#print(movie0.shape)
#seuil = movie0['language'].value_counts()
#seuil = seuil[seuil ==1]
#s =seuil.index
#for index , e in movie0.iterrows():
 #   if e['language'] in s:
  #      movie0.loc[index,'language']= np.nan
#movie0['language'].value_counts()
#t1 = x1[len(percent1):]
#len(t1)
movie2 = movie1.copy()
movie2.content_rating.value_counts()
multi_distrib(['title_year','cast_total_facebook_likes'],movie2, typeg ='boxplot')
#movie2 =movie2.drop('budget',1)
movieEtape2 = movie2.copy()
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math
   # ly = LabelEncoder()
   # movie2['plot_keywords'] = movie2['plot_keywords'].fillna('NaN')
   # ly.fit(movie2['plot_keywords'].unique())
   # movie2['plot_keywords'] = ly.transform(movie2['plot_keywords'])
   # movie2['plot_keywords'] = movie2['plot_keywords'].replace(ly.transform(['NaN']) , np.nan)
#lx = LabelEncoder()
#lx.fit(movie['genres'].unique())
#movie2['genres'] = lx.transform(movie2['genres'])
movie2.shape
movie2= movie2.dropna(subset=['director_name'])
movieEtape2[movieEtape2['director_name'].isnull()]
movie2.shape
reverse_la = ['actor_1_name','actor_2_name','actor_3_name']
movie2['dirc_director'] = movie2.director_name.map(movie2.director_name.value_counts())
counts = pd.concat([movie2[l] for l in reverse_la]).value_counts()
#counts.head()
movie2['act_1'] = movie2.actor_1_name.map(counts)
movie2['act_2'] = movie2.actor_2_name.map(counts)
movie2['act_3'] = movie2.actor_3_name.map(counts)
from scipy import stats
sns.set(style="darkgrid")
u=np.array(movie2['director_facebook_likes'])
m=np.array(movie2['dirc_director'])
d= pd.DataFrame({'director_facebook_likes': u, 'dirc_director':m})

sns.despine(left=True)

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
b=sns.JointGrid(y='director_facebook_likes',x='dirc_director',data=d,height =10)
b = b.plot_joint(plt.scatter, color="g", s=40, edgecolor="white")
b.set_axis_labels('director_name', 'director_facebook_likes', fontsize=30)
b = b.plot_marginals(sns.distplot, kde=False, color="g")
b = b.annotate(stats.pearsonr, fontsize =20)
from scipy.stats.stats import pearsonr
movie2['director_facebook_likes'].corr(movie2['dirc_director'])
movie2.loc[:,['director_facebook_likes','dirc_director']].corr(method='pearson',min_periods=0)
#ridley scoot est un realisateur extrèmement connue
movie2[movie2['director_name']== 'ridley scott'].loc[:,['director_name','director_facebook_likes']]

movie2['dirc_director'] = movie2['dirc_director'].apply(lambda x : math.log(x+1) if x !=np.nan else x )
movie2['act_1'] = movie2['act_1'].apply(lambda x : math.log(x+1) if x !=np.nan else x )
movie2['act_2'] = movie2['act_2'].apply(lambda x : math.log(x+1) if x !=np.nan else x )
movie2['act_3'] = movie2['act_3'].apply(lambda x : math.log(x+1) if x !=np.nan else x )
#movie2.director_name.value_counts()
movie2['director_name'][0]
movie2['act_1'].head(10)
movie2.language.head()
encode_l=[]
label_enc=['content_rating','country','color','language']
diclab_enc={}


for i in range(len(label_enc)):
    diclab_enc[label_enc[i]]=i
    encode_l.append(LabelEncoder())
    encode = encode_l[i]
    label  = label_enc[i]
    encode.fit(movie2[label].fillna('NaN').unique())
    movie2[label] = movie2[label].fillna('NaN')
    movie2[label] = encode.transform(movie2[label])
    movie2[label] = movie2[label].replace(encode.transform(['NaN']) , np.nan)

title_enc = LabelEncoder()
title_enc.fit(movie2.movie_title)
movie2.movie_title = title_enc.transform(movie2.movie_title)

#color_enc = LabelEncoder()
#color_enc.fit(movie2.color.fillna('NaN').unique())
#movie2.color=movie2.color.fillna('NaN')
#movie2.color = color_enc.transform(movie2.color)
#movie2.color = movie2.color.replace(color_enc.transform(['NaN']) , np.nan)

#l_actor=['actor_1_name','actor_2_name','actor_3_name']
#all_actor =pd.concat([movie2[l] for l in l_actor])
#actor_enc = LabelEncoder()
#actor_enc.fit(all_actor.fillna('NaN').unique())
#movie2.loc[:,l_actor] = movie2.loc[:,l_actor].fillna('NaN')
#for e in l_actor:
#    movie2[e] = actor_enc.transform(movie2[e])
#    movie2[e] = movie2[e].replace(actor_enc.transform(['NaN']) , np.nan)

#movie02.color.str.strip().unique()
encode_l[diclab_enc['color']].transform(['NaN'])
movie2.select_dtypes(include=['O']).columns
movie2.shape

movie3 = movie2.copy()
movie4 = movie3.drop(['genres','plot_keywords','movie_imdb_link'],1)
movie4.content_rating.value_counts()
movie4= movie4.drop(['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','director_facebook_likes'],1)
movie4= movie4.drop(['aspect_ratio','facenumber_in_poster','num_voted_users'],1)
movie4= movie4.drop(['actor_1_name','actor_2_name','actor_3_name','director_name'],1)
movie4.columns[:20]
# We'll treat fillna as a regression / classification problem here.
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def reg_class_fill(df, column, classifier):
# on separe la colonne ou on veut retrouver
    ndf = df.dropna(subset=[col for col in df.columns if col != column])
    nullmask = ndf[column].isnull()
    train, test  = ndf[~nullmask], ndf[nullmask]
    train_x, train_y = train.drop(column, axis=1), train[column]
    classifier.fit(train_x, train_y)
    if len(test) > 0:
        test_x, test_y = test.drop(column, axis=1), test[column]
        values = classifier.predict(test_x)
        test_y = values
        new_x, new_y = pd.concat([train_x, test_x]), pd.concat([train_y, test_y])
        newdf = new_x[column] = new_y
        return newdf
    else:
        return ndf
r, c = KNeighborsRegressor, KNeighborsClassifier
#impute_order = [('director_name', c), ('title_year', r),
#                ('actor_1_name', c), ('actor_2_name', c), ('actor_3_name', c),('director_facebook_likes', c),
#                ('gross', r),('actor_1_facebook_likes', c),('actor_2_facebook_likes', c),('actor_3_facebook_likes', c),
#                ('content_rating', c),('country',c),('aspect_ratio',r)]
impute_order = [('title_year', c),
                ('act_1', r), ('act_2', r), ('act_3', r),
                ('content_rating', c),('country',c),('duration',r),('imdb_score',r)]
for col, classifier in impute_order:
    movie4 = reg_class_fill(movie4, col, classifier())
    print(col, 'Done')
movie4.shape
df =movie4[movie4.columns].isnull().sum()
df[df!=0]
#reverse_la = ['actor_1_name','actor_2_name','actor_3_name']

#for a in reverse_la:
#    movie4[a] = actor_enc.inverse_transform(movie4[a].astype(int))
movie4.movie_title = title_enc.inverse_transform(movie4.movie_title)
reverse_la2=['content_rating','country','color','language']
for a in reverse_la2:
    movie4[a] = encode_l[diclab_enc[a]].inverse_transform(movie4[a].astype(int))
movie4.content_rating.value_counts()
#act_ct = pd.concat([movie4[l] for l in reverse_la]).value_counts()
#act_ct[act_ct >1].index
#movie4['actor_1_name'].sample(5)
#pd.concat([movie4[l] for l in reverse_la]).str.contains('jon lovitz')
#len(act_ct[act_ct >1].index)
#len(act_ct)
movieEtape3 = movie4
movieEtape3.shape
movie5= movieEtape3.copy()
ert =movie5.language.value_counts()
ert[ert >1].index
onehot_l =['color','content_rating','language']
pref_lab =[]
def one_hot_enc(label, data):

    for l in label:
        prefix=''
        prefix = l[0]+l[1]+l[2]+'_'
        pref_lab.append(prefix)
        act_ct = data[l].value_counts()
        unique_seuil =act_ct[act_ct >1].index
        #print(unique_labels)
        for a in unique_seuil:
            data[prefix+a.replace(' ','-')] = data[l].str.contains(a).astype(int)

one_hot_enc(onehot_l,movie5)
#movie5.groupby(['director_name','director_facebook_likes'])['movie_title'].count().sort_values(ascending=False).head(5)
#act_ct2[act_ct2 >1].index
pref_all = pref_lab
movie5.columns
movie5.shape
movie6 = movie5.copy()

pref_lab2=[]
def one_hot_act(label, data):
    dfactor=[]
    p=label[0]
    print(p)
    prefix=''
    prefix = p[0]+p[np.random.randint(1,len(p)-1)]+p[np.random.randint(1,len(p)-1)]+'_'
    print(prefix)
    pref_lab2.append(prefix)
    print(pref_lab2)
    act_ct = pd.concat([data[l] for l in label]).value_counts()
    for index, item in data.iterrows():
        actor=''
        for l in label:
            if actor=='':
                actor=data.loc[index,l]
            else:
                actor+='|'+data.loc[index,l]
        dfactor.append(actor)
    data['all_actor']=dfactor
    #print(data['all_actor'].shape)
    #print(data['all_actor'])
    unique_seuil =act_ct[act_ct >1].index
    #print(unique_labels)
    for a in unique_seuil:
        data[prefix+a.replace(' ','-')] = data['all_actor'].str.contains(a).astype(int)
movie6.shape
#act_ct = pd.concat([movieEtape4[l] for l in reverse_la]).value_counts()
#act_ct[act_ct >1]
#one_hot_act(reverse_la, movie6)
pref_all= pref_all + pref_lab2
movieEtape4 = movie6
movie7 = movieEtape4.copy()
movie7.shape
movie7 =movie7.reset_index(drop=True)
multi_distrib(['title_year'],movie7, typeg ='boxplot')
multi_distrib(['title_year'],movie7, typeg ='hist')
from collections import Counter
#seul title year est interessant
title_class=['1','2','3','4','5']
new_year=[]
for i in movie7.title_year:
    if i <2020 and i >=1998:
        new_year.append(title_class[0])
    elif i <1998 and i >=1978:
        new_year.append(title_class[1])
    elif i <1978 and i >=1958:
        new_year.append(title_class[2])
    elif i <1958 and i >=1938:
        new_year.append(title_class[3])
    elif i <1938:
        new_year.append(title_class[4])
#pas d'erreur avec counter si la clé n'est pas présente
Counter(new_year)
        
Counter(new_year).items()
movie7['periode']=new_year
movie8= movie7.copy()
movie8.shape
for a in title_class:
    movie8['yer_'+a.replace(' ','-')] = movie8['periode'].str.contains(a).astype(float)
movie8 =movie8.drop('periode',1)
movie8.shape
pref= pref_all.copy()
pref.extend(['yer_','pkw_','grs_','dirc_','act_'])
pref

label_all = onehot_l.copy()
#label_all.extend(['actor_name','title_year','plot_keywords','genres'])
label_all.extend(['title_year','plot_keywords','genres','director','actor'])
print(label_all)
d ={}
for i in range(len(pref)):
    d[label_all[i]]=[pref[i]]
d
dlf = pd.DataFrame(data=d)
dlf.to_csv("labelonehot.csv")
movieEtape5 = movie8.copy()
movie9 = movieEtape5
movie9.columns[:27]


drop_l =['color', 'num_critic_for_reviews', 'duration',
         'cast_total_facebook_likes', 'num_user_for_reviews', 'language', 'country',
       'content_rating', 'title_year', 'movie_facebook_likes']
for l in drop_l:
    movie9 = movie9.drop(l,1)
    
movie9.columns[:20]

movie9.shape
movie10 = movie9.copy()
movieEtape6 = movie10
movie10.to_csv("finalmovieCleaned.csv")
#from random import uniform
#import time
#from IPython.display import display, clear_output
#
#def black_box():
#    i = 1
#    n =100
#    nbar=42
#    while i<(n+2):
#        clear_output(wait=True)
#        if i<(n+1):
#            viewb=''
#            percent =i*100/n
#            bar = round(percent*nbar/100)
#            viewb+='|'*bar
#            viewb+=' '*(nbar-bar)
#            viewb+='|--'
#        if i == n:
#            viewb+='|--Done--|'
#        display(str(round(percent))+'% : '+viewb)
#        time.sleep(0.1)
#        i+=1
#black_box()
#import scipy as sc
#def data_sparsMatrix(Data):
#    D = Data.values
#    n,m = D.shape
#    D0 = np.ones((n,1), dtype='O')
#    t = list(range(n))
#    for i in t:
#        D0[i][0]= i
#    Dnew = np.hstack((D0,D))
#    i,j,data=[],[],[]
#    
#    size = Dnew.shape[1]
#    #start = time.time()
#
#    
#    for col in range(1,size):
#        e=col
#        i.extend(Dnew[:,0])
#        j.extend(np.zeros(Dnew.shape[0],int)+(col-1))
#        data.extend(Dnew[:,col])
#    #timer = round(time.time() - start ,2)
#    M=sc.sparse.csr.csr_matrix((data,(i,j)),shape=(n,m))
#    #print(" %s sec"%timer)
#    return M
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import linear_kernel
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#def get_recommendations(title, data, cosine_sim):
#    idx = indices[title]
#    print('idx',idx)
#
#    sim_scores = list(enumerate(cosine_sim[idx]))
#
#    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#    sim_scores = sim_scores[0:11]
#    movie_indices = [i[0] for i in sim_scores]
#    return data['movie_title'].iloc[movie_indices]
#import math
##on veut retrouver la liste des colonne enfonction du préfix.
#def recom_col(lab, assignw, data, label = dlf):
#
#    
#    nbdoc = data.shape[0]
#    #tfid
#    for tfid in lab:
#        x = label[tfid][0]
#        keycolumns= pd.Series(data.columns)[pd.Series(data.columns).str.contains(x)].values
#
#        for col in keycolumns:
#            #print(col)
#            dt= data[col].sum()
#            data[col]= data[col].apply(lambda x : math.log(x* nbdoc/dt) if x > 0 else x)
#    #ponderation hyper param
#    for el in assignw:
#        y = label[el[0]][0]
#        #print(y)
#        key2columns= pd.Series(data.columns)[pd.Series(data.columns).str.contains(y)].values
#
#        for col2 in key2columns:
#            #print(col2)
#            data[col2]= data[col2].apply(lambda x : x*el[1] if x > 0 else x)
#            #if 'yer' in col2 :
#                #print(data[data[col2]>= 1].loc[:,col2])
#    M1= data_sparsMatrix(data)
#    M1.eliminate_zeros()
#    return cosine_similarity(M1, M1)
#movieEtape1.columns[:31]
#dlf
#def new_cosine(EtapeData, tfid =['plot_keywords'],surate = [[ 'genres',4]],indexcol = '' ):
#    data = EtapeData
#    data = data.dropna(thresh=100)
#    data = data.reset_index(drop=True)
#
#    if indexcol == '':
#        data0 = data
#    else:
#        where = data.columns.get_loc(indexcol)
#        data0 = data.iloc[:,where+1:]
#    data0 = data0.replace(np.nan,0)
#    return recom_col(tfid,surate, data0)
#movieEtape2.shape
#new_cosine1 = new_cosine(movieEtape2)
#indices = pd.Series(movieEtape2.index, index=movieEtape2['movie_title'])
#get_recommendations(movieEtape2['movie_title'][0],movieEtape2, new_cosine1)
#etape =movieEtape3.copy()
#etape.shape
#new_cosine1 = new_cosine(etape)
#indices = pd.Series(etape.index, index=etape['movie_title'])
#get_recommendations(etape['movie_title'][0],etape, new_cosine1)
#etape =movieEtape4.copy()
#etape = etape.reset_index(drop=True)
#etape.shape
#new_cosine1 = new_cosine(etape, tfid =['plot_keywords'],surate = [[ 'genres',6],[ 'title_year',6],
#                                                                  [ 'actor',0.1],[ 'director',0.1]])

#indices = pd.Series(etape.index, index=etape['movie_title'])
#get_recommendations(etape['movie_title'][3],etape, new_cosine1)
#movieEtape2[movieEtape2['movie_title'].str.contains('john carter')].loc[:,['genres','plot_keywords']]
 #pd.Series(etape.columns)[pd.Series(etape.columns).str.contains('yer_')].values
#movieEtape6.columns
#etape5 =movieEtape6.drop([ 'movie_title','imdb_score'],1)
#etape5 = etape5.reset_index(drop=True)
#etape5.columns
#etape5.shape
#new_cosine5 = new_cosine(etape5, tfid =['plot_keywords'],surate = [[ 'genres',8],[ 'title_year',6]
#                                                                   ,[ 'content_rating',5],[ 'language',2]])

#indices = pd.Series(movieEtape6.index, index=movieEtape6['movie_title'])
#get_recommendations(movieEtape6['movie_title'][3],movieEtape6, new_cosine5)
#get_recommendations(movieEtape6['movie_title'][0],movieEtape6, new_cosine5)
#movieEtape6[movieEtape6.lan_french == 1]
#get_recommendations(movieEtape6['movie_title'][3255],movieEtape6, new_cosine5)
#movieEtape1[movieEtape1.movie_title == 'les visiteurs' ]
#movieEtape6[movieEtape6.grs_comedy == 1]
