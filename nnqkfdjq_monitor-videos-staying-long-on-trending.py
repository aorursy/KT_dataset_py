import pandas as pd
import numpy as np
import holoviews as hv
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
hv.extension('bokeh')
%matplotlib inline
CAvideos = pd.read_csv(r'../input/youtube-new/CAvideos.csv')
DEvideos = pd.read_csv(r'../input/youtube-new/DEvideos.csv')
FRvideos = pd.read_csv(r'../input/youtube-new/FRvideos.csv')
GBvideos = pd.read_csv(r'../input/youtube-new/GBvideos.csv')
USvideos = pd.read_csv(r'../input/youtube-new/USvideos.csv')
df42 = pd.read_csv(r'../input/youtubeadd/DF4.csv')
with open(r'../input/youtube-new/US_category_id.json') as fb:
    m = fb.read()
    categorydict2 = json.loads(m) 
with open(r'../input/youtubeadd/Youtubepunchmaintain.json', 'rb') as fb:
    mnmb = fb.read().decode('utf-8')
    mvmv = json.loads(mnmb)
with open(r'../input/youtubeadd/df4made.json', 'rb') as fb:
    mnmb = fb.read()
    updowndict = json.loads(mnmb)
GBlen = pd.DataFrame(mvmv['GBvideos'] , columns = list(mvmv['GBvideos'].keys()))
USlen = pd.DataFrame(mvmv['USvideos'] , columns = list(mvmv['USvideos'].keys()))
CAlen = pd.DataFrame(mvmv['CAvideos'] , columns = list(mvmv['CAvideos'].keys()))
DElen = pd.DataFrame(mvmv['DEvideos'] , columns = list(mvmv['DEvideos'].keys()))
FRlen = pd.DataFrame(mvmv['FRvideos'] , columns = list(mvmv['FRvideos'].keys()))
categorydict3 = {}
for j in categorydict2['items']:
    categorydict3[j['id']] = j['snippet']['title']
dictionarydict = {}
for i in USvideos['video_id'].unique():
    dictionarydict[i] = len(USvideos[USvideos['video_id'] == i])
pd.Series(dictionarydict).sort_values(ascending = False).head(10)
listlist = [DEvideos,CAvideos,FRvideos,GBvideos,USvideos]                     #assign Position for each video. 
listlist2 = ['DEvideos','CAvideos','FRvideos','GBvideos','USvideos']  
listlist3 = zip(listlist, listlist2)                               
for i , m in listlist3:
        i['Region'] = m
df4 = pd.concat(listlist)
df4.loc[:,'Position'] = 0
for i in listlist2:
    for j in df4['trending_date'].unique():
        m = len(df4.loc[(df4['Region'] == i)&(df4['trending_date'] == j),'Position'])
        df4.loc[(df4['Region'] == i)&(df4['trending_date'] == j),'Position'] = list(range(1, m+1))
        if m != 200:
            print("{0}{1}is not full length{2}".format(i,j,m))    
(df4['Position'] == 0).sum() # 0 means every single data obtained proper Position value. 
df4 = df4.loc[df4['trending_date'] != '29441'] #drop incorrectly retrieved data 
def trending_date_formater(x):                          #trending_date , publish_time into datetime format
    try:
        mn = datetime.strptime(x, "%y.%d.%m")
    except:
        return(datetime(1900,9,1))
    return(mn)
df4['trending_date'] = df4['trending_date'].map(trending_date_formater)
def published_date_formater(x):
    try:
        mn = datetime.strptime(x[:10], "%Y-%m-%d")
    except:
        return(datetime(1900,9,1))
    return(mn)
df4['publish_time'] = df4['publish_time'].map(published_date_formater)
df4['Position'] =df4['Position'].astype(np.float64) #minor modification 
df4['views'] =df4['views'].astype(np.float64)
df4_index = list(df4['trending_date'].unique())
df4_columns = list(df4['Region'].unique())
IndexPosition = list(df4['Position'].unique())
mn = dict()
for i in df4_columns:
    s = pd.DataFrame({'Region':i ,'trending_date':df4_index})
    mn[i] = s
mlist = list(mn.values())
keylist = pd.concat(mlist, axis = 0)
%output size = 150
%opts HeatMap (cmap="Viridis")
%opts HeatMap [height=420 width=250, colorbar=True,colorbar_position='right', tools=['hover'], toolbar='right']
%output size = 150
%opts Curve [height=250 width=300, tools=['hover'], toolbar='right', invert_yaxis=True]
#https://github.com/ioam/holoviews/issues/2154 invert_yaxis plot option! 
def get_visual2ed(title):
    new= df4[df4['video_id'] == title]
    srP = new.loc[:,['Region','trending_date','Position']]
    testlist2 = pd.merge(keylist, srP, on = ['Region','trending_date'] , how = 'left')
    testsrds2 = hv.Dataset(testlist2, kdims=['Region', 'trending_date'], vdims=['Position'])
    curvelist = []
    df4_indexm = pd.DataFrame(df4_index)
    df4_indexm.columns = ['trending_date'] 
    for i in df4_columns:
        test = df4[df4['Region'] == i]
        test2 = test[test['video_id'] == title]
        test2 = pd.merge(df4_indexm, test2, how='left', on = 'trending_date')
        testsrds3 = hv.Dataset(test2, kdims=['trending_date'] , vdims=['Position'])
        curvelist.append(hv.Curve(testsrds3, kdims = ['trending_date'] , vdims = ['Position'], label = i))
    print(new.iloc[0,2])
    return hv.HeatMap(testsrds2, kdims = ['Region','trending_date'], vdims=['Position'], label = "Title : " + new.iloc[0,2]).opts(plot={'xrotation': 90}) + hv.Overlay.from_values(curvelist) 
get_visual2ed('M4ZoCHID9GI')
get_visual2ed('i_nLsG_asQg')
def trending_date_formater2(x):                          
    try:
        mn = datetime.strptime(x, "%Y-%m-%d")
    except:
        return(datetime(1900,9,1))
    return(mn)
df42['trending_date'] = df42['trending_date'].map(trending_date_formater2)
df42['Ftrending'] = df42['Ftrending'].map(trending_date_formater2)
df42['publish_time'] = df42['publish_time'].map(published_date_formater)
day_diff_to_int = lambda x : x.days
df42['day_diff'] = df42['trending_date'] - df42['Ftrending']
df42['day_diff'] = df42['day_diff'].map(day_diff_to_int)
okay = df42.loc[df42['Region'] == 'USvideos',['Position','day_diff']].groupby('Position').mean()
okay  = okay.reset_index()
%opts Bars [height=300 width=450, tools=['hover'], toolbar='right']
hv.Bars(okay , kdim = 'Position')
fgx, exx = plt.subplots(2,2 , sharey = False, sharex = True, figsize = (20,20))
fulist2 = []
for i in USlen['video_id']:
    nbm = df4[(df4['Region'] == 'USvideos')&(df4['video_id'] == i)].loc[:,['trending_date','video_id','views','likes','dislikes','comment_count']].set_index('trending_date')
    fulist2.append(nbm)
def fm(x):
    return x.diff()
mfulist2 = []
for m in fulist2:
    mfulist2.append(m.iloc[:,1:].apply(fm, axis = 0))
bbb = 5
bbmb = mfulist2[bbb]
fgx.suptitle('video_id = {0} counts'.format(fulist2[bbb].iloc[0,0]) , fontsize = 20)
exx[0, 0].plot(bbmb['views'])
exx[0, 0].set_title('views diff', fontdict={'fontsize': 20})
exx[0, 1].plot(bbmb['likes'])
exx[0, 1].set_title('likes diff', fontdict={'fontsize': 20})
exx[1, 0].plot(bbmb['dislikes'])
exx[1, 0].set_title('dislikes diff', fontdict={'fontsize': 20})
exx[1, 1].plot(bbmb['comment_count'])
exx[1, 1].set_title('comment_count diff', fontdict={'fontsize': 20})
fgx
fgx, exx = plt.subplots(2,2 , sharey = False, sharex = True, figsize = (20,20))
fulist2 = []
for i in USlen['video_id']:
    nbm = df4[(df4['Region'] == 'USvideos')&(df4['video_id'] == i)].loc[:,['trending_date','video_id','views','likes','dislikes','comment_count']].set_index('trending_date')
    fulist2.append(nbm)
def fm(x):
    return x.diff()
mfulist2 = []
for m in fulist2:
    mfulist2.append(m.iloc[:,1:].apply(fm, axis = 0))
bbb = 1
bbmb = mfulist2[bbb]
fgx.suptitle('video_id = {0} counts'.format(fulist2[bbb].iloc[0,0]) , fontsize = 20)
exx[0, 0].plot(bbmb['views'])
exx[0, 0].set_title('views diff', fontdict={'fontsize': 20})
exx[0, 1].plot(bbmb['likes'])
exx[0, 1].set_title('likes diff', fontdict={'fontsize': 20})
exx[1, 0].plot(bbmb['dislikes'])
exx[1, 0].set_title('dislikes diff', fontdict={'fontsize': 20})
exx[1, 1].plot(bbmb['comment_count'])
exx[1, 1].set_title('comment_count diff', fontdict={'fontsize': 20})
fgx
updowndict2 = {}
for i in listlist2:
    updowndict2[i] = pd.DataFrame(updowndict[i]).T
    updowndict2[i].columns = ['ups','categories']
    updowndict2[i].loc[:,'categories'] = updowndict2[i].loc[:,'categories'].map(categorydict3)
pd.DataFrame(updowndict2['USvideos']).sort_values(by = 'ups', ascending = False).head(20) 
df4_indexm = pd.DataFrame(df4_index)
df4_indexm.columns = ['trending_date'] 
%opts Bars [height=250 width=300, tools=['hover'], toolbar='right']
df4_columns = list(df4['Region'].unique())
def get_visual3(x):
    def thatlen(mn): #긴 순서대로 배치해서 그림에서 색깔 다 나오게 함. 
        return len(df4[(df4['Region'] == mn)&(df4['video_id'] == x)])
    df4_columns.sort(reverse = True, key=thatlen) 
    barlist = []
    refere = pd.DataFrame(columns = ['trending_date'])
    for i in df4_columns:
        testtestdd = df4[(df4['Region'] == i)&(df4['video_id'] == x)]
        testtestdd.loc[:,'views'] = testtestdd.loc[:,'views'].astype(np.float64) 
        testtestdd.loc[:,'viewsdiff'] = testtestdd.loc[:,'views'].diff()/1000000
        testtestdd2= pd.merge(df4_indexm, testtestdd, how='left', on = 'trending_date')
        testtestdd2 = testtestdd2.loc[:,['trending_date','viewsdiff']]
        barlist.append(hv.Bars(testtestdd2, label = i))                                        
        mnm = df4[(df4['Region'] == i)&(df4['video_id'] == x)]                     
        mnm = mnm.loc[:,['trending_date','views']]
        mnm.loc[:,'views'] = mnm.loc[:,'views'].astype(np.float64) 
        mnm.loc[:,'viewsdiff'] = mnm.loc[:,'views'].diff()
        mnm = mnm.loc[:,['trending_date','viewsdiff']]
        refere = pd.merge(refere , mnm, on = 'trending_date', how = 'outer')
    refere = refere.set_index('trending_date')
    refere.columns = [x for x in df4_columns]
    print(refere)
    return hv.Overlay.from_values(barlist)
get_visual3('HuwPCUzqwKs')
get_visual2ed('HuwPCUzqwKs')
pd.DataFrame(updowndict2['CAvideos']).sort_values(by = 'ups', ascending = False).head(20) 
pd.DataFrame(updowndict2['DEvideos']).sort_values(by = 'ups', ascending = False).head(20) 
pd.DataFrame(updowndict2['GBvideos']).sort_values(by = 'ups', ascending = False).head(20) 
pd.DataFrame(updowndict2['FRvideos']).sort_values(by = 'ups', ascending = False).head(20) 
clrs = {} #assign color to categories. 
clrs[1] = sns.color_palette('hls', 5)  
clrs[2] = sns.color_palette('Set2', 10) 
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
clrs[3] = sns.color_palette(flatui , 6)
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
clrs[4] = sns.xkcd_palette(colors)
clrs[5] = sns.color_palette("cubehelix", 8)
clrs[6] = sns.color_palette("BrBG", 7)
colorlist = sum(list(clrs.values()), [])
colordict = {} #colordict 제작 
d = set(colorlist)
for i in list(categorydict3.values()): 
    cmm =  random.choice(list(d))
    colordict[i] = cmm
    d.remove(cmm)
fgm, ax = plt.subplots(5,1, figsize=(20, 50), sharex = True, sharey = False)
masterlist = [GBlen,USlen,CAlen,DElen,FRlen]
for numnum, mdmd in enumerate(masterlist):  #one cycle draw one plot
    listlist3 = ['count' ,'maintenance' ,'Position' , 'views' ,'likes', 'dislikes' ,'comment_count']
    for i in listlist3:
        mdmd[i] = mdmd[i].astype('float64')
    mdmd['category_id'] = mdmd['category_id'].map(categorydict3)
    mdmdstackdata = pd.DataFrame(mdmd.groupby(['category_id', 'maintenance'])['count'].sum()).reset_index().sort_values(by = 'category_id' , ascending = False)
    mdmdindexmaintenance = pd.DataFrame(list(range(1,len(mdmdstackdata['maintenance'].unique()) + 1)), columns = ['maintenance'])
    ind = np.arange(1, len(mdmdstackdata['maintenance'].unique()) + 1)
    width = 0.35 
    plotdict = {}
    testtuple = tuple(np.zeros(len(mdmdstackdata['maintenance'].unique())))
    for i in list(mdmdstackdata['category_id'].unique()):   #plot을 그림. 
        mergedstackdata = pd.merge(mdmdindexmaintenance, mdmdstackdata[mdmdstackdata['category_id'] == i], on = 'maintenance', how = 'left')
        categorya = tuple(mergedstackdata['count'].fillna(0))
        plotdict[i] = ax[numnum].bar(ind, categorya, bottom = testtuple, color = colordict[i])
        testtuple = tuple(map(sum, zip(testtuple, categorya)))
        ax[numnum].tick_params(axis = 'both', which = 'major', labelsize = 20)
    axlist = []
    axlist2 = []
    for i in list(mdmdstackdata['category_id'].unique()):    #legend 추가. 
        axlist.append(plotdict[i][0])
        axlist2.append(str(i))
    axlist = axlist[::-1]
    axlist2 = axlist2[::-1]
    ax[numnum].legend(tuple(axlist), tuple(axlist2) ,fontsize=15) 
    print("{0} is done".format(numnum))
    ax[numnum].set_title(listlist2[numnum], fontdict = {'fontsize':30})
    ax[numnum].set_xlabel("maintenance" , fontsize = '20')
fgm.suptitle('How long do they stay on trending' , fontsize = 30)
fgm

GBlen[:10]
USlen[:10]
CAlen[:10]
DElen[:10]
FRlen[:10]
