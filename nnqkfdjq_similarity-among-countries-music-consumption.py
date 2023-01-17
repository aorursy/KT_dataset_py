import numpy as np
import pandas as pd
import holoviews as hv
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from datetime import datetime

hv.extension('bokeh')
spo = pd.read_csv('../input/data.csv')
dicname = {'ar':'Argentina', 'at':'Austria', 'au':'Australia', 'be':'Belgium',
       'bo':'Bolivia', 'br':'Brazil', 'ca':'Canada', 'ch':'Switzerland', 
       'cl':'Chile', 'co':'Columbia', 'cr':'CostaRica', 'cz':'CzechRepublic',
       'de':'Germany', 'dk':'Denmark', 'do':'DominicanRepublic', 'ec':'Ecuador', 
       'ee':'Estonia', 'es':'Spain', 'fi':'Finland', 'fr':'France', 
       'gb':'UnitedKingdom', 'global':'World', 'gr':'Greece', 'gt':'Guatemala',
       'hk':'HongKong', 'hn':'Honduras', 'hu':'Hungary', 'id':'Indonesia', 
       'ie':'Ireland', 'is':'Iceland', 'it':'Italy', 'jp':'Japan', 
       'lt':'Lithuania', 'lu':'Luxemborg', 'lv':'Latvia', 'mx':'Mexico', 
       'my':'Malaysia', 'nl':'Netherlands', 'no':'Norway', 'nz':'NewZealand', 
       'pa':'Panama', 'pe':'Peru', 'ph':'Philippines', 'pl':'Poland', 
       'pt':'Portugal', 'py':'Paraguay', 'se':'Sweden', 'sg':'Singapore', 
       'sk':'Slovakia', 'sv':'ElSalvador', 'tr':'Turkey', 'tw':'Taiwan', 'us':'USA', 'uy':'Uruguay',}  
spo['Region'] = spo['Region'].replace(dicname)
spo['Position'] =spo['Position'].astype(np.float64)  
spo_index = list(spo['Date'].unique())
spo_columns = list(spo['Region'].unique())
m = dict()
for i in spo_columns:
    s = pd.DataFrame({'Region':i ,'Date':spo_index})
    m[i] = s
mlist = list(m.values())
keylist = pd.concat(mlist, axis = 0)
%output size = 150
%opts HeatMap (cmap="Viridis")
%opts HeatMap [height=400 width=550, colorbar=True,colorbar_position='right', tools=['hover'], toolbar='above']
def get_visual(title):
    new= spo[spo['Track Name'] == title]
    srP = new.loc[:,['Region','Date','Position']]
    testlist2 = pd.merge(keylist, srP, on = ['Region','Date'] , how = 'left')
    testsrds2 = hv.Dataset(testlist2, kdims=['Region', 'Date'], vdims=['Position'])
    return hv.HeatMap(testsrds2, kdims = ['Region', 'Date'], vdims=['Position'], label = str(new.loc[:,'Artist'].unique())+ title).opts(plot={'xrotation': 45})
get_visual('New Rules')  
get_visual('PLAYING WITH FIRE')
get_visual('GPS') 
spo_Track_Region = spo.groupby(['Track Name','Region']).Streams.sum()
spo_Track_Region_Table = spo_Track_Region.unstack().loc[:,spo_columns]
spo_Track_Region_Table['Counter'] = np.float64(0) 
def f(x): 
    z = np.float64(54) - x.isnull().sum()
    spo_Track_Region_Table.loc[x.name, 'Counter'] = z 
spo_Track_Region_Table.apply(f, axis = 1)
#levellist = [] #얘 없어도 되지 싶은데.. 
levellistdictionary = {}
level1 = spo_Track_Region_Table[spo_Track_Region_Table['Counter'] == 1]
level1 = level1.drop(['Counter'], axis = 1)
level1 = level1.T
def all_levelfunc(x):     
    get =  tuple(level1.index[x.notnull()])
    levellist.append(get)
for i in range(1,55):
    levellist = []
    level = spo_Track_Region_Table[spo_Track_Region_Table['Counter'] == i]
    level = level.drop(['Counter'], axis = 1)
    level = level.T
    level.apply(all_levelfunc)
    levellist = pd.Series(levellist)
    levellistdictionary[i] = levellist
newnewlist = []
for i in levellistdictionary.values():
    newnewlist.append(list(i))
newnewlist2 = sum(newnewlist, [])
newnewlist2series = pd.Series(newnewlist2)
levellistdictionary[2].value_counts().head(30)
levellistdictionary[3].value_counts().head(30)
levellistdictionary[4].value_counts().head(30)
PP = spo_Track_Region_Table[spo_Track_Region_Table['Philippines'].isnull()==False]
PPdictionary = {}
def what_close(x):  
    get =  tuple(level.index[x.notnull()])
    PPlist.append(get)
for i in range(1,55):
    PPlist = []
    level1 = PP[PP['Counter'] == i]
    level1 = level1.drop(['Counter'], axis = 1)
    level1 = level1.T
    level1.apply(what_close)
    PPlist = pd.Series(PPlist)
    PPdictionary[i] = PPlist
PPdictionary[2].value_counts()  
PPdictionary[4].value_counts()  
newnewlist2series.iloc[12777:].value_counts().head(20)

spo["Date"] = spo["Date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
%opts Curve [width=600, height=600, tools=['hover'], toolbar='above']
%opts Curve (line_alpha=0.4)
streaming_per_country = spo.groupby(['Region','Date']).Streams.sum().reset_index()
countrydict = {}
countrylist = list(streaming_per_country['Region'].unique())
def countfunc(m): 
    x = m['Streams']
    return (x.values - x.min())/(x.max() - x.min())
streaming_per_country.loc[:,'Streamsnorm'] = 0
for i in countrylist:
    streaming_per_country.loc[streaming_per_country['Region'] == i,'Streamsnorm'] = countfunc(streaming_per_country[streaming_per_country['Region'] == i])
streaming_per_country3= streaming_per_country.drop('Streams', axis = 1)
streaming_per_country4 = streaming_per_country3.pivot_table(['Streamsnorm'], index = ['Region'], columns = ['Date']).dropna() 
labels = list(streaming_per_country4.Streamsnorm.index)
Z = hac.linkage(streaming_per_country4, method='single', metric='correlation')
hv.Curve(streaming_per_country4.loc['Argentina'] , 'Date')
plt.figure(figsize=(20, 7))
plt.title('Hierarchical Clustering countries with their streaming pattern' , size = 30)
plt.xlabel('Region')
plt.ylabel('distance')
hac.dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=20., # font size for the x axis labels
    labels=labels ,
)
plt.show()
max_d = 0.18
clusters = fcluster(Z, max_d, criterion='distance')
clusterSeries = pd.Series(clusters , index = streaming_per_country4.Streamsnorm.index , name = 'clusters')
streaming_per_country41 = pd.concat([streaming_per_country4, clusterSeries], axis = 1) 
flist = list(streaming_per_country41[streaming_per_country41['clusters'] == 5].index) #여기 클러스터번호 넣어주면 됨. 
clusterlist = []
for i in flist:
    clusterlist.append(hv.Curve(streaming_per_country[streaming_per_country['Region'] == i], 'Date', 'Streamsnorm', label = i))
hv.Overlay.from_values(clusterlist)
i = '2017-01-01'
spofast = spo.loc[:, ['Position','Track Name','Region','Date']]
field = spofast[spofast['Date'] == i]
countrylist = list(spo.Region.unique())
CvC = pd.DataFrame(0 , index = countrylist , columns = countrylist)
for m in countrylist:
    for k in countrylist:
        if m == k:
            continue
        test1 = field[field.Region == m]
        test2 = field[field.Region == k]
        test1 = test1.drop_duplicates(['Track Name'])
        test2 = test2.drop_duplicates(['Track Name'])
        MG = set(test1['Track Name'])&set(test2['Track Name'])
        T = 200 - len(MG) 
        testlist = []
        for i in MG:
            testlist.append(np.abs(int(test1[test1['Track Name']== i]['Position']) - int(test2[test2['Track Name']== i]['Position'])))
        CvC.loc[m,k] = (sum(testlist) + T*200)/100 
import scipy.spatial.distance as ssd
distArray = ssd.squareform(CvC)
Z = hac.linkage(distArray)
labels = countrylist
CvC 
plt.figure(figsize=(20, 7))
plt.title('Chart similarity 2017-01-01' , size = 30)
plt.xlabel('Region')
plt.ylabel('distance')
hac.dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=20., # font size for the x axis labels
    labels=labels ,
)
plt.show()
i2 = '2018-01-01'
field2 = spofast[spofast['Date'] == i2]
CvC2 = pd.DataFrame(0 , index = countrylist , columns = countrylist)
for m in countrylist:
    for k in countrylist:
        if m == k:
            continue
        test1 = field2[field2.Region == m]
        test2 = field2[field2.Region == k]
        test1 = test1.drop_duplicates(['Track Name'])
        test2 = test2.drop_duplicates(['Track Name'])
        MG = set(test1['Track Name'])&set(test2['Track Name'])
        T = 200 - len(MG) 
        testlist = []
        for i in MG:
            testlist.append(np.abs(int(test1[test1['Track Name']== i]['Position']) - int(test2[test2['Track Name']== i]['Position'])))
        CvC2.loc[m,k] = (sum(testlist) + T*200)/100 
distArray = ssd.squareform(CvC2)
Z2 = hac.linkage(distArray)
CvC2
plt.figure(figsize=(20, 7))
plt.title('Chart similarity 2018-01-01' , size = 30)
plt.xlabel('Region')
plt.ylabel('distance')
hac.dendrogram(
    Z2,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=20., # font size for the x axis labels
    labels=labels ,
)
plt.show()







test2 = spo.groupby(['Region','Track Name']).Streams.sum()
test2_Series = pd.Series(np.zeros(54),index = spo_columns)
for i in list(test2_Series.index):
    test2_Series[i] = len(test2[i])
test2_Series.sort_values(ascending = False) 
spo['New Track Name'] = spo['Track Name'] + spo['URL']
countrylist = list(spo.Region.unique())
spoproject4 = spo.groupby(['Region','Date','Position','New Track Name']).sum()
spoproject4 = spoproject4.reset_index('New Track Name').drop('Streams' , axis = 1).unstack()
spoproject4 = spoproject4.loc[:,'New Track Name']
spoproject4label = pd.Series(0, index = spoproject4.index)
for C in countrylist:
    df = spoproject4.loc[C]
    L = len(df)-1
    for i in range(L):
        testtest = df.iloc[i]
        testtest2 = df.iloc[i+1]
        testtest[testtest.isnull().values] = 'R'
        testtest2[testtest2.isnull().values] = 'R'
        Flen = (testtest != 'R').sum()
        Slen = (testtest2 != 'R').sum()
        mace1 = set(testtest) - set('R')
        mace2 = set(testtest2) - set('R')
        intsecmace = mace1&mace2
        Fmace = mace1 - mace2&mace1
        Smace = mace2 - mace1&mace2
        onlylist2 = []
        intersectlist2 = []
        for i in Fmace:
            onlylist2.append(int(Flen +1 - list(testtest.index[testtest == i])[0])) 
        for i in Smace:
            onlylist2.append(int(Slen +1 - list(testtest2.index[testtest2 == i])[0]))
        for I in intsecmace:
            mac = list( testtest.index[testtest == I] )[0]
            mac2 = list( testtest2.index[testtest2 == I] )[0]
            sum = np.abs(mac - mac2)
            intersectlist2.append(sum)
        spoproject4label.loc[(C, testtest.name)] = pd.Series(intersectlist2).sum() + pd.Series(onlylist2).sum()
sp4 = spoproject4label
sp4.columns = ['Region', 'Date', 'distance']
sp4.groupby('Region').sum().sort_values(ascending = False)





