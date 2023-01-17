import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
import difflib
from wordcloud import WordCloud
import pylab as pl
import math

init_notebook_mode()
shanghai = pd.read_csv(("../input/shanghaiData.csv"))
cwur = pd.read_csv('../input/cwurData.csv')
timesData = pd.read_csv('../input/timesData.csv')
def getRankPhoto(data,name,lens):
    data[name] = data[name].fillna(0)
    data = data[data[name]!=0]

    university = {}
    for index,content in data.iterrows():
        if content[name] not in university:
            university.setdefault(content[name],{})
        university[content[name]].setdefault(content['year'],0)
        university[content[name]][content['year']] = content['world_rank']

    rank = pd.DataFrame(university)
    rank = rank.fillna(0)

    top10 = {}
    for s in range(lens):
        for i in rank.iloc[s].iteritems():
            if i[1]!=0:
                try:
                   if int(i[1])<10:
                       if i[0] not in top10:
                           top10.setdefault(i[0],{})
                       for j in (rank[i[0]]).iteritems():
                           top10[i[0]].setdefault(j[0],j[1])
                except:
                    continue
    top10 = pd.DataFrame(top10)
    top10 = top10.fillna(11)
    plt.figure(figsize=(10,10))
    plt.grid(True)
    for name in (top10.columns):
        plt.plot(top10[name].astype(int),label=name)
    plt.legend(loc='upper left')
getRankPhoto(shanghai,'university_name',11)
getRankPhoto(cwur,'institution',4)
year = 2015
cs = cwur[cwur['year']==year]
coun = cs['country']
my_wordcloud = WordCloud(
            background_color='white',    # 设置背景颜色
 #           mask = abel_mask,        # 设置背景图片
            max_words = 200,            # 设置最大现实的字数
        #     stopwords = STOPWORDS,        # 设置停用词
            width=512,
            height=384,
            max_font_size = 50,            # 设置字体最大值
            random_state = 30,            # 设置有多少种随机生成状态，即有多少种配色方案
            scale=.5
                ).generate(" ".join(coun))

# 根据图片生成词云颜色
#image_colors = ImageColorGenerator(abel_mask)
#my_wordcloud.recolor(color_func=image_colors)

# 以下代码显示图片
plt.subplots(figsize=(8,8))
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()
year = int(year)
info = dict()
for i in np.arange(2012,2016):
    info.setdefault(i,{})
    for index,content in cwur.iterrows():
        if content['year'] == i:
            if content['country'] not in info[i]:
                info[i].setdefault(content['country'],0)
            info[i][content['country']]+=1

info = pd.DataFrame(info)
info = info.sort_index(by=year, ascending=False)
info = info.fillna(0)
count = ([j for i,j in info[year].iteritems()])

sns.set(font_scale=0.9) 

f, ax = plt.subplots(figsize=(4.5, 20))

colors_cw = sns.color_palette('coolwarm', len(info.index))
sns.barplot(count, info.index, palette = colors_cw[::-1])
Text = ax.set(xlabel='total university', title='count university')
info['state'] = info.index
info.index = range(len(info))

data = [ dict(
        type = 'choropleth',
        locationmode = 'country names',  #选择模式 城市全名
        locations = info['state'],      #根据模式 城市的全名
        z = info[year],                 #要计数的值
         #       text = info['state'],   可加  对地址的描述
        marker = dict(
            line = dict(color = 'rgb(0,0,0)', width = 1)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'University rank')
            )
       ]

layout = dict(
    title = 'University Rank',
    geo = dict(
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )

fig = dict(data=data, layout=layout)
iplot(fig, validate=False, filename='map.html')
data = [ dict(
        type='choropleth',
        colorscale = 'Portland',
        autocolorscale = False,
        reversescale = True,
        locationmode = 'country names',  #选择模式 城市全名
        locations = info['state'],      #根据模式 城市的全名
        z = info[year],                 #要计数的值

        marker = dict(
            line = dict (
                color = 'rgb(102,102,102)',
                width = 1
            ) ),
        colorbar = dict(
            title = "Count")
        ) ]

layout = dict(
        title = '%s University rank' % year,
        geo = dict(
            scope='world',
            projection=dict( type='Mercator' ),
            showlakes = True,
            lakecolor = 'rgb(74, 129, 179)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='thismap.html' )
import plotly.figure_factory as ff
timeData = timesData
timeData = timeData[timeData['female_male_ratio']!='-']
timeData = timeData.dropna()

dataframe = timeData[timeData.year == 2015]
data2015 = dataframe[["research","international", "total_score",'female_male_ratio']]
data2015["index"] = np.arange(1,len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015, diag='histogram', index='index',colormap='Portland',colormap_type='cat',height=700, width=700)
iplot(fig)
allLast = {}
allTop = {}
for year in np.arange(2011,2017):
    nowData = timesData[timesData['year']==year]
    retio = nowData['female_male_ratio']   #女男比例
    retio = retio.fillna(0)
    s = {}
    for i,c in retio.items():
        if c!=0:
            try:
                part1,part2 = (c.split(":"))
            except:
                continue
            if int(part1) == 0:
                s[i] = 0
            elif int(part2) == 0:
                s[i] = 10
            else:
                c = int(part1)/int(part2)
                s[i] = c
    sorte = (sorted(zip(s.values(),s.keys())))
    Last5 = sorte[:5]
    top5 = sorte[-5:]
    def toDict(data):
        for i in np.arange(len(data)):
            data[i] = list(data[i])
            temp = data[i][0]
            data[i][0] = data[i][1]
            data[i][1] = {timesData['university_name'].iloc[data[i][0]]:temp}
            data[i][0] = timesData['country'].iloc[data[i][0]]


    toDict(Last5)
    allLast[year] = dict(Last5)
    toDict(top5)
    allTop[year] = dict(top5)
countrys={}
for i in allLast.items():
    for j in i[1]:
        if j not in countrys:
            countrys.setdefault(j,0)
        countrys[j]+=1
val = list(countrys.values())
key = list(countrys.keys())
fig,ax = plt.subplots(figsize=(10,6))
plt.title("The first ten of the country for men")
sns.barplot(key,val)
l = plt.setp(ax.get_xticklabels(),rotation=90)
countrys = {}
for i in allTop.items():
    for j in i[1]:
        if j not in countrys:
            countrys.setdefault(j,0)
        countrys[j]+=1
val = list(countrys.values())
key = list(countrys.keys())
fig,ax = plt.subplots(figsize=(20,8))
plt.title("The first ten of the country for women")
sns.barplot(key,val)
l = plt.setp(ax.get_xticklabels(),rotation=90)
school = {}
for i in allTop.items():
    for j in i[1].items():
        for a in j[1].items():
            if a[0] not in school:
                school.setdefault(a[0],0)
            school[a[0]] = a[1]
val = list(school.values())
key = list(school.keys())
fig,ax = plt.subplots(figsize=(20,8))
plt.title("The first ten of the universities for women")
sns.barplot(key,val)
l = plt.setp(ax.get_xticklabels(),rotation=90)
school = {}
for i in allLast.items():
    for j in i[1].items():
        for a in j[1].items():
            if a[0] not in school:
                school.setdefault(a[0],0)
            school[a[0]] = a[1]
val = list(school.values())
key = list(school.keys())
fig,ax = plt.subplots(figsize=(10,5))
plt.title("The first ten of the universities for men")
sns.barplot(key,val)
l = plt.setp(ax.get_xticklabels(),rotation=90)
sum = pd.DataFrame()
First = dict()
allyears = set(shanghai['year'])
for year in allyears:
    First.setdefault(year,{})
    one = shanghai[shanghai['year']==(year)]
    one.world_rank=np.arange(1,len(one)+1)
    alist = []
    for i in one.national_rank:
        try:
            if '-' in i:
                alist.append(i)
        except:
            continue
    blist = list(one.national_rank)
    for i in alist:
        blist.remove(i)
    one = one[one.national_rank.isin(blist)]
    one = one[one.national_rank.astype(float) == 1]
    for i,j in one.iterrows():
        First[year].setdefault(j.university_name,[])
        First[year][j.university_name].append(j.world_rank)

for i in First.items():
    for j in i[1].items():
        alllist = list(timesData.university_name)
        diff={}
        test = {}
        for s in range(len(alllist)):
            if j[0] in alllist[s]:
                try:
                    if len(j[1])==1:
                        j[1].append(timesData.iloc[int(s)].country)
                        break
                except:
                    if len(j[1])==1:
                        j[1].append(alllist[s])
                        break
            elif 'The' in j[0]:
                name = j[0][4:]
                if name in alllist[s]:
                    try:
                        if len(j[1])==1:
                            j[1].append(timesData.iloc[int(s)].country)
                            break
                    except:
                        if len(j[1])==1:
                            j[1].append(alllist[s])
                            break
            else:
                if alllist[s] not in diff:
                    diff[s] = (difflib.SequenceMatcher(None, j[0], alllist[s]).quick_ratio())
                    test[alllist[s]] = (difflib.SequenceMatcher(None, j[0], alllist[s]).quick_ratio())
        if  len(j[1])==1:
            sorte = (sorted(zip(diff.values(),diff.keys())))
            index = (sorte[-1][1])
            coun = (timesData.iloc[int(index)].country)
            j[1].append(coun)
            
            sorte1 = (sorted(zip(test.values(),test.keys())))
            print(i[0])
            print(j[0])
            print(sorte1[-1][1])
            print(sorte1[-1][0])
            print("=============")
            
                    

for i in First.items():    
    for j in i[1].items():
        dic = {}
        dic[j[1][0]] = j[1][1]
        First[i[0]][j[0]] = dic

year = 2009
for i in First.items():
    alllist = []
    data = pd.DataFrame()
    for j in i[1].items():
        list1 = []
        list1.append(j[0])
        for s in j[1].items():
            list1.append(s[0])
            list1.append(s[1])
        alllist.append(list1)
    data = pd.DataFrame(alllist)
    #print(data.head())
    data.columns = ['university','rank','country']
    #print(d['rank'])
    data = data.sort_values(by="rank" , ascending=True) 
    rand = np.array([math.floor(int(i)/20)+2 for i in list(data['rank'])])

    data.index = data['rank']
    data.drop('rank',1,inplace = True)
    colors = rand
    area = np.pi * (rand)**2  # 0 to 15 point radii
    plt.figure(figsize=(20,8))
    plt.title(i[0])
    plt.scatter(data['university'], data['country'], s=area, c=colors, alpha=0.5)
    pl.xticks(rotation=90)
time = 1
for i in First.items():
    val = (list(i[1].values()))
    for v in range(len(val)):
        vals = list(val[v].values())[0]
        key = list(val[v].keys())[0]
        val[v][vals] = val[v].pop(key)
        #print(key)
        val[v][vals] = key
       # print(val[v])

time = 1
for i in First.items():
    val = (list(i[1].values()))
    dic = {}
    for j in val:
        dic.update(j)
    keys = list(dic.keys())
    vals = list(dic.values())
    fig,ax = plt.subplots(figsize=(20,8))
    plt.title(i[0])
    sns.stripplot(keys,vals)
    l = plt.setp(ax.get_xticklabels(),rotation=90)
    plt.grid(True)
    time+=1
        
