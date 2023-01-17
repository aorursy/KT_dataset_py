import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import matplotlib
from matplotlib import rc
 
font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)
#plt.xticks(x, labels, rotation='vertical')

def drawGr(x,y,fileName,x_label,y_label,tit):
    matplotlib.rcParams.update({'font.size': 15})
    n_groups = x.size
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = ax.bar(index, y, bar_width,
                alpha=opacity, color='g')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x)
    plt.xticks(rotation=90)

    ax.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(tit)
    fig.tight_layout()
    #plt.savefig(fileName)
    plt.show()

def drawLine(header,Array,my_xticks,fileName,x_label,y_label,tit):
    x = np.arange(0,my_xticks.size,1)
    plt.figure(figsize=(25,15))
    plt.xticks(x, my_xticks,rotation='vertical')
    color = ['red','black','navy','brown','orange','olive','gray','skyblue']
    column = Array.shape[1]
    for i in range(0,column):
        plt.legend(header[i])
        if i< len(color):
            crColor = color[i]
        else:
            crColor = 'red'
        plt.plot(x, Array[:,i],crColor,label=header[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(tit)
    plt.legend(loc='upper left')
    plt.grid(True)
    #plt.savefig(fileName)
    plt.show()

def drawPoint(Xval,Yval,step,fileName,x_label,y_label,tit):
    plt.figure(figsize=(20,15))
    axes = plt.gca().set_xlim([np.min(Xval)-1,np.max(Xval)+1])
    plt.plot(Xval, Yval, 'ro')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(tit)
    #plt.savefig(fileName)
    plt.show()
import pandas as pn
from ast import literal_eval
import numpy as np

def buildСhart(fldName,fileName,x_label,y_label,tit):
    md = pn.read_csv('../input/movies_metadata.csv',low_memory=False)
    md[fldName] = md[fldName].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    flat_list = [item for sublist in md[fldName] for item in sublist]
    result,count = np.unique(flat_list,return_counts=True)
    resIndex= count.argsort()[-10:][::-1]
    drawGr(result[resIndex],count[resIndex],fileName,x_label,y_label,tit)

#какие фильмы выпускают    
fldName = 'genres'
fileName = '/home/dima/Machine_Learning/movie/genreMovie.png'
buildСhart(fldName,fileName,'Жанр','Количество штук','Какой жанр фильма популярен')
#где выпускают
fldName = 'production_countries'
fileName = '/home/dima/Machine_Learning/movie/cnMovie.png'
buildСhart(fldName,fileName,'Страна','Количество штук','В какой стране производят фильмы')
import pandas as pn
import numpy as np
import dateutil
from ast import literal_eval
#https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

def groupDt(fildParam,arrayParam,fileName,x_label,y_label,tit):
    md = pn.read_csv('../input/movies_metadata.csv',low_memory=False)    
    md = md.dropna(subset=['release_date'])
    df1 = md[['release_date',fildParam]]

    df1['release_date'] = df1['release_date'].apply(lambda x:  x.split('-')[0] if ( int(x.split('-')[0]) > 1990 and int(x.split('-')[0]) < 2018) else 0).apply(int)
    df1 = df1[df1['release_date'] != 0]
    df1[fildParam] = df1[fildParam].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    resultArr = np.array([])
    group = df1.groupby(['release_date']).groups.keys()
    for k,v in df1.groupby(['release_date']).groups.items():
        flat_list = [item for sublist in df1[fildParam][v] for item in sublist]
        result,count = np.unique(flat_list,return_counts=True)
        indexElement = np.in1d(result,arrayParam)
        indexDict =np.in1d(arrayParam,result)
        count = count[indexElement]
        result = result[indexElement]
        Index = np.argsort(result)
        Sum=np.zeros(len(arrayParam))
        Sum[indexDict]= count[Index]
        if resultArr.size==0:
            resultArr = np.array(Sum.copy()).reshape((1,len(arrayParam)))
        else:
            resultArr=np.append(resultArr,np.array(Sum.copy()).reshape((1,len(arrayParam))),axis=0)
    drawLine(np.array(arrayParam),resultArr,np.array(list(group)),fileName,x_label,y_label,tit)

fildParam = 'genres'
arrayParam = sorted(['Drama','Comedy','Thriller','Romance','Action'])
fileName='/home/dima/Machine_Learning/movie/genreMoviebyYear.png'
groupDt(fildParam,arrayParam,fileName,'Год','Количество штук','Производство жанров по годам')

fildParam = 'production_countries'
arrayParam = sorted(['United States of America','United Kingdom','France','Germany','Italy'])
fileName='/home/dima/Machine_Learning/movie/cnMoviebyYear.png'
groupDt(fildParam,arrayParam,fileName,'Год','Количество штук','Динамика производства фильмов по странам')
import pandas as pn
import numpy as np
cs = pn.read_csv('../input/movies_metadata.csv',low_memory=False)

cs['budget'] = cs['budget'].apply(lambda x: float(x) if x.isdigit() else 0)
cs = cs[cs['vote_average']!=0]
cs = cs[cs['budget']!=0]
fileName='/home/dima/Machine_Learning/movie/retMovie.png'
drawPoint(cs['vote_average'],np.log2(cs['budget']),1,fileName,'Рейтинг фильма','log(бюджет)','Сколько потрачено - на сколько оценили')


import pandas as pn
import numpy as np
from ast import literal_eval

def peopleMoneyVote(cs,fldName,fileName,x_label,y_label,tit):
    cs['id'] = pn.to_numeric(cs['id'], errors='coerce').fillna(0).astype(np.int64)
    cs = cs.as_matrix(columns = ['id',fldName])
    cs = cs[cs[:,1].argsort()[-4500:][::-1],:]
    df = pn.read_csv('../input/credits.csv',low_memory=False)
    df = df[np.in1d(df['id'],cs[:,0])]
    df = df['cast'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    flat_list = [item for sublist in df for item in sublist]
    result,count = np.unique(flat_list,return_counts=True)
    resIndex = count.argsort()[-20:][::-1]
    drawGr(result[resIndex],count[resIndex],fileName,x_label,y_label,tit)

# выбор 4500 самых дорогих фильмов и посмотрел какие люди там учавствуют и сколько 
cs = pn.read_csv('../input/movies_metadata.csv',low_memory=False)
cs['budget'] = cs['budget'].apply(lambda x: float(x) if x.isdigit() else 0)
fldName = 'budget'
fileName = '/home/dima/Machine_Learning/movie/moneyPeople.png'
peopleMoneyVote(cs,fldName,fileName,'Имя актера','Сколько раз снимался в высокобюджетном фильме','Богатые актеры')

# выбор 4500 самых рейтинговых фильмов и посмотрел какие люди там учавствуют и сколько
fldName = 'vote_average'
fileName='/home/dima/Machine_Learning/movie/votePeople.png'
peopleMoneyVote(cs,fldName,fileName,'Имя актера','Сколько раз снимался в хорошем фильме','Хорошие актеры')

import pandas as pn
import numpy as np
from ast import literal_eval

md = pn.read_csv('../input/movies_metadata.csv',low_memory=False)
md['id'] = pn.to_numeric(md['id'], errors='coerce').fillna(0).astype(np.int64)
md = md.dropna(subset=['release_date'])
md['release_date'] = md['release_date'].apply(lambda x:  x.split('-')[0] if ( int(x.split('-')[0]) > 1990 and int(x.split('-')[0]) < 2018) else 0).apply(int)
md = md[md['release_date'] != 0]
md['budget'] = md['budget'].apply(lambda x: float(x) if x.isdigit() else 0)

df = pn.read_csv('../input/credits.csv',low_memory=False)
df['cast'] = df['cast'].fillna('[]').apply(literal_eval).apply(lambda x: [i['gender'] for i in x] if isinstance(x, list) else [])
group = md.groupby(['release_date']).groups.keys()

# денег по годам
money = md.groupby(['release_date'])['budget'].mean()
y = [m for m in money]
fileName='/home/dima/Machine_Learning/movie/yearMoney.png'
drawGr(np.array(list(group)),np.log2(y),fileName,'Год','log(бюджет)','Сколько потрачено в киноиндустрии')

resultArr = np.array([])
for k,v in md.groupby(['release_date']).groups.items():
    gender = df[np.in1d(df['id'],md['id'][v])]['cast']
    flat_list = [item for sublist in gender for item in sublist]
    result,count = np.unique(flat_list,return_counts=True)
    if resultArr.size==0:
        resultArr = np.array(count[1:3]).reshape((1,2))
    else:
        resultArr=np.append(resultArr,np.array(count[1:3]).reshape((1,2)),axis=0)

fileName='/home/dima/Machine_Learning/movie/femaleMale.png'
drawLine(np.array(['female','male']),resultArr,np.array(list(group)),fileName,'Год','Количество','Сколько женщин и мужчина задействованы в киноиндустрии')


#https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
import pandas as pn
import numpy as np
from ast import literal_eval
md = pn.read_csv('../input/movies_metadata.csv',low_memory=False)
md['production_companies'] = md['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
flat_list = [item for sublist in md['production_companies'] for item in sublist]
result,count = np.unique(flat_list,return_counts=True)
resIndex = count.argsort()[-20:][::-1]
fileName = '/home/dima/Machine_Learning/movie/production_companies.png'
drawGr(result[resIndex],count[resIndex],fileName,'Название компании','Кол-во выпущенных фильмов','Производительные компании')


md['budget'] = md['budget'].apply(lambda x: float(x) if x.isdigit() else 0)
df = pn.DataFrame(pn.concat([pn.Series(row['budget'], row['production_companies']) for _, row in md.iterrows()]))
df = df.reset_index()
df.columns = ['production_companies','budget']

df = pn.DataFrame(df.groupby(['production_companies'])['budget'].sum())
df = df.reset_index()
df.columns = ['production_companies','budget']
resIndex = df['budget'].argsort()[-20:][::-1]

fileName = '/home/dima/Machine_Learning/movie/moneyCompanies.png'
drawGr(df['production_companies'][resIndex],np.log2(df['budget'][resIndex]),fileName,'Название компании','Денег потрачено log(бюджет)','Богатые компании')

