# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import matplotlib.pyplot as plt
from bokeh.io import  show, output_notebook#, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool, WheelZoomTool, PanTool, ColumnDataSource, LogColorMapper, LinearColorMapper
from bokeh.palettes import Viridis256 as palette
from wordcloud import WordCloud, STOPWORDS
palette.reverse()
output_notebook()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
datsta = pd.read_csv('../input/StateNames.csv')
datnat = pd.read_csv('../input/NationalNames.csv')
datsta.head()
datnat.head()
# Group by 'Year' and 'Name', then normalized by number of baby each year to get percentages for each names.
ntop=10
popbyyearM = datnat[datnat['Gender']=='M'].groupby(['Year','Name']).sum()
popbyyearM = popbyyearM.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popbyyearM = popbyyearM.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count',ascending=False).head(ntop))['Count']
popbyyearF = datnat[datnat['Gender']=='F'].groupby(['Year','Name']).sum()
popbyyearF = popbyyearF.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popbyyearF = popbyyearF.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count',ascending=False).head(ntop))['Count']
pnamesM1 = []
pnamesM2 = []
pnamesF1 = []
pnamesF2 = []
for year in range(1880,2015):
    pnamesM1.append(popbyyearM[year].index.tolist()[0:5])
    pnamesM2.append(popbyyearM[year].index.tolist()[5:10])
    pnamesF1.append(popbyyearF[year].index.tolist()[0:5])
    pnamesF2.append(popbyyearF[year].index.tolist()[5:10])

x = [i-int(i/10)*10 for i in range(1880,2015)]
y = [int(i/10)*10 for i in range(1880,2015)]
source = ColumnDataSource(
    data=dict(
        X=x,
        Y=y,
        pnamesM1 = pnamesM1,
        pnamesM2 = pnamesM2,
        pnamesF1 = pnamesF1,
        pnamesF2 = pnamesF2
        )
    )
hover = hover = HoverTool(
    tooltips=[('Male', '@pnamesM1'),
              ('   ', '@pnamesM2'),
              ('Female', '@pnamesF1'),
              ('     ', '@pnamesF2'),
             ]
)
p = figure(title="Top 10 Popular Names by Each Year",  tools=[hover],
           toolbar_location="above", plot_width=450, plot_height=600)
p.square('X', 'Y', source=source, size=30, fill_alpha=0.4, line_alpha=0.4)
p.xaxis.ticker = [i for i in range(0,10)]
p.xaxis.axis_label= 'Year'
p.xaxis.axis_label_text_font_size='12pt'
p.yaxis.axis_label_text_font_size='12pt'
p.yaxis.axis_label= 'Decade'
tic = [i for i in range(1880,2020,10)]
dtic = {}
for i in tic:
    dtic[i] = str(i)+'s'
p.yaxis.ticker = tic
p.yaxis.major_label_overrides = dtic
p.xaxis.major_label_text_font_size='12pt'
p.yaxis.major_label_text_font_size='12pt'
print("Please click on each year for the top 10 popular names for each year")

#output_file('Pop_Name_by_Year.html')
show(p)
def showname(popnames):
    names = popnames.sort_values(ascending=False).index.values
    freqs = popnames.sort_values(ascending=False).values
    words = {}
    max_words = 30 if names.size > 30 else names.size
    width = 1.5
    offset = 0.1
    for i in range(names.size):
        #words[names[i]] = (freqs[i]-freqs[max_words])/(freqs[0]-freqs[max_words])*width + offset
        words[names[i]] = freqs[i]/freqs[0]

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_font_size=120,  background_color='white', max_words=max_words,
                          width=800, height=450, stopwords=stopwords)
    wordcloud.generate_from_frequencies(words)
    # Plotting
    plt.figure(figsize=(8,4.5))
    plt.axes([0.0, 0.0, 0.8, 1.0])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.axes([0.8, 0.15, 0.2, 0.70])
    max_words = 20 if names.size > 20 else names.size
    plt.text(0.2, 1, 'Order:', color='blue', alpha=0.7)
    for i in range(max_words):
        plt.text(0.2,(1-(i+1)/max_words),str(i+1)+'. '+names[i], color='blue', alpha=0.7)
    plt.axis("off")

    plt.show()
popalltime = datnat[datnat['Gender']=='M'].groupby(['Name']).sum()['Count']
popalltime = popalltime.apply(lambda x: x/popalltime.sum())
print("Top popular male names all time")
#print(popalltime.sort_values(ascending=False).head(10))
showname(popalltime)

popalltime = datnat[datnat['Gender']=='F'].groupby(['Name']).sum()['Count']
popalltime = popalltime.apply(lambda x: x/popalltime.sum())
print('')
print("Top popular female names all time")
#print(popalltime.sort_values(ascending=False).head(10))
showname(popalltime)
poprc = datnat[(datnat['Year'] > 2009) & (datnat['Gender']=='M')].groupby(['Name']).sum()['Count']
poprc = poprc.apply(lambda x: x/poprc.sum())
print("Top popular male names recently 5 years")
#print(poprc.sort_values(ascending=False).head(10))
showname(poprc)
poprc = datnat[(datnat['Year'] > 2009) & (datnat['Gender']=='F')].groupby(['Name']).sum()['Count']
poprc = poprc.apply(lambda x: x/poprc.sum())
print("Top popular female names recently 5 years")
#print(poprc.sort_values(ascending=False).head(10))
showname(poprc)
print('Male names having the most time in the top list:')
mpopnameM = popbyyearM.groupby(level=1, group_keys=False).count().sort_values(ascending=False).head(20)
print(mpopnameM.head(10))
showname(mpopnameM)
print('Female names having the most time in the top list:')
mpopnameF = popbyyearF.groupby(level=1, group_keys=False).count().sort_values(ascending=False).head(20)
print(mpopnameF.head(10))
showname(mpopnameF)
# How dominant the popular name, number of baby born and name used by year
popbyyear = datnat.groupby(['Year','Name']).sum()
popbyyear = popbyyear.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popbyyear = popbyyear.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count',ascending=False).head(ntop))['Count']
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,3))
ax1.barh(popbyyear.index.levels[0], popbyyear.values[0::ntop]*100, color='orange')
ax1.set_title('Percentage of The Most Popular Names')
ax1.set_xlabel('Percentage of the most popular name (%)')
ax1.set_ylabel('Year')
ax1.set_xlim(0.0, 5.0)
ax1.set_ylim(1880, 2015)

tmp=datnat.groupby('Year').count()['Count'].apply(lambda x: x/1000)
ax2.plot(tmp.index, tmp.values, label='# Names', color='C1')
tmp=datnat.groupby('Year').sum()['Count'].apply(lambda x: x/100000)
ax2.plot(tmp.index, tmp.values, label='# Babies', color='C2')
ax2.legend()
ax2.set_label('Number of Babies and Names')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Names (x1k), Baby (x100k)')
plt.show()
nbygender = datnat.groupby(['Gender','Year']).sum()['Count']
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))

ax1.plot(nbygender['M'].index, nbygender['M'].values/1000, label='Male')
ax1.plot(nbygender['F'].index, nbygender['F'].values/1000, label='Female')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Baby (x1000)')
ax1.legend()
ax2.plot(nbygender['F'].index, nbygender['M'].values/nbygender['F'].values, color='red', label='M/F ratio')
ax2.set_xlabel('Year')
ax2.set_ylabel('Male to Female baby born ratio')
ax2.legend()
plt.show()
namenow = datnat[(datnat['Year']==2014) & (datnat['Gender']=='M')].groupby('Name').sum()
namenow = namenow[namenow['Count'] > 5000]['Count']
namenowM = namenow.index.tolist()
namenow = datnat[(datnat['Year']==2014) & (datnat['Gender']=='F')].groupby('Name').sum()
namenow = namenow[namenow['Count'] > 5000]['Count']
namenowF = namenow.index.tolist()

recentpop = datnat[datnat['Year'] > 2009].groupby(['Year','Name']).sum()
recentpop = recentpop.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
recentpop = recentpop.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count', ascending=False))['Count']
# Using a simple linear regression to find which name having the largest increasing slopes.
model = linear_model.LinearRegression()
slopdictM={}
for name in namenowM:
    tmp = recentpop[:,name]
    nsample = tmp.size
    X = tmp.index.values.reshape(nsample,1)
    Y = tmp.values.reshape(nsample,1)
    model.fit(X, Y)
    slopdictM[name] = model.coef_[0][0]
    
slopdictF={}
for name in namenowF:
    tmp = recentpop[:,name]
    nsample = tmp.size
    X = tmp.index.values.reshape(nsample,1)
    Y = tmp.values.reshape(nsample,1)
    model.fit(X, Y)
    slopdictF[name] = model.coef_[0][0]
print('The most trending male names:')
#print(pd.Series(slopdictM).sort_values(ascending=False).head(10))
showname(pd.Series(slopdictM).sort_values(ascending=False).head(10))
print('The most trending female names:')
#print(pd.Series(slopdictF).sort_values(ascending=False).head(10))
showname(pd.Series(slopdictF).sort_values(ascending=False).head(10))
# Popular names of each state whole time and recently for Male and Female
popname = datsta[(datsta['Year'] > 2009) & (datsta['Gender']=='M')].groupby(['State','Name']).sum()
popname = popname.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popnameMrc = popname.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count', ascending=False).head(5))['Count']
popname = datsta[datsta['Gender']=='M'].groupby(['State','Name']).sum()
popname = popname.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popnameM = popname.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count', ascending=False).head(5))['Count']

popname = datsta[(datsta['Year'] > 2009) & (datsta['Gender']=='F')].groupby(['State','Name']).sum()
popname = popname.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popnameFrc = popname.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count', ascending=False).head(5))['Count']
popname = datsta[datsta['Gender']=='F'].groupby(['State','Name']).sum()
popname = popname.groupby(level=0, group_keys=False).transform(lambda x: x/x.sum())
popnameF = popname.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values(by='Count', ascending=False).head(5))['Count']

popname = datsta.groupby(['State','Name']).sum()
numbaby = popname.groupby(level=0, group_keys=False).sum()['Count']
# Relocate and scale patch of state
def relocateandscale(lons, lats, newpos, scale):
    lons = np.array(lons)
    for i in range(lons.size):
        if lons[i] > 0:
            lons[i] -= 360
    lats = np.array(lats)
    clon = np.nanmean(lons)
    clat = np.nanmean(lats)
    lons = lons - clon
    lats = lats - clat
    lons = (lons*scale + newpos[0])
    lats = (lats*scale + newpos[1])
    return (lons, lats)    

# Preparing to plot 51 states on map with moved AK and HI
stlist = popname.index.levels[0].tolist()
#Temporarily revmove AK and HI
stlist.remove('AK')
stlist.remove('HI')

from bokeh.sampledata import us_states
us_states = us_states.data.copy()
state_xs = [us_states[code]["lons"] for code in stlist]
state_ys = [us_states[code]["lats"] for code in stlist]

# Moving and scaling AK and HI
newpos = [-106, 26.5]
lons = us_states['HI']['lons']
lats = us_states['HI']['lats']
(lons, lats) = relocateandscale(lons, lats, newpos, 0.2)
stlist.append('HI')
state_xs.append(lons.tolist())
state_ys.append(lats.tolist())

newpos = [-120, 26.5]
lons = us_states['AK']['lons']
lats = us_states['AK']['lats']
(lons, lats) = relocateandscale(lons, lats, newpos, 0.2)
stlist.append('AK')
state_xs.append(lons.tolist())
state_ys.append(lats.tolist())

# dividing line for HI and AK
hiline_x = [-110, -110, -103.5, -103.5, -110]
hiline_y = [25, 28, 28, 25, 25]
akline_x = [-124, -124, -111.5, -111.5, -124]
akline_y = [25, 30, 30, 25, 25]
nbaby = []
for state in stlist:
    nbaby.append(numbaby[state])
pnamesM = []
for state in stlist:
    pnamesM.append(popnameM[state].index.tolist())
pnamesMrc = []
for state in stlist:
    pnamesMrc.append(popnameMrc[state].index.tolist())
pnamesF = []
for state in stlist:
    pnamesF.append(popnameF[state].index.tolist())
pnamesFrc = []
for state in stlist:
    pnamesFrc.append(popnameFrc[state].index.tolist())

source = ColumnDataSource(
    data=dict(
        X=state_xs,
        Y=state_ys,
        pnamesM=pnamesM,
        pnamesMrc=pnamesMrc,
        pnamesF=pnamesF,
        pnamesFrc=pnamesFrc,
        state=stlist,
        nbaby=nbaby
        )
    )
hover = HoverTool(
    tooltips=[('State','@state'),
              ('Recent 5y: Male', '@pnamesMrc'),
              ('Recent 5y: Female', '@pnamesFrc'),
              ('All time: Male','@pnamesM'),              
              ('All time: Female','@pnamesF'),
             ]
)
color_mapper = LogColorMapper(palette=palette)
#color_mapper = LinearColorMapper(palette=palette)
# init figure
p = figure(title="Most Popular Names in Each State Recently and All Time", tools=[hover],
           toolbar_location="above", plot_width=825, plot_height=525)
# Fill each state with number of baby
p.patches('X', 'Y', source=source,fill_alpha=0.5, line_color="#884444", line_width=1.5,
         fill_color={'field':'nbaby', 'transform': color_mapper})
p.line(akline_x, akline_y, line_color='black', line_dash = (6,3))
p.line(hiline_x, hiline_y, line_color='black', line_dash = (6,3))

# output to static HTML file
#output_file("PopularNamesbyStates.html")

# show results
print("Please click on each state to see the popular names of each state. The state color represent the number of baby born in that state.")
show(p)
def nameneutrality(df):
    if 'M' not in df.index:
        nn = -1
    elif 'F' not in df.index:
        nn = 1
    else:
        nn = (df['M'] - df['F'])/(df['M'] + df['F'])
    return nn
allname = datnat.groupby('Name').sum()['Count']
# exclude some too unique/weird names which have been named for less than 1000 babies in more than 100 years
allname = allname[allname >= 1000].index.tolist()
groupname = datnat.groupby(['Name','Gender']).sum()['Count']
nameneu = {}
for name in allname:
    nameneu[name] = nameneutrality(groupname[name])
nndf = pd.Series(nameneu)
fig = plt.figure(figsize=(6,4))
plt.hist(nndf.values, bins=50,color='C1')
plt.xlabel('Name Neutrality', fontsize=12)
plt.ylabel('Number of Names', fontsize=12)
a = plt.axes([0.29,0.29,0.45,0.45])
plt.hist(nndf.values, bins=50,color='C0')
plt.ylim(0,200)
plt.show()
print('The Most Femalely Names:')
nndf[nndf<-0.95].sort_values(ascending=True).head(20)
print('The Most Malely Names:')
nndf[nndf > 0.95].sort_values(ascending=False).head(20)
print('Gender neutral Names:')
nndf[(nndf < 0.02) & (nndf > -0.02)]