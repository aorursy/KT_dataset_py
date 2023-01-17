import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import os
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
print(os.listdir("../input"))
heropowers = pd.read_csv("../input/super_hero_powers.csv")
heropowers.head()
heroInfo = pd.read_csv("../input/heroes_information.csv",index_col=0)
heroInfo.head()
heroInfo.replace('-',np.nan,inplace=True)
heroInfo.replace(-99.0,np.nan,inplace=True)
heroInfo.rename(columns={'Eye color':'Eyecolor','Hair color':'Haircolor','Skin color':'Skincolor'},inplace=True)
heroInfo.head()
heroInfo.info()
race = heroInfo.groupby('Race').Race.count()[heroInfo.groupby('Race').Race.count()==1]
heroInfo[heroInfo.Race.isin(list(race.index))].loc[:,['name','Race']]
eye = heroInfo.groupby('Eyecolor').Eyecolor.count()[heroInfo.groupby('Eyecolor').Eyecolor.count()==1]
heroInfo[heroInfo.Eyecolor.isin(list(eye.index))].loc[:,['name','Eyecolor']]
skin = heroInfo.groupby('Skincolor').Skincolor.count()[heroInfo.groupby('Skincolor').Skincolor.count()==1]
heroInfo[heroInfo.Skincolor.isin(list(skin.index))].loc[:,['name','Skincolor']]
unknownpercent = heroInfo.isnull().astype('int64').sum() / heroInfo.isnull().count() * 100
data = [go.Bar(x=(list(unknownpercent.index)),y=unknownpercent.values)]
layout=go.Layout(yaxis=dict(title='Percentage %'),xaxis=dict(title='Attributes'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
mhero = heroInfo[heroInfo.Gender=='Male'].name
fhero = heroInfo[heroInfo.Gender=='Female'].name
mlist = heropowers[heropowers.hero_names.isin(mhero.values)]
flist = heropowers[heropowers.hero_names.isin(fhero.values)]
mlist = mlist.loc[:,mlist.columns!='hero_names'].astype('Int64').sum().sort_values(ascending=False)
flist = flist.loc[:,flist.columns!='hero_names'].astype('Int64').sum().sort_values(ascending=False)
data = [go.Bar(x=list(mlist.index),y=mlist.values,name='Male Heros'),go.Bar(x=list(flist.index),y=flist.values,name='Female Heros')]
layout=go.Layout(yaxis=dict(title='Number of Hero\'s'),xaxis=dict(title='powers'),barmode='stack')
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
mlist = heropowers[heropowers.hero_names.isin(mhero.values)]
flist = heropowers[heropowers.hero_names.isin(fhero.values)]
mlist = mlist.loc[:,mlist.columns!='hero_names'].astype('Int64').sum()
flist = flist.loc[:,flist.columns!='hero_names'].astype('Int64').sum()
mflist = pd.DataFrame(data=[mlist,flist],index=['Male','Female']).transpose()
mflist = mflist[mflist.Male==0]
mflist = mflist[mflist.Female!=0]
data = [go.Bar(x=list(mflist['Female'].index),y=mflist['Female'].values,name='Unique Female only powers')]
layout=go.Layout(yaxis=dict(title='Number of Hero\'s'),xaxis=dict(title='powers'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
mlist = heropowers[heropowers.hero_names.isin(mhero.values)]
flist = heropowers[heropowers.hero_names.isin(fhero.values)]
mlist = mlist.loc[:,mlist.columns!='hero_names'].astype('Int64').sum()
flist = flist.loc[:,flist.columns!='hero_names'].astype('Int64').sum()
mflist = pd.DataFrame(data=[mlist,flist],index=['Male','Female']).transpose().sort_values(by='Male',ascending=False)
mflist = mflist[mflist.Male!=0]
mflist = mflist[mflist.Female==0]
data = [go.Bar(x=list(mflist['Male'].index),y=mflist['Male'].values,name='Unique Female only powers')]
layout=go.Layout(yaxis=dict(title='Number of Hero\'s'),xaxis=dict(title='powers'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
pub = heroInfo.groupby('Publisher').name.count()
data = [go.Pie(labels=list(pub.index),values=pub.values)]
py.iplot(data,filename='styled_pie_chart')
heights=heroInfo.groupby('Height').name.count()
data=[go.Scatter(x=list(heights.index),y=heights.values,name='Number of Heros'),go.Bar(x=list(heights.index),y=heights.values,name='Number of Heros')]
layout=go.Layout(yaxis=dict(title='Number of Hero\'s'),xaxis=dict(title='Height'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
heightpower=heropowers[heropowers.hero_names.isin(heroInfo[heroInfo.Height==183].name.values)]
heightpower = heightpower.loc[:,heightpower.columns!='hero_names'].astype("Int64")
heightpower = heightpower.sum().sort_values(ascending=False)[heightpower.sum().sort_values(ascending=False)!=0]
data = [go.Bar(x=list(heightpower.index),y=heightpower.values,name='Common Powers')]
layout=go.Layout(yaxis=dict(title='Number of Hero\'s'),xaxis=dict(title='powers'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
maxpow = heropowers.loc[:,heropowers.columns!='hero_names'].astype("Int64").sum(axis=1)
maxpow.index =(heropowers['hero_names'].values)
maxpow=maxpow.sort_values(ascending=False).head(20)
data=[go.Bar(x=list(maxpow.index),y=maxpow.values,name='Most Number of power')]
layout=go.Layout(yaxis=dict(title='Number of Powers'),xaxis=dict(title='Super Hero\'s'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
resistances = heropowers.columns[heropowers.columns.str.contains('Resistance')]
heroWithresis = heropowers.loc[:,list(resistances)].astype("Int64").sum(axis=1)
heroWithresis.index=heropowers['hero_names'].values
heroWithresis = heroWithresis[heroWithresis.values!=0].sort_values(ascending=False).head(20)
data=[go.Bar(x=list(heroWithresis.index),y=heroWithresis.values,name='Most Number of Resistance')]
layout=go.Layout(yaxis=dict(title='Number of Powers'),xaxis=dict(title='Super Hero\'s'))
py.iplot(go.Figure(data=data,layout=layout),filename='basic_bar')
karma = heroInfo.groupby('Alignment').name.count()
py.iplot([go.Pie(labels=list(karma.index),values=karma.values)],filename="basic_pie_chart")