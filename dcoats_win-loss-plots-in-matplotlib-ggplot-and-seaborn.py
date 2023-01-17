import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
#matplotlib.style.use('fivethirtyeight')
import pylab
pylab.ion()
pylab.show()
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
import pandas
### Data ###
byu= pandas.read_csv('../input/byuData.csv',header=0,index_col=None)
colors= {'L':'indianred','W':'mediumturquoise'}
def colfunc(x): return colors[x]

byu['Wins']= pandas.cut(byu["win"],2,labels=["L","W"])

byu
### Haws Points Plot ###
byu.plot(kind='scatter',x='Unnamed: 0',y='hawsp',s=byu['margin']*20,c=byu['Wins'].apply(colfunc),alpha=.8)
pylab.xlim(0,36)
pylab.ylim(0,36)
pylab.ylabel('Points')
pylab.xlabel('Games (2014-2015) Season')
pylab.title('Tyler Haws Points by Game')
red_patch= mpatches.Patch(color='indianred',label='Loss')
blue_patch= mpatches.Patch(color='mediumturquoise',label='Win')
legend1= plt.legend(handles=[red_patch,blue_patch],title='Outcome',loc=4)
plt.gca().add_artist(legend1)
l1 = plt.scatter([],[], s=5*20, edgecolors='none',color='white')
l2 = plt.scatter([],[], s=20*20, edgecolors='none',color='white')
l3 = plt.scatter([],[], s=35*20, edgecolors='none',color='white')
l4 = plt.scatter([],[], s=50*20, edgecolors='none',color='white')
labels = ["5", "20", "35", "50"]
plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True,
		                handlelength=2, loc = 8, borderpad = 1.1,
				                handletextpad=1, title='Margin of Victory',scatterpoints =1)

### Team Points Plot ###
byu.plot(kind='scatter',x='Unnamed: 0',y='teampoints',s=byu['margin']*20,c=byu['Wins'].apply(colfunc), alpha=.8)
pylab.xlim(0,35.5)
pylab.ylim(59,125)
pylab.ylabel('Points')
pylab.xlabel('Games (2014-2015) Season')
pylab.title('BYU Points by Game')
red_patch= mpatches.Patch(color='indianred',label='Loss')
blue_patch= mpatches.Patch(color='mediumturquoise',label='Win')
first_legend= plt.legend(handles=[red_patch,blue_patch],title='Outcome',loc=1)
plt.gca().add_artist(first_legend)
l1 = plt.scatter([],[], s=5*20, edgecolors='none',color='white')
l2 = plt.scatter([],[], s=20*20, edgecolors='none',color='white')
l3 = plt.scatter([],[], s=35*20, edgecolors='none',color='white')
l4 = plt.scatter([],[], s=50*20, edgecolors='none',color='white')
labels = ["5", "20", "35", "50"]
plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True,
		handlelength=2, loc = 9, borderpad = 1.1,
		handletextpad=1, title='Margin of Victory',scatterpoints =1)

### Opponent's BPI Plot ###
byu.plot(kind='scatter',x='Unnamed: 0',y='oppbpi',s=byu['margin']*20,c=byu['Wins'].apply(colfunc),alpha=.8)
pylab.xlim(0,35.5)
#pylab.ylim(59,125)
pylab.ylabel('Basketball Performance Index Score')
pylab.xlabel('Games (2014-2015) Season')
pylab.title("Opponent's BPI by Game")
red_patch= mpatches.Patch(color='indianred',label='Loss')
blue_patch= mpatches.Patch(color='mediumturquoise',label='Win')
legend1= plt.legend(handles=[red_patch,blue_patch],title='Outcome',loc=4)
plt.gca().add_artist(legend1)
l1 = plt.scatter([],[], s=5*20, edgecolors='none',color='white')
l2 = plt.scatter([],[], s=20*20, edgecolors='none',color='white')
l3 = plt.scatter([],[], s=35*20, edgecolors='none',color='white')
l4 = plt.scatter([],[], s=50*20, edgecolors='none',color='white')
labels = ["5", "20", "35", "50"]
plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True,
		                handlelength=2, loc = 8, borderpad = 1.1,
					                handletextpad=1, title='Margin of Victory',scatterpoints =1)


from plotnine import *
(ggplot(byu)
 + aes(x=range(1,36),y='oppbpi', color='Wins') 
 + geom_point(aes(size='margin'))  
 + xlab("Games (2014-2015 Season)") 
 + ylab("Basketball Performance Index Score") 
 + ggtitle("Opponent's BPI by Game") 
 + scale_colour_discrete(name="Outcome",breaks=["L", "W"],labels=["Loss", "Win"]) 
 + scale_size_area(name="Margin"))
import seaborn as sns
import statsmodels
import patsy
#sns.set(style="darkgrid")
colorpal= ['faded red','pale teal']
sns.scatterplot(x=range(1,36), y='oppbpi',hue='Wins',size='margin',data=byu, palette=sns.xkcd_palette(colorpal))
sns.lmplot(x='teampoints',y='win',y_jitter=.02,data=byu,logistic=True)
colorpal= ['faded red','pale teal']
#sns.palplot(sns.xkcd_palette(colorpal))
dat= byu.loc[:,['Unnamed: 0','teampoints','hawsp','oppbpi','Wins']]
sns.pairplot(dat,hue='Wins',palette=sns.xkcd_palette(colorpal))