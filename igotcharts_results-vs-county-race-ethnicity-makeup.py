#import the things you need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import division
from pandas import DataFrame
import matplotlib as mpl

from matplotlib.font_manager import FontManager
import matplotlib.patches as patches
import matplotlib.colors as col
from matplotlib import cm
from pylab import *
%pylab inline
#read the results and county_facts files as Pandas DataFrames
results = pd.read_csv('../input/primary_results.csv')
county_facts= pd.read_csv('../input/county_facts.csv')
"""
A little clean-up so we can join the two DataFrames.. Also noticed that NH has no fips values
for the results. We'll need those too for the merge.

We'll create a new column that only has the County name, then filter by state and pull the fips
values from the County Facts frame.

"""
results['fips']=results['fips'].fillna(0)
results['fips']=results['fips'].astype('int')
county_facts['county']=[(county_facts['area_name'].str.split(' County', 1)[x])[0] for x in range(len(county_facts))]
NH = results[results['state']=='New Hampshire'][['county','fips']]
NH_facts = county_facts[county_facts['state_abbreviation']=='NH'][['county','fips']]
NH = pd.merge(NH,NH_facts,how='left',on='county').set_index(NH.index)
results['fips'][results['state']=='New Hampshire'] =NH['fips_y']

#let's only take the County Facts we want 

def county_facts_func():
    
    county_facts_use = county_facts[['fips','RHI225214', #black
                                     'RHI725214', #his/lat
                                     'RHI825214', #white NH
                                     'RHI425214', #asian
                                     'RHI325214',#american indian
                                     
                                    ]]
    county_facts_use.columns =['fips','Black','His/Lat','White NH','Asian','AmInd']
    
    return county_facts_use


#join the County Facts you pulled with the results

def result_merge():
    return pd.merge(results,county_facts_func(),on='fips')
    

results = result_merge()
#create a frame with votes for each county

county_agg = results.pivot_table(['votes'],index=['fips'],columns=['candidate'],aggfunc='sum',margins='True')
county_agg = county_agg['votes']
county_agg.columns.values


candidates = ['Ben Carson', 'Bernie Sanders',
       'Carly Fiorina', 'Chris Christie', 'Donald Trump',
       'Hillary Clinton', 'Jeb Bush', 'John Kasich', 'Marco Rubio',
       'Martin O\'Malley', 'Mike Huckabee', 'Rand Paul', 'Rick Santorum',
       'Ted Cruz']

"""
create a dictionary of Data Frame that contains the percent of votes for each candidate

we'll keep all candidates for now.. in case you want to look at others
"""

d = {item: pd.DataFrame({'fips':county_agg.index,
                         'percent':(county_agg[item]/county_agg['All']) *100}).reset_index(drop=True) for item in candidates}

#create a dictionary of Data Frames for each candidate from our results Data Frame
d2 = {item: results[results['candidate']==item] for item in candidates}
#create new data frame for Trump,Clinton,Sanders and Cruz that merge percents won for each candidate with the results frames
trump = pd.merge(d["Donald Trump"],d2["Donald Trump"],on='fips')
clinton = pd.merge(d["Hillary Clinton"],d2["Hillary Clinton"],on='fips')
sanders = pd.merge(d["Bernie Sanders"],d2["Bernie Sanders"],on='fips')
cruz = pd.merge(d["Ted Cruz"],d2["Ted Cruz"],on='fips')
#Here are some formatting functions that we'll be re-using in our plots

def set_up():
    
    myFontName = 'Droid Sans'
    mpl.rc('font',family=myFontName)
    fig.suptitle(ttl,fontsize = 18, alpha = .9, ha = 'center', va = 'bottom', style = 'normal', weight = 'bold',color = 'black')
    fig.subplots_adjust(hspace=.5)
    plt.subplots_adjust(top=.9)
    
def spine_ticks():
    ax.grid(linestyle='-',zorder=1,alpha = .2,color='#7570b3')
    ax.yaxis.set_ticks_position('none') 
    ax.xaxis.set_ticks_position('none')

    ax.spines["top"].set_visible(False)  #remove spines
    ax.spines["right"].set_visible(False)  
    ax.spines["left"].set_visible(False)  
    ax.spines["bottom"].set_visible(False)  

def labels (xlab,ylab):
    
    
    
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel(xlab, ha = 'center',fontsize=8, alpha = .8, color='black')
    ax.set_ylabel(ylab, ha = 'center',fontsize=8, alpha = .8,color='black')
    
    ax.xaxis.set_label_coords(x=.5,y=-.15)
    ax.yaxis.set_label_coords(x=-.18,y=.5)
    
#in case you want to change the range of the x-axis, and need the labels to shift as well

def ticks (xlabels = ['']+[str((i)*20)+'%' for i in range(1,5)]):
    
    
   

    ylabels = ['']+[str((i)*20)+'%' for i in range(1,5)]
    ax.set_yticklabels(ylabels,size = 10, alpha=.8,horizontalalignment = 'center',color = 'black', position=(-.06,.5))

    ax.set_xticklabels(xlabels, size = 10,alpha=.8, horizontalalignment = 'center',color = 'black',position=(.5,-.02))

def patch():
    
    left, width = 0, 1
    bottom, height = 0,1
    right = left + width
    top = bottom + height

    p = patches.Rectangle(
    (left, bottom), width, height,
    transform=ax.transAxes, color = '#ebebeb', clip_on=False, alpha = .1,zorder=1
    )

    ax.add_patch(p)
"""
lets visualize each county by percent the candidate won vs 
the percent of residents who are Latino/Hispanic
"""

ttl = '% Latino/Hispanic vs\n% of Votes Cast for Presidential Candidates\nby County'
xlab = 'Percent Latino/Hispanic in County'
ylab = 'Percent of Votes'

fig = plt.figure(1,figsize=(11,8),frameon=False)
set_up()

ax = fig.add_subplot(221,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Trump',color='#666666',alpha=.8)
spine_ticks()
plt.scatter(x=trump['His/Lat'],y=trump['percent'],alpha=.3,color='#666666',zorder=2)
labels(xlab,ylab)
ticks()
patch()

ax = fig.add_subplot(222,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Clinton',color='#1b9e77',alpha=.8)
spine_ticks()
plt.scatter(x=clinton['His/Lat'],y=clinton['percent'],alpha=.3,color='#1b9e77',zorder=2)
ticks()
patch()



ax = fig.add_subplot(223,frame_on= False,xlim=(0,100),ylim=(0,100),title='Cruz')
ax.set_title('Cruz',color='#e7298a',alpha=.8)
spine_ticks()
plt.scatter(x=cruz['His/Lat'],y=cruz['percent'],alpha=.3,color='#e7298a',zorder=2)
ticks()
patch()

ax = fig.add_subplot(224,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Sanders',color='#d95f02',alpha=.8)
spine_ticks()
plt.scatter(x=sanders['His/Lat'],y=sanders['percent'],alpha=.3,color='#d95f02',zorder=2)
ticks()
patch()

"""
Key initial observations: Trump's numbers among counties with >60% Hispanic/Latino population is very low

Pretty much around 20%, there's a negative correlation. 
I'll look at the exact correlation at various points in the future (if I don't get distracted by something else first).

Hillary on the other hand see strong gains when the population exceeds 60% and goes up from there
"""

"""
lets visualize each county by percent the candidate won vs 
the percent of residents who are White Non-Latino/Hispanic
"""

ttl = '% White Non-Hispanic vs\n% of Votes Cast for Presidential Candidates\nby County'
xlab = 'Percent White Non-Hispanic in County'
ylab = 'Percent of Votes'

fig = plt.figure(1,figsize=(11,8),frameon=False)
set_up()

ax = fig.add_subplot(221,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Trump',color='#666666',alpha=.8)
spine_ticks()
plt.scatter(x=trump['White NH'],y=trump['percent'],alpha=.3,color='#666666',zorder=2)
labels(xlab,ylab)
ticks()
patch()

ax = fig.add_subplot(222,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Clinton',color='#1b9e77',alpha=.8)
spine_ticks()
plt.scatter(x=clinton['White NH'],y=clinton['percent'],alpha=.3,color='#1b9e77',zorder=2)
ticks()
patch()



ax = fig.add_subplot(223,frame_on= False,xlim=(0,100),ylim=(0,100),title='Cruz')
ax.set_title('Cruz',color='#e7298a',alpha=.8)
spine_ticks()
plt.scatter(x=cruz['White NH'],y=cruz['percent'],alpha=.3,color='#e7298a',zorder=2)
ticks()
patch()

ax = fig.add_subplot(224,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Sanders',color='#d95f02',alpha=.8)
spine_ticks()
plt.scatter(x=sanders['White NH'],y=sanders['percent'],alpha=.3,color='#d95f02',zorder=2)
ticks()
patch()


#Key initial observation: Hillary may have a White problem in November
"""
lets visualize each county by percent the candidate won vs 
the percent of residents who are Black
"""

ttl = '% Black vs\n% of Votes Cast for Presidential Candidates\nby County'
xlab = 'Percent Black in County'
ylab = 'Percent of Votes'

fig = plt.figure(1,figsize=(11,8),frameon=False)
set_up()

ax = fig.add_subplot(221,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Trump',color='#666666',alpha=.8)
spine_ticks()
plt.scatter(x=trump['Black'],y=trump['percent'],alpha=.3,color='#666666',zorder=2)
labels(xlab,ylab)
ticks()
patch()

ax = fig.add_subplot(222,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Clinton',color='#1b9e77',alpha=.8)
spine_ticks()
plt.scatter(x=clinton['Black'],y=clinton['percent'],alpha=.3,color='#1b9e77',zorder=2)
ticks()
patch()



ax = fig.add_subplot(223,frame_on= False,xlim=(0,100),ylim=(0,100),title='Cruz')
ax.set_title('Cruz',color='#e7298a',alpha=.8)
spine_ticks()
plt.scatter(x=cruz['Black'],y=cruz['percent'],alpha=.3,color='#e7298a',zorder=2)
ticks()
patch()

ax = fig.add_subplot(224,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Sanders',color='#d95f02',alpha=.8)
spine_ticks()
plt.scatter(x=sanders['Black'],y=sanders['percent'],alpha=.3,color='#d95f02',zorder=2)
ticks()
patch()

"""

Key initial observation  - Quite a positive correlation for Hillary once the Black population
exceeds about 10%. A pretty major (and well reported) reason why Bernie lost the nomination

"""
"""
lets visualize each county by percent the candidate won vs 
the percent of residents who are Asian
"""

ttl = '% Asian vs\n% of Votes Cast for Presidential Candidates\nby County'
xlab = 'Percent Asian in County'
ylab = 'Percent of Votes'

fig = plt.figure(1,figsize=(11,8),frameon=False)
set_up()

ax = fig.add_subplot(221,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Trump',color='#666666',alpha=.8)
spine_ticks()
plt.scatter(x=trump['Asian'],y=trump['percent'],alpha=.3,color='#666666',zorder=2)
labels(xlab,ylab)
ticks()
patch()

ax = fig.add_subplot(222,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Clinton',color='#1b9e77',alpha=.8)
spine_ticks()
plt.scatter(x=clinton['Asian'],y=clinton['percent'],alpha=.3,color='#1b9e77',zorder=2)
ticks()
patch()



ax = fig.add_subplot(223,frame_on= False,xlim=(0,100),ylim=(0,100),title='Cruz')
ax.set_title('Cruz',color='#e7298a',alpha=.8)
spine_ticks()
plt.scatter(x=cruz['Asian'],y=cruz['percent'],alpha=.3,color='#e7298a',zorder=2)
ticks()
patch()

ax = fig.add_subplot(224,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Sanders',color='#d95f02',alpha=.8)
spine_ticks()
plt.scatter(x=sanders['Asian'],y=sanders['percent'],alpha=.3,color='#d95f02',zorder=2)
ticks()


    
patch()

"""
Key initial observation: Not much when it comes the primary results.. But it shows how relatively
small the Asian population in this country is. The county with the highest percent of Asian residents
is Honolulu County at 42.4%.

"""
"""
lets visualize each county by percent the candidate won vs 
the percent of residents who are American Indian or Native Alaskan
"""

ttl = '% American Indian/Alaskan Native vs\n% of Votes Cast for Presidential Candidates\nby County'
xlab = 'Percent American Indian/AK Native in County'
ylab = 'Percent of Votes'

fig = plt.figure(1,figsize=(11,8),frameon=False)
set_up()

ax = fig.add_subplot(221,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Trump',color='#666666',alpha=.8)
spine_ticks()
plt.scatter(x=trump['AmInd'],y=trump['percent'],alpha=.3,color='#666666',zorder=2)
labels(xlab,ylab)
ticks()
patch()

ax = fig.add_subplot(222,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Clinton',color='#1b9e77',alpha=.8)
spine_ticks()
plt.scatter(x=clinton['AmInd'],y=clinton['percent'],alpha=.3,color='#1b9e77',zorder=2)
ticks()
patch()



ax = fig.add_subplot(223,frame_on= False,xlim=(0,100),ylim=(0,100),title='Cruz')
ax.set_title('Cruz',color='#e7298a',alpha=.8)
spine_ticks()
plt.scatter(x=cruz['AmInd'],y=cruz['percent'],alpha=.3,color='#e7298a',zorder=2)
ticks()
patch()

ax = fig.add_subplot(224,frame_on= False,xlim=(0,100),ylim=(0,100))
ax.set_title('Sanders',color='#d95f02',alpha=.8)
spine_ticks()
plt.scatter(x=sanders['AmInd'],y=sanders['percent'],alpha=.3,color='#d95f02',zorder=2)
ticks()

patch()

"""
Key intial observation: As the darkness of the plots show, the vast majority of counties have
very low American Indian populations. However, there are a handful of counties with very high
American Indian populations. Those counties had much higher turnout for Clinton and Bernie

"""