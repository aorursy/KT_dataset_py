import math as mt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_pcdata=pd.read_json("../input/weeklyRivensPC.json")

df_xboxdata=pd.read_json("../input/weeklyRivensXB1.json")

df_playstationdata=pd.read_json("../input/weeklyRivensPS4.json")

df_switchdata=pd.read_json("../input/weeklyRivensSWI.json")
print("pc data shape: ",df_pcdata.shape)

print("xbox one data shape: ",df_xboxdata.shape)

print("play station 4 data shape: ",df_playstationdata.shape)

print("switch data shape: ",df_switchdata.shape)
df_pcdata.isnull().values.any()

df_pcdata.isnull().sum()
df_pcdata['cv']=round(df_pcdata['stddev']/df_pcdata['avg']*100,0)

df_xboxdata['cv']=round(df_xboxdata['stddev']/df_xboxdata['avg']*100,0)

df_playstationdata['cv']=round(df_playstationdata['stddev']/df_playstationdata['avg']*100,0)

df_switchdata['cv']=round(df_switchdata['stddev']/df_switchdata['avg']*100,0)
df_pcdata.head()
df_pcdata.describe()
sns.set_style("white")

#cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)



fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="median",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata,

                     label='median')



ax = sns.scatterplot(x="compatibility", 

                     y="avg", 

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata,

                     label='mean')



ax.set(xticklabels=[])



ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



plt.title('\nMedian & mean Price for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="median",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata,

                     label='median')



ax = sns.scatterplot(x="compatibility", 

                     y="stddev", 

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata,

                     label='stddev',

                     color='r')



ax.set(xticklabels=[])

ax.legend()



ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)







plt.title('\nMedian & standard deviation of Price for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="cv",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata,

                     label='median')



ax.set(xticklabels=[])

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



plt.title('\nCoefficient of variation (CV) for all compatibilities(PC)\n', fontsize=14, fontweight='bold')

plt.xlabel('\nCompatibility\n', fontsize=12)

plt.ylabel('\nCV\n', fontsize=12);
df_underXcv = df_pcdata.copy()

df_underXcv = df_underXcv.loc[df_underXcv['cv']<100]

#df_underXcv.head()
#sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="median",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_underXcv,

                     label='median')



ax = sns.scatterplot(x="compatibility", 

                     y="avg", 

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_underXcv,

                     label='mean')



ax.set(xticklabels=[])



ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



plt.title('\nMedian & mean Price for all compatibilities(PC) for CV<100\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
df_pcdata_multi = df_pcdata.copy()

df_pcdata_multi = df_pcdata_multi.loc[(df_pcdata_multi['min']!=df_pcdata_multi['max'])&(df_pcdata_multi['min']!=df_pcdata_multi['avg'])&(df_pcdata_multi['max']!=df_pcdata_multi['avg'])]

#df_pcdata_multi.head()
df_pcdata.describe()
df_underXcv.describe()


df_pcdata_multi.describe()
print("The total median of all compability-medians (PC data) is: ", df_pcdata['median'].median())

print("The total median of all compability-medians (cv below 100) is: ", df_underXcv['median'].median())

print("The total median of all compability-medians (2+ entries) is: ", df_pcdata_multi['median'].median())
print("The total median of all compability-maxima (PC data) is: ", df_pcdata['max'].median())

print("The total median of all compability-maxima (cv below 100) is: ", df_underXcv['max'].median())

print("The total median of all compability-maxima (2+ entries) is: ", df_pcdata_multi['max'].median())
df_pcdata_types = df_pcdata.groupby(['itemType']).agg({'avg':'count'}).rename(columns={'avg':'count'})

df_pcdata_types.head()


fig, ax = plt.subplots(figsize=(12,12))

df_pcdata_types.plot(kind='pie', 

                     y = 'count', 

                     ax=ax, 

                     autopct='%1.1f%%', 

                     startangle=90, 

                     shadow=False, 

                     labels=df_pcdata['itemType'], 

                     legend = False, 

                     fontsize=10)

# Equal aspect ratio ensures that pie is drawn as a circle

ax.axis('equal')

plt.tight_layout()



plt.xlabel('')

plt.ylabel('')

plt.title('\nPie chart: Distrubution of compatibilities by itemTypes (PC)\n', fontsize=14, fontweight='bold');
red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.', markersize='8')



df_pcdata.boxplot(column='median', 

                  by='itemType',

                  flierprops=red_square,

                  vert=False,

                  figsize=(14,6))



plt.xlabel('\nmedian price', fontsize=12)

plt.ylabel('itemType\n', fontsize=12)

plt.title('\nBoxplot: Median prices by itemType (PC)\n', fontsize=14, fontweight='bold')



# get rid of automatic boxplot title

plt.suptitle('');

plt.grid(True, which='major', color='#666666', linestyle='-',linewidth=0.25, alpha=0.4)
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,6))

red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.', markersize='8')

ax = sns.boxplot(x='avg', 

            y='itemType',                

            hue='rerolled',

            flierprops=red_square,

            data=df_pcdata)



plt.xlabel('\nmedian price', fontsize=12)

plt.ylabel('itemType\n', fontsize=12)

plt.title('\nBoxplot: Average unrolled & rerolled prices by itemType (PC)\n', fontsize=14, fontweight='bold')





plt.grid(True, which='major', color='#666666', linestyle='-',linewidth=0.25, alpha=0.4);
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,6))

red_square = dict(markerfacecolor='salmon', markeredgecolor='salmon', marker='.', markersize='8')

ax = sns.boxplot(x='median', 

            y='itemType',                

            hue='rerolled',

            flierprops=red_square,

            data=df_pcdata)



plt.xlabel('\nmedian price', fontsize=12)

plt.ylabel('itemType\n', fontsize=12)

plt.title('\nBoxplot: Median unrolled & rerolled prices by itemType (PC)\n', fontsize=14, fontweight='bold')





plt.grid(True, which='major', color='#666666', linestyle='-',linewidth=0.25, alpha=0.4);
df_pcdata_unrolled = df_pcdata.loc[df_pcdata['rerolled']==False].copy()

df_pcdata_rerolled = df_pcdata.loc[df_pcdata['rerolled']==True].copy()
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="median",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata_unrolled,

                     label='median unrolled')



ax = sns.scatterplot(x="compatibility", 

                     y="median", 

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata_rerolled,

                     label='median rerolled')



ax.set(xticklabels=[])



ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)

ax.axhline(df_pcdata_unrolled['median'].median(), ls='-.', color='C0')

ax.axhline(df_pcdata_rerolled['median'].median(), ls='--', color='C1')

ax.text(x=305, y=df_pcdata_unrolled['median'].median(), s='median unr.', alpha=0.7, color='C0')

ax.text(x=305, y=df_pcdata_rerolled['median'].median(), s='median rer.', alpha=0.7, color='C1')



plt.title('\nMedian unrolled & rerolled prices for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
print("The total median of all compability-medians is: ", df_pcdata['median'].median())

print("The total median of all compability-medians (unrolled) is: ", df_pcdata_unrolled['median'].median())

print("The total median of all compability-medians (rerolled) is: ", df_pcdata_rerolled['median'].median())
print("The total median of all compability-medians for Kitguns is: ", df_pcdata['median'].loc[df_pcdata['itemType']=='Kitgun Riven Mod'].median())

print("The total median of all compability-medians (unrolled) for Kitguns is: ", df_pcdata_unrolled['median'].loc[df_pcdata['itemType']=='Kitgun Riven Mod'].median())

print("The total median of all compability-medians (rerolled) for Kitguns is: ", df_pcdata_rerolled['median'].loc[df_pcdata['itemType']=='Kitgun Riven Mod'].median())
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="max",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata_unrolled,

                     label='max unrolled')



ax = sns.scatterplot(x="compatibility", 

                     y="max", 

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata_rerolled,

                     label='max rerolled')



ax.set(xticklabels=[])



ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)

ax.axhline(df_pcdata_unrolled['max'].max(), ls='-.', color='C0')

ax.axhline(df_pcdata_rerolled['max'].max(), ls='--', color='C1')

ax.text(x=305, y=df_pcdata_unrolled['max'].max(), s='max unr.', alpha=0.7, color='C0')

ax.text(x=305, y=df_pcdata_rerolled['max'].max(), s='max rer.', alpha=0.7, color='C1')



plt.title('\nMaximum unrolled & rerolled prices for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
print("The total max of all compability-max is: ", df_pcdata['max'].max())

print("The total max of all compability-max (unrolled) is: ", df_pcdata_unrolled['max'].max())

print("The total max of all compability-max (rerolled) is: ", df_pcdata_rerolled['max'].max())
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,10))



ax = sns.scatterplot(x="compatibility", 

                     y="avg",                    

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata_unrolled,

                     label='average unrolled')



ax = sns.scatterplot(x="compatibility", 

                     y="avg", 

                     #size='rerolled', 

                     #sizes=(90, 30),

                     #hue='itemType', 

                     #palette=cmap,

                     data=df_pcdata_rerolled,

                     label='average rerolled')



ax.set(xticklabels=[])



ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)

ax.axhline(df_pcdata_unrolled['avg'].mean(), ls='-.', color='C0')

ax.text(x=305, y=df_pcdata_unrolled['avg'].mean(), s='tot. avg unr.', alpha=0.7, color='C0')

ax.axhline(df_pcdata_rerolled['avg'].mean(), ls='--', color='C1')

ax.text(x=305, y=df_pcdata_rerolled['avg'].mean(), s='tot. avg rer.', alpha=0.7, color='C1')



plt.title('\nMean unrolled & rerolled prices for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
print("The total average of all compability-max is: ", df_pcdata['avg'].max())

print("The total average of all compability-max (unrolled) is: ", df_pcdata_unrolled['avg'].max())

print("The total average of all compability-max (rerolled) is: ", df_pcdata_rerolled['avg'].max())
df_pcdata_unrolled_gr = df_pcdata_unrolled.groupby(['itemType']).agg({'avg':'mean',

                                            'itemType':'count',

                                            'median':'median',

                                            'max':'max',

                                            'min':'min',

                                            'pop':'median',

                                             'cv':'median'})

df_pcdata_unrolled_gr.head()
df_pcdata_rerolled_gr = df_pcdata_rerolled.groupby(['itemType']).agg({'avg':'mean',

                                            'itemType':'count',

                                            'median':'median',

                                            'max':'max',

                                            'min':'min',

                                            'pop':'median',

                                              'cv':'median'})

df_pcdata_rerolled_gr.head()
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,15))

ax = sns.scatterplot(y="median", 

                     x="compatibility", 

                     size='rerolled', 

                     sizes=(90, 30),

                     hue='itemType',

                     data=df_pcdata)

ax.yaxis.grid()



ax.set(xticklabels=[])



plt.title('\nOverview median Price vs compatibility by rerolled state (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nmedian price', fontsize=12)

plt.xlabel('compatibility\n', fontsize=12);
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,15))

sns.scatterplot(y="median", 

                     x="compatibility", 

                     label='PC',

                     data=df_pcdata)

sns.scatterplot(y="median", 

                     x="compatibility", 

                     label='XBOX1',

                     data=df_xboxdata)

sns.scatterplot(y="median", 

                     x="compatibility", 

                     label='PS4',

                     data=df_playstationdata)

sns.scatterplot(y="median", 

                     x="compatibility", 

                     label='SWITCH',

                     data=df_switchdata)

ax.yaxis.grid()



ax.set(xticklabels=[])



ax.axhline(df_pcdata['median'].median(), ls='-.', color='C0')

ax.axhline(df_xboxdata['median'].median(), ls='-.', color='C1')

ax.axhline(df_playstationdata['median'].median(), ls='-.', color='C2')

ax.axhline(df_switchdata['median'].median(), ls='-.', color='C3')

ax.text(x=305, y=df_pcdata['median'].median(), s='total medians', alpha=0.7, color='black')



plt.title('\nMedian Price for all compabilities and all platforms\n', fontsize=14, fontweight='bold')

plt.ylabel('\nmedian price', fontsize=12)

plt.xlabel('compatibility\n', fontsize=12);
print("The total median of all compability-medians for PC is: ", df_pcdata['median'].median())

print("The total median of all compability-medians for XBOX1 is: ", df_xboxdata['median'].median())

print("The total median of all compability-medians for PS4 is: ", df_playstationdata['median'].median())

print("The total median of all compability-medians for SWITCH is: ", df_switchdata['median'].median())
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,15))

sns.scatterplot(y="max", 

                     x="compatibility", 

                     label='PC',

                     data=df_pcdata)

sns.scatterplot(y="max", 

                     x="compatibility", 

                     label='XBOX1',

                     data=df_xboxdata)

sns.scatterplot(y="max", 

                     x="compatibility", 

                     label='PS4',

                     data=df_playstationdata)

sns.scatterplot(y="max", 

                     x="compatibility", 

                     label='SWITCH',

                     data=df_switchdata)

ax.yaxis.grid()



ax.set(xticklabels=[])



ax.axhline(df_pcdata['max'].max(), ls='-.', color='C0')

ax.text(x=305, y=df_pcdata['max'].max(), s='max PC', alpha=0.7, color='C0')

ax.axhline(df_xboxdata['max'].max(), ls='-.', color='C1')

ax.text(x=305, y=df_xboxdata['max'].max(), s='max XB', alpha=0.7, color='C1')

ax.axhline(df_playstationdata['max'].max(), ls='-.', color='C2')

ax.text(x=305, y=df_playstationdata['max'].max(), s='max PS', alpha=0.7, color='C2')

ax.axhline(df_switchdata['max'].max(), ls='-.', color='C3')

ax.text(x=305, y=df_switchdata['max'].max(), s='max SW', alpha=0.7, color='C3')



plt.title('\nMedian Price for all compabilities and all platforms\n', fontsize=14, fontweight='bold')

plt.ylabel('\nmedian price', fontsize=12)

plt.xlabel('compatibility\n', fontsize=12);
print("The max of all compability-maxima for PC is: ", df_pcdata['max'].max())

print("The max of all compability-maxima for XBOX1 is: ", df_xboxdata['max'].max())

print("The max of all compability-maxima for PS4 is: ", df_playstationdata['max'].max())

print("The max of all compability-maxima for SWITCH is: ", df_switchdata['max'].max())
sns.set_style("white")



fig, ax = plt.subplots(figsize=(20,15))

sns.scatterplot(y="avg", 

                     x="compatibility", 

                     label='PC',

                     data=df_pcdata)

sns.scatterplot(y="avg", 

                     x="compatibility", 

                     label='XBOX1',

                     data=df_xboxdata)

sns.scatterplot(y="avg", 

                     x="compatibility", 

                     label='PS4',

                     data=df_playstationdata)

sns.scatterplot(y="avg", 

                     x="compatibility", 

                     label='SWITCH',

                     data=df_switchdata)

ax.yaxis.grid()



ax.set(xticklabels=[])



ax.axhline(df_pcdata['avg'].mean(), ls='-.', color='C0')

ax.text(x=305, y=df_pcdata['avg'].mean(), s='avg PC', alpha=0.7, color='C0')

ax.axhline(df_xboxdata['avg'].mean(), ls='-.', color='C1')

ax.text(x=305, y=df_xboxdata['avg'].mean(), s='avg XB', alpha=0.7, color='C1')

ax.axhline(df_playstationdata['avg'].mean(), ls='-.', color='C2')

ax.text(x=305, y=df_playstationdata['avg'].mean(), s='avg PS', alpha=0.7, color='C2')

ax.axhline(df_switchdata['avg'].mean(), ls='-.', color='C3')

ax.text(x=305, y=df_switchdata['avg'].mean(), s='avg SW', alpha=0.7, color='C3')



plt.title('\nMedian Price for all compabilities and all platforms\n', fontsize=14, fontweight='bold')

plt.ylabel('\nmedian price', fontsize=12)

plt.xlabel('compatibility\n', fontsize=12);
print("The total average of all compability-averages for PC is: ", round(df_pcdata['avg'].mean(),2))

print("The total average of all compability-averages for XBOX1 is: ", round(df_xboxdata['avg'].mean(),2))

print("The total average of all compability-averages for PS4 is: ", round(df_playstationdata['avg'].mean(),2))

print("The total average of all compability-averages for SWITCH is: ", round(df_switchdata['avg'].mean(),2))
fig, ax = plt.subplots()

df_pcdata.head(50).sort_values(['compatibility','rerolled']).plot.bar(x='compatibility', 

               y='median', 

               ax=ax, 

               legend=True,

               color='C0',            

               figsize=(18, 14))



df_xboxdata.head(50).sort_values(['compatibility','rerolled']).plot.bar(x='compatibility', 

               y='median', 

               ax=ax,

               color='C1')



df_playstationdata.head(50).sort_values(['compatibility','rerolled']).plot.bar(x='compatibility', 

               y='median', 

               ax=ax,

               color='C2')



df_switchdata.head(50).sort_values(['compatibility','rerolled']).plot.bar(x='compatibility', 

               y='median', 

               ax=ax,

               color='C3')



plt.grid(True, which='major', color='#666666', linestyle='-',linewidth=0.25, alpha=0.4)

plt.legend(['PC', 'XBox1','PS4','Switch'])

plt.title('\nPlatform comparison: Median Prices for all compabilities\n', fontsize=14, fontweight='bold')

plt.ylabel('\nmedian price', fontsize=12)

plt.xlabel('compatibility\n', fontsize=12);
#df_pcdata.head(50).sort_values(['compatibility','rerolled'])

#df_switchdata.head(50).sort_values(['compatibility','rerolled'])
df_pcdata_rerolledmerge = df_pcdata.copy()



df_pcdata_rerolledmerge =df_pcdata_rerolledmerge[['compatibility',

                                              'avg', 

                                              'median', 

                                              'max', 

                                              'min', 

                                              'pop', 

                                              'stddev',

                                              'cv']].groupby('compatibility').agg({'avg':'mean',

                                                                                        'median':'median',

                                                                                        'max':'max',

                                                                                        'min':'min', 

                                                                                        'pop':'mean', 

                                                                                        'stddev':'median',

                                                                                        'cv':'median'})

# Note that I built the median price for **median**, mean for **avg**, max for **max**, min for **min**, mean for pop** (this is a constant anyways) and median for stddev for both rerolled (True and False) prices.  



df_pcdata_rerolledmerge =df_pcdata_rerolledmerge.reset_index(drop=False)
df_pcdata_rerolledmerge.head()
topx = 30
df_pcdata_mediantop = df_pcdata_rerolledmerge.nlargest(topx, 'median')
df_pcdata_mediantop.head()
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='median',

            yerr=df_pcdata_mediantop['stddev'],

            data=df_pcdata_mediantop,

            label= 'median')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=16, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nMedian & standard deviation  of Price for all compatibilities\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
df_pcdata_avgtop = df_pcdata_rerolledmerge.nlargest(topx, 'avg')
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='avg',

            yerr=df_pcdata_avgtop['stddev'],

            data=df_pcdata_avgtop,

            label= 'avg')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=16, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nAverage & standard deviation  of Price for all compatibilities\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
df_pcdata_maxtop = df_pcdata_rerolledmerge.nlargest(topx, 'max')
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='max',

            yerr=df_pcdata_maxtop['stddev'],

            data=df_pcdata_maxtop,

            label= 'max')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=12, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nMaximum & standard deviation of Price for all compatibilities\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
df_pcdata_meantop = df_pcdata_rerolledmerge.nlargest(topx, 'avg')
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='avg',

            yerr=df_pcdata_meantop['stddev'],

            data=df_pcdata_meantop,

            label= 'average')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=12, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nAverage & standard deviation of Price for all compatibilities\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
df_pcdata_mintop = df_pcdata_rerolledmerge.nlargest(topx, 'min')
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='min',

            #yerr=df_pcdata_mintop['stddev'],

            data=df_pcdata_mintop,

            label= 'min')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=12, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nMinimum & standard deviation of Price for all compatibilities\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12)

plt.xlabel('Compatibility\n', fontsize=12);
df_pcdata_mediantop_unrolled = df_pcdata_unrolled.nlargest(topx, 'median')
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='median',

            yerr=df_pcdata_mediantop_unrolled['stddev'],

            data=df_pcdata_mediantop_unrolled,

            label= 'median')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=16, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nMedian & standard deviation of price UNROLLED for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12);
df_pcdata_mediantop_rerolled = df_pcdata_rerolled.nlargest(topx, 'median')
fig, ax = plt.subplots(figsize=(20,6))



chart = sns.barplot(x='compatibility',

            y='median',

            yerr=df_pcdata_mediantop_rerolled['stddev'],

            data=df_pcdata_mediantop_rerolled,

            label= 'median')



chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

ax.yaxis.grid(True, 

         which='major', 

         color='#666666', 

         linestyle='-',

         linewidth=0.25, 

         alpha=0.4)



for p in ax.patches:

             ax.annotate("%.0f" % p.get_height(), 

                         (p.get_x() + p.get_width() / 2., 

                          p.get_height()),

                          ha='center', 

                          va='center', 

                          fontsize=16, 

                          color='black', 

                          xytext=(0, 20),

                          textcoords='offset points')



plt.title('\nMedian & standard deviation of price REROLLED for all compatibilities (PC)\n', fontsize=14, fontweight='bold')

plt.ylabel('\nPrice', fontsize=12);
print('Rubico unrolled median: ',df_pcdata_unrolled['median'].loc[df_pcdata_unrolled['compatibility']=='RUBICO'].values[0])

print('Rubico rerolled median: ',df_pcdata_rerolled['median'].loc[df_pcdata_rerolled['compatibility']=='RUBICO'].values[0])