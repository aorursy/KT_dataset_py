import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



def cleanOuterAxis(outerAxis):

    plt.subplots_adjust(top=0.85,hspace=0.6)

    outerAxis.spines['top'].set_color('none')

    outerAxis.spines['bottom'].set_color('none')

    outerAxis.spines['left'].set_color('none')

    outerAxis.spines['right'].set_color('none')

    outerAxis.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
worldData=pd.read_csv('../input/master.csv')

brazilData=worldData[worldData['country']=='Brazil']
worldData.head()
brazilData.head()
fig=plt.figure(figsize=(16, 20))

outerAxis=fig.add_subplot(1,1,1)

axisArray=[fig.add_subplot(4,2,1),fig.add_subplot(4,2,2),

            fig.add_subplot(4,2,3),fig.add_subplot(4,2,4),

            fig.add_subplot(4,2,5),fig.add_subplot(4,2,6),

            fig.add_subplot(4,2,(7,8))]



outerAxis.set_xlabel('Age interval',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)

outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)

fig.suptitle("Number of suicides according to age and generation for worldwide data", fontsize=18)

order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']

hueOrder=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z']



i=0

for generation in hueOrder: 

    sns.barplot(data=worldData[worldData['generation']==generation].groupby([worldData['age']]).sum().reset_index(level=['age']),

                x='age',

                y='suicides_no',

                #hue='generation',

                ax=axisArray[i],

                order=order,

                hue_order=hueOrder,

                ci=None)

    axisArray[i].set_title(str(generation), fontsize=16)

    i+=1



sns.barplot(data=worldData['suicides_no'].groupby([worldData['age'],worldData['generation']]).sum().reset_index(level=['generation','age']),

            x='age',

            y='suicides_no',

            hue='generation',

            ax=axisArray[6],

            order=order,

            hue_order=hueOrder,

            ci=None)

axisArray[6].set_title("World data", fontsize=16)



for axis in axisArray:

    axis.set_xlabel('')

    axis.set_ylabel('')

    sns.despine(ax=axis)

 

cleanOuterAxis(outerAxis)
fig=plt.figure(figsize=(16, 20))

outerAxis=fig.add_subplot(1,1,1)

axisArray=[fig.add_subplot(4,2,1),fig.add_subplot(4,2,2),

            fig.add_subplot(4,2,3),fig.add_subplot(4,2,4),

            fig.add_subplot(4,2,5),fig.add_subplot(4,2,6),

            fig.add_subplot(4,2,(7,8))]



outerAxis.set_xlabel('Age interval',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)

outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)

fig.suptitle("Number of suicides according to age and generation for Brazil data", fontsize=18)

order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']

hueOrder=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z']



i=0

for generation in hueOrder: 

    sns.barplot(data=brazilData[brazilData['generation']==generation].groupby([brazilData['age']]).sum().reset_index(level=['age']),

                x='age',

                y='suicides_no',

                #hue='generation',

                ax=axisArray[i],

                order=order,

                hue_order=hueOrder,

                ci=None)

    axisArray[i].set_title(str(generation), fontsize=16)

    i+=1



sns.barplot(data=brazilData['suicides_no'].groupby([brazilData['age'],brazilData['generation']]).sum().reset_index(level=['generation','age']),

            x='age',

            y='suicides_no',

            hue='generation',

            ax=axisArray[6],

            order=order,

            hue_order=hueOrder,

            ci=None)

axisArray[6].set_title("World data", fontsize=16)



for axis in axisArray:

    axis.set_xlabel('')

    axis.set_ylabel('')

    sns.despine(ax=axis)

 

cleanOuterAxis(outerAxis)
fig=plt.figure(figsize=(16, 9))

outerAxis=fig.add_subplot(1,1,1)

outerAxis.set_xlabel('Generations (from oldest to newest)',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)

outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)

axisArray=[fig.add_subplot(2,2,1),fig.add_subplot(2,2,2),fig.add_subplot(2,2,(3,4))]

fig.suptitle("Suicide rate according to generation", fontsize=18)



order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z']



sns.barplot(data=brazilData['suicides_no'].groupby(brazilData['generation']).sum().reset_index(level='generation'),

            x='generation',

            y='suicides_no',

            ax=axisArray[0],

            order=order,

            ci=None)

axisArray[0].set_title("Brazil data", fontsize=16)



sns.barplot(data=worldData['suicides_no'].groupby(worldData['generation']).sum().reset_index(level='generation'),

            x='generation',

            y='suicides_no',

            ax=axisArray[1],

            order=order,

            ci=None)

axisArray[1].set_title("Worldwide data", fontsize=16)



df=pd.concat([brazilData['suicides_no'].groupby(brazilData['generation']).sum().reindex(order).rename('Brazil'),

              worldData['suicides_no'].groupby(worldData['generation']).sum().reindex(order).rename('World')],

             axis='columns',

             sort=False).reset_index(level='generation')

sns.barplot(data=df.melt(id_vars='generation',value_vars=['Brazil','World'],var_name='scope',value_name='suicides_no'),

            x='generation',

            y='suicides_no',

            hue='scope',

            ax=axisArray[2],

            ci=None)

axisArray[2].set_title("Comparision", fontsize=16)



for axis in axisArray:

    axis.set_xlabel('')

    axis.set_ylabel('')

    sns.despine(ax=axis)



cleanOuterAxis(outerAxis)
fig=plt.figure(figsize=(16, 9))

outerAxis=fig.add_subplot(1,1,1)

axisArray=[fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)]

outerAxis.set_xlabel('Age interval',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)

outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)

fig.suptitle("Number of suicides according to age and sex", fontsize=18)

order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']



sns.barplot(data=brazilData['suicides_no'].groupby([brazilData['age'],brazilData['sex']]).sum().reset_index(level=['sex','age']),

            x='age',

            y='suicides_no',

            hue='sex',

            ax=axisArray[0],

            order=order,

            ci=None)

axisArray[0].set_title("Brazil data", fontsize=16)



sns.barplot(data=worldData['suicides_no'].groupby([worldData['age'],worldData['sex']]).sum().reset_index(level=['sex','age']),

            x='age',

            y='suicides_no',

            hue='sex',

            ax=axisArray[1],

            order=order,

            ci=None)

axisArray[1].set_title("Worldwide data", fontsize=16)



for axis in axisArray:

    axis.set_xlabel('')

    axis.set_ylabel('')

    sns.despine(ax=axis)

    

cleanOuterAxis(outerAxis)
fig=plt.figure(figsize=(16, 9))

outerAxis=fig.add_subplot(1,1,1)

axisArray=[fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)]

outerAxis.set_xlabel('Year',fontsize=18); outerAxis.xaxis.set_label_coords(0.5,-0.2)

outerAxis.set_ylabel('Number of suicides',fontsize=18); outerAxis.yaxis.set_label_coords(-0.075,0.5)

fig.suptitle("Number of suicides versus time", fontsize=18)



sns.lineplot(data=brazilData['suicides_no'].groupby(brazilData['year']).sum().reset_index(level='year'),

            x='year',

            y='suicides_no',

             ax=axisArray[0])

axisArray[0].set_title("Brazil data", fontsize=18)



sns.lineplot(data=worldData['suicides_no'].groupby(worldData['year']).sum().reset_index(level='year'),

            x='year',

            y='suicides_no',

            ax=axisArray[1])

axisArray[1].set_title("Worldwide data", fontsize=18)



for axis in axisArray:

    axis.set_xlabel('')

    axis.set_ylabel('')

    sns.despine(ax=axis)

    

cleanOuterAxis(outerAxis)