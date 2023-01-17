#matplotlab inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
# Data loading functions

def loadTrainData():

    building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

    train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

    weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")  

    return building_metadata,train,weather_train



def loadTestData():

    test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

    weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

    return test,weather_test



def summary(dfs):

    for df in dfs:

        print(df[1]+':\n')

        print(df[0].info(null_counts=True))

        print('\n\n')

        

def convertType(df): 

    for col in df.columns:

        colType = str(df[col].dtypes)

        if colType[:3]=='int':

            if df[col].max()<=np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif df[col].max()<=np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif df[col].max()<=np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

        elif colType[:5]=='float':

            if df[col].max()<=np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif df[col].max()<=np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

    return df







# Visualize Trend (Moving Average) and Seasonality (Serial Correlation)

def plotTrendSeasonality(axes,df,building_id=0,meter=0,span=20,plotRaw=True):

    if plotRaw:

        s = df[(df['building_id']==building_id)&(df['meter']==meter)]['meter_reading']

        axes[0].plot(s,label='Raw')



    # Exponentially-weighted moving average (EWMA)

    rolling = s.ewm(span=span)

    axes[0].plot(rolling.mean(),label='EWMA')



    # Autocorrelation

    if not s.empty:

        axes[1].acorr(s-rolling.mean(),usevlines=True, normed=True, maxlags=500)

        

        

           

# Detect the constant meter reading for select meter and building

def getSpanIdx(df,building_id,meter,maxLength=24):

    tempData = df[(df['building_id']==building_id)&(df['meter']==meter)]['meter_reading']

    x= tempData.values

    idx = np.where(x[:-1]!=x[1:])[0]

    idx = np.concatenate((idx,[len(x)-1]))

    sp = np.concatenate(([idx[0]+1],np.diff(idx)))

    if any(sp>maxLength):

        return tempData.index[np.concatenate([range(x-y+1,x+1) for (x,y) in zip(idx[sp>maxLength],sp[sp>maxLength])])]

    else:

        return None



def plotConstantSpan(ax,df,building_id=0,meter=0,maxLength=24,plotRaw=True):

    if plotRaw:

        ax.plot(df[(df['building_id']==building_id)&(df['meter']==meter)]['meter_reading'],label='Raw')

    idx = getSpanIdx(df=df,building_id=building_id,meter=meter,maxLength=maxLength)

    if idx is not None:

        ax.scatter(x=df_train.loc[idx].index,y=df_train.loc[idx]['meter_reading'],c='r',label='Constant Span')

        

        

        

# Visualize random building on select/random meter and site

def plotRandBuild(df_train,df_building_metadata,site_id=None, meter=None, maxLength=24):

    if site_id is None:

        rng = np.random.default_rng()

        site_id = rng.integers(0,df_building_metadata.site_id.max()+1)

    if meter is None:

        rng = np.random.default_rng()

        meter = rng.integers(0,4)

    building_id = df_building_metadata[df_building_metadata['site_id']==site_id]['building_id'].sample(n=1).values[0]

    

    fig,axes = plt.subplots(1,2,figsize=[30,5])

    axes[0].set_title('Site_id = {}, Building_id = {}, Meter = {}'.format(site_id,building_id,meter))

    axes[1].set_title('Autocorrelation of residual')

    axes[1].set_xlabel('Lags')

    plotTrendSeasonality(axes,df=df_train,building_id=building_id,meter=meter,plotRaw=True)

    plotConstantSpan(ax=axes[0],df=df_train,building_id=building_id,meter=meter,plotRaw=False)

    axes[0].legend()
df_building_metadata, df_train, df_weather_train = loadTrainData()

summary([(df_building_metadata,'building_metadata'), (df_train,'train'), (df_weather_train,'weather_train')])
df_building_metadata = convertType(df_building_metadata)

df_train = convertType(df_train) 

df_weather_train = convertType(df_weather_train)

summary([(df_building_metadata,'building_metadata'), (df_train,'train'), (df_weather_train,'weather_train')])
df_train_merge = df_train.merge(df_building_metadata, how='left',on='building_id').merge(df_weather_train,how='left',on=['site_id','timestamp'])

summary([(df_train_merge,'building_metadata')])
print(df_building_metadata.describe())

sns.heatmap(df_building_metadata.corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)

sns.pairplot(df_building_metadata)

sns.catplot(y='primary_use',kind='count',data=df_building_metadata.sort_values('primary_use'),color='c')

sns.catplot(y='primary_use',kind='count',data=df_building_metadata.sort_values('primary_use'),color='c',col='site_id',height=5,aspect=.3)

sns.catplot(x='site_id',y='square_feet',data=df_building_metadata,kind='bar')
print(df_weather_train.describe())

print('\n')

print('Timeseries data count per site: \n')

print(df_weather_train.site_id.value_counts(sort=False))

sns.heatmap(df_weather_train.corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)

sns.pairplot(df_weather_train)

sns.catplot(x='site_id',y='cloud_coverage',data=df_weather_train,kind='box')

g = sns.FacetGrid(df_weather_train,subplot_kws=dict(projection='polar'),despine=False,height=10,sharex=False, sharey=False)

g.map(sns.countplot,'wind_direction',alpha=.5)

g = sns.FacetGrid(df_weather_train,col='site_id',subplot_kws=dict(projection='polar'),despine=False,col_wrap=4)

g.map(sns.countplot,'wind_direction',alpha=.5)

# sns.relplot(x='timestamp',y='air_temperature',data=df_weather_train,col='site_id')

# sns.relplot(x='timestamp',y='precip_depth_1_hr',data=df_weather_train,col='site_id')

# sns.relplot(x='timestamp',y='sea_level_pressure',data=df_weather_train,col='site_id')

# sns.relplot(x='timestamp',y='wind_speed',data=df_weather_train,col='site_id')
# Overall for all meters

print(df_train_merge.describe())

print('\n')

print('Timeseries data count per meter: \n')

print(df_train_merge.meter.value_counts(sort=False))

sns.heatmap(df_train_merge.corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)



# Overall for each meter

sns.catplot(x='meter',y='meter_reading',data=df_train_merge,kind='box',showfliers=False)



plt.figure(figsize=(50,10))

for i in range(4):

    plt.subplot(1,4,i+1)

    sns.heatmap(df_train_merge[df_train_merge.meter==i].corr(),vmin=-1,vmax=1,linewidths=.5,annot=True)

    

sns.catplot(x='meter_reading',y='primary_use',data=df_train_merge,col='meter',kind='bar',sharex=False)

sns.relplot(x='year_built',y='meter_reading',data=df_train_merge,hue='meter',kind='line',ci=None,aspect=5)

sns.catplot(y='meter_reading',x='site_id',data=df_train_merge,col='meter',kind='bar',sharey=False)
# Each meter per site

sns.catplot(x='meter_reading',y='primary_use',data=df_train_merge,col='meter',row='site_id',kind='bar',sharex=False)

sns.relplot(x='year_built',y='meter_reading',data=df_train_merge,hue='meter',kind='line',ci=None,aspect=5,row='site_id',facet_kws=dict(sharey=False))
# Timeseries for each meter

df_train_merge['timestamp'] = pd.to_datetime(df_train_merge['timestamp'])



plt.figure(figsize=[30,5])

plt.subplot(131)

sns.lineplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.month,'meter']).mean().reset_index(),ci=None,hue='meter').set_xlabel('Month')

plt.subplot(132)

sns.lineplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.week,'meter']).mean().reset_index(),ci=None,hue='meter').set_xlabel('Week')

plt.subplot(133)

sns.lineplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.dayofyear,'meter']).mean().reset_index(),ci=None,hue='meter').set_xlabel('Day of Year')



sns.relplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.dayofyear,'meter']).mean().reset_index(),

            ci=None,row='meter',kind='line',facet_kws=dict(sharey=False),aspect=4)

sns.relplot(y='meter_reading',x='timestamp', data=df_train_merge.groupby([df_train_merge['timestamp'].dt.dayofyear,'meter','site_id']).mean().reset_index(),

            ci=None,col='meter',row='site_id',kind='line',facet_kws=dict(sharey=False),aspect=4)



sns.relplot(x='timestamp',y='meter_reading',data=df_train_merge.set_index(['site_id','meter']).loc[(9,0)],kind='line',ci=None,aspect=4)
# Trend, seasonality and constant span for random meters

plotRandBuild(df_train,df_building_metadata)

plotRandBuild(df_train,df_building_metadata,meter=0)

plotRandBuild(df_train,df_building_metadata,site_id=6,meter=1)

plotRandBuild(df_train,df_building_metadata,site_id=13,meter=2)

plotRandBuild(df_train,df_building_metadata,site_id=15)