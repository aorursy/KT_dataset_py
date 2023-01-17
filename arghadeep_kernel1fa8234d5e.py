import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df=pd.read_csv('/kaggle/input/covid19-community-mobility-dataset/community_dataset_with_infection_count.csv')
#countries with infected count against date
sns.lmplot(x='DATE_CT', y='CT', hue='COUNTRY_CODE_ISO',data=df)
corr = df.corr()
# Heatmap for the correlation values
sns.heatmap(corr,annot=True,cmap="YlGnBu")
#countries to maintain Mobility trends for places like grocery markets, food warehouses, farmers markets, specialty food shops, drug stores,and pharmacies.
sns.lmplot(x='DATE_CT', y='GROSSARY_AND_PHARMA', hue='COUNTRY_CODE_ISO',data=df,height=10)
#countries to maintain Mobility trends for places like public transport hubs such as subway, bus, and train stations.
sns.lmplot(x='DATE_CT', y='TRANSIT_STATIONS', hue='COUNTRY_CODE_ISO',data=df,height=10)
#countries to maintain Mobility trends for places like national parks,public beaches, marinas, dog parks, plazas,and public gardens
sns.lmplot(x='DATE_CT', y='PARKS', hue='COUNTRY_CODE_ISO',data=df,height=10)
#countries to maintain Mobility trends for places like restaurants,cafes, shopping centers, theme parks,museums, libraries, and movie theaters.
sns.lmplot(x='DATE_CT', y='RETAIL_AND_RECREATION', hue='COUNTRY_CODE_ISO',data=df,height=10)
#countries to maintain Mobility trends for places of work
sns.lmplot(x='DATE_CT', y='WORKPLACE', hue='COUNTRY_CODE_ISO',data=df,height=10)
#countries to maintain Mobility trends for places of residence
sns.lmplot(x='DATE_CT', y='RESIDENTIAL', hue='COUNTRY_CODE_ISO',data=df,height=10)

#Joint Distribution Plot
#how world count impacting for each countries count
sns.set(style="darkgrid")

df_ctry=df['COUNTRY_CODE_ISO'].unique()
for x in df_ctry:
    ctry_fl=df[df['COUNTRY_CODE_ISO']==x]
    g = sns.jointplot("CT", "WORLD_CT", data=ctry_fl,
                  kind="reg", truncate=False,
                  color="m", height=7)
    #sns.jointplot(x='CT', y='WORLD_CT', data=ctry_fl)
    plt.suptitle('Country Code '+x)
#IMPACT OF EACH MOBILITY FACTOR IN WORLD COUNT 
df_mob_fact=['GROSSARY_AND_PHARMA','RESIDENTIAL','TRANSIT_STATIONS','PARKS','RETAIL_AND_RECREATION','WORKPLACE']
for x in df_mob_fact:
    g = sns.jointplot("WORLD_CT",x,  data=df,
                  kind="reg", truncate=False,
                  color="m", height=7)
    #sns.jointplot(x='CT', y='WORLD_CT', data=ctry_fl)
    plt.suptitle('Mobility Factor '+x)
#IMPACT OF EACH MOBILITY FACTOR EACH COUNTRIES INFECTION COUNT 
for x in df_mob_fact:
    sns.lmplot(x=x, y='CT', hue='COUNTRY_CODE_ISO',data=df,height=7, truncate=False)
    plt.suptitle('Mobility Factor '+x)
ax = sns.scatterplot(x="DATE_CT", y="CT",
                     hue="COUNTRY_CODE_ISO",
                     data=df)
sns.set(style="whitegrid")
sns.residplot(df['DATE_CT'], df['WORLD_CT'], color="g")
sns.boxplot(data=df)
