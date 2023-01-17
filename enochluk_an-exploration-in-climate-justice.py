import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import geopandas as gpd
damage = pd.read_csv("../input/houstondamage/DR-4332_TX_-_Housing_Damage_by_Block_Group.csv")
demographics = pd.read_csv("../input/city-of-houston-hurricane-harvey-damage-assessment/Harvey_BG.csv")
svi=pd.read_csv("../input/texas-social-vulnerability-by-census-tract/Texas.csv")
demographics.head()
dropped=demographics.drop(columns=['Geography','State','Shape_Leng','Shape_Leng','Count_Af_1'])
dropped['county_name']=dropped.County.replace(to_replace = [157,201,339],value = ['Fort Bend','Harris','Montgomery'])
dropped['Percent_White']=pd.Series(dropped['NH_White']/dropped['Pop_Total'])
dropped['Percent_Black']=pd.Series(dropped['NH_Black']/dropped['Pop_Total'])
dropped['Percent_AmInd']=pd.Series(dropped['NH_AmInd_A']/dropped['Pop_Total'])
dropped['Percent_Asian']=pd.Series(dropped['NH_Asian']/dropped['Pop_Total'])
dropped['Percent_Native']=pd.Series(dropped['NH_Native_']/dropped['Pop_Total'])
dropped['Percent_Mixed']=pd.Series(dropped['NH_Two_Mor']/dropped['Pop_Total'])
dropped['Percent_Hispanic']=pd.Series(dropped['Hispanic']/dropped['Pop_Total'])
dropped['Percent_Child']=pd.Series(dropped['Children_U']/dropped['Pop_Total'])
dropped['Percent_Senior']=pd.Series(dropped['SeniorCiti']/dropped['Pop_Total'])
dropped['PopDensity']=pd.Series(dropped['Pop_Total']/dropped['Shape_Area'])


dropped
dropped=dropped.drop(columns=['NH_White','NH_Black','NH_AmInd_A','NH_Asian','NH_Native_','NH_Some_Ot','NH_Two_Mor','Hispanic'])
dropped
dropped['Percent_Aff']=dropped['Count_Affe']/dropped['Housing_Un']
corr=dropped.corr()
plt.figure(figsize=(10,10))
thing = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
        square=True,
)
thing.set_xticklabels(labels=thing.get_xticklabels(),rotation=80)
sns.scatterplot(x='MHI_ACS',y='Percent_Aff',hue='county_name',data=dropped)

plt.figure(figsize=(5,10))
sns.heatmap(dropped.drop(columns=['Count_Affe','BLKGRP','Vacant','County','SeniorCiti','Tract']).corr()[['Percent_Aff']].sort_values(by=['Percent_Aff'],ascending=False),
            vmin=-1,
            cmap='coolwarm',
            annot=True);


bycounty = dropped.groupby(['county_name'])

thing=bycounty.Count_Affe.agg(sum)
print(thing)
data=pd.DataFrame(thing).reset_index()
print(data)
sns.barplot(x='county_name',y='Count_Affe',data=data)
zipper=[]
for thing in dropped['MHI_ACS']:
    zipper.append(thing//20000 * 20000)
demographics['IncomeBracket']=zipper
brack=demographics.groupby('IncomeBracket').agg(sum)
sns.barplot(x=brack.index,y='NH_Black',data=brack)

chart=sns.barplot(x=brack.index,y='Count_Affe',data=brack)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

sns.barplot(x=brack.index,y='Count_Affe',data=brack)

blipper=[]
for thing in dropped['Percent_Aff']:
    blipper.append(thing>0.5)
demographics["Maj_Aff"]=blipper
demographics.groupby('Maj_Aff').agg(sum)
brack
brack = brack[["Hispanic","NH_White","NH_Black","NH_Asian"]]

brack.iloc[0].plot.bar()

brack.iloc[1].plot.bar()
dropped

pog=[]
for thing in svi.LOCATION:
    string=thing
    pog.append(int(float(string[string.index("Tract")+6:string.index(",")])*100))

svi['Tract']=pog
result=pd.merge(dropped,svi,on='Tract')

plt.figure(figsize=(5,40))
sns.heatmap(result.drop(columns=['Count_Affe','STCNTY','FIPS']).corr()[['Percent_Aff']].sort_values(by=['Percent_Aff'],ascending=False),
            vmin=-1,
            cmap='coolwarm',
            annot=True)

demographics=demographics.rename(columns={'Tract':'TRACT'})

result=pd.merge(damage,demographics,on='TRACT')
result=result.drop(columns=['FID','STATE','GEOID','STUSAB','Shape_Leng','Households','Occupied'])
result
result['Percent_Aff']=result['Count_Affe']/result['Housing_Un']