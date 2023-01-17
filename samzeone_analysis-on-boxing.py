#standard
import pandas as pd
import numpy as nm
#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option("display.max_columns",None)
df=pd.read_csv('../input/bouts_out_new.csv')
df.head()
df.info()
df.describe()
fil=((df.height_A<224)&(df.height_A>147)&(df.height_B<224)&(df.height_B>147)&(df.reach_A<250)&(df.reach_A>130)&(df.weight_A>70)
     &(df.weight_B>70)&(df.reach_B<250)&(df.reach_B>130)&(df.age_A<60)&(df.age_A>14)&(df.age_B<60)&(df.reach_B>14))
df=df[fil]
df.head()
df.info()
fig, ax= plt.subplots(4,2,figsize=(12,20))
sns.distplot(df.age_A,bins=20,ax=ax[0][0])
sns.distplot(df.age_B,bins=20,ax=ax[0][1])
sns.distplot(df.height_A,bins=20,ax=ax[1][0])
sns.distplot(df.height_B,bins=20,ax=ax[1][1])
sns.distplot(df.weight_A,bins=20,ax=ax[2][0])
sns.distplot(df.weight_B,bins=20,ax=ax[2][1])
sns.distplot(df.reach_A,bins=20,ax=ax[3][0])
sns.distplot(df.reach_B,bins=20,ax=ax[3][1])
df['Diff_age'] = df.age_A - df.age_B
df[['Diff_age','result']].groupby('result').mean()
g = sns.FacetGrid(df,hue='result',size=7)
g.map(plt.scatter,'age_A','age_B',edgecolor="w")
g.add_legend()
df['Diff_Height']=df.height_A-df.height_B
df[['Diff_Height','result']].groupby('result').mean()
g = sns.FacetGrid(df,hue='result',size=7)
g.map(plt.scatter,'height_A','height_B',edgecolor="w")
g.add_legend()

df['Reach']=df.reach_A-df.reach_B
df[['Reach','result']].groupby('result').mean()
g = sns.FacetGrid(df,hue='result',size=6)
g.map(plt.scatter,'reach_A','reach_B',edgecolor="w")
g.add_legend();
df['Total_Fight_A']=df.won_A+df.lost_A+df.drawn_A
df['Total_Fight_B']=df.won_B+df.lost_B+df.drawn_B
df['Exp']=df['Total_Fight_A']-df['Total_Fight_B']
df[['Exp','result']].groupby('result').mean()
df.Exp.hist(bins=100)
g = sns.FacetGrid(df,hue='result',size=6)
g.map(plt.scatter,'Total_Fight_A','Total_Fight_B',edgecolor="w")
g.add_legend()
df['Win_A']=df.won_A / df.Total_Fight_A
df.loc[df.Total_Fight_A==0,'Win_A']=0  #maybe it is the first fight
df['Win_B']=df.won_B / df.Total_Fight_B
df.loc[df.Total_Fight_B==0,'Win_B']=0
df['KOs_P_A']=df.kos_A / df.won_A
df.loc[df.Win_A==0,'KOs_P_A']=0
df['KOs_P_B']=df.kos_B / df.won_B
df.loc[df.Win_B==0,'KOs_P_B']=0
fig,ax=plt.subplots(1,2,figsize=(12,6))
sns.distplot(df.Win_A,bins=20,ax=ax[0])
sns.distplot(df.Win_B,bins=20,ax=ax[1])
df.KOs_P_A.hist(bins=30)
df.KOs_P_B.hist(bins=30)
df.loc[df.stance_A == df.stance_B,'Stance'] = 0
df.loc[(df.stance_A =='orhtodox') & (df.stance_B =='southpaw'),'Stance'] = 1
df.loc[(df.stance_B =='orthodox') & (df.stance_A =='southpaw'),'Stance'] = -1
pd.crosstab(df.Stance, df.result)
