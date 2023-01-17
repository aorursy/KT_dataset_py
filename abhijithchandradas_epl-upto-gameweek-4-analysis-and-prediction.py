import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import dataframe_image as dfi

from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/english-premier-leage-season-202021/EPL_GW4_standard.csv')
df_fix=pd.read_csv('../input/english-premier-leage-season-202021/EPL_fixture.csv')
#xG and xGA per match home and away 
df['xGpm_h']=df.xG_h/df.MP_h
df['xGpm_a']=df.xG_h/df.MP_a
df['xGApm_h']=df.xGA_h/df.MP_h
df['xGApm_a']=df.xGA_h/df.MP_a
#Aggregate data
df['MP']=df['MP_h']+df['MP_a']
df['xG']=df['xG_h']+df['xG_a']
df['xGA']=df['xGA_h']+df['xGA_a']
df['delta_xG']=df['xG']-df['xGA']
df['GF']=df.GF_a+df.GF_h
df['GA']=df.GA_a+df.GA_h
df['xaG']=df.GF-df.xG
df['xaGA']=df.xGA-df.GA
#Aggregate per match
df['xGpm']=df['xG']/df['MP']
df['xGApm']=df['xGA']/df['MP']
df['delta_xGpm']=df['delta_xG']/df['MP']
df['GFpm']=df.GF/df.MP
df['GApm']=df.GA/df.MP
df['xaGpm']=df.xaG/df.MP
df['xaGApm']=df.xaGA/df.MP
print("Total Matches : {}".format(df.MP_h.sum()))
print("Home Team Win : {}".format(df.W_h.sum()))
print("Away Team Win : {}".format(df.L_h.sum()))
print("Draw : {}".format(df.D_h.sum()))
x=[df.W_h.sum(), df.D_h.sum(), df.L_h.sum()]
labels=["Home Win", "Draw", "Away Win"]
explode=[0.01,0.01,0.01]

plt.pie(x=x, labels=labels,explode=explode, startangle=90,
        autopct='%1.2f%%',wedgeprops={"width":0.6})
plt.savefig('h_vs_a.png')
plt.show()
labels=["Home","Away"]
explode=[0.01,0.01]
plt.figure(figsize=(8,8))
plt.subplot(221)
plt.title("Goals Scored")
x=[sum(df.GF_h),sum(df.GF_a)]
plt.pie(x=x, labels=labels,explode=explode, startangle=90,
        autopct='%1.2f%%',wedgeprops={"width":0.6})

plt.subplot(222)
plt.title("Goals Conceded")
x=[sum(df.GA_h),sum(df.GA_a)]
plt.pie(x=x, labels=labels,explode=explode, startangle=90,
        autopct='%1.2f%%',wedgeprops={"width":0.6})

plt.subplot(223)
plt.title("xG Scored")
x=[sum(df.xG_h),sum(df.xG_a)]
plt.pie(x=x, labels=labels,explode=explode, startangle=90,
        autopct='%1.2f%%',wedgeprops={"width":0.6})

plt.subplot(224)
plt.title("xG Conceded")
x=[sum(df.xGA_h),sum(df.xGA_a)]
plt.pie(x=x, labels=labels,explode=explode, startangle=90,
        autopct='%1.2f%%',wedgeprops={"width":0.6})
plt.savefig('pie_xg.png')
plt.show()
print("Average xG per match(Home) : {}"
     .format(df.xGpm_h.mean()))
print("Average xG per match(Away) : {}"
     .format(df.xGpm_a.mean()))
print("Average xG Conceaded per match(Home) : {}"
     .format(round(df.xGApm_h.mean(),2)))
print("Average xG Conceaded per match(Away) : {}"
     .format(round(df.xGApm_a.mean(),2)))

plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x='xGApm_h', y='xGpm_h', 
                color='blue', label='Home', alpha=0.5)
sns.scatterplot(data=df,x='xGApm_a', y='xGpm_a', 
                color='red', label='Away', alpha=0.5)
plt.show()
df_agg=df[['Squad','MP', 'xG','xGA', 'delta_xG', 'xGpm', 'xGApm',
           'delta_xGpm']]
#df_agg.sort_values(by='delta_xGpm', ascending=False)
plt.figure(figsize=(8,5))
#plt.suptitle("EPL 2020/21 SEASON UPTO GW4")

plt.subplot(1,2,1)
plt.title("xG Scored per match")
sns.barplot(orient='h', x='xGpm',y='Squad',
            data=df_agg.sort_values(by='xGpm',ascending=False))
plt.grid(which='both', axis='x')

plt.subplot(1,2,2)
plt.title("xG Conceaded per match")
sns.barplot(orient='h', x='xGApm',y='Squad',
            data=df_agg.sort_values(by='xGApm',ascending=True))
plt.grid(which='both', axis='x')

plt.tight_layout()
plt.savefig('xg_xa.png')
plt.show()
plt.figure(figsize=(6,4))
plt.title("xG Scored - xG Conceded")
sns.barplot(orient='h', x='delta_xGpm',y='Squad', 
            data=df_agg.sort_values(by='delta_xGpm', ascending=False))
plt.grid(which='both', axis='x')
plt.xlabel('Delta xG')
plt.tight_layout()
plt.savefig('delta_xg.png')
plt.show()
plt.figure(figsize=(10,6))
plt.title("xG Scored Vs xG Conceded")
sns.scatterplot(data=df_agg, x='xGApm', y='xGpm')
for i in range(df_agg.shape[0]):
    plt.text(df_agg.xGApm[i]+0.01, df_agg.xGpm[i]+0.01, 
             df_agg.Squad[i], fontdict={'fontsize':8})
plt.xlabel("xG conceded Per match")
plt.ylabel("xG Scored Per match")
plt.plot([0,3],[0,3],'r--')
plt.xlim(df_agg.xGApm.min()-0.2,df_agg.xGApm.max()+0.2)
plt.ylim(df_agg.xGpm.min()-0.2,df_agg.xGpm.max()+0.2)
plt.savefig('scatter_xg_xa.png')
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x=df.xGpm, y=df.GFpm)
for i in range(df.shape[0]):
    plt.text(x=df.xGpm[i]+0.01, y=df.GFpm[i]+0.01, 
             s=df.Squad[i], fontsize=8)
plt.plot([-3,3],[-3,3],'r--')
plt.xlabel("Expected Goals per match")
plt.ylabel("Goals per match")
plt.xlim(df.xGpm.min()-0.2, df.xGpm.max()+0.2)
plt.ylim(df.GFpm.min()-0.2, df.GFpm.max()+0.2)
plt.tight_layout()
#plt.savefig('expvsact_g.png')
plt.show()
plt.figure(figsize=(4,5))
sns.barplot(orient='h', y='Squad',x='xaGpm', 
            data=df.sort_values(by='xaGpm', ascending=False))
plt.grid(which='both', axis='x')
#plt.savefig('expvsact_gbar.png')
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x=df.xGApm, y=df.GApm)
for i in range(df.shape[0]):
    plt.text(x=df.xGApm[i], y=df.GApm[i], 
             s=df.Squad[i], fontsize=7)
plt.plot([0,3],[0,3],'r--')
plt.xlabel('Expected Goals Conceded per match')
plt.ylabel('Goals conceded per match')
plt.tight_layout()
#plt.savefig('xaGA.png')
plt.show()
plt.figure(figsize=(4,5))
sns.barplot(orient='h', y='Squad',x='xaGApm', 
            data=df.sort_values(by='xaGApm', ascending=False))
plt.grid(which='both', axis='x')
#plt.savefig('xaGA_b.png')
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(x='xaGApm', y='xaGpm', data=df, hue='delta_xGpm',
                size='delta_xGpm')
plt.grid(which='both')
plt.legend()
plt.plot([5,-5],[0,0], 'k--')
plt.plot([0,0],[5,-5], 'k--')
plt.xlim(df.xaGApm.min()-0.2,df.xaGApm.max()+0.3)
plt.ylim(df.xaGpm.min()-0.3,df.xaGpm.max()+0.2)
for i in range(df.shape[0]):
    plt.text(x=df.xaGApm[i]+0.02, y=df.xaGpm[i]+0.02, 
             s=df.Squad[i], fontsize=7)
    
plt.text(x=0.1, y=1, s="Q1\nOverperformed xG\nOverperformed xA", 
         alpha=0.7,fontsize=9, color='red')
plt.text(x=0.1, y=-0.5, s="Q4\nUnder performed xG\nOverperformed xA", 
         alpha=0.7,fontsize=9, color='red')
plt.text(x=-1, y=1, s="Q2\nOverperformed xG\nUnderperformed xA", 
         alpha=0.7,fontsize=9, color='red')
plt.text(x=-1, y=-0.5, s="Q3\nUnder performed xG\nUnderperformed xA", 
         alpha=0.7,fontsize=9, color='red')
plt.xlabel("Goals Coceded(Expected-Actual) per match")
plt.ylabel("Goals Scored(Actual-Expected) per match")
plt.tight_layout()
#plt.savefig('op_up.png')
plt.show()
df_fix['G_home']=0.0
df_fix['G_away']=0.0

for i in range(df_fix.shape[0]):
    df_fix.G_home[i]=(df_agg.xGpm[df_agg.Squad==df_fix.Home[i]].sum()+
                   df_agg.xGApm[df_agg.Squad==df_fix.Away[i]].sum())/2
    df_fix.G_away[i]=(df_agg.xGpm[df_agg.Squad==df_fix.Away[i]].sum()+
                   df_agg.xGApm[df_agg.Squad==df_fix.Home[i]].sum())/2

df_fix['GD']=df_fix['G_home']-df_fix['G_away']
df_fix['GS']=df_fix['G_home']+df_fix['G_away']
df_fix=df_fix.sort_values(by='GD', ascending=False)
df_styled=df_fix.iloc[:,1:][df_fix.GW==5].style.background_gradient(cmap='RdYlGn',subset=['GD','GS']).hide_index()
#dfi.export(df_styled,"mytable.png")
df_styled
gw_dict={'GW5':5,'GW6':6,'GW7':7,'GW8':8}

df_fdr=pd.DataFrame({'Squad':df.Squad})

for GW in gw_dict.keys():
    df_temp=df_fix[df_fix.GW==gw_dict[GW]]

    df_fdr[GW]=df_fdr.Squad\
    .apply(lambda x:(df_temp[df_temp.Home==x].GD.sum()) 
           if x in (df_temp.Home.unique()) 
           else -df_temp[df_temp.Away==x].GD.sum())
sc=MinMaxScaler()
df_fdr['Mean']=df_fdr.mean(axis=1)
for col in gw_dict.keys():
    df_fdr[col]=sc.fit_transform(np.array(df_fdr[col]).reshape(-1,1))
df_fdr=df_fdr.sort_values(by='Mean', ascending=False)
df_fdr.style.background_gradient(cmap='RdYlGn',
                                 subset=list(gw_dict.keys()))
