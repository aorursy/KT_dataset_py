import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import json

from pandas.io.json import json_normalize
df = pd.read_csv('../input/ted_main.csv')

# Filtering out a talk that is mostly a video

df=df[df["event"]!="TED-Ed"]

df.head()
df['num_speaker_cap'] = "1"

df.loc[df.num_speaker>1,'num_speaker_cap']=">1"



df.groupby('num_speaker_cap').size()
58/2492*100
df['type']="non-tech"

df['speaker_occupation'].fillna("",inplace=True)

a=["tech","research","computer","science","Computer scientist", "Researcher", "inventor", "Engineer", "Inventor", "Researcher", "futurist", "Data", "Science", "Biologist", "biologist", "Technologist", "Neuroscientist", "psychologist", "Neuroscientist", "Ecologist", "health", "psychologist", "Health"]

#a=["tech","research","computer","science","Computer scientist", "Researcher", "Engineer", "Researcher", "futurist", "Data", "Science", "Biologist", "biologist", "Technologist", "Neuroscientist", "psychologist", "Neuroscientist", "Ecologist", "health", "psychologist", "Health"]

df.loc[df.speaker_occupation.str.contains("|".join(a)),'type']="tech"



df.groupby(['type','num_speaker_cap']).size()
m=df.groupby(['type','num_speaker_cap']).size().unstack()

(m.div(m.sum(axis=1),axis=0)*100).round()
g = sns.boxplot(x="num_speaker_cap", y="views", data=df, palette="PRGn")

g.set(ylim=(0, 4000000))
viewdiff=df.groupby('num_speaker_cap').agg({'views':np.median})

viewdiff
stats.mannwhitneyu(df[df.num_speaker>1].views,df[df.num_speaker==1].views)
#(1.707530e+06-1.301605e+06)/1.707530e+06*100

(1131452.5-873904.0)/1131452.5*100
g = sns.boxplot(x="num_speaker_cap", y="views", data=df[df['type']=="tech"], palette="PRGn")

g.set(ylim=(0, 4000000))
df_t=df[df['type']=="tech"]

stats.mannwhitneyu(df_t[df_t.num_speaker>1].views,df_t[df_t.num_speaker==1].views)
df['ratings']=df['ratings'].str.replace("'",'"')

pd.read_json(df['ratings'].iloc[1])[['name','count']]
df=df.merge(df.ratings.apply(lambda x: pd.Series(pd.read_json(x)['count'].values,index=pd.read_json(x)['name'])), 

    left_index=True, right_index=True)

#normalize by Views

for i in range(19,33):

    df.iloc[:,i]=df.iloc[:,i]/df["views"]
# get a copy for the tech talks

df_a=df.copy()

df_t=df[df['type']=="tech"].copy()

for i in range(19,33):

    df_a.iloc[:,i]=(df_a.iloc[:,i] - df_a.iloc[:,i].mean())/df_a.iloc[:,i].std()

    df_t.iloc[:,i]=(df_t.iloc[:,i] - df_t.iloc[:,i].mean())/df_t.iloc[:,i].std()
l_a=df_a[['title', 'num_speaker_cap', 'type', 'main_speaker', 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)

l_a[:10]
l_a_c=df_a[df_a.num_speaker>1][['title', 'num_speaker_cap', 'type', 'main_speaker',"speaker_occupation", 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)

l_a_c[:10]
l_t=df_t[['title', 'num_speaker_cap', 'type', 'main_speaker',"speaker_occupation", 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)

l_t[:10]
l_t_c=df_t[df_t.num_speaker>1][['title', 'num_speaker_cap', 'type', 'main_speaker',"speaker_occupation", 'views', 'Longwinded']].sort_values('Longwinded', ascending=False)

l_t_c[:10]
#df_plot=df.iloc[:,[6,12,14,16,17,18]+list(range(19,33))]

df_a_plot=df_a.iloc[:,[17,18]+list(range(19,33))].melt(id_vars=['num_speaker_cap','type'])

df_t_plot=df_t.iloc[:,[17,18]+list(range(19,33))].melt(id_vars=['num_speaker_cap','type'])

df_a_plot.iloc[list(range(1,10))]
def annotate(g,sign):

    for i in range(0,len(g.axes)):

        col=sns.xkcd_rgb["pale red"]

        if sign[i]>0.05:

            col=sns.xkcd_rgb["denim blue"]

        # significance

        g.axes[i].text(0.1,-2,'p-value='+"{:.2e}".format(sign[i]), color=col)

        #delta

        #g.axes[i].text(0.1,600,'delta='+"{:.2e}".format(delta[i]), color=col)
sign_a=[]

for i in range(19,33):

    sign_a.append(stats.mannwhitneyu(df_a[df_a.num_speaker>1].iloc[:,i],df_a[df_a.num_speaker==1].iloc[:,i]).pvalue*(33-19))

sign_a
g = sns.FacetGrid(df_a_plot, col="variable",  col_wrap=5)

g = g.map(sns.boxplot, "num_speaker_cap", "value")

g.set(ylim=(-2.5, 2))

g.set(xlabel='speaker number', ylabel='normalized votes')

annotate(g,sign_a)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('All talks')
sign_t=[]

for i in range(19,33):

    sign_t.append(stats.mannwhitneyu(df_t[df_t.num_speaker>1].iloc[:,i],df_t[df_t.num_speaker==1].iloc[:,i]).pvalue*(33-19))

sign_t
g = sns.FacetGrid(df_t_plot[df_t_plot.type=="tech"], col="variable",  col_wrap=5)

g = g.map(sns.boxplot, "num_speaker_cap", "value")

g.set(ylim=(-2.5, 2))

g.set(xlabel='speaker number', ylabel='normalized votes')

annotate(g,sign_t)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Tech talks')
def posneg (df):

    df["pos"]=(df["Funny"]+df["Beautiful"]+df["Ingenious"]+df["Courageous"]+df["Informative"]+df["Fascinating"]+df["Persuasive"]+df["Jaw-dropping"]+df["Inspiring"])/9

    df["neg"]=(df["Longwinded"]+df["Confusing"]+df["Unconvincing"]+df["Obnoxious"]/4)

    #zscore

    #df["pos"]=(df["pos"] - df["pos"].mean())/df["pos"].std()

    #df["neg"]=(df["neg"] - df["neg"].mean())/df["neg"].std()

    df["pos_vs_neg"]=df["pos"]-df["neg"]

    #df["pos_vs_neg"]=(df["pos_vs_neg"] - df["pos_vs_neg"].mean())/df["pos_vs_neg"].std()

#Significance

posneg(df_a)

p_a=stats.mannwhitneyu(df_a[df_a.num_speaker>1].pos_vs_neg,df_a[df_a.num_speaker==1].pos_vs_neg).pvalue

print("Mann Whitney ransum p-value: {:.2f}".format(p_a))
g = sns.boxplot(x="num_speaker_cap", y="pos_vs_neg", data=df_a, palette="PRGn")

g.set_title("All talks")

g.set(ylim=(-4, +4))

g.axes.text(0.35,-3,'p-value='+"{:.2f}".format(p_a))

g.set(xlabel='speaker number', ylabel='"Goodness" score')
posneg(df_t)

p_t=stats.mannwhitneyu(df_t[df_t.num_speaker>1].pos_vs_neg,df_t[df_t.num_speaker==1].pos_vs_neg).pvalue

p_t
g = sns.boxplot(x="num_speaker_cap", y="pos_vs_neg", data=df_t, palette="PRGn")

g.set_title("Tech talks")

g.set(ylim=(-4, +4))

g.axes.text(0.35,-3,'p-value='+"{:.2f}".format(p_t))

g.set(xlabel='speaker number', ylabel='"Goodness" score')
j_a=df_a[df_a.num_speaker>1][['title', 'num_speaker', 'type', 'main_speaker', 'views', 'pos_vs_neg']].sort_values('pos_vs_neg', ascending=False)

j_a[:10]
g=sns.lmplot(x="Jaw-dropping", y="Beautiful",data=df_a, hue="num_speaker_cap", fit_reg=False)

g.set(xscale="symlog", yscale="symlog")

g.set(ylim=(-1, 10))
g=sns.lmplot(x="Jaw-dropping", y="Beautiful",data=df_t, hue="num_speaker_cap", fit_reg=False)

g.set(xscale="symlog", yscale="symlog")

g.set(ylim=(-1, 10));
top=10

top_j=df_t[['event','title','main_speaker','num_speaker_cap', 'Jaw-dropping','Beautiful']].sort_values('Jaw-dropping', ascending=False)[:top]

top_f=df_t[['event','title','main_speaker','num_speaker_cap', 'Jaw-dropping','Beautiful']].sort_values('Beautiful', ascending=False)[:top]



top_jf=pd.concat([top_j,top_f])

top_jf.drop_duplicates(inplace=True)

top_jf
g=sns.lmplot(x="Jaw-dropping", y="Beautiful",data=top_jf, hue="num_speaker_cap", fit_reg=False)

#g.set(xscale="symlog", yscale="symlog")